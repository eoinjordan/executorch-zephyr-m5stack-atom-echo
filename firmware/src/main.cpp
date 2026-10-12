/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * "Hey Zephyr" on the M5Stack ATOM Echo: Zephyr + ExecuTorch wake-word spotter.
 *
 *   SPM1423 PDM mic --I2S0--> 16 kHz ring --MFCC 49x10--> ExecuTorch KwsNet --> p(wake)
 *                                                                |
 *                        SK6812 LED (blue listening, green wake) <+--> JSON lines on UART0
 *                                                                     (linux/ha_bridge.py -> Home Assistant)
 *
 * Boot self-tests, no human needed:
 *   1. the C MFCC of an embedded clip against the Python MFCC of the same clip
 *   2. the model on those features against the PyTorch logits
 * Then the live loop: every APP_INFER_PERIOD_MS take the newest second of
 * audio, run the net, and fire a wake event after APP_WAKE_CONSECUTIVE
 * windows above APP_WAKE_THRESHOLD_PERCENT.
 */

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <zephyr/drivers/gpio.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/version.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "features_test.h"
#include "kws_test_input.h"
#include "led.h"
#include "mfcc.h"
#include "mic.h"
#include "model_pte.h"

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;

static_assert(KWS_FRAMES == MFCC_FRAMES && KWS_COEFFS == MFCC_N_COEF, "feature shape mismatch");
static_assert(FEAT_TEST_SAMPLES >= MFCC_WINDOW_SAMPLES, "test clip too short");

#ifndef APP_METHOD_ALLOCATOR_POOL_SIZE
#define APP_METHOD_ALLOCATOR_POOL_SIZE (48 * 1024)
#endif
#ifndef APP_TEMP_ALLOCATOR_POOL_SIZE
#define APP_TEMP_ALLOCATOR_POOL_SIZE (8 * 1024)
#endif

alignas(16) static uint8_t method_pool[APP_METHOD_ALLOCATOR_POOL_SIZE];
alignas(16) static uint8_t temp_pool[APP_TEMP_ALLOCATOR_POOL_SIZE];

static const struct gpio_dt_spec button = GPIO_DT_SPEC_GET_OR(DT_ALIAS(sw0), gpios, {0});

namespace {

struct Runner {
	BufferDataLoader loader{model_pte, model_pte_len};
	MemoryAllocator method_allocator{APP_METHOD_ALLOCATOR_POOL_SIZE, method_pool};
	MemoryAllocator temp_allocator{APP_TEMP_ALLOCATOR_POOL_SIZE, temp_pool};
	std::vector<Span<uint8_t>> planned_spans;
	Program *program = nullptr;
	Method *method = nullptr;
	float *input = nullptr;
	size_t input_numel = 0;
	Tensor *input_tensor = nullptr;

	Error init();
	Error run(const float *features, float *logits, uint32_t *us);
};

Error Runner::init()
{
	Result<Program> program_res = Program::load(&loader);
	if (!program_res.ok()) {
		ET_LOG(Error, "Program::load failed: 0x%x", (unsigned)program_res.error());
		return program_res.error();
	}
	program = new Program(std::move(program_res.get()));

	const char *method_name = program->get_method_name(0).get();
	Result<MethodMeta> meta = program->method_meta(method_name);
	if (!meta.ok()) {
		return meta.error();
	}
	for (size_t id = 0; id < meta->num_memory_planned_buffers(); ++id) {
		size_t size = static_cast<size_t>(meta->memory_planned_buffer_size(id).get());
		uint8_t *buf = static_cast<uint8_t *>(method_allocator.allocate(size, 16));
		if (buf == nullptr) {
			ET_LOG(Error, "planned buffer %zu (%zu B) does not fit the method pool", id, size);
			return Error::MemoryAllocationFailed;
		}
		printk("planned buffer %u: %u bytes\n", (unsigned)id, (unsigned)size);
		planned_spans.push_back({buf, size});
	}
	static HierarchicalAllocator planned_memory({planned_spans.data(), planned_spans.size()});
	static MemoryManager memory_manager(&method_allocator, &planned_memory, &temp_allocator);

	Result<Method> method_res = program->load_method(method_name, &memory_manager);
	if (!method_res.ok()) {
		ET_LOG(Error, "load_method(%s) failed: 0x%x", method_name, (unsigned)method_res.error());
		return method_res.error();
	}
	method = new Method(std::move(method_res.get()));

	Result<TensorInfo> in_meta = meta->input_tensor_meta(0);
	if (!in_meta.ok()) {
		return in_meta.error();
	}
	input_numel = in_meta->nbytes() / sizeof(float);
	input = static_cast<float *>(method_allocator.allocate(in_meta->nbytes(), 16));
	if (input == nullptr) {
		return Error::MemoryAllocationFailed;
	}
	static TensorImpl impl(
		in_meta->scalar_type(), in_meta->sizes().size(),
		const_cast<TensorImpl::SizesType *>(in_meta->sizes().data()), input,
		const_cast<TensorImpl::DimOrderType *>(in_meta->dim_order().data()));
	static Tensor tensor(&impl);
	input_tensor = &tensor;

	printk("method '%s' loaded; input numel %u; pool used %u / %u bytes\n", method_name,
	       (unsigned)input_numel, (unsigned)method_allocator.used_size(),
	       (unsigned)method_allocator.size());
	return Error::Ok;
}

Error Runner::run(const float *features, float *logits, uint32_t *us)
{
	std::memcpy(input, features, input_numel * sizeof(float));
	Error err = method->set_input(EValue(*input_tensor), 0);
	if (err != Error::Ok) {
		return err;
	}
	uint32_t t0 = k_cycle_get_32();
	err = method->execute();
	uint32_t t1 = k_cycle_get_32();
	if (err != Error::Ok) {
		return err;
	}
	*us = k_cyc_to_us_floor32(t1 - t0);
	const EValue &out = method->get_output(0);
	if (!out.isTensor()) {
		return Error::InvalidState;
	}
	const float *data = out.toTensor().const_data_ptr<float>();
	for (size_t i = 0; i < KWS_LABELS; ++i) {
		logits[i] = data[i];
	}
	return Error::Ok;
}

static float features[KWS_FRAMES * KWS_COEFFS];
static int16_t window[MFCC_WINDOW_SAMPLES];

void normalise(float *f)
{
	for (int t = 0; t < KWS_FRAMES; t++) {
		for (int c = 0; c < KWS_COEFFS; c++) {
			f[t * KWS_COEFFS + c] = (f[t * KWS_COEFFS + c] - kFeatMean[c]) / kFeatStd[c];
		}
	}
}

void softmax(const float *logits, float *p)
{
	float m = logits[0];
	for (int i = 1; i < KWS_LABELS; i++) {
		m = fmaxf(m, logits[i]);
	}
	float s = 0.f;
	for (int i = 0; i < KWS_LABELS; i++) {
		p[i] = expf(logits[i] - m);
		s += p[i];
	}
	for (int i = 0; i < KWS_LABELS; i++) {
		p[i] /= s;
	}
}

bool self_test(Runner &runner)
{
	/* 1. front end */
	uint32_t t0 = k_cycle_get_32();
	mfcc_compute(kTestAudio, features);
	uint32_t mfcc_us = k_cyc_to_us_floor32(k_cycle_get_32() - t0);
	float fdiff = 0.f;
	for (int i = 0; i < KWS_FRAMES * KWS_COEFFS; i++) {
		fdiff = fmaxf(fdiff, fabsf(features[i] - kTestMfcc[i]));
	}
	bool feat_ok = fdiff < 2e-2f;
	printk("  mfcc : 49 frames in %u us, max|c-python|=%.2e %s\n", mfcc_us, (double)fdiff,
	       feat_ok ? "ok" : "MISMATCH");

	/* 2. model on the Python-normalised features (the exact test input) */
	float logits[KWS_LABELS], p[KWS_LABELS];
	uint32_t us = 0;
	if (runner.run(kTestInput, logits, &us) != Error::Ok) {
		printk("  kws  : inference failed\n");
		return false;
	}
	float ldiff = 0.f;
	int pred = 0;
	printk("  kws  : logits [");
	for (int i = 0; i < KWS_LABELS; i++) {
		printk("%s%.4f", i ? " " : "", (double)logits[i]);
		ldiff = fmaxf(ldiff, fabsf(logits[i] - kTestLogits[i]));
		if (logits[i] > logits[pred]) {
			pred = i;
		}
	}
	softmax(logits, p);
	printk("] pred=%d (%s) expected=%d max|esp32-pytorch|=%.1e (%u us)\n", pred, kKwsLabels[pred],
	       KWS_TEST_ARGMAX, (double)ldiff, us);
	bool model_ok = pred == KWS_TEST_ARGMAX && ldiff < 1e-3f;

	/* 3. the whole chain: C features, normalised here, through the model */
	normalise(features);
	if (runner.run(features, logits, &us) == Error::Ok) {
		float cdiff = 0.f;
		for (int i = 0; i < KWS_LABELS; i++) {
			cdiff = fmaxf(cdiff, fabsf(logits[i] - kTestLogits[i]));
		}
		printk("  chain: C mfcc -> model, max|logit diff|=%.1e\n", (double)cdiff);
	}
	return feat_ok && model_ok;
}

} // namespace

int main(void)
{
	executorch::runtime::runtime_init();
	mfcc_init();
	led_init();

	printk("\nHey Zephyr on %s (Zephyr %s): ExecuTorch wake word\n", CONFIG_BOARD_TARGET,
	       KERNEL_VERSION_STRING);
	printk("model: %u bytes in flash, method pool %u B, temp pool %u B\n",
	       (unsigned)model_pte_len, (unsigned)APP_METHOD_ALLOCATOR_POOL_SIZE,
	       (unsigned)APP_TEMP_ALLOCATOR_POOL_SIZE);

	static Runner runner;
	if (runner.init() != Error::Ok) {
		printk("FAIL: runtime init\n");
		led_set(LED_ERROR);
		return 1;
	}

	printk("\nself-test:\n");
	bool ok = self_test(runner);
	printk(ok ? "SELF-TEST PASS\n" : "SELF-TEST FAIL\n");
	led_set(ok ? LED_LISTENING : LED_ERROR);

	bool have_button = button.port != nullptr && gpio_is_ready_dt(&button) &&
			   gpio_pin_configure_dt(&button, GPIO_INPUT) == 0;

	if (mic_init() != 0) {
		printk("mic: init failed; staying in self-test mode\n");
		led_set(LED_ERROR);
		for (;;) {
			k_msleep(1000);
		}
	}

	uint32_t seq = 0, above = 0;
	int64_t last_wake = -100000, last_beat = 0, last_stats = 0;
	bool muted = false, btn_was = false;
	float logits[KWS_LABELS], p[KWS_LABELS];

	for (;;) {
		k_msleep(CONFIG_APP_INFER_PERIOD_MS);
		int64_t now = k_uptime_get();

		if (have_button) {
			bool btn = gpio_pin_get_dt(&button) > 0;
			if (btn && !btn_was) {
				muted = !muted;
				led_set(muted ? LED_MUTED : LED_LISTENING);
				printk("{\"seq\":%u,\"t_ms\":%lld,\"muted\":%d}\n", seq++, (long long)now, muted);
			}
			btn_was = btn;
		}

#ifdef CONFIG_APP_MIC_DEBUG
		if (now - last_stats >= 1000) {
			struct mic_stats st;
			mic_get_stats(&st);
			static float power[MFCC_NFFT / 2 + 1];
			int peak = 2;
			if (mic_read_latest(window, MFCC_FRAME_LEN) == MFCC_FRAME_LEN) {
				mfcc_set_dc(window, MFCC_FRAME_LEN);
				mfcc_power_spectrum(window, power);
				for (int k = 2; k < MFCC_NFFT / 2; k++) {
					if (power[k] > power[peak]) {
						peak = k;
					}
				}
			}
			printk("mic: %u words/s %u samples/s pick %d rms lo %.0f hi %.0f peak %d Hz overruns %u\n",
			       st.words_per_s, st.samples_per_s, st.pick, (double)st.rms_lo,
			       (double)st.rms_hi, peak * MFCC_SAMPLE_RATE / MFCC_NFFT, st.overruns);
			if ((now / 1000) % 5 == 0) {
				printk("mic: raw:");
				for (int i = 0; i < 32; i++) {
					printk(" %d", window[i]);
				}
				printk("\n");
			}
			last_stats = now;
		}
#endif

		if (muted || mic_sample_count() < MFCC_WINDOW_SAMPLES) {
			continue;
		}
		if (mic_read_latest(window, MFCC_WINDOW_SAMPLES) != MFCC_WINDOW_SAMPLES) {
			continue;
		}
		mfcc_compute(window, features);
		normalise(features);
		uint32_t us = 0;
		if (runner.run(features, logits, &us) != Error::Ok) {
			printk("kws: inference failed\n");
			continue;
		}
		softmax(logits, p);
		float pw = p[KWS_WAKE_INDEX];
		int pred = 0;
		for (int i = 1; i < KWS_LABELS; i++) {
			if (p[i] > p[pred]) {
				pred = i;
			}
		}

		if (pw * 100.f >= CONFIG_APP_WAKE_THRESHOLD_PERCENT) {
			above++;
		} else {
			above = 0;
		}
		bool holdoff = now - last_wake < CONFIG_APP_WAKE_HOLDOFF_MS;
		if (above >= CONFIG_APP_WAKE_CONSECUTIVE && !holdoff) {
			last_wake = now;
			above = 0;
			led_set(LED_WAKE);
			printk("{\"seq\":%u,\"t_ms\":%lld,\"wake\":1,\"label\":\"%s\",\"p\":%.3f,\"infer_us\":%u}\n",
			       seq++, (long long)now, kKwsLabels[KWS_WAKE_INDEX], (double)pw, us);
		} else if (now - last_wake >= 1000 && now - last_wake < 1000 + CONFIG_APP_INFER_PERIOD_MS) {
			led_set(LED_LISTENING);
		}

		if (now - last_beat >= CONFIG_APP_HEARTBEAT_MS) {
			last_beat = now;
			printk("{\"seq\":%u,\"t_ms\":%lld,\"wake\":0,\"top\":\"%s\",\"p_wake\":%.3f,\"infer_us\":%u,\"samples\":%u}\n",
			       seq++, (long long)now, kKwsLabels[pred], (double)pw, us,
			       (unsigned)mic_sample_count());
		}
	}
	return 0;
}
