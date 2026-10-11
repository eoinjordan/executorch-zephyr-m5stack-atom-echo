/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * PDM microphone capture on the ESP32 (hardware v1 I2S) under Zephyr.
 *
 * Zephyr's I2S driver has no PDM mode, so the trick is: configure it as a
 * standard 16-bit stereo master at 32 kHz, which puts the bit clock at
 * 32000 x 32 = 1.024 MHz, exactly the 64 x 16 kHz PDM clock the SPM1423
 * wants, then flip I2S0 into PDM-to-PCM receive mode with the ESP-IDF HAL
 * before starting it. The clock comes out on the RX word-select pin (GPIO33)
 * and the PDM data goes in on GPIO23; that is how ESP-IDF wires PDM RX too.
 *
 * What lands in the DMA buffers depends on the silicon's mono packing, so the
 * first half second is spent measuring the word rate and picking the halfword
 * stream that gives 16 kHz. The choice and the rates are printed; the
 * APP_MIC_PICK Kconfig option can pin it.
 */

#include "mic.h"

#include <zephyr/device.h>
#include <zephyr/drivers/i2s.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <hal/i2s_ll.h>
#include <soc/i2s_struct.h>

#include <math.h>
#include <string.h>

#define I2S_NODE DT_NODELABEL(i2s0)

#define FRAME_HZ    CONFIG_APP_MIC_I2S_FRAME_HZ /* 32000: bck = 1.024 MHz */
#define BLOCK_WORDS 512                         /* 32-bit words per DMA block */
#define BLOCK_BYTES (BLOCK_WORDS * 4)
#define BLOCK_COUNT 6
#define RING_SAMPLES (MIC_RING_SAMPLES)

K_MEM_SLAB_DEFINE_STATIC(mic_slab, BLOCK_BYTES, BLOCK_COUNT, 4);

static const struct device *const i2s = DEVICE_DT_GET(I2S_NODE);

static int16_t ring[RING_SAMPLES];
static size_t ring_head; /* next write index */
static uint32_t total_samples;
static struct k_spinlock lock;

static struct mic_stats stats;
static int pick = CONFIG_APP_MIC_PICK; /* -1 = auto */

K_THREAD_STACK_DEFINE(mic_stack, 3072);
static struct k_thread mic_thread;

static void push_samples(const int16_t *s, size_t n)
{
	k_spinlock_key_t key = k_spin_lock(&lock);
	for (size_t i = 0; i < n; i++) {
		ring[ring_head] = s[i];
		ring_head = (ring_head + 1) % RING_SAMPLES;
	}
	total_samples += n;
	k_spin_unlock(&lock, key);
}

static float rms_of(const int16_t *s, size_t n, size_t stride)
{
	double acc = 0.0;
	size_t cnt = 0;
	for (size_t i = 0; i < n; i += stride) {
		acc += (double)s[i] * (double)s[i];
		cnt++;
	}
	return cnt ? (float)sqrt(acc / (double)cnt) : 0.0f;
}

static void mic_thread_fn(void *a, void *b, void *c)
{
	static int16_t tmp[BLOCK_WORDS * 2];
	int64_t t0 = k_uptime_get();
	uint32_t words = 0, picked = 0;
	int64_t last_rate = t0;

	for (;;) {
		void *block;
		size_t size;
		int ret = i2s_read(i2s, &block, &size);
		if (ret != 0) {
			/* An RX overrun parks the Zephyr I2S stream in its error state and
			 * every read fails from then on: drop the queue and start again. */
			stats.overruns++;
			i2s_trigger(i2s, I2S_DIR_RX, I2S_TRIGGER_DROP);
			k_msleep(2);
			if (i2s_trigger(i2s, I2S_DIR_RX, I2S_TRIGGER_START) != 0) {
				k_msleep(20);
			}
			continue;
		}
		const int16_t *h = block;
		size_t nwords = size / 4;
		words += nwords;

		stats.rms_lo = rms_of(h, nwords * 2, 2);
		stats.rms_hi = rms_of(h + 1, nwords * 2 - 1, 2);

		int64_t now = k_uptime_get();
		if (pick < 0 && now - t0 >= 500) {
			/* ~16000 words/s: each word is one 16 kHz frame, take the louder half.
			 * ~8000 words/s: two mono samples per word, take every halfword. */
			uint32_t wps = (uint32_t)(words * 1000 / (uint32_t)(now - t0));
			if (wps < 12000) {
				pick = 2;
			} else {
				pick = stats.rms_hi > stats.rms_lo * 1.5f ? 1 : 0;
			}
			printk("mic: %u DMA words/s in the first %lld ms -> pick mode %d\n", wps,
			       (long long)(now - t0), pick);
		}
		if (pick >= 0) {
			size_t n = 0;
			if (pick == 2) {
				memcpy(tmp, h, nwords * 4);
				n = nwords * 2;
			} else {
				for (size_t i = 0; i < nwords; i++) {
					tmp[i] = h[2 * i + pick];
				}
				n = nwords;
			}
			push_samples(tmp, n);
			picked += n;
		}
		k_mem_slab_free(&mic_slab, block);

		if (now - last_rate >= 1000) {
			stats.words_per_s = (uint32_t)(words * 1000 / (uint32_t)(now - last_rate));
			stats.samples_per_s = (uint32_t)(picked * 1000 / (uint32_t)(now - last_rate));
			stats.pick = pick;
			words = 0;
			picked = 0;
			last_rate = now;
		}
	}
}

int mic_init(void)
{
	if (!device_is_ready(i2s)) {
		printk("mic: i2s0 not ready\n");
		return -ENODEV;
	}

	struct i2s_config cfg = {
		.word_size = 16,
		.channels = 2,
		.format = I2S_FMT_DATA_FORMAT_I2S,
		.options = I2S_OPT_FRAME_CLK_MASTER | I2S_OPT_BIT_CLK_MASTER,
		.frame_clk_freq = FRAME_HZ,
		.mem_slab = &mic_slab,
		.block_size = BLOCK_BYTES,
		.timeout = 1000,
	};
	int ret = i2s_configure(i2s, I2S_DIR_RX, &cfg);
	if (ret != 0) {
		printk("mic: i2s_configure failed (%d)\n", ret);
		return ret;
	}

	/* Standard mode is configured; now make the receiver a PDM-to-PCM one.
	 * Mirrors i2s_hal_pdm_set_rx_slot() / i2s_hal_pdm_enable_rx_channel()
	 * in ESP-IDF for the ESP32. */
	i2s_dev_t *hw = &I2S0;
	i2s_ll_rx_set_sample_bit(hw, 16, 16);
	i2s_ll_rx_enable_mono_mode(hw, true);
	i2s_ll_rx_select_pdm_slot(hw, I2S_PDM_SLOT_LEFT);
	i2s_ll_rx_force_enable_fifo_mod(hw, true);
	i2s_ll_rx_enable_msb_right(hw, false);
	i2s_ll_rx_enable_right_first(hw, false);
	i2s_ll_rx_enable_msb_shift(hw, false);
	i2s_ll_rx_set_pdm_dsr(hw, I2S_PDM_DSR_8S);
	i2s_ll_rx_enable_pdm(hw, true);

	ret = i2s_trigger(i2s, I2S_DIR_RX, I2S_TRIGGER_START);
	if (ret != 0) {
		printk("mic: i2s start failed (%d)\n", ret);
		return ret;
	}

	k_thread_create(&mic_thread, mic_stack, K_THREAD_STACK_SIZEOF(mic_stack), mic_thread_fn, NULL,
			NULL, NULL, K_PRIO_COOP(4), 0, K_NO_WAIT);
	k_thread_name_set(&mic_thread, "mic");
	printk("mic: SPM1423 PDM on I2S0 (clk GPIO33, data GPIO23), bck %u Hz, pick %s\n",
	       FRAME_HZ * 32, pick < 0 ? "auto" : "fixed");
	return 0;
}

size_t mic_read_latest(int16_t *out, size_t n)
{
	if (n > RING_SAMPLES) {
		n = RING_SAMPLES;
	}
	k_spinlock_key_t key = k_spin_lock(&lock);
	size_t avail = total_samples < RING_SAMPLES ? (size_t)total_samples : RING_SAMPLES;
	if (n > avail) {
		n = avail;
	}
	size_t start = (ring_head + RING_SAMPLES - n) % RING_SAMPLES;
	for (size_t i = 0; i < n; i++) {
		out[i] = ring[(start + i) % RING_SAMPLES];
	}
	k_spin_unlock(&lock, key);
	return n;
}

uint32_t mic_sample_count(void)
{
	k_spinlock_key_t key = k_spin_lock(&lock);
	uint32_t n = total_samples;
	k_spin_unlock(&lock, key);
	return n;
}

void mic_get_stats(struct mic_stats *out)
{
	*out = stats;
	out->pick = pick;
}
