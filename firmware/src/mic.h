/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * The ATOM Echo's SPM1423 PDM microphone through the ESP32's I2S0 PDM-to-PCM
 * path: 16 kHz, 16-bit mono into a ring buffer that the keyword spotter reads
 * one-second windows from.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mic_stats {
	uint32_t words_per_s;   /* 32-bit DMA words per second, measured */
	uint32_t samples_per_s; /* picked samples per second, measured */
	uint32_t overruns;
	int pick;               /* 0 low halfword, 1 high halfword, 2 every halfword */
	float rms_lo, rms_hi;   /* RMS of each halfword stream, last block */
};

int mic_init(void);

/* Copy the newest n samples (oldest first). Returns the number copied. */
size_t mic_read_latest(int16_t *out, size_t n);

/* Total samples captured so far (for "has a new hop arrived" checks). */
uint32_t mic_sample_count(void);

void mic_get_stats(struct mic_stats *out);

#ifdef __cplusplus
}
#endif
