/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * MFCC front end matching train/features.py. See that file for the numbers.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MFCC_SAMPLE_RATE 16000
#define MFCC_FRAME_LEN   480
#define MFCC_HOP         320
#define MFCC_NFFT        512
#define MFCC_N_MEL       40
#define MFCC_N_COEF      10
#define MFCC_FRAMES      49
#define MFCC_WINDOW_SAMPLES (MFCC_HOP * (MFCC_FRAMES - 1) + MFCC_FRAME_LEN)

/* Build the window, twiddles, mel filterbank and DCT tables. Call once. */
void mfcc_init(void);

/* audio: at least MFCC_WINDOW_SAMPLES int16 samples.
 * out: MFCC_FRAMES * MFCC_N_COEF floats, frame-major. */
void mfcc_compute(const int16_t *audio, float *out);

/* The window mean that mfcc_power_spectrum() subtracts (PDM mics have a DC offset);
 * mfcc_compute() sets it from the whole window. */
void mfcc_set_dc(const int16_t *audio, size_t n);

/* Power spectrum of one frame starting at audio[0]; out: MFCC_NFFT/2 + 1 floats.
 * Exposed for the microphone debug mode (peak-frequency readout). */
void mfcc_power_spectrum(const int16_t *audio, float *out);

#ifdef __cplusplus
}
#endif
