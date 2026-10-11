/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * MFCC in plain C float: periodic Hann window, 512-point radix-2 complex FFT
 * (real input in the real part, so bins 0..256 are the spectrum), 40 mel
 * triangles built from bin centre frequencies, natural log, orthonormal DCT-II.
 * The same recipe as train/features.py; the device checks the two agree at boot.
 */

#include "mfcc.h"

#include <math.h>
#include <string.h>

#define NBINS (MFCC_NFFT / 2 + 1)
#define PI_F  3.14159265358979f

static float win[MFCC_FRAME_LEN];
static float twr[MFCC_NFFT / 2], twi[MFCC_NFFT / 2];
static uint16_t bitrev[MFCC_NFFT];
/* Each mel filter is non-zero on a contiguous bin range: store start, count, weights. */
static uint16_t mel_start[MFCC_N_MEL], mel_count[MFCC_N_MEL];
static float mel_w[MFCC_N_MEL][24];
static float dct[MFCC_N_COEF][MFCC_N_MEL];

static float re[MFCC_NFFT], im[MFCC_NFFT];

static float hz_to_mel(float f)
{
	return 2595.0f * log10f(1.0f + f / 700.0f);
}

static float mel_to_hz(float m)
{
	return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
}

void mfcc_init(void)
{
	for (int n = 0; n < MFCC_FRAME_LEN; n++) {
		win[n] = 0.5f - 0.5f * cosf(2.0f * PI_F * (float)n / (float)MFCC_FRAME_LEN);
	}
	for (int k = 0; k < MFCC_NFFT / 2; k++) {
		twr[k] = cosf(-2.0f * PI_F * (float)k / (float)MFCC_NFFT);
		twi[k] = sinf(-2.0f * PI_F * (float)k / (float)MFCC_NFFT);
	}
	for (int i = 0; i < MFCC_NFFT; i++) {
		unsigned r = 0, v = (unsigned)i;
		for (int b = 0; b < 9; b++) { /* log2(512) */
			r = (r << 1) | (v & 1u);
			v >>= 1;
		}
		bitrev[i] = (uint16_t)r;
	}

	float edges[MFCC_N_MEL + 2];
	float mlo = hz_to_mel(20.0f), mhi = hz_to_mel(4000.0f);
	for (int m = 0; m < MFCC_N_MEL + 2; m++) {
		edges[m] = mel_to_hz(mlo + (mhi - mlo) * (float)m / (float)(MFCC_N_MEL + 1));
	}
	for (int m = 0; m < MFCC_N_MEL; m++) {
		float lo = edges[m], c = edges[m + 1], hi = edges[m + 2];
		int first = -1, count = 0;
		for (int k = 0; k < NBINS; k++) {
			float f = (float)k * ((float)MFCC_SAMPLE_RATE / (float)MFCC_NFFT);
			float up = (f - lo) / (c - lo);
			float down = (hi - f) / (hi - c);
			float w = up < down ? up : down;
			if (w > 1.0f) {
				w = 1.0f;
			}
			if (w > 0.0f) {
				if (first < 0) {
					first = k;
				}
				if (count < 24) {
					mel_w[m][count++] = w;
				}
			} else if (first >= 0) {
				break;
			}
		}
		mel_start[m] = (uint16_t)(first < 0 ? 0 : first);
		mel_count[m] = (uint16_t)count;
	}

	for (int i = 0; i < MFCC_N_COEF; i++) {
		float scale = sqrtf(2.0f / (float)MFCC_N_MEL) * (i == 0 ? 1.0f / sqrtf(2.0f) : 1.0f);
		for (int j = 0; j < MFCC_N_MEL; j++) {
			dct[i][j] = scale * cosf(PI_F / (float)MFCC_N_MEL * ((float)j + 0.5f) * (float)i);
		}
	}
}

static void fft512(void)
{
	for (int i = 0; i < MFCC_NFFT; i++) {
		int j = bitrev[i];
		if (j > i) {
			float t = re[i];
			re[i] = re[j];
			re[j] = t;
			t = im[i];
			im[i] = im[j];
			im[j] = t;
		}
	}
	for (int len = 2; len <= MFCC_NFFT; len <<= 1) {
		int half = len >> 1;
		int step = MFCC_NFFT / len;
		for (int i = 0; i < MFCC_NFFT; i += len) {
			for (int k = 0; k < half; k++) {
				float wr = twr[k * step], wi = twi[k * step];
				int a = i + k, b = i + k + half;
				float xr = re[b] * wr - im[b] * wi;
				float xi = re[b] * wi + im[b] * wr;
				re[b] = re[a] - xr;
				im[b] = im[a] - xi;
				re[a] += xr;
				im[a] += xi;
			}
		}
	}
}

static float dc_offset;

void mfcc_set_dc(const int16_t *audio, size_t n)
{
	int64_t acc = 0;
	for (size_t i = 0; i < n; i++) {
		acc += audio[i];
	}
	dc_offset = n ? (float)acc / (float)n : 0.0f;
}

void mfcc_power_spectrum(const int16_t *audio, float *out)
{
	for (int n = 0; n < MFCC_FRAME_LEN; n++) {
		re[n] = ((float)audio[n] - dc_offset) * (1.0f / 32768.0f) * win[n];
		im[n] = 0.0f;
	}
	memset(&re[MFCC_FRAME_LEN], 0, sizeof(float) * (MFCC_NFFT - MFCC_FRAME_LEN));
	memset(&im[MFCC_FRAME_LEN], 0, sizeof(float) * (MFCC_NFFT - MFCC_FRAME_LEN));
	fft512();
	for (int k = 0; k < NBINS; k++) {
		out[k] = re[k] * re[k] + im[k] * im[k];
	}
}

void mfcc_compute(const int16_t *audio, float *out)
{
	static float power[NBINS];
	static float logmel[MFCC_N_MEL];

	mfcc_set_dc(audio, MFCC_WINDOW_SAMPLES);
	for (int t = 0; t < MFCC_FRAMES; t++) {
		mfcc_power_spectrum(&audio[t * MFCC_HOP], power);
		for (int m = 0; m < MFCC_N_MEL; m++) {
			float e = 0.0f;
			const float *w = mel_w[m];
			const float *p = &power[mel_start[m]];
			for (int k = 0; k < mel_count[m]; k++) {
				e += w[k] * p[k];
			}
			logmel[m] = logf(e + 1e-6f);
		}
		float *o = &out[t * MFCC_N_COEF];
		for (int i = 0; i < MFCC_N_COEF; i++) {
			float c = 0.0f;
			for (int j = 0; j < MFCC_N_MEL; j++) {
				c += dct[i][j] * logmel[j];
			}
			o[i] = c;
		}
	}
}
