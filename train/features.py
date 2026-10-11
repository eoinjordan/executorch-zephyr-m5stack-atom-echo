# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""The MFCC front end, in numpy, written to match firmware/src/mfcc.c step for step.

One second of 16 kHz audio -> 49 frames x 10 coefficients:

  frame   30 ms (480 samples), hop 20 ms (320), periodic Hann window
  FFT     512 points, power spectrum |X|^2 on bins 0..256
  DC      the mean of the one-second window is subtracted first (PDM mics sit on an offset)
  mel     40 triangular filters, 20 Hz to 4 kHz, HTK mel scale, unit peak,
          weights computed from bin centre frequencies (no integer bin edges,
          so float32 on the device and float64 here land on the same filters)
  log     ln(E + 1e-6)
  DCT-II  orthonormal, coefficients 0..9

Both sides are checked against each other on the device: the firmware embeds
the clip from test_audio() and this module's output for it (features_test.h)
and reports the maximum difference at boot.
"""

import numpy as np

SAMPLE_RATE = 16000
FRAME_LEN = 480
HOP = 320
NFFT = 512
N_MEL = 40
N_COEF = 10
FRAMES = 49
F_MIN = 20.0
F_MAX = 4000.0
LOG_EPS = 1e-6
WINDOW_SAMPLES = HOP * (FRAMES - 1) + FRAME_LEN  # 15840; the device keeps 16000


def hann(n: int = FRAME_LEN) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """(N_MEL, NFFT//2 + 1) triangular weights, unit peak."""
    edges = mel_to_hz(np.linspace(hz_to_mel(F_MIN), hz_to_mel(F_MAX), N_MEL + 2))
    f_bins = np.arange(NFFT // 2 + 1) * (SAMPLE_RATE / NFFT)
    fb = np.zeros((N_MEL, NFFT // 2 + 1))
    for m in range(N_MEL):
        lo, c, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (f_bins - lo) / (c - lo)
        down = (hi - f_bins) / (hi - c)
        fb[m] = np.clip(np.minimum(up, down), 0.0, 1.0)
    return fb


def dct_matrix() -> np.ndarray:
    """(N_COEF, N_MEL) orthonormal DCT-II."""
    j = np.arange(N_MEL)
    d = np.zeros((N_COEF, N_MEL))
    for i in range(N_COEF):
        d[i] = np.sqrt(2.0 / N_MEL) * np.cos(np.pi / N_MEL * (j + 0.5) * i)
    d[0] *= 1.0 / np.sqrt(2.0)
    return d


_WIN = hann()
_FB = mel_filterbank()
_DCT = dct_matrix()


def mfcc(audio_int16: np.ndarray) -> np.ndarray:
    """int16 samples (>= WINDOW_SAMPLES) -> (FRAMES, N_COEF) float32."""
    x = np.asarray(audio_int16, dtype=np.float64)
    x = (x - x[:WINDOW_SAMPLES].mean()) / 32768.0  # DC removal, as mfcc.c does over the window
    out = np.zeros((FRAMES, N_COEF), dtype=np.float64)
    for t in range(FRAMES):
        frame = x[t * HOP:t * HOP + FRAME_LEN] * _WIN
        spec = np.fft.rfft(frame, NFFT)
        power = spec.real ** 2 + spec.imag ** 2
        mel_e = _FB @ power
        out[t] = _DCT @ np.log(mel_e + LOG_EPS)
    return out.astype(np.float32)


def test_audio() -> np.ndarray:
    """A deterministic one-second clip for the device self-test: a chirp, a
    gated 1 kHz tone and a little seeded noise, so no mel band is empty."""
    n = SAMPLE_RATE
    t = np.arange(n) / SAMPLE_RATE
    f0, f1 = 200.0, 3500.0
    chirp = 0.30 * np.sin(2.0 * np.pi * (f0 * t + (f1 - f0) / (2.0 * t[-1]) * t * t))
    gate = ((t > 0.35) & (t < 0.65)).astype(np.float64)
    tone = 0.20 * gate * np.sin(2.0 * np.pi * 1000.0 * t)
    rng = np.random.default_rng(2026)
    noise = 0.02 * rng.standard_normal(n)
    y = np.clip(chirp + tone + noise, -0.999, 0.999)
    return (y * 32767.0).astype(np.int16)


if __name__ == "__main__":
    a = test_audio()
    f = mfcc(a)
    print("test clip", a.shape, "mfcc", f.shape, "c0 range", float(f[:, 0].min()), float(f[:, 0].max()))
