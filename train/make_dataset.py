#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Build the "hey zephyr" training set: synthetic positives from piper TTS, confuser
phrases from the same voices, real speech negatives from Google Speech Commands,
and noise/silence clips. Everything is 16 kHz, 16-bit mono, exactly one second.

  python make_dataset.py --voices C:/ws/data/piper-voices --speech-commands C:/ws/data/speech_commands

Output (default train/data/hey_zephyr/):
  positive/NNNNN.wav   "hey zephyr" (label 2)
  negative/NNNNN.wav   confusers + Speech Commands words (label 1)
  silence/NNNNN.wav    room noise, pink/white noise, digital silence (label 0)
  manifest.csv         path,label,label_name,source,voice,speaker,text

Augmentation on every clip: random placement in the window, gain, background
noise at a random SNR, an occasional one-pole low-pass for "far from the mic".
Needs the TTS venv:  pip install piper-tts soundfile numpy   (piper 1.8+ API)
"""

import argparse
import csv
import json
import pathlib
import random
import wave

import numpy as np

try:
    from piper import PiperVoice, SynthesisConfig
except ImportError:  # the training venv does not need piper, only this script does
    PiperVoice = None

SR = 16000
CLIP = SR

POSITIVE_TEXTS = ["hey zephyr", "hey, zephyr", "Hey Zephyr!", "hey zephyr.", "Hey, Zephyr?"]
CONFUSER_TEXTS = [
    "hey jeffrey", "hey there", "hey jasper", "hey google", "hey siri", "hey zebra", "hey sephora",
    "hey chef", "hey jenny", "hey zed", "hey", "zephyr", "a zephyr", "say zephyr", "the zephyr blew",
    "okay zephyr", "hi zephyr", "hey zephyr's friend", "jeffrey", "safari", "zephyrus", "hey yes",
    "turn on the lights", "what time is it", "play some music", "hello", "good morning", "stop",
    "it is raining today", "set a timer for five minutes", "how are you", "thanks", "no thanks",
    "yes please", "turn it off", "hey you", "hey man", "hey mum", "hey dad", "heyday",
]


def write_wav(path: pathlib.Path, x: np.ndarray) -> None:
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_wav(path: pathlib.Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    if sr != SR:
        x = resample(x, sr, SR)
    return x


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Low-pass (windowed sinc) then linear interpolation. Good enough below 4 kHz."""
    if sr_in == sr_out:
        return x
    if sr_out < sr_in:
        cutoff = 0.45 * sr_out / sr_in  # normalised to sr_in
        n = 64
        k = np.arange(-n, n + 1)
        h = 2 * cutoff * np.sinc(2 * cutoff * k) * np.hanning(2 * n + 1)
        h /= h.sum()
        x = np.convolve(x, h, mode="same")
    t_out = np.arange(int(len(x) * sr_out / sr_in)) * (sr_in / sr_out)
    return np.interp(t_out, np.arange(len(x)), x)


def trim(x: np.ndarray, thresh_db: float = -40.0) -> np.ndarray:
    """Cut leading/trailing near-silence."""
    if len(x) == 0:
        return x
    env = np.abs(x)
    peak = env.max() + 1e-9
    thr = peak * 10 ** (thresh_db / 20)
    idx = np.where(env > thr)[0]
    if len(idx) == 0:
        return x
    a, b = max(0, idx[0] - 400), min(len(x), idx[-1] + 400)
    return x[a:b]


def fit_window(x: np.ndarray, rng: random.Random, max_len: float = 0.96) -> np.ndarray:
    """Place a clip (<= max_len s, time-compressed if longer) at a random offset in 1 s."""
    max_n = int(max_len * SR)
    if len(x) > max_n:
        x = np.interp(np.linspace(0, len(x) - 1, max_n), np.arange(len(x)), x)
    out = np.zeros(CLIP)
    off = rng.randint(0, CLIP - len(x))
    out[off:off + len(x)] = x
    return out


class Noise:
    def __init__(self, sc_root: pathlib.Path):
        self.clips = []
        d = sc_root / "_background_noise_"
        if d.exists():
            for p in sorted(d.glob("*.wav")):
                self.clips.append((p.stem, read_wav(p)))

    def segment(self, rng: random.Random, n: int = CLIP):
        if not self.clips:
            return "white", rng_white(rng, n)
        name, x = rng.choice(self.clips)
        if len(x) <= n:
            return name, np.resize(x, n)
        a = rng.randint(0, len(x) - n)
        return name, x[a:a + n]


def rng_white(rng: random.Random, n: int) -> np.ndarray:
    g = np.random.default_rng(rng.randint(0, 2 ** 31))
    return g.standard_normal(n) * 0.05


def augment(x: np.ndarray, noise: Noise, rng: random.Random, snr_range=(5.0, 30.0)) -> np.ndarray:
    x = x * rng.uniform(0.3, 1.0)
    if rng.random() < 0.3:  # one-pole low-pass: "across the room"
        a = np.exp(-2 * np.pi * rng.uniform(2000, 5000) / SR)
        y = np.zeros_like(x)
        acc = 0.0
        for i in range(len(x)):
            acc = (1 - a) * x[i] + a * acc
            y[i] = acc
        x = y
    if rng.random() < 0.75:
        _, nz = noise.segment(rng)
        sig_p = np.mean(x ** 2) + 1e-9
        nz_p = np.mean(nz ** 2) + 1e-9
        snr = rng.uniform(*snr_range)
        x = x + nz * np.sqrt(sig_p / (nz_p * 10 ** (snr / 10)))
    peak = np.abs(x).max()
    if peak > 0.98:
        x = x * (0.98 / peak)
    return x


class Voices:
    def __init__(self, root: pathlib.Path):
        self.voices = []
        for onnx in sorted(root.rglob("*.onnx")):
            cfg = json.loads(onnx.with_suffix(".onnx.json").read_text(encoding="utf-8"))
            self.voices.append((onnx.stem, PiperVoice.load(str(onnx)), int(cfg.get("num_speakers", 1))))
        if not self.voices:
            raise SystemExit(f"no piper voices under {root}")
        print("voices:", ", ".join(f"{n} ({s} spk)" for n, _, s in self.voices))

    def say(self, text: str, rng: random.Random):
        name, voice, n_spk = rng.choice(self.voices)
        spk = rng.randrange(n_spk) if n_spk > 1 else None
        cfg = SynthesisConfig(speaker_id=spk, length_scale=rng.uniform(0.8, 1.25),
                              noise_scale=rng.uniform(0.4, 0.8), noise_w_scale=rng.uniform(0.5, 1.0))
        chunks = list(voice.synthesize(text, syn_config=cfg))
        audio = np.concatenate([c.audio_float_array.astype(np.float64) for c in chunks])
        sr = chunks[0].sample_rate
        return name, spk, trim(resample(audio, sr, SR))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voices", type=pathlib.Path, required=True)
    ap.add_argument("--speech-commands", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(__file__).parent / "data" / "hey_zephyr")
    ap.add_argument("--n-pos", type=int, default=3000)
    ap.add_argument("--n-neg-tts", type=int, default=1500)
    ap.add_argument("--n-neg-sc", type=int, default=4500)
    ap.add_argument("--n-silence", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = args.out
    for d in ("positive", "negative", "silence"):
        (out / d).mkdir(parents=True, exist_ok=True)
    noise = Noise(args.speech_commands)
    voices = Voices(args.voices)
    rows = []

    def add(kind, label, label_name, source, voice, speaker, text, x):
        i = sum(1 for r in rows if r[0].startswith(kind))
        p = out / kind / f"{i:05d}.wav"
        write_wav(p, x)
        rows.append((f"{kind}/{p.name}", label, label_name, source, voice, speaker if speaker is not None else "", text))

    print(f"positives: {args.n_pos}")
    for i in range(args.n_pos):
        text = rng.choice(POSITIVE_TEXTS)
        voice, spk, x = voices.say(text, rng)
        add("positive", 2, "hey_zephyr", "tts", voice, spk, text, augment(fit_window(x, rng), noise, rng))
        if i % 200 == 0:
            print(f"  {i}", flush=True)

    print(f"tts confusers: {args.n_neg_tts}")
    for i in range(args.n_neg_tts):
        text = rng.choice(CONFUSER_TEXTS)
        voice, spk, x = voices.say(text, rng)
        x = fit_window(x, rng) if len(x) <= int(1.5 * SR) else fit_window(x[: int(1.2 * SR)], rng)
        add("negative", 1, "_unknown_", "tts", voice, spk, text, augment(x, noise, rng))
        if i % 200 == 0:
            print(f"  {i}", flush=True)

    words = sorted(p for p in args.speech_commands.iterdir() if p.is_dir() and not p.name.startswith("_"))
    files = [p for w in words for p in w.glob("*.wav")]
    rng.shuffle(files)
    print(f"speech commands negatives: {args.n_neg_sc} of {len(files)} clips in {len(words)} words")
    for i, p in enumerate(files[: args.n_neg_sc]):
        x = read_wav(p)
        add("negative", 1, "_unknown_", "speech_commands", p.parent.name, "", p.parent.name,
            augment(fit_window(trim(x), rng), noise, rng))

    print(f"silence/noise: {args.n_silence}")
    for i in range(args.n_silence):
        r = rng.random()
        if r < 0.15:
            x, src = np.zeros(CLIP), "digital_silence"
        elif r < 0.30:
            x, src = rng_white(rng, CLIP) * rng.uniform(0.02, 1.0), "white"
        else:
            src, x = noise.segment(rng)
            x = x * rng.uniform(0.05, 1.0)
        add("silence", 0, "_silence_", src, "", "", "", x)

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "label_name", "source", "voice", "speaker", "text"])
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r[2]] = counts.get(r[2], 0) + 1
    print("done:", counts, "->", out / "manifest.csv")


if __name__ == "__main__":
    main()
