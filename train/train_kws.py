#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Train KwsNet on the clips from make_dataset.py and save train/out/kws.pt.

  python train_kws.py                      # uses train/data/hey_zephyr/manifest.csv
  python export_kws.py --checkpoint out/kws.pt

Features are computed once with features.py (the same recipe as the device)
and cached in out/features.npz. The checkpoint carries the per-coefficient
mean and std used for normalisation, which export_kws.py bakes into the
firmware header. Reports accuracy and, at the firmware threshold, the false
reject and false accept rates on the held-out split.
"""

import argparse
import csv
import json
import pathlib
import time
import wave

import numpy as np
import torch
from torch import nn

from features import FRAMES, N_COEF, mfcc
from kws_model import LABELS, KwsNet

HERE = pathlib.Path(__file__).resolve().parent


def load_clip(path: pathlib.Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if len(x) < 16000:
        x = np.pad(x, (0, 16000 - len(x)))
    return x[:16000]


def build_features(manifest: pathlib.Path, cache: pathlib.Path):
    if cache.exists() and cache.stat().st_mtime > manifest.stat().st_mtime:
        z = np.load(cache)
        return z["x"], z["y"], z["src"]
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    xs = np.zeros((len(rows), FRAMES, N_COEF), dtype=np.float32)
    ys = np.zeros(len(rows), dtype=np.int64)
    src = np.array([r["source"] for r in rows])
    t0 = time.time()
    for i, r in enumerate(rows):
        xs[i] = mfcc(load_clip(manifest.parent / r["path"]))
        ys[i] = int(r["label"])
        if i % 1000 == 0:
            print(f"  features {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=xs, y=ys, src=src)
    return xs, ys, src


def spec_augment(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Small time shift and a couple of masked coefficients, per batch."""
    shift = int(torch.randint(-2, 3, (1,), generator=rng))
    if shift:
        x = torch.roll(x, shifts=shift, dims=2)
    if float(torch.rand(1, generator=rng)) < 0.5:
        c = int(torch.randint(0, N_COEF, (1,), generator=rng))
        x = x.clone()
        x[:, :, :, c] = 0.0
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=pathlib.Path, default=HERE / "data" / "hey_zephyr" / "manifest.csv")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "out")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threshold", type=float, default=0.85, help="firmware wake threshold for FRR/FAR")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    x, y, src = build_features(args.manifest, args.out / "features.npz")
    print(f"{len(y)} clips; labels {np.bincount(y).tolist()} ({LABELS})")

    # stratified 85/15 split
    idx_tr, idx_va = [], []
    g = np.random.default_rng(args.seed)
    for c in range(len(LABELS)):
        idx = np.where(y == c)[0]
        g.shuffle(idx)
        n_va = max(1, int(0.15 * len(idx)))
        idx_va += idx[:n_va].tolist()
        idx_tr += idx[n_va:].tolist()
    idx_tr, idx_va = np.array(idx_tr), np.array(idx_va)

    mean = x[idx_tr].reshape(-1, N_COEF).mean(0)
    std = x[idx_tr].reshape(-1, N_COEF).std(0) + 1e-3
    xn = (x - mean) / std
    xt = torch.from_numpy(xn)[:, None]  # (N, 1, 49, 10)
    yt = torch.from_numpy(y)

    model = KwsNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"KwsNet {n_params} parameters; train {len(idx_tr)} / val {len(idx_va)}")
    weights = torch.tensor([1.0, 1.0, 1.5])
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        perm = idx_tr[torch.randperm(len(idx_tr), generator=rng).numpy()]
        tot, correct, n = 0.0, 0, 0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            xb = spec_augment(xt[b], rng)
            yb = yt[b]
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
            correct += int((out.argmax(1) == yb).sum())
            n += len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            out = model(xt[idx_va])
            acc = float((out.argmax(1) == yt[idx_va]).float().mean())
        print(f"epoch {epoch + 1}/{args.epochs}: loss {tot / n:.4f} train acc {correct / n:.4f} val acc {acc:.4f}")
        if acc > best:
            best = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(xt[idx_va]), 1)[:, LABELS.index("hey_zephyr")].numpy()
    yv = y[idx_va]
    wake = yv == LABELS.index("hey_zephyr")
    frr = float((p[wake] < args.threshold).mean())
    far = float((p[~wake] >= args.threshold).mean())
    far_tts = float((p[(~wake) & (src[idx_va] == "tts")] >= args.threshold).mean())
    metrics = {"val_accuracy": round(best, 4), "frr_at_threshold": round(frr, 4), "far_at_threshold": round(far, 4),
               "far_confusers_at_threshold": round(far_tts, 4), "threshold": args.threshold,
               "n_train": int(len(idx_tr)), "n_val": int(len(idx_va)), "params": n_params}
    print("held-out:", json.dumps(metrics))

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feat_mean": mean.tolist(), "feat_std": std.tolist(),
                "labels": LABELS, "metrics": metrics}, args.out / "kws.pt")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {args.out / 'kws.pt'}")


if __name__ == "__main__":
    main()
