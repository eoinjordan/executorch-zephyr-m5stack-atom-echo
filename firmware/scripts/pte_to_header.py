#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Convert an ExecuTorch .pte into a C header with a 16-byte aligned const array.

The array lands in .rodata, which on the ESP32-S3 is memory-mapped flash, so
the model costs no SRAM beyond the runtime's planned buffers.
"""

import argparse
import pathlib


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pte", required=True, type=pathlib.Path)
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.add_argument("--symbol", default="model_pte")
    args = p.parse_args()

    data = args.pte.read_bytes()
    lines = [
        f"/* Generated from {args.pte.name} ({len(data)} bytes). Do not edit. */",
        "#pragma once",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        f"static const size_t {args.symbol}_len = {len(data)}u;",
        f"static const uint8_t __attribute__((aligned(16))) {args.symbol}[{len(data)}] = {{",
    ]
    for i in range(0, len(data), 24):
        lines.append("    " + ", ".join(f"0x{b:02x}" for b in data[i : i + 24]) + ",")
    lines += ["};", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
