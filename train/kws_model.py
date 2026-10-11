# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Depthwise-separable CNN for "hey zephyr" spotting on 49x10 MFCC frames.

Only ops with portable ExecuTorch kernels (conv2d, depthwise conv2d, relu,
mean, linear), so the .pte runs on the ESP32's Xtensa LX6 with the selective
build and no delegate. CH and BLOCKS set the speed/accuracy trade-off; the
firmware prints the inference time at boot so you can tune them.
"""

import torch
from torch import nn

from features import FRAMES, N_COEF

LABELS = ["_silence_", "_unknown_", "hey_zephyr"]
WAKE_LABEL = "hey_zephyr"
CH = 16
BLOCKS = 2


class DSConv(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return torch.relu(self.pw(torch.relu(self.dw(x))))


class KwsNet(nn.Module):
    def __init__(self, ch: int = CH, blocks: int = BLOCKS, n_labels: int = len(LABELS)):
        super().__init__()
        self.stem = nn.Conv2d(1, ch, (10, 4), stride=(2, 2), padding=(5, 1))
        self.blocks = nn.Sequential(*[DSConv(ch) for _ in range(blocks)])
        self.head = nn.Linear(ch, n_labels)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)


def example_input():
    return (torch.zeros(1, 1, FRAMES, N_COEF),)
