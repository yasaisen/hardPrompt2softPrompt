"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2505051557
"""

import torch
import torch.nn as nn
import os

from ...common.utils import log_print, get_trainable_params


class ValueHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
    def forward(self, hs):
        # hs: [B, L, D]
        return self.net(hs).squeeze(-1)  # [B, L]
    











    