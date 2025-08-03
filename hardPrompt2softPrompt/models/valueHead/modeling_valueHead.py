"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2505091525
"""

import torch
import torch.nn as nn
import os

from ...common.utils import log_print, get_trainable_params


class ValueHead(nn.Module):
    def __init__(self, 
        d_model: int,
        device: str="cuda", 
    ):
        super().__init__()
        self.state_name = 'ValueHead'
        self.device = device
        print()
        log_print(self.state_name, f"Building...")

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

        log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self)}")
        self.to(self.device)
        log_print(self.state_name, f"...Done\n")

    def forward(self, hs): # hs: [B, L, D]
        output = self.net(hs).squeeze(-1)  # [B, L]

        return output
    











    