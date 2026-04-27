#!/usr/bin/env python3
"""Per-column PWL sag model (baseline). Primary sag only."""

import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn
from piecewise import FixedPWL


class Model(nn.Module):
    """Per-column piecewise linear model of same-side row sum.

    y_ij ≈ P_j(s_i)  where P_j is a per-column PWL with shared breakpoints.
    """

    def __init__(self, b, s, **kw):
        super().__init__()
        num_channels = b.shape[0]
        s_min = s.amin()
        scale = (s.amax() - s_min).item()
        bp = torch.tensor([.004, .007, .015], dtype=s.dtype, device=s.device) * scale + s_min
        self.primary = FixedPWL(bp, num_channels)

    def forward(self, b, s):
        return self.primary(s)

    def init_params(self, y, b, s):
        with torch.no_grad():
            self.primary.biases.data = y.mean(dim=0)

    def post_step(self):
        with torch.no_grad():
            self.primary._slopes.data[:, 0] = 0

    def get_param_groups(self, s, lr=1.0):
        s_scale = (s.amax() - s.amin()).item()
        return [
            {'params': [self.primary.biases],  'lr': lr},
            {'params': [self.primary._slopes], 'lr': lr / s_scale},
        ]
