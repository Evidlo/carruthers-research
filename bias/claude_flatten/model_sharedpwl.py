#!/usr/bin/env python3
"""Shared-shape PWL sag model. Primary sag only.

y_ij = (x_ij + b_j) − PWL(s_i) · (x_ij + b_j − c_j)

PWL(s) is a single piecewise linear function of the row sum with shared
learned slopes. Per-column variation comes from (b_j − c_j) scaling.

For OOB columns where x_ij ≈ 0:
    y_ij = b_j − PWL(s_i) · (b_j − c_j)

Learned: PWL slopes (~4 scalars)
Pre-computed: b_j, c_j
"""

import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn
from piecewise import FixedPWL


class Model(nn.Module):

    def __init__(self, b, s, c=None, **kw):
        super().__init__()
        if c is not None:
            self.register_buffer('c', c)
        else:
            self.register_buffer('c', torch.zeros_like(b))

        s_min = s.amin()
        s_scale = (s.amax() - s_min).item()
        bp = torch.tensor([.004, .007, .015], dtype=s.dtype, device=s.device) * s_scale + s_min

        # Single-channel PWL: shared sag shape
        self.pwl = FixedPWL(bp, num_channels=1)
        # Pin bias to 0: no sag at low s
        self.pwl.biases.requires_grad_(False)

        self.register_buffer('_b_scale', b.abs().mean().clone())

    def forward(self, b, s):
        b_eff = (b - self.c).unsqueeze(0)  # (1, cols)
        sag_shape = self.pwl(s)  # (rows, 1)
        return b.unsqueeze(0) - sag_shape * b_eff

    def init_params(self, y, b, s):
        pass

    def post_step(self):
        with torch.no_grad():
            self.pwl._slopes.data[:, 0] = 0

    def get_param_groups(self, s, lr=1.0):
        s_scale = (s.amax() - s.amin()).item()
        return [
            {'params': [self.pwl._slopes], 'lr': lr / s_scale / self._b_scale},
        ]
