#!/usr/bin/env python3
"""Parametric sag model from AGENT.md. Primary sag only.

y_ij = (x_ij + b_j) - (σ·(x_ij + b_j)·s_i + β) · 1(s_i > α)

For OOB columns where x_ij ≈ 0:
    y_ij = b_j - (σ·b_j·s_i + β) · 1(s_i > α)

Learned scalars: σ, β, α
Pre-computed: b_j (per-column bias)
"""

import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, b, s, **kw):
        super().__init__()
        s_scale = (s.amax() - s.amin()).item()
        self.register_buffer('_s_scale', torch.tensor(s_scale))
        self.register_buffer('_s_min', s.amin().clone())

        # σ ≈ sag / (b * s) ≈ 10 / (2700 * 3e6) ≈ 1e-9
        self.log_sigma = nn.Parameter(torch.tensor(-21.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.alpha_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(self, b, s):
        """
        Args:
            b: (num_cols,) per-column bias values
            s: (num_rows, 1) same-side row sums
        Returns:
            (num_rows, num_cols) predicted y values
        """
        sigma = self.log_sigma.exp()
        alpha = self._s_min + torch.sigmoid(self.alpha_logit) * self._s_scale

        # Soft threshold: smooth approximation of 1(s > α)
        gate = torch.sigmoid((s - alpha) / (self._s_scale * 0.002))

        sag = (sigma * b.unsqueeze(0) * s + self.beta) * gate
        return b.unsqueeze(0) - sag

    def init_params(self, y, b, s):
        pass

    def post_step(self):
        pass

    def get_param_groups(self, s, lr=1.0):
        return [
            {'params': [self.log_sigma], 'lr': lr * 0.1},
            {'params': [self.beta],      'lr': lr},
            {'params': [self.alpha_logit], 'lr': lr * 0.1},
        ]
