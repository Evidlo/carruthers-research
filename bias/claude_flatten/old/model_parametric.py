#!/usr/bin/env python3
"""Parametric sag model from AGENT.md with per-column c_j correction.

y_ij = (x_ij + b_j) − (σ·(x_ij + b_j − c_j)·s_i + β) · 𝟙(s_i > α)

For OOB columns where x_ij ≈ 0:
    y_ij = b_j − (σ·(b_j − c_j)·s_i + β) · 𝟙(s_i > α)

Learned scalars: σ, β, α
Pre-computed: b_j (per-column bias), c_j (per-column sag correction)
"""

import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, b, s, c=None, **kw):
        super().__init__()
        s_scale = (s.amax() - s.amin()).item()
        self.register_buffer('_s_scale', torch.tensor(s_scale))
        self.register_buffer('_s_min', s.amin().clone())
        if c is not None:
            self.register_buffer('c', c)
        else:
            self.register_buffer('c', torch.zeros_like(b))

        self.log_sigma = nn.Parameter(torch.tensor(-21.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.alpha_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(self, b, s):
        sigma = self.log_sigma.exp()
        alpha = self._s_min + torch.sigmoid(self.alpha_logit) * self._s_scale
        gate = torch.sigmoid((s - alpha) / (self._s_scale * 0.002))

        b_eff = b.unsqueeze(0) - self.c.unsqueeze(0)
        sag = (sigma * b_eff * s + self.beta) * gate
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
