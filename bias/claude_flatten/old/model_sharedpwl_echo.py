#!/usr/bin/env python3
"""Shared-shape PWL primary sag + parametric echo sag.

y_ij = b_j − PWL(s_i)·(b_j − c_j) − σ'·(b_j − c_j)·s'_i·sigmoid((s'_i − α')/scale')

Primary sag: shared PWL in s_i (same as model_sharedpwl)
Echo sag:    pixel-proportional, linear in s'_i, with soft threshold α'

Set echo_trim = 0 in run.py to include echo rows in the fit.

Learned: PWL slopes, log_sigma_echo, alpha_echo_logit
Pre-computed: b_j, c_j, s'_i (opposite-half row sums, passed as s_prime)
"""

import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn
from piecewise import FixedPWL


class Model(nn.Module):

    def __init__(self, b, s, s_prime=None, c=None, **kw):
        super().__init__()
        if c is not None:
            self.register_buffer('c', c)
        else:
            self.register_buffer('c', torch.zeros_like(b))

        s_min = s.amin()
        s_scale = (s.amax() - s_min).item()
        bp = torch.tensor([.004, .007, .015], dtype=s.dtype, device=s.device) * s_scale + s_min

        self.pwl = FixedPWL(bp, num_channels=1)
        self.pwl.biases.requires_grad_(False)

        self.register_buffer('_b_scale', b.abs().mean().clone())

        sp = s_prime if s_prime is not None else torch.zeros_like(s)
        self.register_buffer('s_prime', sp)
        sp_scale = (sp.amax() - sp.amin()).clamp(min=1.0).item()
        self.register_buffer('_sp_min', sp.amin().clone())
        self.register_buffer('_sp_scale', torch.tensor(sp_scale, dtype=s.dtype, device=s.device))

        self.log_sigma_echo = nn.Parameter(torch.tensor(-24.0))
        self.alpha_echo_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(self, b, s):
        b_eff = (b - self.c).unsqueeze(0)          # (1, cols)
        sag_shape = self.pwl(s)                     # (rows, 1)
        primary = b.unsqueeze(0) - sag_shape * b_eff

        sigma_echo = self.log_sigma_echo.exp()
        alpha_echo = self._sp_min + torch.sigmoid(self.alpha_echo_logit) * self._sp_scale
        gate = torch.sigmoid((self.s_prime - alpha_echo) / (self._sp_scale * 0.002))
        echo = sigma_echo * b_eff * self.s_prime * gate  # (rows, cols)

        return primary - echo

    def init_params(self, y, b, s):
        pass

    def post_step(self):
        with torch.no_grad():
            self.pwl._slopes.data[:, 0] = 0

    def get_param_groups(self, s, lr=1.0):
        s_scale = (s.amax() - s.amin()).item()
        return [
            {'params': [self.pwl._slopes],      'lr': lr / s_scale / self._b_scale},
            {'params': [self.log_sigma_echo],    'lr': lr / self._sp_scale / self._b_scale},
            {'params': [self.alpha_echo_logit],  'lr': lr * 0.1},
        ]

    def to_params(self):
        return {
            'pwl_slopes': self.pwl._slopes.detach().cpu().numpy(),
            'log_sigma_echo': float(self.log_sigma_echo.detach().cpu()),
            'alpha_echo_logit': float(self.alpha_echo_logit.detach().cpu()),
        }

    @classmethod
    def from_params(cls, b, s, global_p=None, per_img=None):
        c = torch.tensor(global_p['cj'], dtype=torch.float32) if global_p else None
        m = cls(b, s, c=c)
        if global_p:
            m.pwl._slopes.data = torch.tensor(global_p['pwl_slopes'], dtype=torch.float32)
            m.log_sigma_echo.data = torch.tensor(global_p['log_sigma_echo'], dtype=torch.float32)
            m.alpha_echo_logit.data = torch.tensor(global_p['alpha_echo_logit'], dtype=torch.float32)
        m.eval()
        return m
