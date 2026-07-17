#!/usr/bin/env python3
"""Shared-shape PWL primary sag."""

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
        # self.bp = [.004, .007, .020]
        # self.bp = [.020, .021]
        self.bp = [.012, .017]

        _bp = torch.tensor(self.bp, dtype=s.dtype, device=s.device) * s_scale + s_min
        self.pwl = FixedPWL(_bp, num_channels=1)
        self.pwl.biases.requires_grad_(False)

        self.register_buffer('_b_scale', b.abs().mean().clone())

    def forward(self, b, s):
        b_eff = (b - self.c).unsqueeze(0)          # (1, cols)
        sag_shape = self.pwl(s)                     # (rows, 1)
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

    def to_params(self):
        return {
            'pwl_slopes': self.pwl._slopes.detach().cpu().numpy(),
        }

    @classmethod
    def from_params(cls, b, s, global_p=None, per_img=None):
        c = torch.tensor(global_p['cj'], dtype=torch.float32) if global_p else None
        m = cls(b, s, c=c)
        if per_img and 'pwl_slopes' in per_img:
            m.pwl._slopes.data = torch.tensor(per_img['pwl_slopes'], dtype=torch.float32)
        m.eval()
        return m
