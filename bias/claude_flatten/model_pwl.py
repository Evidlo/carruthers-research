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
        self.bp = [.012, .017]
        # bp = torch.tensor([.004, .007, .015], dtype=s.dtype, device=s.device) * scale + s_min
        _bp = torch.tensor(self.bp, dtype=s.dtype, device=s.device) * scale + s_min
        self.primary = FixedPWL(_bp, num_channels)

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

    def to_params(self):
        return {
            'slopes': self.primary._slopes.detach().cpu().numpy(),
            'biases': self.primary.biases.detach().cpu().numpy(),
        }

    @classmethod
    def from_params(cls, b, s, global_p=None, per_img=None):
        m = cls(b, s)
        if per_img:
            m.primary._slopes.data = torch.tensor(per_img['slopes'], dtype=torch.float32)
            m.primary.biases.data = torch.tensor(per_img['biases'], dtype=torch.float32)
        m.eval()
        return m


    def __repr__(self):
        return f'{self.__name__}(bp={self.bp})'