#!/usr/bin/env python3
"""SharedPWL variant: c_j scales only the below-riser slope, not the riser base level."""

from model_sharedpwl import Model as _Base


class Model(_Base):

    def forward(self, b, s):
        b_u = b.unsqueeze(0)
        p = self.pwl(s)                                               # (rows, 1)
        bp_last = self.pwl.breakpoints[0][-1]
        last = self.pwl._slopes[0, -1] * (s - bp_last).clamp(min=0)  # below-riser only
        return b_u - (p - last) * b_u - last * (b - self.c).unsqueeze(0)
