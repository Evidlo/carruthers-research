#!/usr/bin/env python3

import torch as t
from sph_raytracer.loss import Loss, CheaterLoss
import math

class ConditionLoss(Loss):

    kind = 'regularizer'

    def __init__(self, m, **kwargs):
        self.m = m
        super().__init__(**kwargs)

    def compute(self, f, y, d, c):
        """"""

        # make measurements and flatten all LOS into single dim
        # (atoms, vantage, measrow, meascol) → (atoms, vantage*measrow*meascol)
        meas = f(self.m()).flatten(start_dim=-3)
        result = t.linalg.norm(meas, ord='fro')
        # normalize loss by number of LOS, dict
        result = result / math.prod(f.range_shape) * self.m.num_atoms

        return result

class DirectL2Loss(Loss):

    kind = 'fidelity'

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        super().__init__(**kwargs)

    def compute(self, f, y, d, c):
        return t.mean((self.dataset - d)**2)