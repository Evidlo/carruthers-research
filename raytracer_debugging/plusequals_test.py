#!/usr/bin/env python3

import numpy as np
# import torch as np
# np.array = np.tensor

x = np.zeros(10)

inds = np.array((0, 0, 3, 3, 6, 6, 9, 9))
vals = np.array((1, 1, 1, 1, 1, 1, 1, 1))

x[inds] += vals

print(x)
# desired value: [2, 0, 0, 2, 0, 0, 2, 0, 0, 2]
#  actual value: [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]