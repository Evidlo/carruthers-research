#!/usr/bin/env python3

from common import *

# ----- Flattened Raytracing -----

# flatten everything
inds_flat = (obs * shape**3 + r * shape**2 + e * shape + a).flatten()
lens_flat = lens.flatten()
x_flat = x.flatten()

check_mem()
result_squeezed = x_flat[inds_flat]
result_squeezed *= lens_flat
result_squeezed = result_squeezed.view(lens.shape).sum(axis=-1)
check_mem('Flattened raytracer')
