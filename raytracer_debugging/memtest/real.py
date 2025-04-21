#!/usr/bin/env python3
# Investigating why raytracer uses so much memory
#
#

from common import *

# ----- Real Raytracing -----
# from sph_raytracer import *
# grid = SphericalGrid(shape=(shape, shape, shape))
# geom = ConeRectGeom(shape=(num_pix, num_pix), pos=(1, 0, 0))
# geom = sum([geom] * num_obs)
# check_mem()
# op = Operator(grid, geom, device=x.device, _flatten=False, dynamic=True)
# result_real = op(x)
# check_mem('Real raytracer')


from sph_raytracer import *
from sph_raytracer.raytracer import trace_indices
grid = SphericalGrid(shape=(shape, shape, shape))
geom = ConeRectGeom(shape=(num_pix, num_pix), pos=(1, 0, 0))
geom = sum([geom] * num_obs)

prof_start()
check_mem()

inds_real, lens_real = trace_indices(
    grid, geom.ray_starts, geom.rays,
    ftype=t.float64, itype=t.int64, device=x.device
)


r, e, a = inds_real
obs_real = t.arange(len(x), **int_spec)[:, None, None, None]

# result_real = x[(obs_real, r, e, a)]
result_real = x[obs_real, r, e, a]
# result_real = x[obs, r, e, a]
# result_real *= lens_real
# result_real = result_real.sum(axis=-1)

check_mem('real')
prof_save('real')
