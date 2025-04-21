#!/usr/bin/env python3

import torch as t
import numpy as np
from sph_raytracer.raytracer import *
from contexttimer import Timer

np.random.seed(0)
los = 100*50*50
xs = np.random.random((los, 3))
rays = np.random.random((los, 3))

ftype = t.float64
itype = t.int64

xs_t = tr.asarray(xs)
rays_t = tr.asarray(rays)
xs_tc = tr.asarray(xs, dtype=ftype, device='cuda')
rays_tc = tr.asarray(rays, dtype=ftype, device='cuda')

print('-----------')
rs = np.linspace(0, 1, 60)

t.cuda.empty_cache()
# with Timer(prefix='r'):
#     r(xs, rays, rs)
with Timer(prefix='r_torch CUDA'):
    r_torch(rs, xs_tc, rays_tc, ftype, itype, 'cuda')
with Timer(prefix='r_torch CPU'):
    r_torch(rs, xs_t, rays_t, ftype, itype, 'cpu')

print('-----------')
phis = np.linspace(0, 1, 19)

t.cuda.empty_cache()
# with Timer(prefix='e'):
#     e(xs, rays, phis)
with Timer(prefix='e_torch CUDA'):
    e_torch(phis, xs_tc, rays_tc, ftype, itype, 'cuda')
with Timer(prefix='e_torch CPU'):
    e_torch(phis, xs_t, rays_t, ftype, itype, 'cpu')

print('-----------')
thetas = np.linspace(0, 1, 37)

t.cuda.empty_cache()
# with Timer(prefix='e'):
#     e(xs, rays, thetas)
with Timer(prefix='a_torch CUDA'):
    a_torch(thetas, xs_tc, rays_tc, ftype, itype, 'cuda')
with Timer(prefix='a_torch CPU'):
    a_torch(thetas, xs_t, rays_t, ftype, itype, 'cpu')