#!/usr/bin/env python3

import time
from contexttimer import Timer
import torch as t
from torch import optim

t.manual_seed(0)
t.cuda.empty_cache()
t.cuda.reset_peak_memory_stats()

def check_mem(desc=None):
    """Print additional memory used since last call (peak)"""
    global mem_last
    mem_peak = t.cuda.max_memory_allocated()
    mem_curr = t.cuda.memory_allocated()
    t.cuda.reset_peak_memory_stats()
    if desc is not None:
        print(f'{desc}:', (mem_peak - mem_last) / 1e9, 'GB')
    mem_last = mem_curr
mem_last = t.cuda.memory_allocated()

# ----- Array Setup -----
# integer and floating point datatypes
fspec = {'device':'cuda', 'dtype':t.float64}
ispec = {'device':'cuda', 'dtype':t.int64}

# shape of volume being raytraced
shape = 50
# volume array which will be raytraced
vol = t.rand((shape, shape, shape), **fspec)
# maximum number of voxels that can be intersected by a single ray
num_vox = 2 * vol.shape[0] + 2 * vol.shape[1] + vol.shape[2]
# camera resolution.  there are num_pix² rays from each observation point
num_pix = 64
# number of observations.
num_obs = 50

check_mem()
# voxel indices intersecting each ray
ind = t.randint(shape, (3, num_obs, num_pix, num_pix, num_vox), **ispec)
check_mem('Index Array Memory')
# intersection lengths of rays with voxels
lens = t.rand(num_obs, num_pix, num_pix, num_vox, **fspec)
check_mem('Length Array Memory')

# ----- Regular Indexing -----

print('----- Raytracing -----')

# image stack.  one num_pix² image from each observation point
# shape (num_obs, num_pix, num_pix)
r, e, a = ind
result = (vol[r, e, a] * lens).sum(axis=-1)
check_mem('Raytracer Memory')

# time for 100x raytrace operations
with Timer(prefix='Index time (100x)'):
    for _ in range(100):
        result = (vol[r, e, a] * lens).sum(axis=-1)

# autograd
vol.requires_grad_()
optimizer = optim.Adam([vol])
with Timer(prefix='Autograd time (100x)'):
    for _ in range(100):
        optimizer.zero_grad()
        result = (vol[r, e, a] * lens).sum(axis=-1)
        truth = t.zeros_like(result)
        ((result - truth)**2).sum().backward()
        optimizer.step()

# ----- Flattened Indexing -----
# %% flat

print('----- Raytracing Flat -----')

# flatten everything
inds_flat = (r * shape**2 + e * shape + a).flatten()
lens_flat = lens.flatten()
vol_flat = vol.flatten().detach()

# result_flat = vol_flat[inds_flat]
# result_flat *= lens_flat
# result_flat = result_flat.view(lens.shape).sum(axis=-1)
result_flat = (vol_flat[inds_flat] * lens_flat).view(lens.shape).sum(axis=-1)
check_mem('Raytracer Memory')

# time for 100x raytrace operations
with Timer(prefix='Index time (100x)'):
    for _ in range(100):
        result = (vol_flat[inds_flat] * lens_flat).view(lens.shape).sum(axis=-1)

# autograd
vol_flat.requires_grad_()
optimizer = optim.Adam([vol_flat])
with Timer(prefix='Autograd time (100x)'):
    for _ in range(100):
        optimizer.zero_grad()
        result_flat = (vol_flat[inds_flat] * lens_flat).view(lens.shape).sum(axis=-1)
        truth_flat = t.zeros_like(result_flat)
        ((result_flat - truth_flat)**2).sum().backward()
        optimizer.step()
