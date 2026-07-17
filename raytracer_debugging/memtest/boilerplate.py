#!/usr/bin/env python3

from common import *
check_mem()

t.cuda.empty_cache()

# ideally integers would be int8, but pytorch requires them to be int64
int_spec = {'device': 'cuda', 'dtype': t.int64}
float_spec = {'device': 'cuda', 'dtype': t.float64}

# ----- Input Tensors Setup -----
# these are all placeholder tensors, but their sizes/dtypes are correct

# number of camera locations to raytrace from
num_obs = 30
# dynamic 3D volume being raytraced
shape = 50
x = t.rand((num_obs, shape, shape, shape), **float_spec)
# width of camera
num_pix = 64
# maximum number of voxels intersected by a single ray
num_vox = 2 * (shape + 1) + 2 * (shape + 1) + (shape + 1)




prof_start()
check_mem()
# voxel indices where rays intersect (for each of the 3 dimensions)
obs = t.arange(len(x), **int_spec)[:, None, None, None]
r = t.randint(shape, (num_obs, num_pix, num_pix, num_vox), **int_spec)
e = t.randint(shape, (num_obs, num_pix, num_pix, num_vox), **int_spec)
a = t.randint(shape, (num_obs, num_pix, num_pix, num_vox), **int_spec)
# inds = t.randint(shape, (num_obs, num_pix, num_pix, num_vox, 3), **int_spec)
# inds = inds.moveaxis(-1, 0)
# r, e, a = inds
check_mem('Index tensor allocated:')

# intersection lengths of each ray with each voxel
lens = t.rand(num_obs, num_pix, num_pix, num_vox, **float_spec)
inf = t.full(lens.shape[:-1] + (1,), float('inf'), **float_spec)
lens = lens.diff(dim=-1, append=inf)
check_mem('Length tensor allocated:')