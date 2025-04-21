#!/usr/bin/env python3

from common import *

# ----- Regular Raytracing -----

# look up voxel indices for each ray and multiply by intersection length, then sum

# intersection lengths of each ray with each voxel
# lens = t.rand(num_obs, num_pix, num_pix, num_vox, **float_spec)
# inf = t.full(lens.shape[:-1] + (1,), float('inf'), **float_spec)
# lens = lens.diff(dim=-1, append=inf)
# check_mem('Length tensor allocated:')


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
# check_mem('Index tensor allocated:')

result = x[obs, r, e, a]
# result *= lens
# result = result.sum(axis=-1)
# check_mem('Regular raytracer:')

check_mem('regular')
prof_save('regular')