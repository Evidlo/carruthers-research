#!/usr/bin/env python3
import torch as t
t.cuda.empty_cache()

def prof_start():
   t.cuda.memory._record_memory_history(
       max_entries=100000,
       stacks='python',
   )

def prof_save(file_prefix):
    import pickle
    filename = f"{file_prefix}.pickle"
    t.cuda.memory._dump_snapshot(filename)
    # Stop recording memory snapshot history.
    t.cuda.memory._record_memory_history(enabled=None)

# ----- Array Setup -----

# ideally integers would be int8, but pytorch requires them to be int64
int_spec = {'device': 'cuda', 'dtype': t.int64}
float_spec = {'device': 'cuda', 'dtype': t.float32}

# number of camera locations to raytrace from
num_obs = 30
# temporal 3D volume being raytraced
shape = 50
x = t.rand((num_obs, shape, shape, shape), **float_spec)
# width of camera
num_pix = 64
# maximum number of voxels intersected by a single ray
num_vox = 2 * (shape + 1) + 2 * (shape + 1) + (shape + 1)

# ----- Raytracing -----

prof_start()

# voxel indices where rays intersect (for each of the 3 dimensions)
r = t.randint(shape, (num_obs, num_pix, num_pix, num_vox), **int_spec)
e = t.randint(shape, (num_obs, num_pix, num_pix, num_vox), **int_spec)
a = t.randint(shape, (num_obs, num_pix, num_pix, num_vox), **int_spec)

# raytrace a different 3D volume from our 4D stack for each observation
# obs = t.arange(len(x), **int_spec)[:, None, None, None]
# obs = t.arange(len(x), **int_spec)[:, None, None, None].expand_as(r)
obs = t.randint(num_obs, (num_obs, num_pix, num_pix, num_vox), **int_spec)
result = x[obs, r, e, a]

obs2 = t.arange(len(x), **int_spec)
# result2 = x[:, r, e, a][obs2, obs2, :, :, :]
result2 = x[:, r, e, a]

prof_save('record')