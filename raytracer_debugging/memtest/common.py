#!/usr/bin/env python3

import torch as t

t.cuda.empty_cache()

def check_mem(desc=None):
    """Print additional memory used since last call (peak)"""
    global mem_last
    mem_peak = t.cuda.max_memory_allocated()
    mem_curr = t.cuda.memory_allocated()
    t.cuda.reset_peak_memory_stats()
    if desc is not None:
        print(f'{desc}:', (mem_peak - mem_last) / 1e9, 'GB')
    mem_last = mem_curr
check_mem()

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


def prof_start():
   t.cuda.memory._record_memory_history(
       max_entries=100000,
       stacks='python',
   )

def prof_save(file_prefix):
    import pickle
    filename = f"{file_prefix}.pickle"
    t.cuda.memory._dump_snapshot(filename)
    d = pickle.load(open(filename, 'rb'))
    # d['segments'] = [s for s in d['segments'] if len(s['frames']) != 0]

    # pickle.dump(d, open(filename, 'wb'))

    # Stop recording memory snapshot history.
    t.cuda.memory._record_memory_history(enabled=None)


# voxel indices where rays intersect (for each of the 3 dimensions)
obs = t.arange(len(x), **int_spec)[:, None, None, None]
r = t.randint(shape, (num_obs, num_pix, num_pix, num_vox), **int_spec)
e = t.randint(shape, (num_obs, num_pix, num_pix, num_vox), **int_spec)
a = t.randint(shape, (num_obs, num_pix, num_pix, num_vox), **int_spec)