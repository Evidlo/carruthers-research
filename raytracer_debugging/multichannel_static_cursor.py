#!/usr/bin/env python3

from memtest.common import *
t.cuda.empty_cache()

from sph_raytracer import *

# spec = {'device':'cuda', 'dtype':t.float}
# typ =
dev = dict(device='cuda')
check_mem()

x = t.rand((264, 200, 45, 60), **dev)

check_mem('dataset')

vg = ConeCircGeom((50, 100), (10, 0, 0))
grid = SphericalGrid(x.shape[1:])

op = Operator(grid, vg, **dev)

check_mem('operator')

r, e, a = op.regs
lens = op.lens

# ----- Raytracer -----
import torch as t

# Array shapes:
# density: (264, 200, 45, 60)
# r: (50, 100, 556)
# e: (50, 100, 556)
# a: (50, 100, 556)

result = (x[..., r, e, a] * lens).sum(axis=-1)

check_mem('raytrace')
print('    ', result.sum())

# ----- Batch Processing -----
# %% batch
batch_size = 32  # Adjust this based on your GPU memory
result_batched = []
for i in range(0, x.shape[0], batch_size):
    batch = x[i:i+batch_size]
    batch_result = (batch[..., r, e, a] * lens).sum(axis=-1)
    result_batched.append(batch_result)
result_batched = t.cat(result_batched, dim=0)

check_mem('raytrace_batched')
print('    ', result_batched.sum())

del result_batched, batch
t.cuda.empty_cache()

# ----- Double Batch Processing -----
# %% double
batch_size_x = 32  # First dimension
batch_size_v = 25  # View dimension (50/100)
result_double = t.zeros((x.shape[0], vg.shape[0], vg.shape[1]), device=x.device)

for i in range(0, x.shape[0], batch_size_x):
    for j in range(0, vg.shape[0], batch_size_v):
        r_slice = r[j:j+batch_size_v]
        e_slice = e[j:j+batch_size_v]
        a_slice = a[j:j+batch_size_v]
        lens_slice = lens[j:j+batch_size_v]
        
        batch = x[i:i+batch_size_x]
        batch_result = (batch[..., r_slice, e_slice, a_slice] * lens_slice).sum(axis=-1)
        result_double[i:i+batch_size_x, j:j+batch_size_v] = batch_result

check_mem('raytrace_double_batch')
print('    ', result_double.sum())

del r_slice, e_slice, a_slice, lens_slice, batch, batch_result, result_double
t.cuda.empty_cache()

# ----- Einsum -----
# %% einsum
# Reshape x to combine the middle dimensions
x_reshaped = x.reshape(x.shape[0], -1, x.shape[-1])
# Reshape indices to combine first two dimensions
r_reshaped = r.reshape(-1, r.shape[-1])
e_reshaped = e.reshape(-1, e.shape[-1])
a_reshaped = a.reshape(-1, a.shape[-1])
lens_reshaped = lens.reshape(-1, lens.shape[-1])

# After indexing: (264, 5000, 556) * (5000, 556) -> (264, 5000)
result_einsum = t.einsum('bvr,vr->bv', 
    x_reshaped[..., r_reshaped, e_reshaped, a_reshaped],
    lens_reshaped
).reshape(x.shape[0], vg.shape[0], vg.shape[1])

check_mem('raytrace_einsum')
print('    ', result_einsum.sum())
