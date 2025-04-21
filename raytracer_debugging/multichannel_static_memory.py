#!/usr/bin/env python3

from memtest.common import *
import torch as t
t.cuda.empty_cache()
from contexttimer import Timer

from sph_raytracer import *

# spec = {'device':'cuda', 'dtype':t.float}
# typ =
dev = dict(device='cuda')
check_mem()

x = t.rand((300, 200, 45, 60), dtype=t.float64, **dev)

check_mem('dataset')

vg = ConeCircGeom((50, 100), (10, 0, 0))
grid = SphericalGrid(x.shape[1:])

op = Operator(grid, vg, **dev)

check_mem('operator')

r, e, a = op.regs
lens = op.lens

# ----- Raytracer -----

# with Timer() as tim:
#     result = (x[..., r, e, a] * lens).sum(axis=-1)


# check_mem('raytrace')
# print('    ', tim, 's')
# print('    ', result.sum())

# del result
# t.cuda.empty_cache()
# check_mem()


# ----- VMAP -----

# Assuming x and lens are PyTorch tensors
def batched_lookup(r, e, a, x, lens):
    return (x[..., r, e, a] * lens).sum(dim=-1)

batched_fn = t.vmap(
    batched_lookup,
    in_dims=(0, 0, 0, None, 0),
    chunk_size=8
)
with Timer() as tim:
    result = batched_fn(r, e, a, x, lens)

check_mem('vmap1')
print('    ', tim, 's')
print('    ', result.sum())
print()

# ----- VMAP2 -----

del result
t.cuda.empty_cache()
check_mem()

batched_fn = t.vmap(
    t.vmap(
        batched_lookup,
        in_dims=(0, 0, 0, None, 0),
        chunk_size=8
    ),
    in_dims=(0, 0, 0, None, 0),
    chunk_size=8
)
with Timer() as tim:
    result = batched_fn(r, e, a, x, lens)

check_mem('vmap2')
print('    ', tim, 's')
print('    ', result.sum())
print()

# ----- Index Select -----

def _():
    del result
    t.cuda.empty_cache()
    check_mem()

    # First, gather along the last three axes separately
    x_r = t.index_select(x, dim=-3, index=r.reshape(-1))  # Select along axis -3
    x_re = t.index_select(x_r, dim=-2, index=e.reshape(-1))  # Select along axis -2
    x_rea = t.index_select(x_re, dim=-1, index=a.reshape(-1))  # Select along axis -1

    # Reshape back and multiply by lens
    x_rea = x_rea.view(*r.shape)  # Restore original shape
    result = (x_rea * lens).sum(dim=-1)

    check_mem('index select')
    print('    ', result.sum())

# ----- Gather -----

def _():
    del result
    t.cuda.empty_cache()
    check_mem()

    # Compute flattened indices
    indices = r * x.shape[-2] * x.shape[-1] + e * x.shape[-1] + a  # Shape: (50, 100, 556)

    # Expand indices to match the batch size of `x_flat`
    indices = indices.unsqueeze(0).expand(x.shape[0], -1, -1, -1)  # Shape: (264, 50, 100, 556)

    # Flatten `x` for efficient gathering
    x_flat = x.view(*x.shape[:-3], -1)  # Shape: (264, 540000)

    # Gather values
    x_selected = t.gather(x_flat, dim=-1, index=indices.flatten(1))

    # Compute final result
    result = (x_selected * lens.flatten()).sum(dim=-1)

    check_mem('gather')
    print('    ', result.sum())