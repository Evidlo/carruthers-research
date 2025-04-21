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

# ----- VMAP -----

# Assuming x and lens are PyTorch tensors
def batched_lookup(x, r, e, a, lens):
    return (x[..., r, e, a] * lens).sum(dim=-1)

batched_fn = t.vmap(
    t.vmap(
        batched_lookup,
        in_dims=(None, 0, 0, 0, 0),
        out_dims=1,
        chunk_size=8
    ),
    in_dims=(None, 0, 0, 0, 0),
    out_dims=1,
    chunk_size=32
)
with Timer() as tim:
    # result = batched_fn(x, r, e, a, lens)
    result = op(x)

check_mem('vmap1')
print('     shape:', result.shape)
print('    ', tim, 's')
print('    ', result.shape)
print('    ', result.sum())
print()
