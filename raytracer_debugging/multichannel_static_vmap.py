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

x = t.rand((1, 200, 45, 60), dtype=t.float64, **dev)

check_mem('Dataset')

vg = ConeCircGeom((512, 512), (10, 0, 0))
grid = SphericalGrid(x.shape[1:])

with Timer(prefix='Operator'):
    op = Operator(grid, vg, **dev)

check_mem('Operator')

r, e, a = op.regs
lens = op.lens

# ----- VMAP -----

# Assuming x and lens are PyTorch tensors
def batched_lookup(x, r, e, a, lens):
    return (x[..., r, e, a] * lens).sum(dim=-1)

batch_single = t.vmap(
    batched_lookup,
    in_dims=(None, 0, 0, 0, 0),
    out_dims=1,
    chunk_size=16
)
batch_double = t.vmap(
    batch_single,
    in_dims=(None, 0, 0, 0, 0),
    out_dims=1,
    chunk_size=16
)
print()
with Timer(prefix='Unbatched') as tim:
    result = op(x)
check_mem('Unbatched')
print()
with Timer(prefix='Single') as tim:
    result = batch_single(x, r, e, a, lens)
check_mem('Single')
print()
with Timer(prefix='Double') as tim:
    result = batch_double(x, r, e, a, lens)
check_mem('Double')

# print('     shape:', result.shape)
# print('    ', tim, 's')
# print('    ', result.shape)
# print('    ', result.sum())
# print()
