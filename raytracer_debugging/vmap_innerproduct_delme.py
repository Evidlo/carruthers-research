#!/usr/bin/env python3

# Computing raytracer inner product with vmap
# Paying special attention to output return shape, and checking
# for errors related to `.item()` error when using fancy indexing inside vmap
#
# This code targets lessmem3 branch with vmap and `chunk` kwarg on Operator

from memtest.common import *
import torch as t
from contexttimer import Timer

from tomosphero import *

# spec = {'device':'cuda', 'dtype':t.float}
# typ =
dev = dict(device='cuda')
check_mem()


cases = [
    (
        (50, 50, 50), # grid shape
        (64, 64),     # geom shape
        False,        # dynamic?
        (64, 64),     # output shape
    ),
    (
        (50, 50, 50),
        (10, 64, 64),
        False,
        (10, 64, 64)
    ),
    (
        (10, 50, 50, 50),
        (10, 64, 64),
        True,
        (10, 64, 64)
    ),
]

for grid, geom, dynamic, outshape in cases:
    t.cuda.empty_cache()

    grid = SphericalGrid(grid)
    geom = ViewGeom((0, 0, 0), t.rand((*geom, 3)))
    x = t.rand(grid.shape, **dev)

    print('---------------------------------------------')
    print('grid:', tuple(grid.shape))
    print('geom:', geom.shape)
    print('dynamic:', dynamic)
    print('---------------------------------------------')

    with Timer(prefix='Operator'):
        op = Operator(grid, geom, chunk=False, **dev)
        result = op(x)

    check_mem('Operator')

    assert result.shape == outshape, f"Result shape incorrect: {result.shape}"
