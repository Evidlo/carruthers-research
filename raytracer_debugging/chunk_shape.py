#!/usr/bin/env python3

from sph_raytracer import *
from sph_raytracer.geometry import ViewGeom
import torch as t

a = {'chunk': True, 'chunk_size': 2}
# a = {'chunk': False, 'chunk_size': 2}

def create_geom(shape):
    return ViewGeom(t.rand(shape + (3,)), t.rand(shape + (3,)))

# op = Operator(SphericalGrid((50, 50, 50)), create_geom((64, 64)), **a)
# op(t.rand(op.grid.shape))

# op = Operator(SphericalGrid((50, 50, 50)), create_geom((10, 64, 64)), **a)
# op(t.rand(op.grid.shape))

# ------------- Multi-Channel --------------

# op = Operator(SphericalGrid((50, 50, 50)), create_geom((5, 64, 64)), **a)
# op(t.rand((10, 50, 50, 50)))

# op = Operator(SphericalGrid((50, 50, 50)), create_geom((64, 64)), **a)
# op(t.rand((10, 50, 50, 50)))

# op = Operator(SphericalGrid((50, 50, 50)), ViewGeom([1, 1, 1], [1, 1, 1]), **a)
# FIXME? 0d broken but 1d works
# op = Operator(SphericalGrid((50, 50, 50)), ViewGeom([[1, 1, 1]], [[1, 1, 1]]), **a)
# op(t.rand((10, 50, 50, 50)))

# ------------- Dynamic --------------

# op = Operator(SphericalGrid((10, 50, 50, 50)), create_geom((10, 64, 64)), **a)
# op(t.rand(op.grid.shape))

op = Operator(SphericalGrid((10, 50, 50, 50)), create_geom((10,)), **a)
result = op(t.rand(op.grid.shape))

print(result)