#!/usr/bin/env python3

from sph_raytracer import *
from sph_raytracer.raytracer import *
from contexttimer import Timer
import torch as t

device = 'cpu'

geom = ConeRectGeom((256, 256), (1, 0, 0))
grid = SphericalGrid((500, 45, 60))

x = t.rand(grid.shape, device=device)

with Timer(prefix='operator'):
    op = Operator(grid, geom, pdevice=device, device=device)

with Timer(prefix='raytrace'):
    y = op(x)

with Timer(prefix='trace_indices'):
    regs, lens = trace_indices(
        grid, geom.ray_starts, geom.rays,
        device=device, pdevice=device,
    )

trace_indices_comp = t.compile(trace_indices)

with Timer(prefix='trace_indices compile'):
    regs, lens = trace_indices_comp(
        grid, geom.ray_starts, geom.rays,
        device=device, pdevice=device,
    )

# %% batch


# def batch(ray_starts, rays):
#     return trace_indices(
#         grid, geom.ray_starts, geom.rays,
#         device=device, pdevice=device,
#     )

# trace_batch = tr.vmap(
#     batch,
#     in_dims=(0, 0),
#     out_dims=1,
#     chunk_size=8
# )

# with Timer(prefix='trace_indices vmap'):
#     regs, lens = trace_batch(
#         geom.ray_starts.broadcast_to(geom.rays.shape),
#         geom.rays,
#     )