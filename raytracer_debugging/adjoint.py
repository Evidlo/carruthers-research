#!/usr/bin/env python3

from sph_raytracer import *
from sph_raytracer.plotting import *
import torch as t

from glide.science.plotting import save_gif
from glide.science.plotting_sph import cardplot

grid = SphericalGrid(shape=(50, 51, 51), spacing='lin')
N = 256
# grid = SphericalGrid(shape=(5, 1, 1), spacing='lin')
# N = 1

geom = (
    ConeRectGeom((N, N), (5, 1, 1), fov=(5, 5))
    + ConeRectGeom((N, N), (1, -5, 1), fov=(5, 5))
    )

op = Operator(grid, geom)

x = t.ones(op.grid.shape, device=op.device)
y = op(x)

# y[N//2:, N//2:] *= 4
# y[:N//2, :N//2] *= 4

xi = op.T(t.ones_like(y))
xi[xi==0] = 1
# x2 = op.T(y) / xi
x2 = op.T(y / op(t.ones_like(x))) / xi

y2 = op(x2)

print('previewing...')
previews = preview3d(x2, grid)
# %% plot
print('plotting...')
# image_stack(previews).save('/www/out.gif')
save_gif('/www/out.gif', previews, rescale='sequence')
# cardplot(x, grid)
plt.savefig('/www/out.png')