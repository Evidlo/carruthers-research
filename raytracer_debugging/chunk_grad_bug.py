#!/usr/bin/env python3
# Evan Widloski - 2025-05-13
# Investigating why enabling chunking breaks gradient descent?

from sph_raytracer import *
from sph_raytracer.plotting import *
import torch as t
from tqdm import tqdm
from glide.science.plotting import save_gif

d = {'device': 'cuda'}

grid = SphericalGrid((50, 50, 50))
geom = ConeRectGeom((128, 128), (3, 0, 0))
op = Operator(grid, geom, **d, chunk=True, chunk_size=10)

x = t.zeros(grid.shape, **d)
x[:, :25, :] = 1

y = op(x)

x_hat = t.rand_like(x, requires_grad=True)
optim = t.optim.Adam([x_hat], lr=1e-1)

for _ in (bar:=tqdm(range(300), desc='iteration')):
    optim.zero_grad()
    loss = ((y - op(x_hat))**2).mean()
    loss.backward()

    bar.set_description(f'loss {loss}')

    optim.step()

# op.plot().figure.savefig('/www/orbit.png')
save_gif('/www/truth.gif', preview3d(x, grid))

save_gif('/www/recon.gif', preview3d(x_hat, grid))
