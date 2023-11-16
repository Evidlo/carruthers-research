#!/usr/bin/env python3

import tomosipo as ts
from tomosipo.torch_support import to_autograd
import torch as t
from tqdm import tqdm
import matplotlib.pyplot as plt

# set up projection geometry.  some voxels have no lines of sight
vol = ts.volume(shape=100, size=1)
proj = ts.concatenate([
    ts.cone(
        angles=25, shape=(256, 256),
        size=5, src_orig_dist=2, src_det_dist=4
    ),
    ts.cone(
        angles=25, shape=(256, 256),
        size=.5, src_orig_dist=2, src_det_dist=4
    ),
])
op = ts.operator(vol, proj)

# simple phantom of two intersecting cubes
x = t.zeros((100, 100, 100), device='cuda')
x[40:80, 40:80, 40:80] = 1
x[20:60, 20:60, 20:60] = 1

y = op(x)

# gradient descent
x_recon = t.zeros_like(x, requires_grad=True)
y.requires_grad_()
op = to_autograd(op)

# minimize mean squared error with gradient descent
optimizer = t.optim.Adam([x_recon], lr=1e-1)
losses = []
for _ in (bar := tqdm(range(500))):
    optimizer.zero_grad()

    loss = t.mean((y - op(x_recon))**2)
    bar.set_description(f'Loss: {loss:.1e}')
    loss.backward(retain_graph=True)
    losses.append(loss.detach().cpu())
    optimizer.step()

from ts_algorithms import *
# x_recon = nag_ls(op, y, num_iterations=500)
# x_recon = sirt(op, y, num_iterations=500)
# x_recon = fbp(op, y)

# %% plot

from tomosipo_test_plot import *
save_gif('preview_truth.gif', preview3d(x))
save_gif('preview_recon.gif', preview3d(x_recon))
ts.svg(vol, proj).save('geom.svg')

print(f'MSE: {t.mean((x - x_recon)**2)}')


from dech import Page, Img

yfix = y.moveaxis((0, 1, 2), (1, 0, 2))
f = '/srv/www/display/tomobug.html'
Page(
    [
        [
            Img(yfix.detach().cpu(), animation=True)
        ]
    ]
).save(f)
print(f"Saved to {f}")