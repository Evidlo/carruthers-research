#!/usr/bin/env python3

from torchpwl import PWL, Calibrator
from piecewise import FixedPWL
import torch as t
from tqdm import tqdm
import numpy as np

from common import load

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def normalize(x):
    xmin, xmax = x.amin(dim=0), x.amax(dim=0)
    x = x - xmin
    return x / (xmax - xmin), xmin, xmax

def denormalize(x, xmin, xmax):
    return x * (xmax - xmin) + xmin

def trimmed_norm(x, keep_ratio=0.9):
    keep_n = int(len(x) * keep_ratio)
    loss, _ = t.topk(x, keep_n, largest=False, dim=0)
    return loss.mean()

def learn_pwl(y, s):
    """Train piecewise linear model over a flat patch of data

    Args:
        y (tensor): flat patch of image values.  (shape (rows, cols))
        s (tensor): row statistics (shape (rows))
    """

    # normalize data before learning PWL function
    s, smin, smax = normalize(s)
    y, ymin, ymax = normalize(y)

    # p = FixedPWL(breakpoints=[.005, .015], num_channels=y.shape[1])
    p = FixedPWL(breakpoints=[.004, .007, .015], num_channels=y.shape[1])


    optim = t.optim.Adam(p.parameters(), lr=1e-1)

    loss_hist = []
    for _ in (bar:=tqdm(range(2000))):
        optim.zero_grad()

        loss = trimmed_norm((y - p(s))**2)
        loss.backward()

        loss_hist.append(float(loss))

        optim.step()
        bar.set_description(f'loss = {float(loss):.2e}')

        p.slopes.data[:, 0] = 0


    # fold normalization into PWL parameters so p operates on raw inputs
    s_scale = smax - smin
    y_scale = ymax - ymin
    with t.no_grad():
        p.x_positions = denormalize(p.x_positions, smin, smax)
        p.slopes.data = p.slopes.data * y_scale.unsqueeze(1) / s_scale
        p.biases.data = denormalize(p.biases.data, ymin, ymax)

    return p


img = t.from_numpy(load(path:='images_20260111/oob_nfi_l0.pkl'))
img_flat_pwl = img.clone()

# row to begin clipping of echo
echo = 150

img_flat_orig = img.clone()
img_flat_orig[echo:512, :400] -= img[echo:250, :400].mean(dim=0, keepdim=True)
img_flat_orig[echo:512, 775:] -= img[echo:250, 775:].mean(dim=0, keepdim=True)
img_flat_orig[-512:-echo, :400] -= img_flat_orig[-250:-echo, :400].mean(dim=0, keepdim=True)
img_flat_orig[-512:-echo, 775:] -= img_flat_orig[-250:-echo, 775:].mean(dim=0, keepdim=True)

# --- training ---
cols = slice(0, 400)
rows = slice(echo, 512)
y = img[rows, cols]
s = img.sum(dim=1, keepdim=True)[rows]
mapping = learn_pwl(y, s)
img_flat_pwl[rows, cols] -= mapping(s)

# %% plot2

from common import plot_profile
plot_profile(s, y, mapping(s))
plt.savefig('/www/profile.png')

# %% foo

# upper row stat
us = s
umapping = mapping

cols = slice(0, 400)
rows = slice(512, -echo)
y = img[rows, cols]
s = img.sum(dim=1, keepdim=True)[rows]
mapping = learn_pwl(y, s)
img_flat_pwl[rows, cols] -= mapping(s)

# lower row stat
bs = s
bmapping = mapping

cols = slice(775, None)
rows = slice(echo, 512)
y = img[rows, cols]
s = img.sum(dim=1, keepdim=True)[rows]
mapping = learn_pwl(y, s)
img_flat_pwl[rows, cols] -= mapping(s)

cols = slice(775, None)
rows = slice(512, -echo)
y = img[rows, cols]
s = img.sum(dim=1, keepdim=True)[rows]
mapping = learn_pwl(y, s)
img_flat_pwl[rows, cols] -= mapping(s)

# ----- plotting -----
# %% plot

# image plots

# clip to bottom dynamic range
clipb = lambda x: t.clip(x, x.min(), x.min() + 50)
plt.close()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# plt.imshow(np.clip(img[rows, cols], None, img.min() + 2000))
plt.imshow(clipb(img_flat_orig).detach())
plt.colorbar()
plt.title("Standard Bias Subtraction: y - b")
plt.clim(-10, 10)

plt.subplot(1, 2, 2)
plt.imshow(clipb(img_flat_pwl).detach())
plt.clim(-10, 10)
plt.colorbar()
plt.title("PWL Bias Subtraction: y - f₂(b, s)")
plt.savefig('/www/out3.png', dpi=400)

# slope relationship plot
import iplot as iplt
iplt.close()
iplt.scatter(bmapping.y_positions[:, 0], umapping.slopes[:, -1])
iplt.savefig('/www/slopes.html')
