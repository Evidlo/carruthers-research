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

num_columns = 400

img = t.from_numpy(load(path:='images_20260111/oob_nfi_l0.pkl'))
# img = t.from_numpy(load(path:='images_20260101/dark_nfi_l0.pkl'))
cols = slice(0, num_columns)
rows = slice(150, 512)
y = img[rows, cols]
s = img.sum(dim=1, keepdim=True)[rows]

m = img[150:250, cols]
mask = np.logical_and(
    m.numpy() > np.percentile(m, 40, axis=0, keepdims=True),
    m.numpy() < np.percentile(m, 60, axis=0, keepdims=True)
)

robbias = np.ma.array(m, mask=~mask).mean(axis=0)

def normalize(x):
    xmin, xmax = x.amin(dim=0), x.amax(dim=0)
    x = x - xmin
    return x / (xmax - xmin), xmin, xmax

def denormalize(x, xmin, xmax):
    return x * (xmax - xmin) + xmin

s, smin, smax = normalize(s)
y, ymin, ymax = normalize(y)

# s = s**(.4)

def trimmed_norm(x, keep_ratio=0.9):
    keep_n = int(len(x) * keep_ratio)
    loss, _ = t.topk(x, keep_n, largest=False, dim=0)
    return loss.mean()

# p = PWL(num_channels=num_columns, num_breakpoints=3)
p = FixedPWL(breakpoints=[.004, .007, .015], num_channels=num_columns)
# p = Calibrator(keypoints=[[0.01, .05, 1]], monotonicity=[-1])
# def p(y, s):
#     pass


# bpoints, slopes, _ = list(p.parameters())
optim = t.optim.Adam(p.parameters(), lr=1e-1)

# constrain
# slopes.data[0, 1] = -10

loss_hist = []
for _ in (bar:=tqdm(range(2000))):
    optim.zero_grad()

    loss = trimmed_norm((y - p(s))**2)
    loss.backward()

    loss_hist.append(float(loss))

    optim.step()
    bar.set_description(f'loss = {float(loss):.2e}')

    # constrain
    # bpoints.data[:, -1] = .015
    # bpoints.data[:, 1] = .005
    p.slopes.data[:, 0] = 0

# %% plot

if __name__ == '__main__':

    from plotting import plot_profile
    fig = plot_profile(s, y, p)
    fig.savefig('/www/out.png')

    plt.close()
    means = y.mean(axis=0)
    import iplot as iplt
    plt.scatter(p(t.tensor(0)).detach() + means, p.slopes[:, -1].detach())
    plt.xlabel('offset')
    plt.ylabel('slope of linear region')
    plt.title(path)
    # plt.ylim([-.75, .25])
    # plt.xlim([0, 1.75])
    plt.savefig('/www/out2.png')

    clip = lambda x: t.clip(x, x.min(), x.min() + 50)
    plt.close()
    iplt.figure(figsize=(10, 4))
    iplt.subplot(1, 2, 1)
    # plt.imshow(np.clip(img[rows, cols], None, img.min() + 2000))
    iplt.imshow(clip(img[rows, cols] - img[rows, cols].mean(axis=0)))
    iplt.colorbar()
    iplt.subplot(1, 2, 2)
    img_flat = img.clone()
    img_flat[rows, cols] -= denormalize(p(s).detach(), ymin, ymax)
    # plt.imshow(np.clip(img_flat[rows, cols], None, img_flat.min() + 2000))
    iplt.imshow(clip(img_flat[rows, cols]))
    iplt.colorbar()
    iplt.savefig('/www/out3.html', dpi=200)


    fig = plot_profile(s, img[rows, cols], denormalize(p(s), ymin, ymax))
    fig.savefig('/www/out4.png')