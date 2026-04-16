#!/usr/bin/env python3

from torchpwl import PWL, Calibrator
from piecewise import FixedPWL
import torch as t
from tqdm import tqdm
import numpy as np

from common import load

import iplot as iplt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

device = 'cuda'

def trimmed_norm(x, keep_ratio=0.9):
    keep_n = int(len(x) * keep_ratio)
    loss, _ = t.topk(x, keep_n, largest=False, dim=0)
    return loss.mean()

def learn_pwl(y, s, iterations=2000):
    """Train piecewise linear model over a flat patch of data

    Args:
        y (tensor): flat patch of image values.  (shape (rows, cols))
        s (tensor): row statistics (shape (rows, 1))
        iterations (int): number of iterations

    Returns:
        FixedPWL: piecewise linear function in raw units
    """
    smin, smax = s.amin(), s.amax()
    s_scale = (smax - smin).item()

    # Breakpoints in raw s units, equivalent to [.004, .007, .015] in normalized space
    bp_raw = t.tensor([.004, .007, .015], dtype=s.dtype, device=device) * s_scale + smin

    p = FixedPWL(
        breakpoints=bp_raw,
        # breakpoints=[.004, .007, .015, .44] * s_scale + smin,
        num_channels=y.shape[1],
    )
    p.to(device)

    with t.no_grad():
        p.biases.data = y.mean(dim=0)

    # Slopes need lr scaled by 1/s_scale: target slope is O(std_y/s_scale),
    # so Adam's step (O(lr)) matches when lr_slopes = base_lr / s_scale.
    base_lr = 1.0
    optim = t.optim.Adam([
        {'params': [p.biases],  'lr': base_lr},
        {'params': [p._slopes], 'lr': base_lr / s_scale},
    ])

    loss_hist = []
    for _ in (bar:=tqdm(range(iterations))):
        optim.zero_grad()

        loss = trimmed_norm((y - p(s))**2, .8)
        loss.backward()

        loss_hist.append(float(loss))

        optim.step()
        bar.set_description(f'loss = {float(loss):.2e}')

        p.slopes.data[:, 0] = 0

    print(f'  loss: {loss_hist[0]:.3e} → {loss_hist[-1]:.3e}')

    return p


img = t.from_numpy(load(path:='images_20260117/oob_nfi_l0.pkl')).to(device)
img_flat_pwl = img.clone()

# row to begin clipping of echo
echo = 150

img_flat_orig = img.clone()
img_flat_orig[echo:512, :400] -= img[echo:250, :400].mean(dim=0, keepdim=True)
img_flat_orig[echo:512, 775:] -= img[echo:250, 775:].mean(dim=0, keepdim=True)
img_flat_orig[-512:-echo, :400] -= img_flat_orig[-250:-echo, :400].mean(dim=0, keepdim=True)
img_flat_orig[-512:-echo, 775:] -= img_flat_orig[-250:-echo, 775:].mean(dim=0, keepdim=True)

# --- training ---
# %% plot2
cols = slice(0, 400)
rows = slice(echo, 512)
y = img[rows, cols]
s = img.sum(dim=1, keepdim=True)[rows]
mapping = learn_pwl(y, s)
from plotting import plot_profile, detach
plot_profile(s, y, mapping)
plt.xlim(detach(s.min()), np.percentile(detach(s), 75))
plt.savefig('/www/profile.png')
img_flat_pwl[rows, cols] -= mapping(s)



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
brows = rows
bcols = cols

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
iplt.close()
iplt.figure(figsize=(12, 5))
iplt.subplot(1, 2, 1)
# plt.imshow(np.clip(img[rows, cols], None, img.min() + 2000))
iplt.imshow(detach(clipb(img_flat_orig)))
iplt.colorbar()
iplt.title("Standard Bias Subtraction: y - b")
iplt.clim(-10, 10)

iplt.subplot(1, 2, 2)
iplt.imshow(detach(img_flat_pwl))
iplt.clim(-10, 10)
iplt.colorbar()
iplt.title("PWL Bias Subtraction: y - f₂(b, s)")
iplt.savefig('/www/flat.html', dpi=400)

# slope relationship plot
import iplot as iplt
iplt.close()
# iplt.scatter(bmapping.y_positions[:, 0], bmapping.slopes[:, -1])
iplt.scatter3d(
    bmapping.y_positions[:, 0],
    umapping.y_positions[:, 0],
    bmapping.slopes[:, -1]
)
iplt.gcf().layout.scene.camera.projection.type = 'orthographic'
iplt.gcf().layout.scene.camera.eye = dict(x=0, y=1, z=0)
iplt.xlabel('Bias')
iplt.ylabel('Bias (opposite side)')
iplt.zlabel('Slope')
iplt.zlim([-20e-6, 0])
iplt.savefig('/www/slopes.html')

# residual computation
xx = bmapping.y_positions[:, 0]
yy = bmapping.slopes[:, -1].to(t.float64)
A = yy[:, None].repeat(1, 2); A[:, 1] = 1
residual = np.zeros(img.shape)
residual[512:, bcols] = (xx - (A @ t.inverse(A.T @ A) @ A.T) @ xx).detach().cpu()
np.save('residual.npy', residual)


# 3D scatter of y, s and s_opp
iplt.close()
echo = 150
rows = slice(echo, 512)
# cols = slice(100, 101)
# the other values of cols are busted? why
# scatter3d only seems to show last?
cols = 200
y = img_flat_orig[rows, cols].cpu()
s = img[rows].sum(axis=1)
s_opp = img[echo+512:].sum(axis=1)
iplt.scatter3d(
    s,
    s_opp,
    y
)
# iplt.xlim((detach(s.min()), np.percentile(detach(s), 70)))
iplt.zlim((y.min(), 5))
iplt.xlabel('Stat.')
iplt.ylabel('Stat. (opp)')
# iplt.zlabel('Y')
iplt.savefig('/www/echo.html')