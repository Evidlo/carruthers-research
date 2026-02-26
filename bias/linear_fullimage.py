#!/usr/bin/env python3

from torchpwl import PWL, Calibrator
from piecewise import FixedPWL
import torch as t
from tqdm import tqdm
import numpy as np

from common import load, mean_bias

import iplot as iplt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

d = dict(device='cuda')

def trimmed_norm(x, keep_ratio=0.9):
    keep_n = int(len(x) * keep_ratio)
    loss, _ = t.topk(x, keep_n, largest=False, dim=0)
    return loss.mean()

def shelfmask(s, s_opp):
    """True where s or s_opp exceeds its threshold."""
    # convert threshold to absolute (each dim uses its own range)
    thresh = lambda x: t.tensor([.007], **d) * (x.amax() - x.amin()) + x.amin()

    thresh_s = thresh(s)
    thresh_s_opp = thresh(s_opp)

    return (s >= thresh_s) | (s_opp >= thresh_s_opp)

def surface(s, s_opp, bias, slope, offset, mask):
    """Linear correction above threshold, 0 below.

    slope:  (2, C) per-column slopes for s and s_opp
    offset: (C,)   per-column intercept
    bias:   (C,)   value below threshold
    s:      (N, 1) row statistic
    Returns: (N, C)
    """
    linear = offset.unsqueeze(0) + slope[0].unsqueeze(0) * s + slope[1].unsqueeze(0) * s_opp  # (N, C)
    return t.where(mask, linear, t.ones_like(linear) * bias)

def learn_linear(y, s, s_opp, iterations=30000):
    """Fit a thresholded linear surface to y as a function of s and s_opp.

    Args:
        y (tensor): flat patch of image values.  (shape (rows, cols))
        s (tensor): row statistics (shape (rows, 1))
        s_opp (tensor): row statistics, opposite (shape (rows, 1))
        iterations (int): number of iterations

    Returns:
        callable: mapping(s, s_opp) -> correction (rows, cols)
    """
    C = y.shape[1]

    slope  = t.zeros(2, C, **d).requires_grad_(True)
    offset = y.mean(dim=0).detach().clone().requires_grad_(True)
    bias   = y.mean(dim=0).detach().clone().requires_grad_(True)
    mask   = shelfmask(s, s_opp)

    # Slopes need lr scaled by 1/s_scale: target slope is O(std_y/s_scale),
    # so Adam's step (O(lr)) matches when lr_slopes = base_lr / s_scale.
    base_lr = 0.1
    optim = t.optim.Adam([
        {'params': [offset], 'lr': base_lr},
        {'params': [bias],   'lr': base_lr},
        {'params': [slope],  'lr': base_lr / max((s.amax() - s.amin()).item(),
                                                  (s_opp.amax() - s_opp.amin()).item())},
    ])

    loss_hist = []
    for _ in (bar:=tqdm(range(iterations))):
        optim.zero_grad()

        surf = surface(s, s_opp, bias, slope, offset, mask)
        # loss = trimmed_norm((y - surf)[mask.squeeze()]**2, .8)
        loss = trimmed_norm((y - surf)**2, .8)
        loss.backward()

        loss_hist.append(float(loss))

        optim.step()
        bar.set_description(f'loss = {float(loss):.2e}')

    print(f'  loss: {loss_hist[0]:.3e} → {loss_hist[-1]:.3e}')

    def mapping(s, s_opp):
        return surface(s, s_opp, bias, slope, offset, shelfmask(s, s_opp))

    return mapping


# img = t.from_numpy(load(path:='images_20260117/oob_nfi_l0.pkl')).to(**d)
img = t.from_numpy(load(path:='images_20251117/oob_nfi_l0.pkl')).to(**d)
img_flat_pwl = img.clone()

# row to begin clipping of echo
echo = 1
echo_bias = 150

img_flat_orig = img.clone()
img_flat_orig -= mean_bias(img_flat_orig, echo_bias, echo_bias)
# img_flat_orig[echo:512, :400] -= img[echo:250, :400].mean(dim=0, keepdim=True)
# img_flat_orig[echo:512, 775:] -= img[echo:250, 775:].mean(dim=0, keepdim=True)
# img_flat_orig[-512:-echo, :400] -= img_flat_orig[-250:-echo, :400].mean(dim=0, keepdim=True)
# img_flat_orig[-512:-echo, 775:] -= img_flat_orig[-250:-echo, 775:].mean(dim=0, keepdim=True)

# --- training ---
# cols = slice(0, 400)
# rows = slice(echo, 512)
# y = img[rows, cols]
# s     = img.sum(dim=1, keepdim=True)[echo:512]
# s_opp = img.sum(dim=1, keepdim=True)[512:-echo]
# mapping = learn_linear(y, s, s_opp)
# img_flat_pwl[rows, cols] -= mapping(s, s_opp)

# from plotting import plot_profile_linear, detach
# plot_profile_linear(s, s_opp, y, mapping, col=200)
# iplt.savefig('/www/profile_linear1.html')

cols = slice(0, 400)
rows = slice(512, -echo)
y = img[rows, cols]
s     = img.sum(dim=1, keepdim=True)[512:-echo]
s_opp = img.sum(dim=1, keepdim=True)[echo:512]
mapping = learn_linear(y, s, s_opp)
img_flat_pwl[rows, cols] -= mapping(s, s_opp)

from plotting import plot_profile_linear, detach
plot_profile_linear(s, s_opp, y, mapping, col=200)
iplt.savefig('/www/profile_linear2.html')

# cols = slice(775, None)
# rows = slice(echo, 512)
# y = img[rows, cols]
# s     = img.sum(dim=1, keepdim=True)[echo:512]
# s_opp = img.sum(dim=1, keepdim=True)[512:-echo]
# mapping = learn_linear(y, s, s_opp)
# img_flat_pwl[rows, cols] -= mapping(s, s_opp)

# cols = slice(775, None)
# rows = slice(512, -echo)
# y = img[rows, cols]
# s     = img.sum(dim=1, keepdim=True)[512:-echo]
# s_opp = img.sum(dim=1, keepdim=True)[echo:512]
# mapping = learn_linear(y, s, s_opp)
# img_flat_pwl[rows, cols] -= mapping(s, s_opp)


# ----- plotting -----
# %% plot

from plotting import detach

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
iplt.savefig('/www/flatlinear.html', dpi=400)


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