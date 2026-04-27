#!/usr/bin/env python3
"""Compare c_j estimated from OOB images vs new off-center star image."""

import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import load, rob_bias
from model_sharedpwl import Model
from fit import fit_model

device = 'cuda'
out_dir = '/www/claude_flatten'

hot_pixels = np.load('hot_pixels.npy')
sag_rel = {'top': slice(212, 362), 'bot': slice(0, 150)}
fit_slices = {'top': slice(150, 512), 'bot': slice(512, 874)}
half_rows = {'top': 0, 'bot': 512}


def load_and_prep(path):
    img_np = load(path).astype(np.float64)
    img_np[hot_pixels[:, 0], hot_pixels[:, 1]] = np.nan
    for c in range(img_np.shape[1]):
        col = img_np[:, c]; mask = np.isnan(col)
        if mask.any(): col[mask] = np.nanmedian(col)
    return img_np


def compute_cj(path, flat_cols):
    """Fit sharedpwl with c=None, return 1024-element c_j arrays for top and bot."""
    img_np = load_and_prep(path)
    img = torch.from_numpy(img_np).to(device)
    bias = torch.from_numpy(rob_bias(img_np, clip_out=150, clip_in=150)).to(device)
    rs = img.sum(dim=1)

    cj_full = {}
    for half in ['top', 'bot']:
        fit_slice = fit_slices[half]
        s = rs[fit_slice].unsqueeze(1)
        y = img[fit_slice][:, flat_cols]
        b = bias[half_rows[half], flat_cols]

        m = Model(b, s).to(device)
        fit_model(m, y, b, s)

        with torch.no_grad():
            pred = m(b, s).cpu().numpy()
            sag_sl = sag_rel[half]
            sag_resid = np.median((y.cpu().numpy() - pred)[sag_sl], axis=0)
            pwl_med = float(m.pwl(s[sag_sl]).mean())

        full = np.zeros(1024)
        full[flat_cols] = sag_resid / pwl_med
        cj_full[half] = full

    return cj_full


print("Fitting star image...")
star_flat = np.arange(900)
cj_star = compute_cj('../images_20260318/star_nfi_l0.pkl', star_flat)

cj_oob = {'top': np.load('cj_top.npy'), 'bot': np.load('cj_bot.npy')}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, half in zip(axes, ['top', 'bot']):
    x = cj_oob[half]
    y = cj_star[half]
    ax.scatter(x, y, s=2, alpha=0.3)
    lim = max(np.abs(x).max(), np.abs(y).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'r-', lw=1, label='y=x')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('c_j (OOB images)')
    ax.set_ylabel('c_j (star image)')
    ax.set_title(f'{half} half')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)

fig.suptitle('c_j consistency: OOB images vs star image', fontsize=13)
fig.tight_layout()
fig.savefig(f'{out_dir}/cj_compare.png', dpi=150)
print(f"Saved {out_dir}/cj_compare.png")
