#!/usr/bin/env python3
"""Check if per-column sag residuals are consistent across images.

Usage: python residual_explore.py model_sharedpwl
"""

import sys
sys.path.insert(0, '..')

import importlib
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, theilslopes

from common import load, rob_bias
from fit import fit_model

device = 'cuda'
out_dir = '/www/claude_flatten'

model_name = sys.argv[1] if len(sys.argv) > 1 else 'model_pwl'
model_module = importlib.import_module(model_name)
print(f"Using model: {model_name}")

image_paths = [
    '../images_20260111/oob_nfi_l0.pkl',
    '../images_20260115/oob_nfi_l0.pkl',
    '../images_20260117/oob_nfi_l0.pkl',
    '../images_20260119/oob_nfi_l0.pkl',
]
hot_pixels = np.load('hot_pixels.npy')
oob_idx = list(range(400)) + list(range(750, 1024))

# Sag rows relative to each half's fit_slice
# top fit: 150-511, sag: 362-511 → relative slice(212, 362)
# bot fit: 512-873, sag: 512-661 → relative slice(0, 150)
sag_rel = {'top': slice(212, 362), 'bot': slice(0, 150)}
fit_slices = {'top': slice(150, 512), 'bot': slice(512, 874)}
half_rows = {'top': 0, 'bot': 512}


def load_and_prep(path):
    img_np = load(path).astype(np.float64)
    img_np[hot_pixels[:, 0], hot_pixels[:, 1]] = np.nan
    for c in range(1024):
        col = img_np[:, c]; m = np.isnan(col)
        if m.any(): col[m] = np.nanmedian(col)
    return img_np


def fit_and_extract(img_np, half='top'):
    """Fit model, return per-column median residual in sag region."""
    img = torch.from_numpy(img_np).to(device)
    bias = torch.from_numpy(rob_bias(img_np, clip_out=150, clip_in=150)).to(device)
    rs = img.sum(dim=1)

    fit_slice = fit_slices[half]
    s = rs[fit_slice].unsqueeze(1)
    y = img[fit_slice][:, oob_idx]
    b = bias[half_rows[half], oob_idx]

    m = model_module.Model(b, s).to(device)
    fit_model(m, y, b, s)

    with torch.no_grad():
        pred = m(b, s).cpu().numpy()
        sag_sl = sag_rel[half]
        sag_residual = np.median((y.cpu().numpy() - pred)[sag_sl], axis=0)
        pwl_sag_med = float(m.pwl(s[sag_sl]).mean())

    return sag_residual, pwl_sag_med


# ---- Fit all images, both halves ----
results = {}
for path in image_paths:
    name = path.split('/')[-2]
    print(f"\n{'='*50}")
    print(f"Processing {name}")
    img_np = load_and_prep(path)
    for half in ['top', 'bot']:
        print(f"  --- {half} half ---")
        resid, pwl_med = fit_and_extract(img_np, half)
        results[(name, half)] = {'residual': resid, 'pwl_med': pwl_med}


# ---- Plot: cross-image residual correlations ----
image_names = [p.split('/')[-2] for p in image_paths]
n_images = len(image_names)

fig, axes = plt.subplots(2, n_images - 1, figsize=(5 * (n_images - 1), 10))

for half_idx, half in enumerate(['top', 'bot']):
    ref_name = image_names[0]
    ref_resid = results[(ref_name, half)]['residual']

    for i, name in enumerate(image_names[1:]):
        ax = axes[half_idx, i]
        other_resid = results[(name, half)]['residual']
        r_p, _ = pearsonr(ref_resid, other_resid)
        r_s, _ = spearmanr(ref_resid, other_resid)

        ax.scatter(ref_resid, other_resid, s=2, alpha=0.3)

        ts_slope, ts_intercept, _, _ = theilslopes(other_resid, ref_resid)
        x_fit = np.array([ref_resid.min(), ref_resid.max()])
        ax.plot(x_fit, ts_slope * x_fit + ts_intercept, 'r-', lw=2,
                label=f'slope={ts_slope:.2f}')
        ax.legend(fontsize=7)

        ax.set_xlabel(f'Sag residual ({ref_name})')
        ax.set_ylabel(f'Sag residual ({name})')
        ax.set_title(f'{half}: pearson={r_p:.3f} spearman={r_s:.3f}')
        ax.set_aspect('equal')

fig.suptitle(f'Cross-image sag residual consistency — {model_name}', fontsize=14)
fig.tight_layout()
fig.savefig(f'{out_dir}/residual_explore_{model_name}.png', dpi=150)
print(f"\nSaved {out_dir}/residual_explore_{model_name}.png")

# ---- Print correlation matrix ----
for half in ['top', 'bot']:
    print(f"\n{half.upper()} half — cross-image residual correlations:")
    for i, n1 in enumerate(image_names):
        for j, n2 in enumerate(image_names):
            if j <= i:
                continue
            r_p, _ = pearsonr(results[(n1, half)]['residual'], results[(n2, half)]['residual'])
            r_s, _ = spearmanr(results[(n1, half)]['residual'], results[(n2, half)]['residual'])
            print(f"  {n1} vs {n2}: pearson={r_p:.3f}  spearman={r_s:.3f}")

# ---- Compute and save c_j (1024 columns, zeros for in-band) ----
# c_j = sag_residual / median(PWL(s_sag)), averaged across images
for half in ['top', 'bot']:
    all_cj = []
    for name in image_names:
        r = results[(name, half)]
        cj_full = np.zeros(1024)
        cj_full[oob_idx] = r['residual'] / r['pwl_med']
        all_cj.append(cj_full)
    cj_avg = np.mean(all_cj, axis=0)
    np.save(f'cj_{half}.npy', cj_avg)
    print(f"\nSaved cj_{half}.npy — mean={cj_avg.mean():.1f}, std={cj_avg.std():.1f}")
