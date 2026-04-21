#!/usr/bin/env python3
"""Analyze per-column PWL parameters to check if slopes are predictable from b_j."""

import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from common import load, rob_bias
from model_pwl import Model
from fit import fit_model

device = 'cuda'
out_dir = '/www/claude_flatten'

# ---- Load data ----
img_np = load('../images_20260115/oob_nfi_l0.pkl').astype(np.float64)
hp = np.load('hot_pixels.npy')
img_np[hp[:, 0], hp[:, 1]] = np.nan
for c in range(1024):
    col = img_np[:, c]; m = np.isnan(col)
    if m.any(): col[m] = np.nanmedian(col)

img = torch.from_numpy(img_np).to(device)
bias = torch.from_numpy(rob_bias(img_np, clip_out=150, clip_in=150)).to(device)
rs = img.sum(dim=1)

oob_idx = list(range(400)) + list(range(750, 1024))

# ---- Fit both halves ----
results = {}
for label, fit_slice, half_row in [('Top', slice(150, 512), 0), ('Bot', slice(512, 874), 512)]:
    print(f"=== Fitting {label} half ===")
    s = rs[fit_slice].unsqueeze(1)
    y = torch.cat([img[fit_slice, :400], img[fit_slice, 750:]], dim=1)
    b = bias[half_row, oob_idx]
    m = Model(b, s).to(device)
    fit_model(m, y, b, s, iterations=3000)

    slopes = m.primary._slopes.detach().cpu().numpy()
    biases_learned = m.primary.biases.detach().cpu().numpy()
    b_np = b.cpu().numpy()

    results[label] = {
        'slopes': slopes,
        'biases_learned': biases_learned,
        'b': b_np,
    }

# ---- Plot ----
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for row_idx, label in enumerate(['Top', 'Bot']):
    r = results[label]
    b_np = r['b']
    slopes = r['slopes']
    biases_learned = r['biases_learned']
    slope_names = ['left_tail', 'seg1', 'seg2', 'right_tail']

    # One scatter+fit per non-zero slope segment
    for col_idx, seg_idx in enumerate([1, 2, 3]):
        ax = axes[row_idx, col_idx]
        name = slope_names[seg_idx]
        seg_slope = slopes[:, seg_idx]
        ax.scatter(b_np, seg_slope, s=2, alpha=0.3)
        coeffs = np.polyfit(b_np, seg_slope, 1)
        x_fit = np.array([b_np.min(), b_np.max()])
        ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r-', lw=2)
        r_val, _ = pearsonr(b_np, seg_slope)
        ax.set_xlabel('b_j (rob_bias)')
        ax.set_ylabel(f'{name} slope')
        ax.set_title(f'{label}: {name} vs b_j (r={r_val:.3f})')

fig.suptitle('PWL Parameter Analysis: Are per-column slopes predictable from b_j?', fontsize=14)
fig.tight_layout()
fig.savefig(f'{out_dir}/pwl_explore.png', dpi=150)
print(f"\nSaved {out_dir}/pwl_explore.png")

# ---- Print summary ----
for label in ['Top', 'Bot']:
    r = results[label]
    right_slope = r['slopes'][:, -1]
    ratio = right_slope / r['b']
    corr, _ = pearsonr(r['b'], right_slope)
    print(f"\n{label}:")
    print(f"  right_slope/b_j: mean={ratio.mean():.3e}, std={ratio.std():.3e}, CV={ratio.std()/abs(ratio.mean()):.2f}")
    print(f"  corr(b_j, right_slope): {corr:.4f}")
