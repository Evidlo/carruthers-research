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

NCOLS = 1024
IMAGE_LIST = [
    # (path, (nonflat_start, nonflat_stop)) — nonflat cols are excluded from fit
    # ('../images_20260111/oob_nfi_l0.pkl', (400, 750)),
    # ('../images_20260115/oob_nfi_l0.pkl', (400, 750)),
    # ('../images_20260117/oob_nfi_l0.pkl', (400, 750)),
    # ('../images_20260119/oob_nfi_l0.pkl', (400, 750)),
    ('../images_20260316/oob_nfi_l0.pkl', (333, 666)),
    ('../images_20260317/oob_nfi_l0.pkl', (333, 666)),
    ('../images_20260318/star_nfi_l0.pkl', (800, 1024)), # off-center
    ('../images_20260319/oob_nfi_l0.pkl', (333, 666)),
]
hot_pixels = np.load('hot_pixels.npy')


def flat_cols(nonflat):
    lo, hi = nonflat
    return np.array(list(range(lo)) + list(range(hi, NCOLS)))

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


def fit_and_extract(img_np, flat_idx, half='top'):
    """Fit model, return per-column median residual in sag region."""
    img = torch.from_numpy(img_np).to(device)
    bias = torch.from_numpy(rob_bias(img_np, clip_out=150, clip_in=150)).to(device)
    rs = img.sum(dim=1)

    fit_slice = fit_slices[half]
    s = rs[fit_slice].unsqueeze(1)
    y = img[fit_slice][:, flat_idx]
    b = bias[half_rows[half], flat_idx]

    m = model_module.Model(b, s).to(device)
    fit_model(m, y, b, s)

    with torch.no_grad():
        pred = m(b, s).cpu().numpy()
        y_np = y.cpu().numpy()
        sag_sl = sag_rel[half]
        sag_residual = np.median((y_np - pred)[sag_sl], axis=0)
        pwl_sag_med = float(m.pwl(s[sag_sl]).mean())

    return sag_residual, pwl_sag_med, y_np - pred


# ---- Fit all images, both halves ----
results = {}
flat_idx_by_name = {}
flat_image_by_name = {}
for path, nonflat in IMAGE_LIST:
    name = path.split('/')[-2]
    flat_idx = flat_cols(nonflat)
    flat_idx_by_name[name] = flat_idx
    print(f"\n{'='*50}")
    print(f"Processing {name}")
    img_np = load_and_prep(path)
    flat_img = np.full((1024, NCOLS), np.nan)
    for half in ['top', 'bot']:
        print(f"  --- {half} half ---")
        resid, pwl_med, residual_block = fit_and_extract(img_np, flat_idx, half)
        results[(name, half)] = {'residual': resid, 'pwl_med': pwl_med}
        rows = np.arange(fit_slices[half].start, fit_slices[half].stop)
        flat_img[np.ix_(rows, flat_idx)] = residual_block
    flat_image_by_name[name] = flat_img


# ---- Plot: cross-image residual correlations ----
image_names = [p.split('/')[-2] for p, _ in IMAGE_LIST]
n_images = len(image_names)

fig, axes = plt.subplots(2, n_images - 1, figsize=(5 * (n_images - 1), 10))

def aligned(n1, n2, half):
    """Return residuals from n1, n2 restricted to columns flat in both."""
    common = np.intersect1d(flat_idx_by_name[n1], flat_idx_by_name[n2])
    r1_full = np.full(NCOLS, np.nan)
    r1_full[flat_idx_by_name[n1]] = results[(n1, half)]['residual']
    r2_full = np.full(NCOLS, np.nan)
    r2_full[flat_idx_by_name[n2]] = results[(n2, half)]['residual']
    return r1_full[common], r2_full[common]


for half_idx, half in enumerate(['top', 'bot']):
    ref_name = image_names[0]

    for i, name in enumerate(image_names[1:]):
        ax = axes[half_idx, i]
        ref_resid, other_resid = aligned(ref_name, name, half)
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
fig.savefig(f'{out_dir}/residual.png', dpi=150)
print(f"\nSaved {out_dir}/residual.png")

# ---- Plot: flattened image per source ----
fig2, axes2 = plt.subplots(1, n_images, figsize=(5 * n_images, 5), squeeze=False)
for i, name in enumerate(image_names):
    ax = axes2[0, i]
    im = ax.imshow(flat_image_by_name[name], cmap='RdBu_r', vmin=-10, vmax=10)
    ax.set_title(name)
    plt.colorbar(im, ax=ax, fraction=0.046)
fig2.suptitle(f'Flattened residual images — {model_name}', fontsize=14)
fig2.tight_layout()
fig2.savefig(f'{out_dir}/residual_images.png', dpi=150)
print(f"Saved {out_dir}/residual_images.png")

# ---- Print correlation matrix ----
for half in ['top', 'bot']:
    print(f"\n{half.upper()} half — cross-image residual correlations:")
    for i, n1 in enumerate(image_names):
        for j, n2 in enumerate(image_names):
            if j <= i:
                continue
            a, b = aligned(n1, n2, half)
            r_p, _ = pearsonr(a, b)
            r_s, _ = spearmanr(a, b)
            print(f"  {n1} vs {n2}: pearson={r_p:.3f}  spearman={r_s:.3f}")

# ---- Compute and save c_j (1024 columns, zeros for in-band) ----
# c_j = sag_residual / median(PWL(s_sag)), averaged across images
for half in ['top', 'bot']:
    all_cj = []
    for name in image_names:
        r = results[(name, half)]
        cj_full = np.full(NCOLS, np.nan)
        cj_full[flat_idx_by_name[name]] = r['residual'] / r['pwl_med']
        all_cj.append(cj_full)
    with np.errstate(all='ignore'):
        cj_avg = np.nanmean(np.stack(all_cj), axis=0)
    cj_avg = np.where(np.isnan(cj_avg), 0.0, cj_avg)
    np.save(f'cj_{half}.npy', cj_avg)
    print(f"\nSaved cj_{half}.npy — mean={cj_avg.mean():.1f}, std={cj_avg.std():.1f}")
