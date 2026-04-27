#!/usr/bin/env python3
"""Standalone global-c_j derivation + flatten with sharedpwl.

Two passes:
  1. Per image in IMAGE_LIST: fit sharedpwl (c=0) on that image's flat columns,
     extract c_j from sagged-row residuals. Non-flat cols left as NaN.
     Aggregate across images with nanmean; uncovered cols fall back to 0.
  2. Refit the image at FLATTEN_INDEX with the global c_j applied, subtract the
     fitted prediction on its flat cols, print grade card, save error plot.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import load, rob_bias
from fit import fit_model
from model_sharedpwl import Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Config ----
NCOLS = 1024
IMAGE_LIST = [
    # (path, (nonflat_start, nonflat_stop)) — nonflat cols are excluded from fit
    ('../images_20260316/oob_nfi_l0.pkl', (333, 666)),
    ('../images_20260317/oob_nfi_l0.pkl', (333, 666)),
    ('../images_20260318/star_nfi_l0.pkl', (800, 1024)), # off-center
    ('../images_20260319/oob_nfi_l0.pkl', (333, 666)),
    # ('../images_20260113/oob_nfi_l0.pkl', (333, 666)),
    # ('../images_20260115/oob_nfi_l0.pkl', (333, 666)),
    # ('../images_20260117/oob_nfi_l0.pkl', (333, 666)),
]
FLATTEN_INDEX = 0


def flat_cols(nonflat):
    lo, hi = nonflat
    return np.array(list(range(lo)) + list(range(hi, NCOLS)))

ECHO_TRIM = 150
FIT_ROWS = {'top': (ECHO_TRIM, 512), 'bot': (512, 1024 - ECHO_TRIM)}
HALF_ROW = {'top': 0, 'bot': 512}
CJ_SAG_REL = {'top': slice(212, 362), 'bot': slice(0, 150)}

HOT_PIXELS = np.load(Path(__file__).parent / 'hot_pixels.npy')


def load_and_prep(path):
    img = load(path).astype(np.float64)
    img[HOT_PIXELS[:, 0], HOT_PIXELS[:, 1]] = np.nan
    for c in range(img.shape[1]):
        col = img[:, c]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)
    return img


def prep_image(path):
    img_np = load_and_prep(path)
    bias_np = rob_bias(img_np, clip_out=100, clip_in=200)
    img_t = torch.from_numpy(img_np).to(device)
    bias_t = torch.from_numpy(bias_np).to(device)
    rs_t = img_t.sum(dim=1)
    return img_np, bias_np, img_t, bias_t, rs_t


def fit_half(img_t, bias_t, rs_t, flat_idx, half, c_full=None):
    """Fit sharedpwl on one half. Returns (model, s, y, b)."""
    r0, r1 = FIT_ROWS[half]
    s = rs_t[r0:r1].unsqueeze(1)
    y = img_t[r0:r1][:, flat_idx]
    b = bias_t[HALF_ROW[half], flat_idx]
    c = torch.from_numpy(c_full[flat_idx]).to(device) if c_full is not None else None

    m = Model(b, s, c=c).to(device)
    fit_model(m, y, b, s)
    return m, s, y, b


# ---- Pass 1: per-image c_j estimates ----
cj_stack = {'top': [], 'bot': []}
for path, nonflat in IMAGE_LIST:
    flat_idx = flat_cols(nonflat)
    print(f'\n[pass 1] {path}')
    _, _, img_t, bias_t, rs_t = prep_image(path)
    for half in ('top', 'bot'):
        m, s, y, b = fit_half(img_t, bias_t, rs_t, flat_idx, half)
        with torch.no_grad():
            pred = m(b, s).cpu().numpy()
            sag_sl = CJ_SAG_REL[half]
            sag_resid = np.median((y.cpu().numpy() - pred)[sag_sl], axis=0)
            pwl_med = float(m.pwl(s[sag_sl]).mean())
        cj_full = np.full(1024, np.nan)
        if abs(pwl_med) > 1e-10:
            cj_full[flat_idx] = sag_resid / pwl_med
        cj_stack[half].append(cj_full)

cj_global = {}
for h in ('top', 'bot'):
    stacked = np.stack(cj_stack[h])
    with np.errstate(all='ignore'):
        cj_global[h] = np.nanmean(stacked, axis=0)
    print(f'{h}: {int(np.isnan(cj_global[h]).sum())} uncovered columns')


# ---- Pass 2: refit selected image with global c_j ----
sel_path, _ = IMAGE_LIST[FLATTEN_INDEX]
sel_flat = np.where(~np.isnan(cj_global['top']) & ~np.isnan(cj_global['bot']))[0]
print(f'\n[pass 2] flattening {sel_path}')
img_np, bias_np, img_t, bias_t, rs_t = prep_image(sel_path)

img_flat = img_np - bias_np
for half in ('top', 'bot'):
    m, s, y, b = fit_half(img_t, bias_t, rs_t, sel_flat, half, c_full=cj_global[half])
    with torch.no_grad():
        pred = m(b, s).cpu().numpy()
    resid = y.cpu().numpy() - pred
    r0, r1 = FIT_ROWS[half]
    rows = np.arange(r0, r1)
    img_flat[np.ix_(rows, sel_flat)] = resid


# ---- Grade card ----
unsag_rows = list(range(150, 362)) + list(range(662, 874))
sag_rows = list(range(362, 512)) + list(range(512, 662))
metric_rows = slice(ECHO_TRIM, 1024 - ECHO_TRIM)

oob = img_flat[metric_rows][:, sel_flat]
oob_unsag = img_flat[unsag_rows][:, sel_flat]
oob_sag = img_flat[sag_rows][:, sel_flat]
mid = 512 - ECHO_TRIM
jump = abs(np.median(oob[mid - 1]) - np.median(oob[mid]))

label = Path(sel_path).parent.name
print('\n' + '=' * 40)
print(f'Grade Card: {label}')
print(f'  Row Flatness unsag (σ): {np.std(np.median(oob_unsag, axis=1)):.3f}  (target: 0.22)')
print(f'  Col Flatness unsag (σ): {np.std(np.median(oob_unsag, axis=0)):.3f}  (target: 0.26)')
print(f'  Row Flatness sag   (σ): {np.std(np.median(oob_sag, axis=1)):.3f}')
print(f'  Col Flatness sag   (σ): {np.std(np.median(oob_sag, axis=0)):.3f}')
print(f'  Half-Half Jump:         {jump:.3f}  (target: < 0.5)')
print('=' * 40)


# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(img_flat, cmap='RdBu_r', vmin=-10, vmax=10)
plt.colorbar(im, ax=ax, label='counts')
ax.set_title(f'Flattened error: {label}')
out_png = '/www/flattened.png'
Path(out_png).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=100, bbox_inches='tight')
print(f'\nSaved {out_png}')
