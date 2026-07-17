#!/usr/bin/env python3

# This file uses a model where s_i is a sum over x_ij + b_j
#
# y_ij = (x_ij + b_j)(1 - σ_j · s_i) + b_0
#
# s_i = Σ_j x_ij + b_j
#
# Underdetermined without prior on x_ij. We assume x_ij = K (per-half scalar)
# everywhere outside the earth box (rows 300-749 ∩ cols 300-749) and outside
# echo rows (first/last 100). σ_j is independent per column per half.
#
# K and b_0 are NOT separately identifiable on non-earth alone (1-param
# family of equivalent fits). We break the degeneracy two ways:
#   FB0: fix b_0 to a known value per half; fit σ_j, K.
#   FK : fix K = 0; fit σ_j, b_0.

import sys
sys.path.insert(0, '..')

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import load, rob_bias

IMG_PATH = '../images_20260113/oob_nfi_l0.pkl'
ECHO_TRIM = 100
EARTH_R0, EARTH_R1 = 300, 750
EARTH_C0, EARTH_C1 = 300, 750
SELECTED_COLS = [100, 400, 600, 900]
HALVES = {'top': (0, 512), 'bot': (512, 1024)}
B0_FIXED = {'top': 270 * 8, 'bot': 203 * 8}
OUT = Path('/www/alex'); OUT.mkdir(parents=True, exist_ok=True)


img = load(IMG_PATH).astype(np.float64)
col_med = np.nanmedian(img, axis=0)
nan_mask = ~np.isfinite(img)
img[nan_mask] = np.broadcast_to(col_med, img.shape)[nan_mask]
b_j = rob_bias(img, 150, 200, percent=40)


def half_data(half):
    r0, r1 = HALVES[half]
    fit_r0, fit_r1 = max(r0, ECHO_TRIM), min(r1, 1024 - ECHO_TRIM)
    s = img[r0:r1].sum(axis=1)[fit_r0 - r0:fit_r1 - r0]
    y = img[fit_r0:fit_r1].copy()
    b = b_j[r0]
    rows_abs = np.arange(fit_r0, fit_r1)
    earth_row = (rows_abs >= EARTH_R0) & (rows_abs < EARTH_R1)
    earth_col = np.zeros(1024, bool); earth_col[EARTH_C0:EARTH_C1] = True
    mask = ~(earth_row[:, None] & earth_col[None, :])
    return s, y, b, mask, fit_r0, fit_r1


def fit_fixed_b0(s, y, b_init, mask, b0, n_iter=40):
    """Fix b_0; fit σ_j, b_j, K. Loss: Σ m ((K+b_j)(1-σ_j s) + b_0 - y)²."""
    sigma = np.zeros(1024)
    bj = b_init.copy()
    K = float(np.median(y[mask] - np.broadcast_to(b_init, y.shape)[mask]) - b0)
    for _ in range(n_iter):
        A = K + bj
        ms = mask * s[:, None]
        # σ_j update
        num = (ms * (A[None, :] + b0 - y)).sum(axis=0)
        den = A * (mask * (s[:, None] ** 2)).sum(axis=0)
        sigma = num / den
        r = 1 - sigma[None, :] * s[:, None]
        mr = mask * r
        # b_j update: b_j = Σ m r (y - b_0) / Σ m r² - K
        bj = (mr * (y - b0)).sum(axis=0) / (mr * r).sum(axis=0) - K
        # K update
        r = 1 - sigma[None, :] * s[:, None]
        mr = mask * r
        K = ((mr * (y - b0)).sum() - (mr * bj[None, :] * r).sum()) / (mr * r).sum()
    return sigma, bj, K


def fit_fixed_K(s, y, b_init, mask, K, n_iter=40):
    """Fix K; fit σ_j, b_j, b_0. Loss: Σ m ((K+b_j)(1-σ_j s) + b_0 - y)²."""
    sigma = np.zeros(1024)
    bj = b_init.copy()
    b0 = 0.0
    for _ in range(n_iter):
        A = K + bj
        ms = mask * s[:, None]
        # σ_j update
        num = (ms * (A[None, :] + b0 - y)).sum(axis=0)
        den = A * (mask * (s[:, None] ** 2)).sum(axis=0)
        sigma = num / den
        r = 1 - sigma[None, :] * s[:, None]
        mr = mask * r
        # b_j update
        bj = (mr * (y - b0)).sum(axis=0) / (mr * r).sum(axis=0) - K
        # b_0 update
        A = K + bj
        r = 1 - sigma[None, :] * s[:, None]
        b0 = (mask * (y - A[None, :] * r)).sum() / mask.sum()
    return sigma, bj, b0


def recover(s, y, bj, sigma, b0):
    return (y - b0) / (1 - sigma[None, :] * s[:, None]) - bj[None, :]


def grade(x, K):
    empty = np.r_[np.arange(0, EARTH_C0), np.arange(EARTH_C1, 1024)]
    sub = x[:, empty] - K
    return float(np.std(np.median(sub, axis=1))), float(np.std(np.median(sub, axis=0)))


# ---- run ----
results = {}
for half in HALVES:
    s, y, b, mask, fit_r0, fit_r1 = half_data(half)
    # FB0: fix b_0
    b0_fix = B0_FIXED[half]
    sig_FB0, bj_FB0, K_FB0 = fit_fixed_b0(s, y, b, mask, b0_fix)
    # FK: fix K=0
    sig_FK, bj_FK, b0_FK = fit_fixed_K(s, y, b, mask, 0.0)
    results[half] = {
        'data': (s, y, b, mask, fit_r0, fit_r1),
        'FB0': (sig_FB0, K_FB0, b0_fix, recover(s, y, bj_FB0, sig_FB0, b0_fix)),
        'FK':  (sig_FK,  0.0,    b0_FK,  recover(s, y, bj_FK,  sig_FK,  b0_FK)),
    }

# ---- grade card ----
print(f'{"":<5} {"top row σ":>10} {"top col σ":>10} {"bot row σ":>10} {"bot col σ":>10} {"jump":>7} {"K_top":>10} {"K_bot":>10} {"b0_top":>10} {"b0_bot":>10}')
for ap in ('FB0', 'FK'):
    xt = results['top'][ap][3]; xb = results['bot'][ap][3]
    Kt = results['top'][ap][1]; Kb = results['bot'][ap][1]
    b0t = results['top'][ap][2]; b0b = results['bot'][ap][2]
    rs_t, cs_t = grade(xt, Kt)
    rs_b, cs_b = grade(xb, Kb)
    empty = np.r_[np.arange(0, EARTH_C0), np.arange(EARTH_C1, 1024)]
    top_last = np.median(xt[-1, empty]) - Kt
    bot_first = np.median(xb[0, empty]) - Kb
    jump = abs(top_last - bot_first)
    print(f'{ap:<5} {rs_t:>10.3f} {cs_t:>10.3f} {rs_b:>10.3f} {cs_b:>10.3f} {jump:>7.3f} {Kt:>10.2f} {Kb:>10.2f} {b0t:>10.2f} {b0b:>10.2f}')


# ---- scatter plot: 2 halves × 4 cols ----
fig, axes = plt.subplots(2, len(SELECTED_COLS), figsize=(5 * len(SELECTED_COLS), 8), squeeze=False)
for hi, half in enumerate(('top', 'bot')):
    s, y, b, mask, _, _ = results[half]['data']
    for ci, col in enumerate(SELECTED_COLS):
        ax = axes[hi, ci]
        m = mask[:, col]
        ax.scatter(s[~m], y[~m, col], s=3, alpha=0.4, color='red', label='earth (excluded)')
        ax.scatter(s[m], y[m, col], s=3, alpha=0.5, color='gray', label='non-earth (fit)')
        ss = np.linspace(s.min(), s.max(), 200)
        for ap, color in (('FB0', 'C0'), ('FK', 'C1')):
            sig, K, b0, _ = results[half][ap]
            ax.plot(ss, (K + b[col]) * (1 - sig[col] * ss) + b0, color=color,
                    label=f'{ap} σ={sig[col]:.2e}')
        ax.set_title(f'{half} col {col}')
        ax.set_xlabel('s_i'); ax.set_ylabel(f'y_i,{col}')
        ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / 'scatter.png', dpi=100)
plt.close(fig)
print(f'\nSaved {OUT / "scatter.png"}')


# ---- x image plot: 2 rows (raw / dev) × 2 approaches ----
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ai, ap in enumerate(('FB0', 'FK')):
    full = np.full_like(img, np.nan)
    dev = np.full_like(img, np.nan)
    for half in ('top', 'bot'):
        d = results[half]['data']
        x = results[half][ap][3]
        K = results[half][ap][1]
        full[d[4]:d[5]] = x
        dev[d[4]:d[5]] = x - K

    im0 = axes[0, ai].imshow(full, cmap='viridis', vmin=np.nanpercentile(full, 1), vmax=np.nanpercentile(full, 99))
    axes[0, ai].set_title(f'{ap}: x (recovered signal)')
    plt.colorbar(im0, ax=axes[0, ai], fraction=0.046)

    im1 = axes[1, ai].imshow(dev, cmap='RdBu_r', vmin=-10, vmax=10)
    axes[1, ai].set_title(f'{ap}: x - K  (flatness)')
    plt.colorbar(im1, ax=axes[1, ai], fraction=0.046)

fig.tight_layout()
fig.savefig(OUT / 'flattened.png', dpi=100)
plt.close(fig)
print(f'Saved {OUT / "flattened.png"}')
