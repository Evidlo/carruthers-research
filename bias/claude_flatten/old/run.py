#!/usr/bin/env python3
"""Shared run script. Usage: python run.py model_pwl  (or model_parametric, etc.)"""

import sys
sys.path.insert(0, '..')

import importlib
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import load, rob_bias
from fit import fit_model

device = 'cuda'
out_dir = '/www/claude_flatten'

# ---- Parse model choice ----
model_name = sys.argv[1] if len(sys.argv) > 1 else 'model_pwl'
model_module = importlib.import_module(model_name)
print(f"Using model: {model_name}")

# ---- Load data ----
image_path = '../images_20260113/oob_nfi_l0.pkl'; flat_idx = np.array(list(range(400)) + list(range(750, 1024)))
image_path = '../images_20260115/oob_nfi_l0.pkl'; flat_idx = np.array(list(range(400)) + list(range(750, 1024)))
# image_path = '../images_20260117/oob_nfi_l0.pkl'; flat_idx = np.array(list(range(400)) + list(range(750, 1024)))
# image_path = '../images_20260318/star_nfi_l0.pkl'; flat_idx = np.array(list(range(850)))
image_path = '../images_20260316/oob_nfi_l0.pkl'; flat_idx = np.array(list(range(400)) + list(range(750, 1024)))
img_np = load(image_path).astype(np.float64)
hot_pixels = np.load('hot_pixels.npy')

img_np[hot_pixels[:, 0], hot_pixels[:, 1]] = np.nan
for c in range(img_np.shape[1]):
    col = img_np[:, c]
    mask = np.isnan(col)
    if mask.any():
        col[mask] = np.nanmedian(col)

img = torch.from_numpy(img_np).to(device)

# ---- Bias ----
bias = torch.from_numpy(rob_bias(img_np, clip_out=150, clip_in=150)).to(device)
img_bias_sub = img - bias

# ---- Row sums ----
row_sums = img.sum(dim=1)

# ---- Configuration ----
echo_trim = 150  # rows to trim from edges for metrics
# flatten_idx = Ellipsis  # columns to fit/correct; Ellipsis = all, or list(range(400))+list(range(750,1024))
flatten_idx =flat_idx
flatten_idx = np.arange(img_np.shape[1])[flatten_idx]

# Fitting rows: skip echo rows (primary sag only)
fit_top = slice(echo_trim, 512)
fit_bot = slice(512, 1024 - echo_trim)
metric_rows = slice(echo_trim, 1024 - echo_trim)

# ---- Load c_j if available ----
import os
def load_cj(half):
    path = f'cj_{half}.npy'
    if os.path.exists(path):
        return torch.from_numpy(np.load(path)).to(device)
    return torch.zeros(1024, dtype=torch.float64, device=device)

# ---- Fit top half ----
print("=== Fitting top half ===")
s_top = row_sums[fit_top].unsqueeze(1)
y_top = img[fit_top][:, flatten_idx]
b_top = bias[0, flatten_idx]
c_top = load_cj('top')[flatten_idx]

model_top = model_module.Model(b_top, s_top, c=c_top).to(device)
loss_top = fit_model(model_top, y_top, b_top, s_top)

# ---- Fit bottom half ----
print("\n=== Fitting bottom half ===")
s_bot = row_sums[fit_bot].unsqueeze(1)
y_bot = img[fit_bot][:, flatten_idx]
b_bot = bias[512, flatten_idx]
c_bot = load_cj('bot')[flatten_idx]

model_bot = model_module.Model(b_bot, s_bot, c=c_bot).to(device)
loss_bot = fit_model(model_bot, y_bot, b_bot, s_bot)

# ---- Apply corrections (OOB columns only) ----
img_corrected = img - bias  # baseline everywhere

with torch.no_grad():
    corr_top = model_top(b_top, s_top)
    img_corrected[fit_top, flatten_idx] = img[fit_top][:, flatten_idx] - corr_top

    corr_bot = model_bot(b_bot, s_bot)
    img_corrected[fit_bot, flatten_idx] = img[fit_bot][:, flatten_idx] - corr_bot


# ---- Save surfaces for debug_interactive ----
image_tag = image_path.split('/')[-2]
surface_path = f'surfaces/{model_name}_{image_tag}.npz'
with torch.no_grad():
    # Recompute predictions on full OOB for both halves
    pred_top = model_top(b_top, s_top).cpu().numpy()
    pred_bot = model_bot(b_bot, s_bot).cpu().numpy()
np.savez(surface_path,
         img=img_np,
         img_corrected=img_corrected.cpu().numpy(),
         s_top=s_top.cpu().numpy().ravel(),
         s_bot=s_bot.cpu().numpy().ravel(),
         pred_top=pred_top,
         pred_bot=pred_bot,
         flatten_idx=flatten_idx,
         fit_top_start=fit_top.start, fit_top_stop=fit_top.stop,
         fit_bot_start=fit_bot.start, fit_bot_stop=fit_bot.stop,
)
print(f"Saved {surface_path}")

# ---- Grade card ----
# Unsagged rows: 150-362 (top), 662-874 (bot) — relative to full image
# Pure sag rows: 362-511 (top), 512-662 (bot)
unsag_rows = list(range(150, 362)) + list(range(662, 874))
sag_rows = list(range(362, 512)) + list(range(512, 662))

def grade_card(img, label=""):
    oob = img[metric_rows][:, flat_idx].cpu().numpy()
    oob_unsag = img[unsag_rows][:, flat_idx].cpu().numpy()
    oob_sag = img[sag_rows][:, flat_idx].cpu().numpy()

    mid = 512 - echo_trim
    jump = np.abs(np.median(oob[mid - 1]) - np.median(oob[mid]))

    print(f"\n{'='*40}")
    print(f"Grade Card: {label}")
    print(f"  Row Flatness unsag (σ): {np.std(np.median(oob_unsag, axis=1)):.3f}  (target: 0.22)")
    print(f"  Col Flatness unsag (σ): {np.std(np.median(oob_unsag, axis=0)):.3f}  (target: 0.26)")
    print(f"  Row Flatness sag   (σ): {np.std(np.median(oob_sag, axis=1)):.3f}")
    print(f"  Col Flatness sag   (σ): {np.std(np.median(oob_sag, axis=0)):.3f}")
    print(f"  Half-Half Jump:         {jump:.3f}  (target: < 0.5)")
    print(f"{'='*40}")


# print("\n--- Before correction (bias subtraction only) ---")
# grade_card(img_bias_sub, "Bias Subtraction")

print(f"\n--- After correction ({model_name}) ---")
grade_card(img_corrected, model_name)

# Print model parameters if few
n_params = sum(p.numel() for p in model_top.parameters())
print(f"\nModel parameters (per half): {n_params}")


# ---- 1D Validation Plots (3 top + 3 bottom) ----
fig, axes = plt.subplots(6, 1, figsize=(10, 18), sharex=False)
sample_cols = [50, 200, 800]

for half_idx, (label, rows, s_fit, b_fit, model) in enumerate([
    ('Top', fit_top, s_top, b_top, model_top),
    ('Bot', fit_bot, s_bot, b_bot, model_bot),
]):
    color = 'steelblue' if half_idx == 0 else 'salmon'
    lc = 'b' if half_idx == 0 else 'r'
    s_np = s_fit.cpu().numpy().ravel()
    sort_idx = np.argsort(s_np)

    with torch.no_grad():
        y_pred_all = model(b_fit, s_fit)

    for j, col_idx in enumerate(sample_cols):
        ax = axes[half_idx * 3 + j]
        ch = int(np.searchsorted(flatten_idx, col_idx))

        y_actual = img[rows, col_idx].cpu().numpy()
        y_pred = y_pred_all[:, ch].cpu().numpy()

        ax.scatter(s_np, y_actual, s=1, alpha=0.3, c=color, label=f'{label} actual')
        ax.plot(s_np[sort_idx], y_pred[sort_idx], f'{lc}-', lw=1.5, label=f'{label} fitted')
        ax.set_ylabel(f'{label} Col {col_idx}')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('Row sum (s)')

fig.suptitle(f'1D Validation: {model_name}', fontsize=14)
fig.tight_layout()
fig.savefig(f'{out_dir}/validation_1d_{model_name}.png', dpi=150)
print(f"Saved {out_dir}/validation_1d_{model_name}.png")


# ---- 2D Plots ----
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))

vmin, vmax = -10, 10
ax = axes2[0, 0]
im = ax.imshow(img_bias_sub.cpu().numpy(), vmin=vmin, vmax=vmax, cmap='RdBu_r', aspect='auto')
ax.set_title('Bias Subtraction Only (error)')
plt.colorbar(im, ax=ax, shrink=0.6)

ax = axes2[0, 1]
im = ax.imshow(img_corrected.cpu().numpy(), vmin=vmin, vmax=vmax, cmap='RdBu_r', aspect='auto')
ax.set_title(f'{model_name} (error)')
plt.colorbar(im, ax=ax, shrink=0.6)

ax = axes2[1, 0]
im = ax.imshow(img_bias_sub.cpu().numpy(), vmin=-20, vmax=20, aspect='auto', cmap='gray')
ax.set_title('Bias Subtraction Only')
plt.colorbar(im, ax=ax, shrink=0.6)

ax = axes2[1, 1]
im = ax.imshow(img_corrected.cpu().numpy(), vmin=-20, vmax=20, aspect='auto', cmap='gray')
ax.set_title(f'{model_name}')
plt.colorbar(im, ax=ax, shrink=0.6)

fig2.suptitle(f'OOB Flattening — {model_name}', fontsize=14)
fig2.tight_layout()
fig2.savefig(f'{out_dir}/flattened_2d_{model_name}.png', dpi=200)
print(f"Saved {out_dir}/flattened_2d_{model_name}.png")

plt.close('all')
print("\nDone.")
