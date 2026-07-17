#!/usr/bin/env python3
# Earth-centered ground truth experiment for f₁
# Use rotational symmetry to get x from safe rows, then compute f₁ = y - b - d - x
# Evan Widloski 2026-05-28

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.signal import savgol_filter
from domrep import document, caption, itemgrid, plot
from dominate.util import raw
import plotly.graph_objects as go

from glide.science_data_processing.L1A import L1A
from glide.common_components.camera import CameraNFI
from glide.validation.cam import load_lab_data
import sys; sys.path.insert(0, '..')
from bias_wavelet import remove_dark_stripes

# ----- Load camera data (mask, flat) -----
cam = CameraNFI()
cam.spec = load_lab_data(cam.spec)
mask, flat = cam.spec.mask_fov, cam.spec.flat

# ----- Load OOB image -----
dataset = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-OOB_20260318_v1.0.nc')).data
earth_center = (512, 512)
dataset = dataset.isel(time=0)

# dataset = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-STR_20260318_v1.0.nc')).data
# earth_center = (512, 1024)
# dataset = dataset.isel(time=1)

# Calibrate: L1A has bias removed, normalize and remove dark stripes
y_raw = (dataset.images / dataset.n_frames).values.squeeze()
z, d = remove_dark_stripes(y_raw, 'NFI')  # z = y - b - d (observed minus bias minus dark)

# Apply mask
z = np.where(mask, z, np.nan)


# ----- Top half analysis -----
top_z = z[:512, :]
top_flat = flat[:512, :]

# Distance from Earth center
row_grid, col_grid = np.mgrid[:512, :1024]
dist = np.sqrt((row_grid - earth_center[0])**2 + (col_grid - earth_center[1])**2)

# Safe rows: far from sag region (rows 150-400 for top half)
SAFE_ROWS = range(150, 400)
safe_mask = np.zeros((512, 1024), dtype=bool)
safe_mask[list(SAFE_ROWS), :] = True

# Fit x/a profile vs radius using safe rows (where f₁ ≈ 0, so y ≈ x)
# Divide by flatfield to get x/a
safe_r = dist[safe_mask].ravel()
safe_x_over_a = (top_z / top_flat)[safe_mask].ravel()
valid = ~np.isnan(safe_x_over_a)
safe_r, safe_x_over_a = safe_r[valid], safe_x_over_a[valid]

# Binned median fit for x/a
r_bins = np.arange(0, safe_r.max() + 1, 2)
bin_centers = (r_bins[:-1] + r_bins[1:]) / 2
bin_idx = np.digitize(safe_r, r_bins) - 1
bin_medians = np.array([np.nanmedian(safe_x_over_a[bin_idx == i]) if np.sum(bin_idx == i) > 0 else np.nan for i in range(len(bin_centers))])
valid_bins = ~np.isnan(bin_medians)
fit_r, fit_x_over_a = bin_centers[valid_bins], bin_medians[valid_bins]

# Smooth and enforce monotonic decreasing
window = min(31, len(fit_x_over_a) // 2 * 2 - 1)
fit_smooth = savgol_filter(fit_x_over_a, window, 3)
fit_mono = np.minimum.accumulate(fit_smooth)

def x_over_a_from_radius(r):
    return np.interp(r, fit_r, fit_mono, left=np.nan, right=np.nan)

# Compute x (ground truth) by: x/a from symmetry, then multiply by flatfield
x_over_a_pred = x_over_a_from_radius(dist)
x_pred = x_over_a_pred * top_flat

# f₁ = z - x = (y - b - d) - x
f1 = top_z - x_pred

# Row sums (for plotting against sᵢ)
s = np.nansum(top_z, axis=1)
# s = np.nansum(x_pred, axis=1)
# s = np.sum(x_pred[:, 200:-200], axis=1)

# ----- Plotting -----
SELECTED_COLS = np.array(range(560, 700, 20))
rng = np.random.default_rng(42)
col_colors = {c: f'rgb({rng.integers(50,255)},{rng.integers(50,255)},{rng.integers(50,255)})' for c in SELECTED_COLS}

# 3D scatter: f₁ vs (s, x)
scatter_fig = go.Figure()
for c in SELECTED_COLS:
    rows = np.arange(512)
    mask = ~np.isnan(f1[:, c]) & ~np.isnan(x_pred[:, c])
    scatter_fig.add_trace(go.Scatter3d(
        x=s[mask], y=x_pred[mask, c], z=f1[mask, c],
        mode='markers', name=f'col={c}',
        marker=dict(size=2, color=col_colors[c]),
        hovertemplate=f'col={c}<br>s: %{{x}}<br>x: %{{y}}<br>f₁: %{{z}}<extra></extra>',
    ))

scatter_fig.update_layout(
    scene=dict(
        xaxis_title='sᵢ (row sum)',
        yaxis_title='x (ground truth)',
        zaxis_title='f₁ = y - b - d - x',
        zaxis=dict(range=[-3, 1]),
    ),
    showlegend=True,
    legend=dict(itemclick='toggleothers', itemdoubleclick='toggle', itemsizing='constant', itemwidth=50),
    margin=dict(l=0, r=0, t=30, b=0),
)

# Build document
with document('Earth f₁ Analysis') as doc:
    with caption('3D Scatter: f₁ vs (sᵢ, x)'):
        raw(scatter_fig.to_html(full_html=False, include_plotlyjs='cdn'))

    with itemgrid(length=3):
        with caption('z = y - b - d'):
            with plot():
                plt.imshow(top_z, cmap='RdBu_r', vmin=-10, vmax=10, origin='upper')
                plt.colorbar()
                plt.axvline(SELECTED_COLS[0], color='blue', linewidth=0.5, alpha=0.5, label='selected cols')
                for c in SELECTED_COLS[1:]:
                    plt.axvline(c, color='blue', linewidth=0.5, alpha=0.5)
                plt.axhline(SAFE_ROWS.start, color='green', linewidth=1, label='safe rows')
                plt.axhline(SAFE_ROWS.stop, color='green', linewidth=1)
                plt.legend(loc='upper right')

        with caption('x (ground truth from symmetry)'):
            with plot():
                plt.imshow(x_pred, cmap='RdBu_r', vmin=-10, vmax=10, origin='upper')
                plt.colorbar()
                plt.axvline(SELECTED_COLS[0], color='blue', linewidth=0.5, alpha=0.5, label='selected cols')
                for c in SELECTED_COLS[1:]:
                    plt.axvline(c, color='blue', linewidth=0.5, alpha=0.5)
                plt.legend(loc='upper right')

        with caption('f₁ = y - b - d - x'):
            with plot():
                plt.imshow(f1, cmap='RdBu_r', vmin=-5, vmax=5, origin='upper')
                plt.colorbar()
                plt.axvline(SELECTED_COLS[0], color='blue', linewidth=0.5, alpha=0.5, label='selected cols')
                for c in SELECTED_COLS[1:]:
                    plt.axvline(c, color='blue', linewidth=0.5, alpha=0.5)
                plt.legend(loc='upper right')

    with caption('Radial x profile (from safe rows)'):
        with plot():
            subsample = rng.choice(len(safe_r), size=min(5000, len(safe_r)), replace=False)
            plt.scatter(safe_r[subsample], safe_x_over_a[subsample], s=1, c='gray', alpha=0.3, label='safe pixels')
            plt.plot(fit_r, fit_mono, 'r-', linewidth=2, label='smoothed fit')
            plt.ylim([-5, 10])
            plt.xlabel('radius from Earth center')
            plt.ylabel('x/a (safe rows)')
            plt.legend()

doc.save('/www/truth/earth_f1.html')
print('Saved to /www/truth/earth_f1.html')

# Save arrays to npz (dict format per AGENTS.md)
data_name = 'OOB_20260318'
np.savez('./earth_f1.npz', truth={data_name: x_pred}, corrupted={data_name: top_z})
print('Saved to ./earth_f1.npz')
