#!/usr/bin/env python3
# Flat-background ground truth experiment for f₁
# Use per-half median to get x, then compute f₁ = y - b - d - x
# Evan Widloski 2026-05-29

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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

# ----- Load flat-background images (same as bj_relate.py, excluding SCI) -----
print('loading...')
dataset1 = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-STR_20251215_v1.0.nc')).data
dataset1 = dataset1.isel(time=[15, 16, 18, 24, 25])
dataset2 = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-DRK_20251215_v1.0.nc')).data
dataset2 = dataset2.isel(time=[0])
dataset = xr.concat([dataset1, dataset2], dim='time')

print('loaded')

# ----- Process each image -----
truth_dict = {}
corrupted_dict = {}

for idx in range(dataset.dims['time']):
    img_data = dataset.isel(time=idx)
    filter_name = str(img_data.filter.values)
    time_str = str(img_data.time.values)[:10].replace('-', '')
    data_name = f"{filter_name}_{time_str}_{idx}"

    # Calibrate: L1A has bias removed, normalize and remove dark stripes
    y_raw = (img_data.images / img_data.n_frames).values.squeeze()
    z, d = remove_dark_stripes(y_raw, 'NFI')  # z = y - b - d

    # Apply mask
    z = np.where(mask, z, np.nan)

    # ----- Top half analysis -----
    top_z = z[:512, :]
    # top_flat = flat[:512, :]

    # Ground truth: median of each half (assumed flat background)
    # top_median = np.nanmedian(top_z / top_flat)
    # x_pred = top_median * top_flat
    x_pred = np.broadcast_to(np.nanmedian(top_z), top_z.shape)

    # f₁ = z - x = (y - b - d) - x
    f1 = top_z - x_pred

    truth_dict[data_name] = x_pred
    corrupted_dict[data_name] = top_z

# ----- Plotting (all images) -----
SELECTED_COLS = C = [562, 632, 702]  # first/middle/last
COLOR_FAMILIES = {
    C[0]: ['rgb(255,200,200)', 'rgb(255,150,150)', 'rgb(255,100,100)', 'rgb(255,50,50)', 'rgb(200,0,0)', 'rgb(150,0,0)'],
    C[1]: ['rgb(200,200,255)', 'rgb(150,150,255)', 'rgb(100,100,255)', 'rgb(50,50,255)', 'rgb(0,0,200)', 'rgb(0,0,150)'],
    C[2]: ['rgb(200,255,200)', 'rgb(150,255,150)', 'rgb(100,255,100)', 'rgb(50,255,50)', 'rgb(0,200,0)', 'rgb(0,150,0)'],
}
data_names = list(truth_dict.keys())

# 3D scatter: f₁ vs (s, x) for all images
scatter_fig = go.Figure()
for img_idx, data_name in enumerate(data_names):
    top_z = corrupted_dict[data_name]
    x_pred = truth_dict[data_name]
    f1 = top_z - x_pred
    s = np.nansum(top_z, axis=1)
    for c in SELECTED_COLS:
        mask_valid = ~np.isnan(f1[:, c]) & ~np.isnan(x_pred[:, c])
        scatter_fig.add_trace(go.Scatter3d(
            x=s[mask_valid], y=x_pred[mask_valid, c], z=f1[mask_valid, c],
            mode='markers', name=f'{data_name} col={c}',
            marker=dict(size=2, color=COLOR_FAMILIES[c][img_idx]),
            hovertemplate=f'{data_name} col={c}<br>s: %{{x}}<br>x: %{{y}}<br>f₁: %{{z}}<extra></extra>',
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
with document('Flat f₁ Analysis') as doc:
    with caption('3D Scatter: f₁ vs (sᵢ, x) - all images'):
        raw(scatter_fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # One row of 2D plots per image
    for data_name in data_names:
        top_z = corrupted_dict[data_name]
        x_pred = truth_dict[data_name]
        f1 = top_z - x_pred

        with itemgrid(length=3):
            with caption(f'z = y - b - d ({data_name})'):
                with plot():
                    plt.imshow(top_z, cmap='RdBu_r', vmin=-10, vmax=10, origin='upper')
                    plt.colorbar()
                    for c in SELECTED_COLS:
                        plt.axvline(c, color='blue', linewidth=0.5, alpha=0.5)

            with caption(f'x (ground truth) ({data_name})'):
                with plot():
                    plt.imshow(x_pred, cmap='RdBu_r', vmin=-10, vmax=10, origin='upper')
                    plt.colorbar()
                    for c in SELECTED_COLS:
                        plt.axvline(c, color='blue', linewidth=0.5, alpha=0.5)

            with caption(f'f₁ = y - b - d - x ({data_name})'):
                with plot():
                    plt.imshow(f1, cmap='RdBu_r', vmin=-5, vmax=5, origin='upper')
                    plt.colorbar()
                    for c in SELECTED_COLS:
                        plt.axvline(c, color='blue', linewidth=0.5, alpha=0.5)

doc.save('/www/truth/flat_f1.html')
print('Saved to /www/truth/flat_f1.html')

# Save arrays to npz (dict format per AGENTS.md)
np.savez('./flat_f1.npz', truth=truth_dict, corrupted=corrupted_dict)
print('Saved to ./flat_f1.npz')
