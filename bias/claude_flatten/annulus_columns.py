#!/usr/bin/env python3

import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from domrep import document, caption, itemgrid, plot
from dominate.util import raw
import sys
sys.path.insert(0, '..')
from common import load, rob_bias

import matplotlib
matplotlib.use('Agg')

# Parameters
SELECTED_COLS = np.array(range(624, 774, 20))
SAFE_ROWS = range(150, 400)  # non-sagged, non-echo rows

# (path, (row_center, col_center))
IMAGES = [
    # ('../images_20260318_1/star_nfi_l0.pkl', (512, 1023)),
    # ('../images_20260318_3/star_nfi_l0.pkl', (512, 1023)),
    ('../images_20260316/oob_nfi_l0.pkl', (512, 512)),
    # ('../images_20260113/oob_nfi_l0.pkl', (512, 595)),
    # ('../images_20260115/oob_nfi_l0.pkl', (512, 595)),
    # ('../images_20260117/oob_nfi_l0.pkl', (512, 595)),
]

HOT_PIXELS = np.load('hot_pixels.npy')

# Random colors per column (shared across images)
unique_cols = np.array(list(SELECTED_COLS))
rng = np.random.default_rng(42)
col_colors = {c: f'rgb({rng.integers(50,255)},{rng.integers(50,255)},{rng.integers(50,255)})' for c in unique_cols}

# 3D scatter figure (will accumulate data from all images)
scatter_fig = go.Figure()

# Store per-image data for image/profile plots
image_data = []

for img_path, earth_center in IMAGES:
    img_name = img_path.split('/')[-2]
    orig = load(img_path)
    orig[HOT_PIXELS[:, 0], HOT_PIXELS[:, 1]] = np.nan
    bias = rob_bias(orig, 150, 250)

    # Top half only
    top_orig = orig[:512, :]
    top_bias = bias[:512, :]

    # Distance from Earth center
    row_grid, col_grid = np.mgrid[:512, :1024]
    dist_full = np.sqrt((row_grid - earth_center[0])**2 + (col_grid - earth_center[1])**2)

    # Fit x+b profile vs radius using safe rows only
    safe_mask = np.zeros((512, 1024), dtype=bool)
    safe_mask[list(SAFE_ROWS), :] = True
    safe_r = dist_full[safe_mask].ravel()
    safe_xb = (top_orig - top_bias)[safe_mask].ravel()
    valid = ~np.isnan(safe_xb)
    safe_r, safe_xb = safe_r[valid], safe_xb[valid]

    # Binned median fit
    r_bins = np.arange(0, safe_r.max() + 1, 2)
    bin_centers = (r_bins[:-1] + r_bins[1:]) / 2
    bin_idx = np.digitize(safe_r, r_bins) - 1
    bin_medians = np.array([np.nanmedian(safe_xb[bin_idx == i]) if np.sum(bin_idx == i) > 0 else np.nan for i in range(len(bin_centers))])
    valid_bins = ~np.isnan(bin_medians)
    fit_r, fit_xb = bin_centers[valid_bins], bin_medians[valid_bins]

    # Smooth with savgol filter
    window = min(31, len(fit_xb) // 2 * 2 - 1)
    fit_xb_smooth = savgol_filter(fit_xb, window, 3)

    # Enforce monotonic decreasing
    fit_xb_mono = np.minimum.accumulate(fit_xb_smooth)

    def xb_from_radius(r):
        return np.interp(r, fit_r, fit_xb_mono, left=np.nan, right=np.nan)

    # Row sums
    s = np.nansum(top_orig, axis=1)

    # Collect data for all pixels in selected columns
    cols_arr = np.array(list(SELECTED_COLS))
    rows_arr = np.arange(512)
    rc = np.array(np.meshgrid(rows_arr, cols_arr, indexing='ij')).reshape(2, -1).T
    data_r, data_c = rc[:, 0], rc[:, 1]
    data_radius = dist_full[data_r, data_c]
    data_z = (top_orig - top_bias)[data_r, data_c]
    data_xb = xb_from_radius(data_radius)
    data_s = s[data_r]

    # Filter out NaN
    valid = ~np.isnan(data_z) & ~np.isnan(data_xb)
    data_r, data_c, data_radius, data_z, data_xb, data_s = (
        data_r[valid], data_c[valid], data_radius[valid], data_z[valid], data_xb[valid], data_s[valid]
    )

    # Add to 3D scatter (one trace per column)
    for c in unique_cols:
        mask = data_c == c
        if not mask.any():
            continue
        scatter_fig.add_trace(go.Scatter3d(
            x=data_s[mask], y=data_xb[mask], z=(data_z - data_xb)[mask],
            mode='markers', name=f'col={c}',
            legendgroup=f'col={c}',
            marker=dict(size=2, color=col_colors[c]),
            hovertemplate=f'{img_name}<br>s: %{{x}}<br>x+b: %{{y}}<br>z: %{{z}}<extra></extra>',
            showlegend=(img_path == IMAGES[0][0])
        ))

    # Store for image/profile plots
    img_display = np.clip(top_orig - top_bias, -10, 10)
    image_data.append({
        'name': img_name,
        'img_display': img_display,
        'safe_r': safe_r,
        'safe_xb': safe_xb,
        'fit_r': fit_r,
        'fit_xb_mono': fit_xb_mono,
    })

# Configure 3D scatter layout
scatter_fig.update_layout(
    scene=dict(
        xaxis_title='s',
        yaxis_title='x+b',
        zaxis_title='z - (x+b) (sag)',
        zaxis=dict(range=[-15, 5]),
        camera=dict(eye=dict(x=1.44, y=1.44, z=1.44))
    ),
    showlegend=True,
    legend=dict(itemclick='toggleothers', itemdoubleclick='toggle', itemsizing='constant', itemwidth=50),
    margin=dict(l=0, r=0, t=30, b=0),
    autosize=True
)

# Build document with domrep
with document('Annulus Column Analysis') as doc:
    with caption('3D Scatter: z vs s, x+b (colored by column)'):
        raw(scatter_fig.to_html(full_html=False, include_plotlyjs='cdn'))

    n_images = len(image_data)
    with itemgrid(length=n_images):
        # Row 1: image plots
        for d in image_data:
            with caption(f"{d['name']} - image"):
                with plot():
                    plt.imshow(d['img_display'], cmap='RdBu_r', vmin=-10, vmax=10, origin='upper')
                    plt.colorbar()
                    for c in SELECTED_COLS:
                        plt.axvline(c, color='blue', linewidth=0.5, alpha=0.5)

        # Row 2: profile plots
        for d in image_data:
            with caption(f"{d['name']} - x+b profile"):
                with plot():
                    subsample_idx = np.random.default_rng(0).choice(len(d['safe_r']), size=min(5000, len(d['safe_r'])), replace=False)
                    plt.scatter(d['safe_r'][subsample_idx], d['safe_xb'][subsample_idx], s=1, c='gray', alpha=0.3, label='raw')
                    plt.plot(d['fit_r'], d['fit_xb_mono'], 'r-', linewidth=2, label='smoothed')
                    plt.ylim([-5, 10])
                    plt.xlabel('radius')
                    plt.ylabel('x+b')
                    plt.legend()

doc.save('/www/annulus_columns.html')
print('Saved to /www/annulus_columns.html')
