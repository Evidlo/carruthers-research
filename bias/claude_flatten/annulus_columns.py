#!/usr/bin/env python3

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
import sys
sys.path.insert(0, '..')
from common import load, rob_bias

# Parameters
EARTH_CENTER = (512, 512)  # (row, col)
SELECTED_COLS = range(250, 362, 5)
SAFE_ROWS = range(150, 362)  # non-sagged, non-echo rows

# Load image and compute bias
orig = load('../images_20260316/oob_nfi_l0.pkl')
HOT_PIXELS = np.load('hot_pixels.npy')
orig[HOT_PIXELS[:, 0], HOT_PIXELS[:, 1]] = np.nan
bias = rob_bias(orig, 150, 250)

# Top half only (rows 0-511)
top_orig = orig[:512, :]
top_bias = bias[:512, :]

# Distance from Earth center for each pixel (full grid)
row_grid, col_grid = np.mgrid[:512, :1024]
dist_full = np.sqrt((row_grid - EARTH_CENTER[0])**2 + (col_grid - EARTH_CENTER[1])**2)

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
window = min(31, len(fit_xb) // 2 * 2 - 1)  # must be odd
fit_xb_smooth = savgol_filter(fit_xb, window, 3)

# Enforce monotonic decreasing
fit_xb_mono = np.minimum.accumulate(fit_xb_smooth)

# Interpolation function for x+b given radius
def xb_from_radius(r):
    return np.interp(r, fit_r, fit_xb_mono)

# Row sums
s = np.nansum(top_orig, axis=1)

# Collect data for all pixels in selected columns
cols_arr = np.array(list(SELECTED_COLS))
rows_arr = np.arange(512)
rc = np.array(np.meshgrid(rows_arr, cols_arr, indexing='ij')).reshape(2, -1).T  # (N, 2)
data_r, data_c = rc[:, 0], rc[:, 1]
data_radius = dist_full[data_r, data_c]
data_z = (top_orig - top_bias)[data_r, data_c]
data_xb = xb_from_radius(data_radius)
data_s = s[data_r]

# Filter out NaN
valid = ~np.isnan(data_z)
data_r, data_c, data_radius, data_z, data_xb, data_s = (
    data_r[valid], data_c[valid], data_radius[valid], data_z[valid], data_xb[valid], data_s[valid]
)

# Random colors per column
unique_cols = np.unique(data_c)
rng = np.random.default_rng(42)
col_colors = {c: f'rgb({rng.integers(50,255)},{rng.integers(50,255)},{rng.integers(50,255)})' for c in unique_cols}

# Image for display
img_display = np.clip(top_orig - top_bias, -10, 10)

# Create figure with 2 rows
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'xy'}, {'type': 'scene'}],
           [{'type': 'xy', 'colspan': 2}, None]],
    subplot_titles=['y - f(0+b, ...)', 'z vs s, x+b (colored by column)', 'x+b profile vs radius'],
    column_widths=[0.4, 0.6],
    row_heights=[0.65, 0.35],
    horizontal_spacing=0.12,
    vertical_spacing=0.1
)

# Left: heatmap
fig.add_trace(
    go.Heatmap(z=img_display, colorscale='RdBu_r', zmid=0, showscale=True,
               colorbar=dict(x=0.35, len=0.8)),
    row=1, col=1
)
fig.update_xaxes(scaleanchor='y', scaleratio=1, row=1, col=1)
fig.update_yaxes(autorange='reversed', row=1, col=1)

# Draw vertical blue lines for selected columns (underneath image via layer='below')
for c in SELECTED_COLS:
    fig.add_vline(x=c, line=dict(color='blue', width=1), layer='below', row=1, col=1)

# Right: 3D scatter by column
for c in unique_cols:
    mask = data_c == c
    fig.add_trace(
        go.Scatter3d(
            x=data_s[mask], y=data_xb[mask], z=data_z[mask],
            mode='markers', name=f'col={c}',
            marker=dict(size=2, color=col_colors[c]),
            hovertemplate='s: %{x}<br>x+b: %{y}<br>z: %{z}<extra>col=%{fullData.name}</extra>'
        ),
        row=1, col=2
    )

# Bottom: x+b profile vs radius
# Subsample raw data for plotting (too many points otherwise)
subsample_idx = np.random.default_rng(0).choice(len(safe_r), size=min(5000, len(safe_r)), replace=False)
fig.add_trace(
    go.Scatter(x=safe_r[subsample_idx], y=safe_xb[subsample_idx], mode='markers',
               marker=dict(size=2, color='gray', opacity=0.3),
               name='raw (safe rows)', showlegend=True),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=fit_r, y=fit_xb_mono, mode='lines',
               line=dict(color='red', width=2),
               name='smoothed (monotonic)', showlegend=True),
    row=2, col=1
)
fig.update_xaxes(title_text='radius', row=2, col=1)
fig.update_yaxes(title_text='x+b', row=2, col=1)

fig.update_layout(
    scene=dict(
        xaxis_title='s',
        yaxis_title='x+b',
        zaxis_title='y - f(0+b, ...)',
        zaxis=dict(range=[-20, 10])
    ),
    height=900,
    width=1200
)

fig.write_html('/www/annulus_columns.html')
print('Saved to /www/annulus_columns.html')
