#!/usr/bin/env python3

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.insert(0, '..')
from common import load, rob_bias

# Parameters
ANNULUS_THICKNESS = 3
INNER_RADIUS = 160
EARTH_CENTER = (512, 512)  # (row, col)

# Load image and compute bias
orig = load('../images_20260316/oob_nfi_l0.pkl')
HOT_PIXELS = np.load('hot_pixels.npy')
orig[HOT_PIXELS[:, 0], HOT_PIXELS[:, 1]] = np.nan
bias = rob_bias(orig, 150, 250)

# Top half only (rows 0-511)
top_orig = orig[:512, :]
top_bias = bias[:512, :]

# Distance from Earth center for each pixel
rows, cols = np.ogrid[:512, :1024]
dist = np.sqrt((rows - EARTH_CENTER[0])**2 + (cols - EARTH_CENTER[1])**2)

# Row sums
s = np.nansum(top_orig, axis=1)

# Collect data from annuli
data_s, data_xb, data_z, data_r, data_c, data_ann = [], [], [], [], [], []
annulus_radii = []

# max_radius = int(dist.max()) + 1
max_radius = 200
for ann_idx, r in enumerate(range(INNER_RADIUS, max_radius, ANNULUS_THICKNESS)):
    mask = (dist >= r) & (dist < r + ANNULUS_THICKNESS)
    if not mask.any():
        continue

    annulus_radii.append(r)

    # Rows present in this annulus
    rows_in_annulus = np.where(mask.any(axis=1))[0]
    if len(rows_in_annulus) == 0:
        continue

    # Reference: T rows furthest from row 512 (smallest row indices)
    ref_rows = rows_in_annulus[:ANNULUS_THICKNESS]
    row_in_ref = np.zeros(512, dtype=bool)
    row_in_ref[ref_rows] = True
    ref_mask = mask & row_in_ref[:, None]
    ref_xb = np.nanmean(top_orig[ref_mask]) if ref_mask.any() else np.nan

    # All pixels in annulus
    ann_rows, ann_cols = np.where(mask)
    for ar, ac in zip(ann_rows, ann_cols):
        z = top_orig[ar, ac] - top_bias[ar, ac]
        data_s.append(s[ar])
        data_xb.append(ref_xb)
        data_z.append(z)
        data_r.append(ar)
        data_c.append(ac)
        data_ann.append(ann_idx)

data_s = np.array(data_s)
data_xb = np.array(data_xb)
data_z = np.array(data_z)
data_ann = np.array(data_ann)

# Color palette for annuli
import plotly.express as px
n_annuli = len(annulus_radii)
colors = px.colors.sample_colorscale('Turbo', np.linspace(0, 1, n_annuli))

# Image for display
img_display = np.clip(top_orig - top_bias, -10, 10)

# Create figure
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'xy'}, {'type': 'scene'}]],
    subplot_titles=['y - f(0+b, ...)', 'z / (x+b) / s'],
    column_widths=[0.4, 0.6],
    horizontal_spacing=0.12
)

# Left: heatmap
fig.add_trace(
    go.Heatmap(z=img_display, colorscale='RdBu_r', zmid=0, showscale=True,
               colorbar=dict(x=0.35, len=0.8)),
    row=1, col=1
)
fig.update_xaxes(scaleanchor='y', scaleratio=1, row=1, col=1)
fig.update_yaxes(autorange='reversed', row=1, col=1)

# Draw semicircles on heatmap
theta = np.linspace(np.pi, 2*np.pi, 100)  # upper semicircle (rows < 512)
for i, r in enumerate(annulus_radii):
    x_arc = EARTH_CENTER[1] + r * np.cos(theta)
    y_arc = EARTH_CENTER[0] + r * np.sin(theta)
    fig.add_trace(
        go.Scatter(x=x_arc, y=y_arc, mode='lines', line=dict(color=colors[i], width=1),
                   showlegend=False, hoverinfo='skip'),
        row=1, col=1
    )

# Right: 3D scatter by annulus
for i, r in enumerate(annulus_radii):
    mask = data_ann == i
    fig.add_trace(
        go.Scatter3d(
            x=data_s[mask], y=data_xb[mask], z=data_z[mask],
            mode='markers', name=f'r={r}',
            marker=dict(size=2, color=colors[i]),
            hovertemplate='s: %{x}<br>x+b: %{y}<br>z: %{z}<extra>r=%{fullData.name}</extra>'
        ),
        row=1, col=2
    )

fig.update_layout(
    scene=dict(
        xaxis_title='s',
        yaxis_title='x+b',
        zaxis_title='y - f(0+b, ...)',
        zaxis=dict(range=[-20, 10])
    ),
    height=600,
    width=1200
)

fig.write_html('/www/annulus.html')
print('Saved to /www/annulus.html')
