#!/usr/bin/env python3
# Investigating variation in b_j across images.
# Is it related to background level?
# Evan Widloski 2026-05-25

from common import load, rob_bias
import numpy as np
import matplotlib.pyplot as plt
from domrep import *
from dominate.tags import pre
from dominate.util import raw
import plotly.graph_objects as go
from importlib import resources
import matplotlib
matplotlib.use('Agg')

import xarray as xr
from glide.science_data_processing.L1A import L1A
from glide.common_components.camera import CameraL1BNFI

mask = CameraL1BNFI().spec.mask_fov

from bias_wavelet import wavelet_destripe, remove_dark_stripes


dataset1 = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-STR_20251215_v1.0.nc')).data
dataset1 = dataset1.isel(time=[15, 16, 18, 24, 25])
dataset2 = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-DRK_20251215_v1.0.nc')).data
dataset2 = dataset2.isel(time=[0])
dtrain = xr.concat([dataset1, dataset2], dim='time')

dataset3 = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-SCI_20251221_v1.0.nc')).data
dataset3 = dataset3.isel(time=0)
dtest = xr.concat([dataset1, dataset2, dataset3], dim='time')

print('loaded')

# ----- Training -----
# %% train
# find slope/offset from training images

# mask, normalize, and remove dark stripes from training images
train = dtrain.images.where(xr.DataArray(mask, dims=['row', 'col']), float('nan'))
train_norm = train / dtrain.n_frames
train_nodark = remove_dark_stripes(train_norm, 'NFI')[0]

b_j = rob_bias(train_nodark, 300, 0)[:, (0, -1)]
# compute backgrounds from safe rows for each half
bkg = np.stack((
    np.nanmedian(train_nodark[:, 300:512], axis=(1, 2)),
    np.nanmedian(train_nodark[:, 512:724], axis=(1, 2)),
), axis=1)[:, :, None].repeat(1024, axis=2)

b_j -= bkg

# fit a line slope/offset to each image half.  use all training images
x = bkg  # shape (n_images, 2, 1024)
y = b_j  # shape (n_images, 2, 1024)
x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)  # (2, 1024)
slope_j = ((x - x_mean) * (y - y_mean)).sum(axis=0) / ((x - x_mean)**2).sum(axis=0)
offset_j = y_mean - slope_j * x_mean

# ----- Apply -----
# apply corrections to test images

# mask, normalize, and remove dark stripes from training images
dtest['images'] = dtest.images.where(xr.DataArray(mask, dims=['row', 'col']), float('nan'))
dtest['images_norm'] = dtest.images / dtest.n_frames
dtest['images_nodark'] = remove_dark_stripes(dtest.images_norm, 'NFI')[0]

# Apply per-half correction: slope_j/offset_j are (2, 1024), broadcast to (1024, 1024)
slope_img = np.concatenate([np.broadcast_to(slope_j[0], (512, 1024)), np.broadcast_to(slope_j[1], (512, 1024))], axis=0)
offset_img = np.concatenate([np.broadcast_to(offset_j[0], (512, 1024)), np.broadcast_to(offset_j[1], (512, 1024))], axis=0)

# apply the model, using xxx
# xxx = dtest.images_nodark # x_ij
xxx = np.repeat(dtest.images_nodark.mean(axis=-1, keepdims=True), 1024, axis=-1) # s_i
# xxx = np.stack(( # xbar (mean bkg) (FIXME: this is wrong for LyaN since not flat)
#     np.nanmedian(dtest.images_nodark[:, :512], axis=(1, 2)),
#     np.nanmedian(dtest.images_nodark[:, 512:], axis=(1, 2)),
# ), axis=1)[:, :, None].repeat(512, axis=1).repeat(1024, axis=2)
dtest['images_nocol'] = dtest.images_nodark - (slope_img * xxx - offset_img)

# plot bounding box
e = (slice(462, 562), slice(0, 100))
ex = (e[1].start, e[1].stop, e[0].start, e[0].stop)

# ----- Plotting -----
# %% plot

clim = (-25, 25)
with document('b_j analysis', style='flex-direction: row;') as doc:
    with itemgrid(flow='row', length=3):
        with caption("Input images (norm. DN/frame)"):
            with dropdown():
                with slider(label='Clipped'):
                    for t, row in dtest.groupby('time'):
                        with plot(label=f'{row.filter.values[0]} ({t})'):
                            plt.imshow(row.images_norm.squeeze())
                            plt.colorbar()
                            plt.tight_layout()
                            plt.clim(clim)
                with slider(label='Whole'):
                    for t, row in dtest.groupby('time'):
                        with plot(label=f'{row.filter.values[0]} ({t})'):
                            plt.imshow(row.images_norm.squeeze())
                            plt.colorbar()
                            plt.tight_layout()

        with caption("Dark removed"):
            with dropdown():
                with slider(label='Clipped'):
                    for t, row in dtest.groupby('time'):
                        with plot(label=f'{row.filter.values[0]} ({t})'):
                            plt.imshow(row.images_nodark.squeeze())
                            plt.colorbar()
                            plt.tight_layout()
                            plt.clim(clim)
                with slider(label='Whole'):
                    for t, row in dtest.groupby('time'):
                        with plot(label=f'{row.filter.values[0]} ({t})'):
                            plt.imshow(row.images_nodark.squeeze())
                            plt.colorbar()
                            plt.tight_layout()

        with caption("Column effect removed"):
            with dropdown():
                with slider(label='Clipped'):
                    for t, row in dtest.groupby('time'):
                        with plot(label=f'{row.filter.values[0]} ({t})'):
                            plt.imshow(row.images_nocol.squeeze())
                            plt.colorbar()
                            plt.tight_layout()
                            plt.clim(clim)
                with slider(label='Whole'):
                    for t, row in dtest.groupby('time'):
                        with plot(label=f'{row.filter.values[0]} ({t})'):
                            plt.imshow(row.images_nocol.squeeze())
                            plt.colorbar()
                            plt.tight_layout()

        print('foo')

        with caption("b_j analysis (train only)"):
            with plot():
                cmap = plt.cm.Paired
                for i, label in enumerate(dtrain.filter.values):
                    for h, half in enumerate(['top', 'bot']):
                        plt.scatter(bkg[i, h], b_j[i, h], label=f'{label} {half}', s=5, c=[cmap(2*i + h)])
                # red line fits (one per half per column)
                x_fit = np.linspace(bkg.min(), bkg.max(), 100)
                for h in range(2):
                    for j in range(1024):
                        plt.plot(x_fit, slope_j[h, j] * x_fit + offset_j[h, j], c='red', alpha=0.2, lw=0.5)
                plt.xlabel('Background level (DN/frame)')
                plt.ylabel('b_j (DN/frame)')
                plt.legend()
                plt.tight_layout()

        with caption('3D scatter (train only)'):
            cols = slice(500, 510)
            col_idx = np.arange(1024)[cols]
            # Meshgrid for (image*half, col) -> flatten all
            x_bkg = bkg[:, 0, cols].flatten()
            y_col = np.broadcast_to(col_idx, bkg[:, 0, cols].shape).flatten()
            z = b_j[:, 0, cols].flatten()
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=x_bkg, y=y_col, z=z, mode='markers',
                marker=dict(size=2, color=y_col, colorscale='Viridis')
            ))
            # red line fits (top half only since scatter shows half 0)
            x_fit = np.linspace(bkg[:, 0, cols].min(), bkg[:, 0, cols].max(), 20)
            for j in col_idx:
                fig.add_trace(go.Scatter3d(
                    x=x_fit, y=np.full_like(x_fit, j), z=slope_j[0, j] * x_fit + offset_j[0, j],
                    mode='lines', line=dict(color='red', width=2), opacity=0.5,
                    showlegend=False
                ))
            fig.update_layout(
                scene=dict(xaxis_title='Background', yaxis_title='Column', zaxis_title='Pixel Value'),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            raw(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    pre(open('bj_relate.py', 'r').read())

doc.save(f:='/www/bj_relate.html')
print(f'Wrote {f}')