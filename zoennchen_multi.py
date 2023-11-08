#!/usr/bin/env python3

import copy
from dech import *
import torch as t
import torch_optimizer as optim
import numpy as np
import tomosipo as ts

from glide.common_components.view_geometry import gen_mission, circular_orbit, CameraWFI
from glide.common_components.cam import CamMode, CamSpec, nadir_wfi_mode, nadir_nfi_mode
from glide.science.orbit import viewgeom2ts
from glide.science.model import SnowmanModel, zoennchen, FullyDenseModel, SphHarmBasisModel, default_vol, density2xr, AxisAlignmentModel, vol2cart
from glide.science.forward import Forwards
from glide.science.plotting import save_gif, preview3d, orbit_svg, imshow, color_negative
from glide.science.recon import nag_ls, sirt, gd
from glide.science.recon.nag_ls import nag_ls_coeff, nag_ls_clip
from glide.science.common import cart2sph

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:500"
# t.cuda.set_per_process_memory_fraction(0.40, 0)

# ----- Density -----

s = 100
vol = default_vol(shape=s, size=50)

# ----- Mission Setup -----
# %% setup

orbit = 'carruthers'
mo = 6
view_geoms = gen_mission(
    num_obs=20, start='2025-07-01',
    cams=[CameraWFI()],
    duration=30*mo,
)

# orbit = 'circular'
# view_geoms = circular_orbit(num_obs=60, radius=280, revolutions=0.5, cam_mode=cm, cam_spec=cs)
# cmodes = [cm] * len(view_geoms)
# cspecs = [cs] * len(view_geoms)


# ----- Forward Model -----
# %% forward

# model_truth = SphHarmBasisModel(vol, num_shells=20, max_l=2, device='cuda')
# coeffs_truth = t.zeros(model_truth.coeffs_shape, device='cuda')
# coeffs_truth[3, 0] = 1
# density_truth = model_truth(coeffs_truth)
density_truth = zoennchen(vol, device='cuda')
# density_truth = AxisAlignmentModel(vol, device='cuda')()
np.random.seed(0)
f_truths = Forwards(view_geoms, vol, use_noise=True, use_grad=False, use_albedo=False, use_aniso=False)
f_truth = f_truths[0]
y_truths = [f(density_truth) for f in f_truths]
y_truth = y_truths[0]

fs = Forwards(view_geoms, vol, use_noise=False, use_grad=True, use_albedo=False, use_aniso=False)
f = fs[0]

import ipdb
ipdb.set_trace()


# ----- Loss Functions and Regularizers -----
square_loss = lambda f, y, d: t.mean((y - f(d))**2)
# r_weight = 1 / cart2sph(vol2cart(vol))[:, :, :, 0]**2
# square_loss_r = lambda f, y, d: t.mean(r_weight * (y - f(d))**2)
def square_rel_loss(f, y, d):
    obs = f(d)
    rel_err = (y - obs) / obs
    rel_err = rel_err.nan_to_num()
    return t.mean(rel_err**2)
cheater_loss = lambda f, y, d: t.mean((d - density_truth)**2)
neg_reg=lambda f, y, d: t.mean(t.abs(d.clip(max=0)))
# reg_fn=lambda f, y, d: 10 * t.mean((t.abs(d) - d)**2)
# reg_fn=lambda f, y, d: t.mean((d - density_truth)**2)

# %% recon

coeffs = {}
models = {}
losses = {}

coeffs['Truth'] = density_truth
models['Truth'] = FullyDenseModel(vol, device='cuda')
losses['Truth'] = None

# models['sph'] = FullyDenseModel(vol, device='cuda')
name = 'sph'
models[name] = SphHarmBasisModel(vol, num_shells=20, max_l=2, device='cuda')
_, losses[name], coeffs[name] = gd(
    fs, y_truths,
    model=models[name],
    num_iterations=2000,
    lr=1e2,
    optimizer=(optimizer:=optim.Yogi),
    loss_fn=square_loss,
    reg_fn=neg_reg,
    loss_history=True
)

# models['adj_dense'] = FullyDenseModel(vol, device='cuda')
# coeffs['adj_dense'] = models['adj_dense'].T(f.T(y_truth))
# scaling = models['adj_dense'].T(f.T(t.ones_like(y_truth)))
# coeffs['adj_dense'] = coeffs['adj_dense'] / scaling.where(scaling.data != 0, t.tensor(1, device='cuda'))

# models['gd_rel'] = SphHarmBasisModel(vol, num_shells=20, max_l=3, device='cuda')
# _, losses['gd_rel'], coeffs['gd_rel'] = gd(
#     f, y_truth, model=models['gd_rel'],
#     num_iterations=600,
#     lr=1e2,
#     optimizer=optim.QHAdam,
#     loss_fn=square_rel_loss,
#     reg_fn=neg_reg,
#     loss_history=True
# )

# models['sph'] = SphHarmBasisModel(vol, num_shells=5, max_l=2, device='cuda')
# coeffs['sph']: nag_ls_clip(
#     f, y_truth, model=models['sph'], num_iterations=200,
#     min_constraint=0,
#     progress_bar=True,
#     l2_regularization=1e0
# )

# models['nag'] = FullyDenseModel(vol, device='cuda')
# coeffs['nag']: nag_ls(
#     f, y_truth, num_iterations=200,
#     min_constraint=0,
#     progress_bar=True,
#     l2_regularization=0
# )

# models['sirt'] = FullyDenseModel(vol, device='cuda')
# coeffs['sirt']: sirt(
#     f, y_truth, num_iterations=200,
#     min_constraint=0,
#     progress_bar=True,
# )

y_list = {}
for title, m in models.items():
    y_list[title] = f(m(coeffs[title]))

# ---------- Plotting ----------
# %% plot

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
matplotlib.use('Agg')
plt.close('all')

# compute errors
density_truth = density2xr(density_truth, vol)
errors = {}
recons = {}
rel_errors = {}
for title, coeff in coeffs.items():
    recon = density2xr(models[title](coeff), vol)
    recons[title] = recon
    err = recon - density_truth
    err = density2xr(err, vol)
    errors[title] = err

    rel_errors[title] = err / density_truth.where(density_truth.data != 0, 1) * 100

# --- Measurements ---
fig_meas = []
for title, y in y_list.items():
    fig_meas.append(
        Figure(f'{title} Meas.', Img(y.detach().cpu().numpy(), animation=True, height=300)),
    )

# --- 3D Densities ---

fig_density = []
for title, coeff in coeffs.items():
    density = models[title](coeff)
    fig_density.append(
        Figure(f'{title} Density', Img(preview3d(density), animation=True, height=300))
    )

# --- Relative Error ---

# norm = colors.CenteredNorm(halfrange=50)
norm = colors.Normalize(vmin=-50, vmax=50)
# boundaries = np.linspace(0, 50, 6)
# norm = colors.BoundaryNorm(boundaries, len(boundaries))
# cmap = plt.get_cmap('Greens')
cmap = plt.get_cmap('seismic')
cmap.set_over(color='black')
cmap.set_under(color='black')
# cmap = plt.get_cmap('PiYG')
# cmap.set_over(color='red')
# cmap.set_under(color='red')
fig_rel_err = []
for title, rel_err in rel_errors.items():
    if title == 'Truth':
        fig_rel_err.append(Img(None, width=300))
        continue

    fig = plt.figure(figsize=(5, 12))
    plt.suptitle(f'H-Density Percent Error')
    a = plt.subplot(3, 1, 1, aspect='equal')
    xr.plot.imshow(rel_err.sel(z=0, method='nearest'), norm=norm, cmap=cmap)
    a.set_title(None)
    a.set_aspect('equal')
    a = plt.subplot(3, 1, 2, aspect='equal')
    xr.plot.imshow(rel_err.sel(y=0, method='nearest'), norm=norm, cmap=cmap)
    a.set_title(None)
    a.set_aspect('equal')
    a = plt.subplot(3, 1, 3, aspect='equal')
    xr.plot.imshow(rel_err.sel(x=0, method='nearest'), norm=norm, cmap=cmap)
    a.set_title(None)
    a.set_aspect('equal')
    # c = plt.colorbar()
    # c.ax.set_ylabel("Percent")
    plt.tight_layout(pad=1)
    fig_rel_err.append(Figure(f'{title} Rel. Err.', Img(fig, width=300)))

    # import plotly.express as px
    # import plotly

    # fig_rel_err.append(HTML(
    #     plotly.io.to_html(
    #         px.imshow(rel_err.sel(x=0, method='nearest')),
    #         include_plotlyjs='cdn',
    #         full_html=False
    #     ))
    # )

# --- Recon Slice ---

fig_recon_slice = []
for title, coeff in coeffs.items():
    sliced = models[title](coeff)[s//2]
    fig_recon_slice.append(
        Figure(f'{title} Slice', Img(sliced, height=300))
    )

# --- Loss ---
fig_loss = []
for title, loss in losses.items():
    if title == 'Truth':
        fig_loss.append(Img(None, width=300))
        continue

    fig = plt.figure(figsize=(3, 3))
    plt.semilogy(loss)
    plt.tight_layout()
    plt.grid(True)

    fig_loss.append(
        Figure(f'{title} Loss', Img(fig, height=300))
    )

# --- Sph. Harm ---
fig_sphharm = []
for title, loss in losses.items():
    if title == 'Truth':
        fig_loss.append(Img(None, width=300))
        continue

    fig = plt.figure(figsize=(3, 3))
    plt.semilogy(loss)
    plt.tight_layout()
    plt.grid(True)

    fig_loss.append(
        Figure(f'{title} Loss', Img(fig, height=300))
    )


noisy_str = 'noisy' if f_truth.use_noise else 'noiseless'
path = f'/srv/www/display/{mo}mo_{noisy_str}_{len(y)}obs_{optimizer.__name__}.html'
Page(
    [
        [
            Figure('Meas. Locations', HTML(orbit_svg(vol, viewgeom2ts(view_geoms[0]), rotate=0).svg_str)),
        ],
        fig_meas,
        fig_density,
        fig_rel_err,
        # fig_abs_err,
        fig_recon_slice,
        fig_loss,
        [
            HTML('<br>'.join([
                f'{f.use_noise=}',
                f'{len(y)=}',
                f'optimizer={optimizer.__name__}',
            ]))
        ]
    ],
    head_extra=f"<script>{img_sync_js}</script>"
).save(path)
# ).save('/srv/www/display.html')
print(f"Wrote to {path}")

plt.close('all')
