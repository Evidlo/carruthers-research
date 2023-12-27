#!/usr/bin/env python3

import copy
from dech import *
import torch as t
import torch_optimizer as optim
import numpy as np
import tomosipo as ts
from pathlib import Path

from glide.common_components.view_geometry import gen_mission, circular_orbit, carruthers_orbit, CameraWFI, CameraNFI
from glide.common_components.cam import CamMode, CamSpec, nadir_wfi_mode, nadir_nfi_mode
from glide.science.orbit import viewgeom2ts
from glide.science.model import SnowmanModel, ZoennchenModel, FullyDenseModel, SphHarmBasisModel, default_vol, density2xr, AxisAlignmentModel, vol2cart, CubesModel, PratikModel
from glide.science.forward import Forward, projection_mask, volume_mask
from glide.science.plotting import save_gif, preview3d, orbit_svg, imshow, color_negative, sphharmplot, carderr
from glide.science.recon import nag_ls, sirt, gd
from glide.science.recon.nag_ls import nag_ls_coeff, nag_ls_clip
from glide.science.common import cart2sph
from glide.science.recon.loss import *

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:500"
# t.cuda.set_per_process_memory_fraction(0.40, 0)

# ----- Density -----

s = 50
vol = default_vol(shape=s, size=50)

# ----- Mission Setup -----
# %% setup

mo = 3
num_obs = 50
cam_mode = nadir_wfi_mode()
cam_spec = CamSpec(cam_mode, true_flat=False)
cam_spec.fov = 3.6
nfi = CameraWFI(cam_mode, cam_spec)
nfi.camID = 'NFI'
cams=[
    # CameraNFI(),
    nfi,
    CameraWFI(),
]
# orbit = 'circular'
# view_geoms = gen_mission(
#     orbit=circular_orbit,
#     num_obs=num_obs, cams=cams,
#     radius=280, revolutions=0.5
# )
orbit = 'carruthers'
view_geoms = gen_mission(
    orbit=carruthers_orbit,
    num_obs=num_obs, cams=cams,
    duration=30*mo,
    # start='2025-07-01', duration=30*mo,
    # start='2025-04-01', duration=30*mo,
    # start='2025-05-15'
    start=(start:='spring'),
    # start=(start:='summer'),
    # start=(start:='fall'),
    # start=(start:='winter'),
)
# view_geoms = [view_geoms[0] + view_geoms[1]]


# ----- Forward Model -----
# %% forward

ys = {}
coeffs = {}
models = {}
losses = {}

# models['Truth'] = SphHarmBasisModel(vol, num_shells=20, max_l=2, device='cuda')
# coeffs['Truth'] = t.zeros(models['Truth'].coeffs_shape, device='cuda')
# coeffs['Truth'][3, 0] = np.random.random(models['Truth'].coeffs_shape)
# density_truth = models['Truth'](coeffs['Truth'])
print('1 model setup')
density_truth = ZoennchenModel(vol, device='cuda')()
# density_truth = PratikModel(vol, date=start, path=None, device='cuda', rinner=3)()
# density_truth = CubesModel(vol, device='cuda')()
# density_truth = AxisAlignmentModel(vol, device='cuda')()
# density_truth = SnowmanModel(vol, device='cuda')()
print('2 truth forward setup')
np.random.seed(0)
f_truth = Forward(view_geoms, vol, use_noise=True, use_grad=False, use_albedo=False, use_aniso=False)
print('3 noise application')
y_truth = f_truth(density_truth)

print('4 recon forward setup')
f = Forward(view_geoms, vol, use_noise=False, use_grad=True, use_albedo=False, use_aniso=False)

# %% recon

# desc = '_torus'
# desc = '_xaligned'
# desc = '_pratik'
# desc = '_deleteme'
desc = ''

ys['Truth'] = y_truth
coeffs['Truth'] = density_truth
models['Truth'] = FullyDenseModel(vol, device='cuda')
losses['Truth'] = None

print('5 model setup')
name = 'sph'
models[name] = SphHarmBasisModel(vol, num_shells=20, max_l=2, axis=(1, 0, 0), device='cuda')
print('6 gradient')
# grid_cart = vol2cart(vol)
# grid_sph = cart2sph(grid_cart)
_, losses[name], coeffs[name], ys[name] = gd(
    f, y_truth,
    model=models[name],
    num_iterations=1000,
    lr=1e2,
    optimizer=(optimizer:=optim.Yogi),
    loss_fn=square_loss,
    reg_fn=neg_reg,
    # loss_fn=square_loss_mask_gen(projection_mask(view_geoms, r=20)),
    # loss_fn=cheater_loss_gen(density_truth),
    # reg_fn=neg_reg_mask_gen(volume_mask(vol, r=20)),
    reg_lam=1e14,
    loss_history=True,
)

# name = 'dense'
# models[name] = FullyDenseModel(vol, device='cuda')
# _, losses[name], coeffs[name] = gd(
#     f, y_truth,
#     model=models[name],
#     num_iterations=300,
#     lr=1e2,
#     optimizer=(optimizer:=optim.Yogi),
#     loss_fn=square_loss,
#     reg_fn=neg_reg,
#     reg_lam=1e10,
#     loss_history=True
# )

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

    rel_errors[title] = err / density_truth.where(density_truth.data != 0, 0) * 100

# --- Measurements ---
fig_meas = []
for title, y in ys.items():
    fig_meas.append(Figure(
        f'{title} Meas.',
        Img(y.detach().cpu().numpy()[:, ::2, ::2], animation=True, height=300)),
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
# norm = colors.Normalize(vmin=-25, vmax=25)
boundaries = np.linspace(-50, 50, 6)
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

    # fig = plt.figure(figsize=(15, 5))
    # plt.suptitle(f'H-Density Percent Error')
    # a = plt.subplot(1, 3, 1, aspect='equal')
    # xr.plot.imshow(rel_err.sel(z=0, method='nearest'), norm=norm, cmap=cmap)
    # a.set_title(None)
    # a.set_aspect('equal')
    # a = plt.subplot(1, 3, 2, aspect='equal')
    # xr.plot.imshow(rel_err.sel(y=0, method='nearest'), norm=norm, cmap=cmap)
    # a.set_title(None)
    # a.set_aspect('equal')
    # a = plt.subplot(1, 3, 3, aspect='equal')
    # xr.plot.imshow(rel_err.sel(x=0, method='nearest'), norm=norm, cmap=cmap)
    # a.set_title(None)
    # a.set_aspect('equal')
    # plt.tight_layout(pad=1)
    # fig_rel_err.append(Figure(f'{title} Rel. Err.', Img(fig, width=800)))

    fig_rel_err.append(Figure(f'{title} Rel. Err.', Img(
        carderr(recon, density_truth, vol),
        width=800
    )))

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
    plt.title(f"Loss={loss[-1]:.3e}")
    plt.semilogy(loss)
    plt.tight_layout()
    plt.grid(True)

    fig_loss.append(
        Figure(f'{title} Loss', Img(fig, height=300))
    )

fig, _ = plt.subplots(figsize=((6, 4)))
sphplot = sphharmplot(coeffs['sph'], models['sph'], figure=fig)
fig_sphcoeffs = Figure('Sph Coeffs', Img(sphplot, width=300))

settings = HTML('<br>'.join([
    f'{f.use_noise=}',
    f'{num_obs=}',
    f'channels={cams}',
    f'density_shape (voxels)={vol.shape}',
    f'density_size (Re)={vol.size}',
    f'optimizer={optimizer.__name__}',
]))
page_code = Code(open('zoennchen_multi.py').read())

noisy_str = 'noisy' if f_truth.use_noise else 'noiseless'
# chans = ''.join(c.camID for c in cams).lower()
display_dir = Path('/srv/www/display')
path = display_dir / f'{mo}mo_{noisy_str}_{num_obs}obs_{start}{desc}.html'
p = Page(
    [
        [
            Figure('Meas. Locations', HTML(orbit_svg(vol, viewgeom2ts(view_geoms))._repr_html_())),
        ],
        fig_meas,
        fig_density,
        fig_rel_err,
        # fig_abs_err,
        fig_recon_slice,
        fig_loss,
        fig_sphcoeffs,
        settings,
        page_code
    ],
    head_extra=f"<script>{img_sync_js}</script>"
)
p.save(path)
print(f"Wrote to {path}")

from datetime import datetime
# archive error plot and all code
Page([
    fig_rel_err,
    fig_sphcoeffs,
    settings,
    page_code,
]).save(display_dir / 'archive' / f'{datetime.now().isoformat()}.html')

plt.close('all')
