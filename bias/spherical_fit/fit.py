#!/usr/bin/env python3

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from glide.common_components.camera import CameraL1BNFI
from glide.common_components.cam import nadir_nfi_mode
from glide.common_components.generate_view_geom import gen_mission
from glide.science.forward_sph import *
from glide.science.model_sph import *
from glide.science.plotting import sphharmplot
from glide.science.plotting_sph import cardplot, cardplotaxes
from glide.science.recon.loss_sph import DiffLoss

from domrep import *
from tomosphero.plotting import loss_plot
from tomosphero.retrieval import gd, LogCallback
from tomosphero.loss import AbsLoss, NegRegularizer

from pathlib import Path
from common import load, rob_bias
from astropy.constants import R_earth as R_earth_const

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch as t

device = 'cuda'

# ----- Load and preprocess -----
img = load(str(Path(__file__).parent.parent / 'images_20260316/oob_nfi_l0.pkl'))
bias = rob_bias(img, 125, 350)
img_debiased = (img - bias).astype(np.float32)

hot_pixels = np.load(str(Path(__file__).parent.parent / 'claude_flatten/hot_pixels.npy'))
img_debiased[hot_pixels[:, 0], hot_pixels[:, 1]] = np.nan

# ----- Spacecraft / viewing geometry -----
rgrid = DefaultGrid((200, 45, 60), size_r=(3, 25), spacing='log')
cam = CameraL1BNFI(nadir_nfi_mode(t_op=360))
sc = gen_mission(num_obs=1, duration=1, start='2026-03-16', cams=[cam])

f = ForwardSph(
    sc, rgrid=rgrid,
    rvg=sum([ScienceGeomFast(s, (100, 50)) for s in sc]),
    device=device,
)

# ----- Row mask: keep rows 125-311 (top) and 712-898 (bottom) -----
row_mask = np.zeros((1024, 1024), dtype=bool)
row_mask[125:312, :] = True
row_mask[712:899, :] = True
row_maskb = rmask2bmask(t.from_numpy(row_mask).to(device), f.rvg[0].bin)
f.proj_maskb = f.proj_maskb * row_maskb.unsqueeze(0)

# ----- Measurements: bin real image into science pixel grid -----
meas_np = f.rvg[0].bin.compute(img_debiased[np.newaxis])  # (1, 100, 50) in Rayleighs
nan_maskb = t.from_numpy(~np.isnan(meas_np[0])).to(device)
f.proj_maskb = f.proj_maskb * nan_maskb.unsqueeze(0)
# unit conversion: Rayleigh → atom·Re/cm³ (same as calibrate() minus instrument cal)
meas_np = meas_np * 1e6 / f.g_factor / float(R_earth_const.to('cm').value)
meas = t.tensor(np.nan_to_num(meas_np, nan=0.0), dtype=t.float32, device=device)

# scale meas to match forward model output range (single reference pass)
with t.no_grad():
    ref_density = t.ones(*rgrid.shape, device=device, dtype=t.float32)
    ref_fwd = f(ref_density.unsqueeze(0))
    valid = f.proj_maskb[0].bool()
    scale = (ref_fwd.flatten()[f.proj_maskb.flatten()].mean() /
             meas.flatten()[f.proj_maskb.flatten()].mean()).item()
meas = meas * scale
print(f'measurement scale factor: {scale:.3f}')

# ----- Recon model (spherically symmetric) -----
mr = SphHarmSplineModel(rgrid, max_l=0, device=device, cpoints=12, spacing='log')

loss_fns = [
    1 * AbsLoss(projection_mask=f.proj_maskb),
    1e2 * NegRegularizer(),
    1e1 * DiffLoss(rgrid),
]

open('/tmp/losses_sphfit.txt', 'w').close()
coeffs, _, losses = gd(
    f, meas, mr, lr=5e-1,
    loss_fns=loss_fns, num_iterations=2000,
    callbacks=[LogCallback('L0sph', '/tmp/losses_sphfit.txt')],
)

retrieved = mr(coeffs)
fitted_meas = f(retrieved)[0].detach().cpu().numpy()  # (100, 50)
actual_meas = meas[0].detach().cpu().numpy()           # (100, 50)
maskb = f.proj_maskb[0].cpu().numpy()

def polar_meas_fig(actual, fitted, mask):
    import matplotlib.pyplot as plt
    theta = np.deg2rad(f.rvg[0].bin.theta_ctrs)  # (50,)
    r = f.rvg[0].bin.rad_ctrs                     # (100,) in image pixels
    T, R = np.meshgrid(theta, r)
    vmin = np.nanmin(actual[mask])
    vmax = np.nanmax(actual[mask])
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 4))
    for ax, data, title in zip(axes, [actual, fitted], ['Actual', 'Fitted']):
        d = np.where(mask, data, np.nan)
        im = ax.pcolormesh(T, R, d, vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax, label='atom·Re/cm³')
        ax.set_title(title)
    fig.tight_layout()
    return fig

# ----- HTML report -----
figset = {'height': 200}

with document('Spherical Fit — 0316 OOB NFI') as doc:
    tags.h1('Spherical Fit — 2026-03-16 OOB NFI (max_l=0)')
    caption(
        'Retrieved density (L=0 spherically symmetric)',
        plot(cardplot(retrieved.squeeze(), rgrid, norm='log'), **figset),
        plot(sphharmplot(mr.sph_coeffs(coeffs), mr), **figset),
    )
    caption(
        'Actual vs fitted measurements (science pixel polar grid)',
        plot(polar_meas_fig(actual_meas, fitted_meas, maskb), height=300),
    )
    tags.details(
        tags.summary('Loss curves + cardplotaxes'),
        plot(loss_plot(losses), **figset),
        plot(cardplotaxes(retrieved.squeeze(), rgrid, yscale='log'), **figset),
    )

outdir = Path('/www/spherical_fit')
outdir.mkdir(parents=True, exist_ok=True)
outfile = outdir / 'fit_0316.html'
outfile.write_text(doc.render())
t.save(coeffs, outdir / 'coeffs_0316.pt')
print(f'Saved to {outfile}')
