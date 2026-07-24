#!/usr/bin/env python3

from glide.science.forward_sph import *
from glide.science.model_sph import *
from glide.science.plotting_sph import *
from glide.science.plotting import sphharmplot
from glide.science.recon.loss_sph import *
from glide.science.common import wipe_gpu
from glide.calibration.column_density import solar_flux_to_g_factor as g

from domrep import *

from pathlib import Path

from tomosphero import ZippedGeom
from tomosphero.plotting import *
from tomosphero.retrieval import gd, LogCallback
from tomosphero.loss import *

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib_torch
matplotlib_torch.activate()
import torch as t

wipe_gpu()

d = {'device': 'cuda'}

# %% load measurements

datapath = Path('/data-products')

desc = 'quiet'
start = np.datetime64('2026-03-01').astype('datetime64[ns]').astype(float)
end = np.datetime64('2026-03-15').astype('datetime64[ns]').astype(float)

dates = np.linspace(start, end, num_obs:=14).astype('datetime64[ns]')

from load import load
nfi, wfi, dates = load(datapath, dates)
N = len(dates)

# (date, camera) pairs, mirroring the leading axes of the zipped rvg below
ims = list(zip(
    nfi.images.values,
    # wfi.images.values
))

# %% forward model

# grid extends far past the science region so the outer shells absorb the WFI
# additive background instead of corrupting 6–15 Re (see AGENTS.md)
rgrid = DefaultGrid((200, 45, 60), size_r=(3, 15), spacing='log')

# albedo table ends at 41.4 Re (NaNs the whole operator beyond) — extend it
# outward holding the outer row, which is already flat at ~1.007
from importlib import resources
alb = xr.open_dataset(resources.files('glide') / 'science/data_files/albedo_GLIDE_CDR.nc')
alb = alb.reindex(r=np.append(alb.r, 1e6), method='ffill')

# zip the two cameras onto a new axis
rvg = ZippedGeom(
    sum([ScienceGeomFast(s, (100, 50), **d) for s in nfi.scraft.values]),
    # sum([ScienceGeomFast(s, (100, 50), **d) for s in wfi.scraft.values]),
)

f = ForwardSph(
    rgrid=rgrid, rvg=rvg,
    g_factor=g(11e11),
    ralbedo=Albedo(alb, rgrid, **d)(),
    **d
)
f.op.regs = None
t.cuda.empty_cache()

# calibrate() bins the L1C images and converts to the retrieval's native units
# (atom·Re/cm³). It assumes Rayleigh input (×1e6)
meas = f.calibrate(ims, disable_noise=True)
# mask = f.rmask * meas.isfinite()
# meas = t.nan_to_num(meas) * mask

# full reconstruction model
mr = SphHarmSplineModel(rgrid, max_l=1, cpoints=8, spacing='log', **d)

# reconstruction model for fast initialization of A00
mrinit = SphHarmSplineModel(
    rgrid, max_l=0,
    cpoints=mr.cpoints, spacing=mr.spacing,
    **d
)

initcoeffs = t.zeros(mr.coeffs_shape, **d)

# sensitivity weighting (Wiener-like): penalize coefficients ∝ 1/‖F·basis‖ so
# modes the geometry barely senses shrink to the prior instead of absorbing
# systematics.  Set W = None to disable (recovers unweighted L1)
sens = sensitivity(f, mr, mask=f.rmask)  # (16, 8) per (l,m,knot)
w = sens.median() / sens
W = t.tensor(np.stack(
    [np.interp(np.log(rgrid.r), np.log(mr.cpoint_locs), row) for row in w.cpu()]
), **d)  # (16, num_shells)

# combined mask, not just rmask: fully-masked bins are NaN→0 and would
# otherwise pull the fit down
initcoeffs.data[0:1, :], _, _ = gd(
    f, t.nan_to_num(meas), mrinit, lr=1e1,
    loss_fns=[1 * HuberLoss(mask=f.rmask), 1e5 * NegRegularizer()],
    num_iterations=2000,
    callbacks=[LogCallback('L0init', '/tmp/losses_baseline.txt')],
)

# stage 2: stronger radial smoothing + A00 anchored to trusted stage-1 values —
# swept on real quiet data and validated against Zoennchen truth (see AGENTS.md)
loss_fns = [
    1 * HuberLoss(mask=f.rmask),
    1e5 * NegRegularizer(),
    5e4 * DiffLoss(rgrid),
    1e3 * SphHarmL1Regularizer(mrinit, weights=W),
    # 1e3 * SphHarmL1Regularizer(mrinit),
    # 1e4 * AnchorRegularizer(initcoeffs),
]

coeffs, retrieved_meas, losses = gd(
    f, t.nan_to_num(meas), mr, lr=1e0,
    loss_fns=loss_fns, num_iterations=3000,
    coeffs=initcoeffs,
    callbacks=[LogCallback('fullL3', '/tmp/losses_baseline.txt')],
)

# zeros/negatives in the far background-absorber shells break log plots
retrieved = mr(coeffs).clamp(min=1e-4)  # (r, e, a)

# %% plot — per-date sliders

labels = [str(d)[:16] for d in dates]

with document('Storm 3D Month Retrieval') as doc:
    tags.h1(f'3D retrieval, {N} dates {labels[0]} … {labels[-1]}')
    with itemgrid(length=2):

        figset = {'width': 720}

        with caption('Recon (cardinal slices)'):
            plot(cardplot(retrieved, rgrid, norm='log'), **figset)

        with caption('Recon (radial profile)'):
            plot(cardplotaxes(retrieved, rgrid, yscale='log'), **figset)

        with caption('Radiance (TP alt) vs Density'):
            with slider(labels=labels):
                items = zip(
                    retrieved[None, ...].repeat(N, 1, 1, 1),
                    meas[:, 0], meas[:, 1],
                    rvg.leaves[0], rvg.leaves[1]
                )
                for ret, nmeas, wmeas, nvg, wvg in items:
                    fig = radiance_v_density(ret, rgrid, nmeas, nvg, wmeas, wvg)
                    plot(fig, **figset)

        with caption('Coefficients'):
            plot(sphharmplot(mr.sph_coeffs(coeffs), mr), **figset)

        plot(loss_plot(losses), **figset)

    tags.h1("Source Code")
    tags.code(tags.pre(open('recon_3D.py').read()))

outfile = Path(f'/www/storm/3D_{desc}.html')
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text(doc.render())
print(f'Saved to {outfile}')

# also archive the reconstruction
from datetime import datetime
outfile = Path(f'/www/sph/archive/{datetime.now().isoformat()}_3D.html')
outfile.write_text(doc.render())
