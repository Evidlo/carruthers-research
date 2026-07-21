#!/usr/bin/env python3

from glide.science.forward_sph import *
from glide.science.model_sph import *
from glide.science.plotting_sph import *
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
# desc = 'january'
# start = np.datetime64('2026-01-19').astype('datetime64[ns]').astype(float)
# end = np.datetime64('2026-01-21').astype('datetime64[ns]').astype(float)
# desc = 'march'
# start = np.datetime64('2026-03-20').astype('datetime64[ns]').astype(float)
# end = np.datetime64('2026-03-22').astype('datetime64[ns]').astype(float)

# FIXME: -----------------------------------------
# FIXME: -----------------------------------------
# FIXME: -----------------------------------------
dates = np.linspace(start, end, num_obs:=30).astype('datetime64[ns]')

from load import load
nfi, wfi, dates = load(datapath, dates)
N = len(dates)

# date-major interleave [nfi_0, wfi_0, nfi_1, wfi_1, ...] for sc/ims — matches
# rvg.geoms order, so calibrate lines up per image
sc = [s for pair in zip(nfi.scraft.values, wfi.scraft.values) for s in pair]
ims = [im for pair in zip(nfi.images.values, wfi.images.values) for im in pair]

# %% forward model

# leading N axis puts the operator in dynamic mode, so each density slice maps
# to one date's NFI+WFI pair instead of broadcasting across all views. Model and
# plotting read only the spatial dims
rgrid = DefaultGrid(
    (N, 200, 45, 60), size_r=(3, 25), spacing='log',
    t=dates, timeunit='ns',
)

# zip the two cameras onto a new axis
rvg = ZippedGeom(
    sum([ScienceGeomFast(s, (100, 50), **d) for s in nfi.scraft.values]),
    sum([ScienceGeomFast(s, (100, 50), **d) for s in wfi.scraft.values]),
)

ralbedo=Albedo(
    xr.open_mfdataset('/home/jackson/glide-sdc/glide/validation/radiative_transfer/pipeline_test/albedo_data_*.nc'),
    rgrid, **d
)
f = ForwardSph(
    sc, rgrid=rgrid, rvg=rvg,
    g_factor=g(11e11),
    ralbedo=ralbedo(),
    **d
)
f.op.regs = None
t.cuda.empty_cache()

# calibrate() bins the L1C images and converts to the retrieval's native units
# (atom·Re/cm³). It assumes Rayleigh input (×1e6)
meas = f.calibrate(ims, disable_noise=True)


mr = SphHarmSplineModel(rgrid, max_l=0, cpoints=8, spacing='log', **d)

loss_fns = [
    # 1 * AbsLoss(mask=f.rmask),
    # 1 * SquareLoss(mask=f.rmask),
    1 * HuberLoss(mask=f.rmask),
    1e5 * NegRegularizer(),
    # 1e5 * DiffLoss(rgrid),
    # 2e4 * DiffLoss(rgrid),            # radial smoothness (was 5e2)
]

open('/tmp/losses_storm.txt', 'w').close()
# leading N dim → model emits (N, r, e, a): one independent reconstruction per date
initcoeffs = t.zeros((N, *mr.coeffs_shape), **d)

coeffs, retrieved_meas, losses = gd(
    f, t.nan_to_num(meas), mr, lr=5e1,
    loss_fns=loss_fns, num_iterations=1000,
    coeffs=initcoeffs,
    callbacks=[LogCallback('L0init', '/tmp/losses_storm.txt')],
)

retrieved = mr(coeffs)  # (N, r, e, a)

# %% plot — per-date sliders

labels = [str(d)[:16] for d in dates]

import time

with document('Storm 1D Month Retrieval') as doc:
    tags.h1(f'1D retrieval, {N} dates {labels[0]} … {labels[-1]}')
    with itemgrid(length=2):

        s = time.time()

        figset = {'width': 720}
        with caption('Recon (cardinal slices)'):
            slider(*[plot(cardplot(retrieved[i], rgrid, norm='log', method='nearest'), **figset)
                    for i in range(N)], labels=labels)

        with caption('Recon (radial profile)'):
            slider(*[plot(cardplotaxes(retrieved[i], rgrid, yscale='log', method='nearest'), **figset)
                    for i in range(N)], labels=labels)

        with caption('Diff from t=0 (cardinal slices)'):
            slider(*[plot(carderr(
                retrieved[i], retrieved[0],
                rgrid, rgrid,
                # norm='log'
                method='nearest',
                title='H Density Percent Change',
            ), **figset) for i in range(N)], labels=labels)

        with caption('Diff from t=0 (radial profile)'):
            slider(*[plot(carderraxes(
                retrieved[i], retrieved[0],
                rgrid,
                # yscale='log'
                method='nearest',
                title='H Density Percent Change',
            ), **figset) for i in range(N)], labels=labels)

        nmeas, wmeas = meas[:, 0], meas[:, 1]
        nvg, wvg = rvg.leaves[0], rvg.leaves[1]
        with caption('Radiance (TP alt) vs Density'):
            with slider(labels=labels):
                for i in range(N):
                    fig = radiance_v_density(
                        retrieved[i], rgrid,
                        nmeas[i], nvg[i], wmeas[i], wvg[i]
                    )
                    plot(fig, **figset)

        with caption('Coefficients'):
            with slider(labels=labels):
                for i in range(N):
                    plot(sphharmplot(mr.sph_coeffs(coeffs)[i], mr), width=500)

        with caption('Diff from t=0. Radiance (TP alt) vs Density'):
            with slider(labels=labels):
                for i in range(N):
                    fig = radiance_v_density_err(
                        retrieved[i], retrieved[0], rgrid,
                        nmeas[i], nmeas[0], nvg[i],
                        wmeas[i], wmeas[0], wvg[i]
                    )
                    plot(fig, **figset)

        plot(loss_plot(losses), **figset)

        with caption('Diff from t=0. Radiance (TP alt) vs Density'):
            with slider(labels=labels):
                for n, w in zip(ims[::2], ims[1::2]):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
                    ax1.imshow(n)
                    ax2.imshow(w)
                    plot(fig, **figset)

    tags.h1("Source Code")
    tags.code(tags.pre(open('recon_1D.py').read()))

outfile = Path(f'/www/storm/1D_{desc}.html')
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text(doc.render())
print(f'Saved to {outfile}')

# also archive the reconstruction
from datetime import datetime
outfile = Path(f'/www/sph/archive/{datetime.now().isoformat()}_storm.html')
outfile.write_text(doc.render())
