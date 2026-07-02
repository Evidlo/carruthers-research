#!/usr/bin/env python3
# 1D fast-init reconstructions over a month of real L1C measurements.
# Each date is an INDEPENDENT spherically-symmetric (max_l=0) reconstruction
# constrained by its NFI/WFI pair. Geometry/cameras come from the stored
# SpaceCraft objects; the L1C images are the measurement (no simulate).
#
# Dynamic operator pairing: a ZippedGeom groups each date's NFI+WFI cameras onto
# a new axis, so the operator maps density slice i -> (NFI_i, WFI_i) directly (no
# tiling). The model emits N slices; autograd sums the NFI+WFI residual gradients
# back into date i's density; regularizers still see the clean N-slice density.

from glide.science.forward_sph import *
from glide.science.model_sph import *
from glide.science.plotting_sph import *
from glide.science.recon.loss_sph import *

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

d = {'device': 'cuda'}

# %% load measurements

datapath = Path('/home/alex/carruthers/pseudo_l1c_data/')

# desc = 'march'
# start = np.datetime64('2026-03-20').astype('datetime64[ns]').astype(float)
# end = np.datetime64('2026-03-22').astype('datetime64[ns]').astype(float)
# desc = 'january'
# start = np.datetime64('2026-01-19').astype('datetime64[ns]').astype(float)
# end = np.datetime64('2026-01-21').astype('datetime64[ns]').astype(float)
desc = 'quiet'
start = np.datetime64('2026-01-15').astype('datetime64[ns]').astype(float)
end = np.datetime64('2026-01-18').astype('datetime64[ns]').astype(float)

dates = np.linspace(start, end, num_obs:=30).astype('datetime64[ns]')
N = len(dates)

from load import load
nfi, wfi = load(datapath, 'NFI', dates), load(datapath, 'WFI', dates)

# date-major interleave [nfi_0, wfi_0, nfi_1, wfi_1, ...] for sc/ims — matches
# rvg.geoms order, so calibrate lines up per image
sc = [s for pair in zip(nfi.scraft.values, wfi.scraft.values) for s in pair]
ims = [np.nan_to_num(im) for pair in zip(nfi.l1c_ims.values, wfi.l1c_ims.values) for im in pair]

# %% forward model

# model grid is 3D — per-date batching comes from a leading N dim on the coeffs
# (the model's einsum carries arbitrary leading dims). The forward needs a
# dynamic (N,...) grid so each density slice maps to one date's NFI+WFI pair.
rgrid = DefaultGrid((200, 45, 60), size_r=(3, 25), spacing='log')
# same spatial grid + a leading N axis, only to put the operator in dynamic mode
rgrid_dyn = DefaultGrid((N, *rgrid.shape), size_r=(3, 25), spacing='log')

# zip the two cameras onto a new axis: density slice i -> (NFI_i, WFI_i), no tiling.
# rvg.leaves[0]/[1] are the NFI/WFI collections; rvg.geoms is their flat date-major
# interleave (matches sc/ims order, so calibrate lines up per image)
rvg = ZippedGeom(
    sum([ScienceGeomFast(s, (100, 50), **d) for s in nfi.scraft.values]),
    sum([ScienceGeomFast(s, (100, 50), **d) for s in wfi.scraft.values]),
)

f = ForwardSph(sc, rgrid=rgrid_dyn, rvg=rvg, **d)
f.op.regs = None
t.cuda.empty_cache()

# calibrate() bins the L1C images and converts to the retrieval's native units
# (atom·Re/cm³). It assumes Rayleigh input (×1e6); our L1C is phot/s/cm²/sr, so
# pre-scale by 4π/1e6 (1e6 cancels the Rayleigh step, 4π takes per-sr→per-[4π sr]).
meas = f.calibrate([im * (4 * np.pi / 1e6) for im in ims], disable_noise=True)

# %% batched 1D fast-init recon (spherically symmetric, max_l=0, per date)

mrinit = SphHarmSplineModel(rgrid, max_l=0, cpoints=8, spacing='log', **d)

loss_fns = [
    1 * AbsLoss(mask=f.rmask),
    1e5 * NegRegularizer(),
    2e4 * DiffLoss(rgrid),            # radial smoothness (was 5e2)
]

open('/tmp/losses_storm.txt', 'w').close()
# leading N dim → model emits (N, r, e, a): one independent reconstruction per date
initcoeffs = t.zeros((N, *mrinit.coeffs_shape), **d)

coeffs, retrieved_meas, losses = gd(
    f, meas, mrinit, lr=5e1,
    loss_fns=loss_fns, num_iterations=1000,
    coeffs=initcoeffs,
    callbacks=[LogCallback('L0init', '/tmp/losses_storm.txt')],
)

retrieved = mrinit(coeffs)  # (N, r, e, a)

# %% plot — per-date sliders

labels = [str(d)[:16] for d in dates]

with document('Storm 1D Month Retrieval') as doc:
    tags.h1(f'1D fast-init retrieval, {N} dates {labels[0]} … {labels[-1]}')
    with itemgrid(length=2):

        figset = {'height': 250}
        with caption('Recon (cardinal slices)'):
            slider(*[plot(cardplot(retrieved[i], rgrid, norm='log'), **figset)
                    for i in range(N)], labels=labels)
        with caption('Recon (radial profile)'):
            slider(*[plot(cardplotaxes(retrieved[i], rgrid, yscale='log'), **figset)
                    for i in range(N)], labels=labels)

        with caption('Diff from t=0 (cardinal slices)'):
            slider(*[plot(carderr(
                retrieved[i], retrieved[0],
                rgrid, rgrid,
                # norm='log'
            ), **figset) for i in range(N)], labels=labels)

        with caption('Diff from t=0 (radial profile)'):
            slider(*[plot(carderraxes(
                retrieved[i], retrieved[0],
                rgrid,
                # yscale='log'
            ), **figset) for i in range(N)], labels=labels)

        with caption('Radiance (TP alt) vs Density'):
            with slider(labels=labels):
                items = zip(retrieved, meas[:, 0], meas[:, 1],
                            rvg.leaves[0].geoms, rvg.leaves[1].geoms)
                for ret, nmeas, wmeas, nvg, wvg in items:
                    fig = radiance_v_density(ret, rgrid, nmeas, nvg, wmeas, wvg)
                    plot(fig)

        plot(loss_plot(losses), **figset)

    tags.h1("Source Code")
    tags.code(tags.pre(open('recon_storm.py').read()))

outfile = Path(f'/www/storm/recon_{desc}.html')
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text(doc.render())
print(f'Saved to {outfile}')

# also archive the reconstruction
from datetime import datetime
outfile = Path(f'/www/sph/archive/{datetime.now().isoformat()}_storm.html')
outfile.write_text(doc.render())
