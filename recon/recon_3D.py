#!/usr/bin/env python3

from glide.science.forward_sph import *
from glide.science.model_sph import *
from glide.science.plotting_sph import *
from glide.science.plotting import sphharmplot
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
desc = '3d'
start = np.datetime64('2026-05-15').astype('datetime64[ns]').astype(float)
end = np.datetime64('2026-05-30').astype('datetime64[ns]').astype(float)

dates = np.linspace(start, end, num_obs:=14).astype('datetime64[ns]')
N = len(dates)

from load import load
nfi, wfi = load(datapath, 'NFI', dates), load(datapath, 'WFI', dates)

# set up spacecraft
sc = [s for pair in zip(nfi.scraft.values, wfi.scraft.values) for s in pair]
ims = [im for pair in zip(nfi.l1c_ims.values, wfi.l1c_ims.values) for im in pair]

# %% forward model

rgrid = DefaultGrid((200, 45, 60), size_r=(3, 25), spacing='log')

# zip the two cameras onto a new axis
rvg = ZippedGeom(
    sum([ScienceGeomFast(s, (100, 50), **d) for s in nfi.scraft.values]),
    sum([ScienceGeomFast(s, (100, 50), **d) for s in wfi.scraft.values]),
)

f = ForwardSph(sc, rgrid=rgrid, rvg=rvg, **d)
f.op.regs = None
t.cuda.empty_cache()

# calibrate() bins the L1C images and converts to the retrieval's native units
# (atom·Re/cm³). It assumes Rayleigh input (×1e6); our L1C is phot/s/cm²/sr, so
# pre-scale by 4π/1e6 (1e6 cancels the Rayleigh step, 4π takes per-sr→per-[4π sr]).
meas = f.calibrate([im * (4 * np.pi / 1e6) for im in ims], disable_noise=True)

# full reconstruction model
mr = SphHarmSplineModel(rgrid, max_l=3, cpoints=8, spacing='log', **d)

# reconstruction model for fast initialization of A00
mrinit = SphHarmSplineModel(
    rgrid, max_l=0,
    cpoints=mr.cpoints, spacing=mr.spacing,
    **d
)

initcoeffs = t.zeros(mr.coeffs_shape, **d)

loss_fns = [
    1 * AbsLoss(mask=f.rmask),
    1e5 * NegRegularizer(),
    5e2 * DiffLoss(rgrid),
    1e1 * SphHarmL1Regularizer(mrinit),
]

initcoeffs.data[0:1, :], _, _ = gd(
    f, meas, mrinit, lr=1e2,
    loss_fns=loss_fns[:-1], num_iterations=1000,
    callbacks=[LogCallback('L0init', '/tmp/losses_baseline.txt')],
)

# do full reconstruction
coeffs, retrieved_meas, losses = gd(
    f, meas, mr, lr=1e1,
    loss_fns=loss_fns, num_iterations=3000,
    coeffs=initcoeffs,
    callbacks=[LogCallback('fullL3', '/tmp/losses_baseline.txt')],
)

retrieved = mr(coeffs)  # (N, r, e, a)

# %% plot — per-date sliders

labels = [str(d)[:16] for d in dates]

with document('Storm 3D Month Retrieval') as doc:
    tags.h1(f'1D fast-init retrieval, {N} dates {labels[0]} … {labels[-1]}')
    with itemgrid(length=2):

        figset = {'height': 250}

        with caption('Recon (cardinal slices)'):
            plot(cardplot(retrieved, rgrid, norm='log'), **figset)

        with caption('Recon (radial profile)'):
            plot(cardplotaxes(retrieved, rgrid, yscale='log'), **figset)

        with caption('Radiance (TP alt) vs Density'):
            with slider(labels=labels):
                items = zip(
                    retrieved[None, ...].repeat(num_obs, 1, 1, 1),
                    meas[:, 0], meas[:, 1],
                    rvg.leaves[0].geoms, rvg.leaves[1].geoms
                )
                for ret, nmeas, wmeas, nvg, wvg in items:
                    fig = radiance_v_density(ret, rgrid, nmeas, nvg, wmeas, wvg)
                    plot(fig)

        with caption('Coefficients'):
            plot(sphharmplot(mr.sph_coeffs(coeffs), mr), **figset)

        plot(loss_plot(losses), **figset)

    tags.h1("Source Code")
    tags.code(tags.pre(open('recon_3D.py').read()))

outfile = Path(f'/www/storm/recon_{desc}.html')
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text(doc.render())
print(f'Saved to {outfile}')

# also archive the reconstruction
from datetime import datetime
outfile = Path(f'/www/sph/archive/{datetime.now().isoformat()}_storm.html')
outfile.write_text(doc.render())
