#!/usr/bin/env python3
# 1D fast-initialization reconstruction from a real 2026-03-20 L1C measurement.
# Geometry comes from the SpaceCraft objects stored in the zarr; the L1C images
# are used directly as the measurement (no simulate/calibrate).

from glide.common_components.generate_view_geom import gen_mission
from glide.science.forward_sph import *
from glide.science.model_sph import *
from glide.science.plotting import *
from glide.science.plotting_sph import cardplot, cardplotaxes
from glide.science.recon.loss_sph import *

from domrep import *

from pathlib import Path
import pickle
import base64

from tomosphero.plotting import *
from tomosphero.retrieval import gd, LogCallback
from tomosphero.loss import *

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import torch as t

device = 'cuda'

# %% load measurement

datapath = Path('/home/alex/carruthers/pseudo_l1c_data/')
date = np.datetime64('2026-03-20')

from load import load
nfi, wfi = load(datapath / 'nfi.zarr', [date]), load(datapath / 'wfi.zarr', [date])

# scrafts carry their own cameras, so feed them straight to the forward model
sc = [nfi.scraft.values[0], wfi.scraft.values[0]]
# out-of-FOV pixels are nan; zero them (FOV itself is masked via input_mask)
ims = [np.nan_to_num(nfi.im.values[0]), np.nan_to_num(wfi.im.values[0])]

# %% forward model

sgrid = DefaultGrid((500, 45, 60), size_r=(3, 25), spacing='log')
rgrid = DefaultGrid((200, 45, 60), size_r=(3, 25), spacing='log')


# mask out-of-FOV pixels (which hold the nans that wreck the loss)
input_mask = (s.sensor.spec.mask_fov for s in sc)

f = ForwardSph(
    sc, sgrid=sgrid,
    rgrid=rgrid,
    rvg=sum([ScienceGeomFast(s, (100, 50)) for s in sc]),
    input_mask=input_mask,
    device=device,
)

# calibrate() ingests L1C images: bins them (rvg.bin) and converts to the
# retrieval's native units (atom·Re/cm³) — the same space f(model(coeffs)) lives
# in. It assumes Rayleigh input (×1e6 → phot/s/cm²/[4π sr]); our L1C is already
# phot/s/cm²/sr, so pre-scale by 4π/1e6: the 1e6 cancels calibrate's Rayleigh
# step and the 4π takes per-sr → per-[4π sr].
meas = f.calibrate([im * (4 * np.pi / 1e6) for im in ims], disable_noise=True)

# %% 1D fast-init recon (spherically symmetric, max_l=0)

mrinit = SphHarmSplineModel(rgrid, max_l=0, cpoints=12, spacing='log', device=device)

# NB: SphHarmL1Regularizer is dropped here — it normalizes by the monopole (A00),
# which is 0 at initcoeffs=0 (→ nan), and is meaningless for a max_l=0 model where
# the monopole is the only term. It belongs in the full max_l>0 reconstruction.
loss_fns = [
    1 * AbsLoss(projection_mask=f.proj_maskb),
    1e4 * NegRegularizer(),
    5e2 * DiffLoss(rgrid),
]

open('/tmp/losses_storm.txt', 'w').close()
initcoeffs = t.zeros(mrinit.coeffs_shape, device=device)

coeffs, retrieved_meas, losses = gd(
    f, meas, mrinit, lr=1e2,
    loss_fns=loss_fns, num_iterations=3000,
    coeffs=initcoeffs,
    callbacks=[LogCallback('L0init', '/tmp/losses_storm.txt')],
)

retrieved = mrinit(coeffs)

# %% plot

with document('Storm 1D Fast-Init Recon') as doc:
    tags.h1(f'1D fast init from {date} measurement')

    figset = {'height': 200}
    caption(
        f"recon={mrinit}",
        plot(loss_plot(losses), **figset),
        tags.br(),
        caption("Recon", plot(cardplot(retrieved.squeeze(), rgrid, norm='log'), **figset)),
        caption("Recon (radial)", plot(cardplotaxes(retrieved.squeeze(), rgrid, yscale='log'), **figset)),
    )

    tags.h1("Source Code")
    tags.code(tags.pre(open('recon_storm.py').read()))

outfile = Path('/www/storm/recon.html')
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text(doc.render())
print(f'Saved to {outfile}')
