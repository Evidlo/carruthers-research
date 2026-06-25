#!/usr/bin/env python3
# 1D fast-init reconstructions over a month of real L1C measurements.
# Each date is an INDEPENDENT spherically-symmetric (max_l=0) reconstruction
# constrained by its NFI/WFI pair. Geometry/cameras come from the stored
# SpaceCraft objects; the L1C images are the measurement (no simulate).
#
# Dynamic operator pairing: the operator maps geom i <-> density slice i (one
# detector per slice). To give each date BOTH cameras, the model emits N slices
# and a thin forward wrapper tiles them to 2N (interleaved [nfi_i, wfi_i]) before
# raytracing. Autograd sums the NFI+WFI residual gradients back into date i's
# density; regularizers still see the clean N-slice density.

from glide.science.forward_sph import *
from glide.science.model_sph import *
from glide.science.plotting import *
from glide.science.plotting_sph import cardplot, cardplotaxes
from glide.science.recon.loss_sph import *

from domrep import *

from pathlib import Path

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

device = 'cuda'

# %% load measurements

datapath = Path('/home/alex/carruthers/pseudo_l1c_data/')

# start = np.datetime64('2026-03-20').astype('datetime64[ns]').astype(float); desc = 'march'
# end = np.datetime64('2026-03-22').astype('datetime64[ns]').astype(float)
# start = np.datetime64('2026-01-19').astype('datetime64[ns]').astype(float); desc = 'january'
# end = np.datetime64('2026-01-21').astype('datetime64[ns]').astype(float)
start = np.datetime64('2026-01-17').astype('datetime64[ns]').astype(float); desc = 'quiet'
end = np.datetime64('2026-01-18').astype('datetime64[ns]').astype(float)

dates = np.linspace(start, end, num_obs:=5).astype('datetime64[ns]')
N = len(dates)

from load import load
nfi, wfi = load(datapath / 'nfi.zarr', dates), load(datapath / 'wfi.zarr', dates)

# interleave cameras per date: sc/ims order = [nfi_0, wfi_0, nfi_1, wfi_1, ...]
# so a repeat_interleave(2) tile of the N-slice density lines up: slice i -> 2i,2i+1
sc = [s for pair in zip(nfi.scraft.values, wfi.scraft.values) for s in pair]
# out-of-FOV pixels are nan; zero them (FOV itself is masked via input_mask)
ims = [np.nan_to_num(im) for pair in zip(nfi.im.values, wfi.im.values) for im in pair]

# %% forward model

# model grid is 3D — per-date batching comes from a leading N dim on the coeffs
# (the model's einsum carries arbitrary leading dims). The forward needs a
# dynamic (4D) grid so its op pairs each of the 2N cameras to a tiled slice.
rgrid = DefaultGrid((200, 45, 60), size_r=(3, 25), spacing='log')
rgrid2 = DefaultGrid((2 * N, 200, 45, 60), size_r=(3, 25), spacing='log')

# mask out-of-FOV pixels (which hold the nans that wreck the loss)
input_mask = (s.sensor.spec.mask_fov for s in sc)

f = ForwardSph(
    sc,
    rgrid=rgrid2,
    rvg=sum([ScienceGeomFast(s, (100, 50)) for s in sc]),
    input_mask=input_mask,
    device=device,
)
f.op.regs = None
t.cuda.empty_cache()


class ForwardTiled:
    """Wrap a ForwardSph so an N-slice density is tiled to the 2N cameras
    ([nfi_i, wfi_i]) before raytracing. gd/model/regularizers stay on N slices."""
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x.repeat_interleave(2, dim=0))

    def __getattr__(self, k):
        return getattr(self.f, k)


ftiled = ForwardTiled(f)

# calibrate() bins the L1C images and converts to the retrieval's native units
# (atom·Re/cm³). It assumes Rayleigh input (×1e6); our L1C is phot/s/cm²/sr, so
# pre-scale by 4π/1e6 (1e6 cancels the Rayleigh step, 4π takes per-sr→per-[4π sr]).
meas = f.calibrate([im * (4 * np.pi / 1e6) for im in ims], disable_noise=True)

# %% batched 1D fast-init recon (spherically symmetric, max_l=0, per date)

mrinit = SphHarmSplineModel(rgrid, max_l=0, cpoints=12, spacing='log', device=device)

loss_fns = [
    1 * AbsLoss(projection_mask=f.proj_maskb),
    1e5 * NegRegularizer(),
    5e2 * DiffLoss(rgrid),
]

open('/tmp/losses_storm.txt', 'w').close()
# leading N dim → model emits (N, r, e, a): one independent reconstruction per date
initcoeffs = t.zeros((N, *mrinit.coeffs_shape), device=device)

coeffs, retrieved_meas, losses = gd(
    ftiled, meas, mrinit, lr=1e2,
    loss_fns=loss_fns, num_iterations=3000,
    coeffs=initcoeffs,
    callbacks=[LogCallback('L0init', '/tmp/losses_storm.txt')],
)

retrieved = mrinit(coeffs)  # (N, r, e, a)

# %% plot — per-date sliders

labels = [str(d)[:16] for d in dates]

with document('Storm 1D Month Retrieval') as doc:
    tags.h1(f'1D fast-init retrieval, {N} dates {labels[0]} … {labels[-1]}')

    figset = {'height': 250}
    plot(loss_plot(losses), height=200)
    with caption('Recon (cardinal slices)'):
        slider(*[plot(cardplot(retrieved[i], rgrid, norm='log'), **figset)
                 for i in range(N)], labels=labels)
    with caption('Recon (radial profile)'):
        slider(*[plot(cardplotaxes(retrieved[i], rgrid, yscale='log'), **figset)
                 for i in range(N)], labels=labels)

    with caption('Radiance (TP alt) vs Density'):
        with slider(labels=labels):
            items = zip(retrieved, meas[::2], meas[1::2], f.rvg[::2], f.rvg[1::2])
            for ret, nmeas, wmeas, nvg, wvg in items:
                fig, ax1 = plt.subplots()

                # --- NFI ---
                # plot measurements vs TP radius
                tprad = t.linalg.norm(tangent_points(nvg.ray_starts, nvg.rays), dim=-1)
                thind = 0 # theta slice to plot
                plt.plot(tprad[:, pind], nmeas[:, pind], label='NFI Radiance')
                # --- WFI ---
                # plot measurements vs TP radius
                tprad = t.linalg.norm(tangent_points(wvg.ray_starts, wvg.rays), dim=-1)
                thind = 0 # theta slice to plot
                plt.plot(tprad[:, pind], wmeas[:, thind], label='WFI Radiance')

                ax1.set_ylim((0, 12000))

                # --- Density ---
                ax2 = ax1.twinx()
                ax2.plot(rgrid.r, ret[:, 22, 30], 'r', label='Density')

                ax2.set_ylim((0, 1300))

                ax1.set_xlabel('Re (TP) / Re')
                fig.legend()
                plot(fig)

    tags.h1("Source Code")
    tags.code(tags.pre(open('recon_storm.py').read()))

outfile = Path(f'/www/storm/recon_{desc}.html')
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text(doc.render())
print(f'Saved to {outfile}')
