#!/usr/bin/env python3
"""L1C -> L2 demo: 1D retrieval, dumped to a pkl, replotted from the pkl alone.

"""

from glide.science.forward_sph import *
from glide.science.model_sph import *
from glide.science.plotting_sph import *
from glide.science.recon.loss_sph import *
from glide.science.common import wipe_gpu
from glide.calibration.column_density import solar_flux_to_g_factor as g

from pathlib import Path

from tomosphero import ZippedGeom
from tomosphero.plotting import *
from tomosphero.retrieval import gd, LogCallback
from tomosphero.loss import *

import pickle
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

# FIXME: -----------------------------------------
# FIXME: Inputs supplied by L1C pipeline
# FIXME: -----------------------------------------

datapath = Path('/data-products')

# desc = 'january_wfi_newgrid'
# start = np.datetime64('2026-01-19').astype('datetime64[ns]').astype(float)
# end = np.datetime64('2026-01-21').astype('datetime64[ns]').astype(float)
# desc = 'march_quiet_wfi_newgrid'
# start = np.datetime64('2026-03-01').astype('datetime64[ns]').astype(float)
# end = np.datetime64('2026-03-15').astype('datetime64[ns]').astype(float)
desc = 'march_storm_wfi_analytic_builtin'
start = np.datetime64('2026-03-20').astype('datetime64[ns]').astype(float)
end = np.datetime64('2026-03-22').astype('datetime64[ns]').astype(float)
# desc = 'march_complete_wfi_newgrid'
# start = np.datetime64('2026-03-01').astype('datetime64[ns]').astype(float)
# end = np.datetime64('2026-04-01').astype('datetime64[ns]').astype(float)
# desc = 'march_8_wfi_newgrid'
# start = np.datetime64('2026-03-08').astype('datetime64[ns]').astype(float)
# end = np.datetime64('2026-03-09').astype('datetime64[ns]').astype(float)

dates = np.linspace(start, end, num_obs:=50).astype('datetime64[ns]')

import sys
sys.path.insert(0, str(Path(__file__).parent / 'recon'))
from load import load
nfi, wfi, dates = load(datapath, dates)

# per-camera lists.  ims is (date, camera) pairs, mirroring the leading axes of
# the zipped rvg below
scraft = [
    # nfi.scraft.values,
    wfi.scraft.values,
]
ims = list(zip(
    # nfi.images.values,
    wfi.images.values
))

albedo_data = xr.open_mfdataset(
    '/home/jackson/glide-sdc/glide/validation/radiative_transfer/pipeline_test/albedo_data_*.nc'
)
solar_flux = 11e11

# FIXME: -----------------------------------------
# FIXME: Begin reconstruction code
# FIXME: -----------------------------------------

# %% forward model

N = len(dates)

# leading N axis puts the operator in dynamic mode, so each density slice maps
# to one date's NFI+WFI pair instead of broadcasting across all views. Model and
# plotting read only the spatial dims
rgrid = DefaultGrid(
    (N, 200, 45, 60), size_r=(3, 25), spacing='log',
    t=dates, timeunit='ns'
)

# zip the cameras onto a new axis.  masklim pins the fitted pixels to the grid
# extent; the analytic tail (ForwardSph tail_slope) supplies signal beyond it
rvg = ZippedGeom(*[
    sum([ScienceGeomFast(s, (100, 50), masklim=rgrid.size.r, **d) for s in cam])
    for cam in scraft
])

ralbedo=Albedo(albedo_data, rgrid, **d)
f = ForwardSph(
    rgrid=rgrid, rvg=rvg,
    g_factor=g(solar_flux),
    ralbedo=ralbedo(),
    tail_slope=2.75,
    **d
)
f.op.regs = None
t.cuda.empty_cache()

# calibrate() bins the L1C images and converts to the retrieval's native units
# (atom·Re/cm³). It assumes Rayleigh input (×1e6)
meas = f.calibrate(ims, disable_noise=True)


mr = SphHarmSplineModel(
    rgrid,
    max_l=0,
    cpoints=8, spacing='log',
    **d
)

loss_fns = [
    # 1 * AbsLoss(mask=f.rmask),
    # 1 * SquareLoss(mask=f.rmask),
    1 * HuberLoss(mask=f.rmask),
    1e5 * NegRegularizer(),
    # 11.2 * DiffLoss(rgrid),           # = old 1e5; DiffLoss now /Δlog r, (3,25)x200
    # 2.25 * DiffLoss(rgrid),           # radial smoothness (= old 2e4)
]

open('/tmp/losses_storm.txt', 'w').close()
# leading N dim → model emits (N, r, e, a): one independent reconstruction per date
initcoeffs = t.zeros((N, *mr.coeffs_shape), **d)

coeffs, retrieved_meas, losses = gd(
    f, t.nan_to_num(meas), mr, lr=1e1,
    loss_fns=loss_fns, num_iterations=1000,
    coeffs=initcoeffs,
    callbacks=[LogCallback('L0init', '/tmp/losses_storm.txt')],
)

# FIXME: -----------------------------------------
# FIXME: save results to pkl
# FIXME: -----------------------------------------

pklfile = Path(f'/tmp/L2_{desc}.pkl')
pklfile.write_bytes(pickle.dumps({
    'coeffs': coeffs, 'model': mr, 'grid': rgrid, 'rvg': rvg,
    'meas': meas, 'ims': ims, 'dates': dates, 'losses': losses,
}))
print(f'Wrote L2 product to {pklfile}')

# FIXME: -----------------------------------------
# FIXME: load results from pkl
# FIXME: -----------------------------------------

l2 = pickle.loads(pklfile.read_bytes())

coeffs, mr, rgrid, rvg = l2['coeffs'], l2['model'], l2['grid'], l2['rvg']
meas, ims, dates, losses = l2['meas'], l2['ims'], l2['dates'], l2['losses']
N = len(dates)

retrieved = mr(coeffs)  # (N, r, e, a)

# ----- Plotting -----
# %% plot

# each date is retrieved independently; figures below show the first one
figures = [
    # primary figures
    cardplot(retrieved[0], rgrid.static, method='nearest'),
    cardplotaxes(retrieved[0], rgrid.static, method='nearest'),
    sphharmplot(mr.sph_coeffs(coeffs)[0], mr),
    loss_plot(losses),
    radiance_v_density(retrieved[0], rgrid, meas[0], [leaf[0] for leaf in rvg.leaves]),
]

# FIXME: -----------------------------------------
# FIXME: remove this section for production, which stops at the pkl above
# FIXME: -----------------------------------------
# ----- Save plots to disk -----

outdir = Path(f'/www/storm/1D_{desc}_demo')
outdir.mkdir(parents=True, exist_ok=True)
for name, fig in zip(
        ['cardplot', 'cardplotaxes', 'sphharmplot', 'lossplot', 'scibinradiance']
        figures
):
    fig.savefig(outdir / f'{name}.png', bbox_inches='tight')
print(f'Saved {len(figures)} figures to {outdir}')
