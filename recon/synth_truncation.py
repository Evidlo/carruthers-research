#!/usr/bin/env python3
"""Does forward-model truncation cause the A00 outer-knot blowup?

Real data cannot separate truncation from an additive background — both add
signal that grows with TP radius. Synthetic can, because we control the truth.

Two runs, identical geometry / grid / knots / losses, differing ONLY in whether
truth H exists beyond the 15 Re retrieval grid:

  full   : Zoennchen truth out to 100 Re, so every LOS carries gas the 15 Re
           model cannot represent  -> truncation present
  trunc  : the same truth zeroed beyond 15 Re                  -> truncation absent

If `full` reproduces the 29.3 -> 63.6 outer-knot upturn seen on real data and
`trunc` does not, truncation is the cause. No noise, no background, no cross-cal
error in either run.
"""

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
from tomosphero.retrieval import gd, LogCallback
from tomosphero.loss import *

import numpy as np, xarray as xr, torch as t
from importlib import resources
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib_torch
matplotlib_torch.activate()

wipe_gpu()
d = {'device': 'cuda'}

# %% real geometry, so the arc limitation is the real one

start = np.datetime64('2026-03-01').astype('datetime64[ns]').astype(float)
end = np.datetime64('2026-03-15').astype('datetime64[ns]').astype(float)
dates = np.linspace(start, end, 14).astype('datetime64[ns]')

from load import load
nfi, wfi, dates = load(Path('/data-products'), dates)
N = len(dates)
print(f'{N} frames')

alb = xr.open_dataset(resources.files('glide') / 'science/data_files/albedo_GLIDE_CDR.nc')
alb = alb.reindex(r=np.append(alb.r, 1e6), method='ffill')

scrafts = [ScienceGeomFast(s, (100, 50), **d) for s in wfi.scraft.values]

def forward(rgrid):
    f = ForwardSph(rgrid=rgrid, rvg=ZippedGeom(sum(scrafts)), g_factor=g(11e11),
                   ralbedo=Albedo(alb, rgrid, **d)(), **d)
    f.op.regs = None
    t.cuda.empty_cache()
    return f

# %% truth on a grid far larger than the retrieval grid

biggrid = DefaultGrid((300, 45, 60), size_r=(3, 100), spacing='log')
truth = Zoennchen24Model(biggrid, **d)()                     # (r, e, a)
rbig = t.as_tensor(biggrid.r, device=truth.device)
truth_trunc = truth * (rbig <= 15)[:, None, None]

fbig = forward(biggrid)
meas = {
    'full':  fbig(truth).detach(),
    'trunc': fbig(truth_trunc).detach(),
}
del fbig
t.cuda.empty_cache()

# how much unmodeled signal does the gas beyond 15 Re actually contribute?
from glide.science.forward_sph import tangent_points
tprad = t.linalg.norm(
    tangent_points(ZippedGeom(sum(scrafts)).leaves[0][0].ray_starts,
                   ZippedGeom(sum(scrafts)).leaves[0][0].rays), dim=-1
).detach().cpu().numpy()
extra = (meas['full'] - meas['trunc'])[0, 0].detach().cpu().numpy()
frac = (extra / meas['full'][0, 0].detach().cpu().numpy())
edges = np.geomspace(3.2, 40, 25)
print('\nunmodeled fraction of signal from H beyond 15 Re (frame 0):')
for j in range(len(edges) - 1):
    sel = (tprad >= edges[j]) & (tprad < edges[j+1]) & np.isfinite(frac)
    if sel.sum() > 20:
        print(f'  TP {np.sqrt(edges[j]*edges[j+1]):5.1f} Re: {np.median(frac[sel]):6.1%}')

# %% retrieve both on the 15 Re grid, identical settings

rgrid = DefaultGrid((200, 45, 60), size_r=(3, 15), spacing='log')
f = forward(rgrid)
mask = f.rmask

def run(y, tag):
    mr = SphHarmSplineModel(rgrid, max_l=1, cpoints=8, spacing='log', **d)
    mrinit = SphHarmSplineModel(rgrid, max_l=0, cpoints=8, spacing='log', **d)
    initcoeffs = t.zeros(mr.coeffs_shape, **d)

    sens = sensitivity(f, mr, mask=mask)
    w = sens.median() / sens
    W = t.tensor(np.stack([np.interp(np.log(rgrid.r), np.log(mr.cpoint_locs), row)
                           for row in w.cpu()]), **d)

    initcoeffs.data[0:1, :], _, _ = gd(
        f, y, mrinit, lr=1e1,
        loss_fns=[1 * HuberLoss(mask=mask), 1e5 * NegRegularizer()],
        num_iterations=2000, callbacks=[LogCallback(tag, '/tmp/losses_synth.txt')])

    coeffs, rmeas, losses = gd(
        f, y, mr, lr=1e0, num_iterations=3000, coeffs=initcoeffs,
        loss_fns=[1 * HuberLoss(mask=mask), 1e5 * NegRegularizer(),
                  # DiffLoss now divides by Δlog r, so the weight drops by
                  # 1/Δlog r² = 8.9e3 relative to the old 5e4 on this grid
                  3.24 * DiffLoss(rgrid), 1e3 * SphHarmL1Regularizer(mrinit, weights=W)],
        callbacks=[LogCallback(tag, '/tmp/losses_synth.txt')])
    return mr, coeffs.detach(), rmeas.detach(), losses

runs = {k: run(meas[k], k) for k in meas}

# %% compare against the known truth spherical mean

rsmall = t.as_tensor(rgrid.r, device=truth.device)
truth_mean = truth.mean((1, 2))                              # (r,) on biggrid
truth_on_small = t.tensor(np.interp(
    rgrid.r, biggrid.r, truth_mean.cpu().numpy()), device=truth.device)

print('\n=== A00 control-point coefficients ===')
for k, (mr, coeffs, _, _) in runs.items():
    a00 = coeffs[0].cpu().numpy()
    print(f'{k:6s}: ' + ' '.join(f'{v:8.1f}' for v in a00))
    print(f'{"":6s}  monotone decreasing: {bool(np.all(np.diff(a00) < 0))}')

print('\n=== retrieved spherical mean vs truth (%) ===')
print(f'{"r (Re)":>8} ' + ' '.join(f'{k:>10}' for k in runs))
for i in range(0, rgrid.shape.r, 20):
    row = f'{rgrid.r[i]:8.2f} '
    for k, (mr, coeffs, _, _) in runs.items():
        rec = mr(coeffs).mean((1, 2))[i].item()
        row += f'{rec/truth_on_small[i].item()-1:+9.1%} '
    print(row)

# %% page

with document('Synthetic truncation test') as doc:
    tags.h1('Does gas beyond the retrieval grid cause the outer-knot blowup?')
    tags.p(f'Zoennchen truth, real WFI geometry, {N} frames, no noise, no '
           f'background. Both runs use the same 3-15 Re grid, 8 log knots and '
           f'the same losses; they differ only in whether truth H exists beyond '
           f'15 Re.')

    figset = {'width': 720}
    with itemgrid(length=2):
        fig, ax = plt.subplots(dpi=150, figsize=(5.5, 3.5))
        for k, (mr, coeffs, _, _) in runs.items():
            ax.semilogx(mr.cpoint_locs, coeffs[0].cpu(), 'o-', label=k)
        ax.set(xlabel='control point (Re)', ylabel='A00 coefficient',
               title='A00 knots: upturn only with truncation?')
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        with caption('A00 control points'):
            plot(fig, **figset)

        fig, ax = plt.subplots(dpi=150, figsize=(5.5, 3.5))
        for k, (mr, coeffs, _, _) in runs.items():
            rec = mr(coeffs).mean((1, 2)).cpu()
            ax.semilogx(rgrid.r, (rec / truth_on_small.cpu() - 1) * 100, label=k)
        ax.axhline(0, c='gray', ls='--')
        ax.set(xlabel='r (Re)', ylabel='error vs truth (%)', ylim=(-60, 60),
               title='Retrieved spherical mean error')
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        with caption('Error vs known truth'):
            plot(fig, **figset)

        for k, (mr, coeffs, _, _) in runs.items():
            with caption(f'Coefficients — {k}'):
                plot(sphharmplot(mr.sph_coeffs(coeffs), mr), **figset)

    tags.h1('Source Code')
    tags.code(tags.pre(open('synth_truncation.py').read()))

outfile = Path('/www/synth/index.html')
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text(doc.render())
print(f'\nSaved to {outfile}')
