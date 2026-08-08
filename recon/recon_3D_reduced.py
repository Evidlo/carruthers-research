#!/usr/bin/env python3
"""recon_3D with an angle-limited basis, run side by side against the current one.

sensitivity.py finding: over this 2-week arc A11 and B11 — the two phases of the
equatorial dipole — are 0.92–0.95 correlated at every radius, with sigma2/sigma1
~ 0.13.  The geometry measures the dipole along ONE azimuth (~111 deg, constant
with radius to 1.6 deg, so it is a viewing property not a density one) and nulls
the orthogonal phase.  Fitting both phases hands the solver an ~8x-amplified null
direction for cross-cal and background residual to occupy.

So: keep A00 and A10, and replace the (A11, B11) pair with a SINGLE dipole locked
to the measurable azimuth.  4 angular harmonics -> 3, radial knots unchanged.
The azimuth is measured from the geometry at run time, not hardcoded, because it
depends on the date window.
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
from tomosphero.plotting import *
from tomosphero.retrieval import gd, LogCallback
from tomosphero.loss import *

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib_torch
matplotlib_torch.activate()
import torch as t

wipe_gpu()

d = {'device': 'cuda'}

# %% load measurements — identical to recon_3D.py

datapath = Path('/data-products')

desc = 'quiet'
start = np.datetime64('2026-03-01').astype('datetime64[ns]').astype(float)
end = np.datetime64('2026-03-15').astype('datetime64[ns]').astype(float)

dates = np.linspace(start, end, num_obs:=14).astype('datetime64[ns]')

from load import load
nfi, wfi, dates = load(datapath, dates)
N = len(dates)

ims = list(zip(wfi.images.values))

# %% forward model — identical to recon_3D.py

rgrid = DefaultGrid((200, 45, 60), size_r=(3, 15), spacing='log')

from importlib import resources
alb = xr.open_dataset(resources.files('glide') / 'science/data_files/albedo_GLIDE_CDR.nc')
alb = alb.reindex(r=np.append(alb.r, 1e6), method='ffill')

rvg = ZippedGeom(
    sum([ScienceGeomFast(s, (100, 50), **d) for s in wfi.scraft.values]),
)

f = ForwardSph(
    rgrid=rgrid, rvg=rvg,
    g_factor=g(11e11),
    ralbedo=Albedo(alb, rgrid, **d)(),
    **d
)
f.op.regs = None
t.cuda.empty_cache()

meas = f.calibrate(ims, disable_noise=True)
mask = f.rmask * meas.isfinite()
measn = t.nan_to_num(meas)

# %% measure the resolvable dipole azimuth from the geometry

def resolved_azimuth(cpoints=8):
    """Azimuth of the equatorial-dipole phase this arc senses, in radians.

    Pushes the A11 and B11 basis vectors through f at each knot and takes the
    leading left singular vector of the 2-column pair.
    """
    probe = SphHarmSplineModel(rgrid, lm=[(1, 1), (1, -1)], cpoints=cpoints,
                               spacing='log', **d)
    eye = t.eye(2 * cpoints, **d).reshape(2 * cpoints, *probe.coeffs_shape)
    cols = t.cat([
        (f(probe(eye[i:i+2])) * mask).flatten(1).double().cpu()
        for i in range(0, 2 * cpoints, 2)
    ]).numpy()  # (2*cpoints, M), harmonic-major

    angles = []
    for k in range(cpoints):
        pair = np.stack([cols[k], cols[cpoints + k]])
        u, sv, _ = np.linalg.svd(pair, full_matrices=False)
        angles.append((np.arctan2(u[1, 0], u[0, 0]) % np.pi, sv[1] / sv[0]))
    t.cuda.empty_cache()
    return probe, np.array(angles)

probe, ang = resolved_azimuth()
psi = ang[:, 0].mean()
print(f'resolved dipole azimuth {np.degrees(psi):.1f} deg '
      f'(spread {np.degrees(np.ptp(ang[:, 0])):.1f}), '
      f'null suppression {ang[:, 1].mean():.3f}')

# %% the two models under comparison

def make_full():
    return SphHarmSplineModel(rgrid, max_l=1, cpoints=8, spacing='log', **d)

def make_reduced():
    """A00, A10, and one dipole rotated to the measurable azimuth"""
    m = SphHarmSplineModel(rgrid, lm=[(0, 0), (1, 0), (1, 1)], cpoints=8,
                           spacing='log', **d)
    # cos(a - psi) = cos(psi) Y11c + sin(psi) Y11s — a rotation about z, so the
    # retained mode is still an exact l=1 harmonic, just phase-locked
    m.grid_harm[..., 2] = (np.cos(psi) * probe.grid_harm[..., 0]
                           + np.sin(psi) * probe.grid_harm[..., 1])
    m.names[2] = f'$D_{{{np.degrees(psi):.0f}}}$'
    return m

# %% retrieval — same two-stage recipe and weights for both

def run(mr, tag, diff=3.24, l1=1e3):    # diff = old 5e4 / Δlog r² on (3,15)x200
    mrinit = SphHarmSplineModel(
        rgrid, max_l=0, cpoints=mr.cpoints, spacing=mr.spacing, **d
    )
    initcoeffs = t.zeros(mr.coeffs_shape, **d)

    sens = sensitivity(f, mr, mask=f.rmask)
    w = sens.median() / sens
    W = t.tensor(np.stack(
        [np.interp(np.log(rgrid.r), np.log(mr.cpoint_locs), row) for row in w.cpu()]
    ), **d)

    initcoeffs.data[0:1, :], _, _ = gd(
        f, measn, mrinit, lr=1e1,
        loss_fns=[1 * HuberLoss(mask=f.rmask), 1e5 * NegRegularizer()],
        num_iterations=2000,
        callbacks=[LogCallback(f'{tag}-L0init', '/tmp/losses_reduced.txt')],
    )

    loss_fns = [
        1 * HuberLoss(mask=f.rmask),
        1e5 * NegRegularizer(),
        diff * DiffLoss(rgrid),
        l1 * SphHarmL1Regularizer(mrinit, weights=W),
    ]

    # gd updates `coeffs` in place, so keep an untouched copy for a00dev
    ref = initcoeffs.clone()
    coeffs, retrieved_meas, losses = gd(
        f, measn, mr, lr=1e0,
        loss_fns=loss_fns, num_iterations=3000,
        coeffs=initcoeffs,
        callbacks=[LogCallback(f'{tag}-full', '/tmp/losses_reduced.txt')],
    )
    retrieved = mr(coeffs).clamp(min=1e-4)
    return dict(mr=mr, coeffs=coeffs, retrieved=retrieved,
                retrieved_meas=retrieved_meas, losses=losses,
                initcoeffs=ref)

# %% metrics (definitions from AGENTS.md stage-2 sweep)

def metrics(res):
    rho = res['retrieved'].detach()                       # (r, e, a)
    valid = rho >= 25
    rel = (rho[1:] - rho[:-1]) / rho[:-1]
    bad = (rel > 0.02) & valid[:-1]
    mono = 1 - bad.any(0).double().mean().item()
    worst = rel[valid[:-1]].max().item() if valid[:-1].any() else float('nan')

    r = t.as_tensor(rgrid.r, device=rho.device)
    band = (r >= 3) & (r <= 10)
    a00 = res['mr'].sph_coeffs(res['coeffs'])[0].detach()
    a00_init = res['mr'].sph_coeffs(res['initcoeffs'])[0].detach()
    a00dev = ((a00[band] - a00_init[band]).abs() / a00_init[band].abs()).median().item()

    resid = ((res['retrieved_meas'] - measn).abs() / measn.abs())[mask.bool()]
    resid = resid[resid.isfinite()].median().item()

    # how much 3D structure actually survived: amplitude of the non-monopole
    # harmonics relative to A00.  If this is ~0 the basis choice cannot matter.
    sc = res['mr'].sph_coeffs(res['coeffs']).detach()
    aniso = (sc[1:].abs().max(0).values[band] / sc[0][band].abs()).median().item()
    return dict(mono=mono, worst=worst, a00dev=a00dev, resid=resid,
                aniso=aniso, valid=valid.double().mean().item())

runs = {
    'current, diff 5e4':        run(make_full(), 'full'),
    'reduced, diff 5e4':        run(make_reduced(), 'reduced'),
    # the production weights may suppress l=1 so hard that the basis is moot;
    # rerun both where AGENTS.md says the null space is actually visible
    'current, diff 5e2 l1 1e2': run(make_full(), 'full-light', diff=3.24e-2, l1=1e2),
    'reduced, diff 5e2 l1 1e2': run(make_reduced(), 'reduced-light', diff=3.24e-2, l1=1e2),
}
scores = {k: metrics(v) for k, v in runs.items()}

for k, v in scores.items():
    print(f'{k:28s} mono {v["mono"]:.2f}  worst {v["worst"]:+.1%}  '
          f'a00dev {v["a00dev"]:.1%}  resid {v["resid"]:.1%}  '
          f'aniso {v["aniso"]:.3f}  valid {v["valid"]:.2f}')

np.savez('/tmp/claude_reduced_coeffs.npz',
         **{k: v['coeffs'].detach().cpu().numpy() for k, v in runs.items()})

# %% page

labels = [str(x)[:16] for x in dates]

with document('3D reduced-basis comparison') as doc:
    tags.h1(f'Angle-limited basis test, {desc}, WFI only, {N} dates')
    tags.p(f'Resolvable dipole azimuth {np.degrees(psi):.1f} deg, '
           f'orthogonal phase suppressed to {ang[:, 1].mean():.2f} of it. '
           f'Reduced basis drops (A11, B11) for one dipole locked to that azimuth.')

    with tags.table(border=1, style='border-collapse:collapse'):
        with tags.tr():
            for h in ('config', 'mono', 'worst incr', 'a00dev', 'resid', 'aniso'):
                tags.th(h, style='padding:4px 10px')
        for k, v in scores.items():
            with tags.tr():
                tags.td(k, style='padding:4px 10px')
                tags.td(f'{v["mono"]:.2f}', style='padding:4px 10px')
                tags.td(f'{v["worst"]:+.1%}', style='padding:4px 10px')
                tags.td(f'{v["a00dev"]:.1%}', style='padding:4px 10px')
                tags.td(f'{v["resid"]:.1%}', style='padding:4px 10px')
                tags.td(f'{v["aniso"]:.3f}', style='padding:4px 10px')

    figset = {'width': 720}
    for k, res in runs.items():
        tags.h2(k)
        with itemgrid(length=2):
            with caption('Recon (cardinal slices)'):
                plot(cardplot(res['retrieved'], rgrid, norm='log'), **figset)
            with caption('Recon (radial profile)'):
                plot(cardplotaxes(res['retrieved'], rgrid, yscale='log'), **figset)
            with caption('Coefficients'):
                plot(sphharmplot(res['mr'].sph_coeffs(res['coeffs']), res['mr']), **figset)
            with caption('Radiance (TP alt) vs Density'):
                with slider(labels=labels):
                    for i in range(N):
                        plot(radiance_v_density(
                            res['retrieved'], rgrid,
                            meas[i], [leaf[i] for leaf in rvg.leaves]
                        ), **figset)

    tags.h1('Source Code')
    tags.code(tags.pre(open('recon_3D_reduced.py').read()))

outfile = Path(f'/www/storm/3D_{desc}_reduced.html')
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text(doc.render())
print(f'Saved to {outfile}')
