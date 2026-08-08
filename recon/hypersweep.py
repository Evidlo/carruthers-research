#!/usr/bin/env python3
"""Regularizer hyperparameter sweep, GP-surrogate active learning.

Two regularizer weights are swept jointly: lam_diff (DiffLoss, radial
smoothness) and lam_l1 (SphHarmL1Regularizer, anisotropy sparsity).  Each
(lam_diff, lam_l1) design point is evaluated for every (truth, fidelity loss)
task, so all tasks share sample locations and every slider position is equally
well-sampled for every line on the plot.

Sample placement is uncertainty-driven, not optimum-driven: the acquisition is
integrated posterior variance reduction summed over tasks, so points go where
the surface is least known rather than where it is lowest.  BO's usual
expected-improvement acquisition would cluster at the minimum and leave the rest
of the surface as unsupported GP extrapolation.

MOCK=True replaces the retrieval with an analytic surface so the report layout
can be reviewed before committing to the real run.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

MOCK = True

# %% sweep configuration

# hyperparameter 1: slider axis.  hyperparameter 2: plot x axis
LAM_DIFF = (5e-3, 5e1)        # DiffLoss weight, production 5.6
                              # (DiffLoss divides by Δlog r; on the (3,25)x200 recon
                              #  grid these are the old 5e1..5e5 / 8.9e3)
LAM_L1 = (1e0, 1e4)           # SphHarmL1Regularizer weight, production 1e3

TRUTHS = ['Zoennchen24', 'Pratik25Storm']
LOSSES = ['SquareLoss', 'HuberLoss']
TASKS = [(t, l) for l in LOSSES for t in TRUTHS]

N_INIT = 12                   # space-filling seed design
N_ACTIVE = 28                 # active-learning additions
SLIDER_STEPS = 50             # lam_diff positions rendered
BAND = (3, 10)                # Re, radial band the error metric is taken over

ARC = ('2026-03-01', '2026-03-15')   # quiet march, WFI only
CLIM = (3, 21)                # knot span; data constrains a knot to ~21 Re
LR = 1e1                      # tail='power' diverges at recon_1D's inherited 5e1
LOGFILE = '/tmp/losses_hypersweep.txt'
CHECKPOINT = '/tmp/claude_hypersweep.npz'   # written each active sample

COLORS = {'Zoennchen24': '#2a78d6', 'Pratik25Storm': '#eb6834'}

# design space is log in both hyperparameters
BOUNDS = np.log10([LAM_DIFF, LAM_L1])


# %% objective

def evaluate_mock(lam_diff, lam_l1):
    """Stand-in error surface, one value per task.  Shapes match `evaluate`."""
    ld, l1 = np.log10(lam_diff), np.log10(lam_l1)
    out = {}
    for truth, loss in TASKS:
        # both truths want a moderate lam_diff; too little is ripple, too much
        # oversmooths the r^-2.75 falloff
        err = 6.0 + 3.0 * (ld - 4.2) ** 2
        if truth == 'Zoennchen24':
            # near-spherical: killing the 3D basis strictly helps
            err += 9.0 / (1 + np.exp(1.4 * (l1 - 1.5)))
        else:
            # storm: real anisotropy, so heavy L1 destroys signal
            err += 7.0 / (1 + np.exp(2.0 * (l1 - 0.8))) + 2.6 * np.clip(l1 - 2.4, 0, None) ** 2
        err *= 0.93 if loss == 'HuberLoss' else 1.0
        out[(truth, loss)] = err + np.random.default_rng(
            abs(hash((truth, loss, round(ld, 6), round(l1, 6)))) % 2**32
        ).normal(0, 0.35)
    return out


def setup():
    """Geometry, truths and noisy measurements.  Built once, reused per sample.

    Returns:
        dict: f, mask, rgrid, meas {truth: tensor}, ref {truth: (r,e,a) density},
              init {(truth, loss): coeffs}
    """
    from glide.science.forward_sph import ForwardSph, Albedo, ScienceGeomFast, geom2mask
    from glide.science.model_sph import (DefaultGrid, SphHarmSplineModel,
                                         Zoennchen24Model, Pratik25StormModel)
    from tomosphero.loss import NegRegularizer
    from glide.calibration.column_density import solar_flux_to_g_factor as gf
    from glide.science.common import wipe_gpu
    from tomosphero import ZippedGeom
    from tomosphero.retrieval import gd, LogCallback
    import torch as t, xarray as xr
    from importlib import resources
    from load import load

    wipe_gpu()
    d = {'device': 'cuda'}

    dates = np.linspace(
        np.datetime64(ARC[0]).astype('datetime64[ns]').astype(float),
        np.datetime64(ARC[1]).astype('datetime64[ns]').astype(float), 14,
    ).astype('datetime64[ns]')
    nfi, wfi, dates = load(Path('/data-products'), dates)
    print(f'{len(dates)} frames')

    alb = xr.open_dataset(resources.files('glide') / 'science/data_files/albedo_GLIDE_CDR.nc')
    alb = alb.reindex(r=np.append(alb.r, 1e6), method='ffill')
    scrafts = list(wfi.scraft.values)

    def forward(grid, sc=None):
        f = ForwardSph(sc=sc, rgrid=grid, g_factor=gf(11e11),
                       rvg=ZippedGeom(sum([ScienceGeomFast(s, (100, 50), **d) for s in scrafts])),
                       ralbedo=Albedo(alb, grid, **d)(), **d)
        f.op.regs = None
        t.cuda.empty_cache()
        return f

    # truth lives on a grid far outside the retrieval grid, so the retrieval sees
    # production's truncation regime rather than a truth that stops where it does
    big = DefaultGrid((300, 45, 60), size_r=(3, 100), spacing='log')
    make = {'Zoennchen24': lambda g: Zoennchen24Model(g, **d)(),
            'Pratik25Storm': lambda g: Pratik25StormModel(g, tail='power', **d)()}

    rgrid = DefaultGrid((200, 45, 60), size_r=(3, 25), spacing='log')
    f = forward(rgrid)
    # decouple the LOS mask from the grid extent -- rmask would also admit the
    # IPH-dominated 24-40 Re pixels (AGENTS.md 2026-07-26)
    mask = f.op.mask.to(d['device']) & geom2mask(f.rvg, 3, 22, r_shadow=3, **d)

    meas, ref = {}, {}
    for name, fn in make.items():
        fbig = forward(big, sc=scrafts)
        meas[name] = t.nan_to_num(fbig.fake_simulate(fn(big)).detach())
        del fbig
        t.cuda.empty_cache()
        ref[name] = fn(rgrid).detach()

    # Huber's delta is in calibrated units, so pin it to the data's own scale --
    # left at 1.0 it can sit far above every residual and silently degenerate
    # into SquareLoss/2, making the two panels identical by construction
    delta = t.stack([v[mask] for v in meas.values()]).abs().median().item() * 0.1
    print(f'huber delta {delta:.4g} (0.1 x median |meas|)')

    # the l=0 warm start does not depend on either regularizer, so it is computed
    # once per (truth, loss) instead of once per design point
    init = {}
    for (truth, loss) in TASKS:
        m0 = SphHarmSplineModel(rgrid, max_l=0, cpoints=8, spacing='log',
                                clim=CLIM, tail='power', **d)
        c0, _, _ = gd(f, meas[truth], m0, lr=LR,
                      loss_fns=[1 * fidelity(loss, mask, delta), 1e5 * NegRegularizer()],
                      num_iterations=2000,
                      callbacks=[LogCallback(f'{truth}-{loss}-L0init', LOGFILE)])
        init[(truth, loss)] = c0.detach()

    return dict(f=f, mask=mask, rgrid=rgrid, meas=meas, ref=ref, init=init,
                delta=delta, d=d)


def fidelity(loss, mask, delta):
    from tomosphero.loss import SquareLoss, HuberLoss
    return {'SquareLoss': lambda: SquareLoss(mask=mask),
            'HuberLoss': lambda: HuberLoss(mask=mask, delta=delta)}[loss]()


def evaluate_real(lam_diff, lam_l1, S=None):
    """One design point, evaluated for every task.  Diverged runs return nan."""
    from glide.science.model_sph import SphHarmSplineModel
    from glide.science.recon.loss_sph import DiffLoss, SphHarmL1Regularizer, sensitivity
    from tomosphero.loss import NegRegularizer
    from tomosphero.retrieval import gd, LogCallback
    import torch as t

    f, mask, rgrid, d = S['f'], S['mask'], S['rgrid'], S['d']
    r = t.as_tensor(np.asarray(rgrid.r), device=d['device'])
    band = (r >= BAND[0]) & (r <= BAND[1])

    out = {}
    for truth, loss in TASKS:
        mr = SphHarmSplineModel(rgrid, max_l=1, cpoints=8, spacing='log',
                                clim=CLIM, tail='power', **d)
        m0 = SphHarmSplineModel(rgrid, max_l=0, cpoints=8, spacing='log',
                                clim=CLIM, tail='power', **d)

        sens = sensitivity(f, mr, mask=mask)
        w = sens.median() / sens
        W = t.tensor(np.stack([np.interp(np.log(np.asarray(rgrid.r)),
                                         np.log(mr.cpoint_locs), row) for row in w.cpu()]), **d)

        coeffs = t.zeros(mr.coeffs_shape, **d)
        coeffs.data[0:1, :] = S['init'][(truth, loss)]

        tag = f'{truth}-{loss}-{lam_diff:.3g}-{lam_l1:.3g}'
        coeffs, rmeas, losses = gd(
            f, S['meas'][truth], mr, lr=LR, num_iterations=3000, coeffs=coeffs,
            loss_fns=[1 * fidelity(loss, mask, S['delta']), 1e5 * NegRegularizer(),
                      lam_diff * DiffLoss(rgrid), lam_l1 * SphHarmL1Regularizer(m0, weights=W)],
            callbacks=[LogCallback(tag, LOGFILE)])

        rec = mr(coeffs).detach()
        err = ((rec[band] / S['ref'][truth][band] - 1).abs().median() * 100).item()

        # tail='power' is known to overshoot at too large an lr, and it shows up
        # as a noisy loss rather than a slow one -- drop those rather than let
        # the GP fit through them
        diverged = not (np.isfinite(losses[-1]) and rec.isfinite().all() and err < 1e3)
        out[(truth, loss)] = float('nan') if diverged else err
        print(f'  {tag}: err {err:6.2f}%' + ('  DIVERGED' if diverged else ''))
        t.cuda.empty_cache()

    return out


# %% GP surrogate + uncertainty-driven design

def fit(X, y):
    """GP over log-hyperparameters.  X (n,2), y (n,).  Diverged runs are nan and
    are dropped rather than fitted through"""
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    kern = (ConstantKernel(np.var(y) + 1e-6, (1e-3, 1e4))
            * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(0.2, 20), nu=2.5)
            + WhiteKernel(0.3, (1e-3, 1e2)))
    return GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=4).fit(X, y)


def post_cov(gp, A, B):
    """Posterior cross-covariance k(A,B) - k(A,Xtr) Ktr^-1 k(Xtr,B)"""
    k = gp.kernel_
    v = np.linalg.solve(k(gp.X_train_) + 1e-10 * np.eye(len(gp.X_train_)), k(gp.X_train_, B))
    return k(A, B) - k(A, gp.X_train_) @ v


def next_point(gps, cand, ref):
    """Candidate maximizing summed integrated-variance reduction over tasks.

    Placing a sample at x* cuts the variance at every reference point x by
    k_post(x,x*)^2 / (k_post(x*,x*) + sigma_n^2), which needs no new observation
    to compute -- so the whole design is chosen from geometry alone.
    """
    gain = np.zeros(len(cand))
    for gp in gps.values():
        kxs = post_cov(gp, ref, cand)                        # (n_ref, n_cand)
        kss = np.clip(np.diag(post_cov(gp, cand, cand)), 0, None)
        noise = gp.kernel_.k2.noise_level
        gain += (kxs ** 2).sum(0) / (kss + noise)
    return cand[gain.argmax()]


def sobol(n, seed=0):
    from scipy.stats import qmc
    u = qmc.Sobol(2, scramble=True, seed=seed).random(n)
    return BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])


def sweep():
    """Returns X (n,2) log-design, Y {task: (n,)}, gps {task: GP}"""
    S = None if MOCK else setup()
    run = (lambda x: evaluate_mock(*10.0 ** x)) if MOCK else (lambda x: evaluate_real(*10.0 ** x, S=S))

    X = sobol(N_INIT)
    Y = {k: [] for k in TASKS}
    for i, x in enumerate(X):
        print(f'seed sample {i+1}/{N_INIT}: '
              f'lam_diff {10**x[0]:.3g}, lam_l1 {10**x[1]:.3g}')
        for k, v in run(x).items():
            Y[k].append(v)
    Y = {k: np.array(v) for k, v in Y.items()}

    cand = sobol(512, seed=1)
    g = np.meshgrid(*[np.linspace(*b, 40) for b in BOUNDS], indexing='ij')
    ref = np.stack([a.ravel() for a in g], -1)

    for i in range(N_ACTIVE):
        gps = {k: fit(X, Y[k]) for k in TASKS}
        x = next_point(gps, cand, ref)
        X = np.vstack([X, x])
        print(f'active sample {i+1}/{N_ACTIVE}: '
              f'lam_diff {10**x[0]:.3g}, lam_l1 {10**x[1]:.3g}')
        for k, v in run(x).items():
            Y[k] = np.append(Y[k], v)
        np.savez(CHECKPOINT, X=X, **{f'{a}|{b}': v for (a, b), v in Y.items()})

    return X, Y, {k: fit(X, Y[k]) for k in TASKS}


# %% report

def panel(gps, X, Y, ld, ax_by_loss, xs):
    """One slider frame: both fidelity panels at a fixed lam_diff"""
    grid = np.stack([np.full_like(xs, ld), xs], -1)
    for loss in LOSSES:
        ax = ax_by_loss[loss]
        for truth in TRUTHS:
            mu, sd = gps[(truth, loss)].predict(grid, return_std=True)
            c = COLORS[truth]
            ax.fill_between(10 ** xs, mu - 2 * sd, mu + 2 * sd, color=c, alpha=.15, lw=0)
            ax.plot(10 ** xs, mu, color=c, lw=2, label=truth, zorder=3)

            # measured points, faded by distance from this slider slice.  The
            # fade width is a fixed fraction of the lam_diff range so it does
            # not shrink to nothing as SLIDER_STEPS grows
            w = np.exp(-((X[:, 0] - ld) / (np.ptp(BOUNDS[0]) / 6)) ** 2)
            near = w > .15
            ax.scatter(10 ** X[near, 1], Y[(truth, loss)][near], s=26, color=c,
                       alpha=w[near], edgecolor='white', linewidth=.8, zorder=4)

        ax.set(xscale='log', xlim=10.0 ** BOUNDS[1], ylim=(0, 50),
               xlabel='λ  SphHarmL1Regularizer', title=loss)
        ax.grid(alpha=.25, lw=.6)
        ax.set_axisbelow(True)


def figure(gps, X, Y, ld):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130, sharey=True)
    xs = np.linspace(*BOUNDS[1], 200)
    panel(gps, X, Y, ld, dict(zip(LOSSES, axes)), xs)
    axes[0].set_ylabel(f'|error| vs truth, {BAND[0]}–{BAND[1]} Re (%)')
    handles = [Line2D([], [], color=COLORS[t], lw=2, label=t) for t in TRUTHS]
    handles.append(Line2D([], [], color='#8a8880', marker='o', ls='none',
                          markersize=5, label='evaluated'))
    axes[1].legend(handles=handles, frameon=False, fontsize=8, loc='upper right')
    fig.suptitle(f'λ  DiffLoss = {10**ld:.2e}', fontsize=11)
    fig.tight_layout()
    return fig


def report(X, Y, gps, outfile):
    from domrep import document, tags, plot, caption, slider

    lds = np.linspace(*BOUNDS[0], SLIDER_STEPS)
    with document('Regularizer hyperparameter sweep') as doc:
        tags.h1('Regularizer hyperparameter sweep')
        tags.p(f'{len(X)} design points ({N_INIT} Sobol + {N_ACTIVE} '
               f'active-learning), each evaluated for all {len(TASKS)} '
               f'(truth × fidelity) tasks. Lines are GP posterior means, bands '
               f'±2σ — where a band is wide the surface is interpolated, not '
               f'measured. Dots are actual retrievals, faded by distance from '
               f'the slider’s λ_diff slice.')
        if MOCK:
            tags.p(tags.b('MOCK DATA — layout review only, no retrievals were run.'),
                   style='color:#c0392b')

        with caption('Error vs truth, sliding λ DiffLoss'):
            slider(*[plot(figure(gps, X, Y, ld), width=1100) for ld in lds],
                   labels=[f'λ_diff {10**ld:.2e}' for ld in lds])

        tags.h1('Source Code')
        tags.code(tags.pre(open('hypersweep.py').read()))

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(doc.render())
    print(f'Saved to {outfile}')


if __name__ == '__main__':
    X, Y, gps = sweep()
    report(X, Y, gps,
           '/www/tmp/hypersweep_mock.html' if MOCK else '/www/storm/hypersweep.html')
