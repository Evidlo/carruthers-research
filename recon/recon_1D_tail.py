#!/usr/bin/env python3

# Before/after for the far-field tail fix.  Two 1D retrievals differing only in
# the forward model's radial extent and how the model continues past the last
# control point.  Data, dates, geometry, losses and fitted pixel set are shared.
#
# Each retrieval runs in its own process (`python recon_1D_tail.py <cfg>`) so the
# first config's operator can't crowd the second off the GPU; with no argument
# the script runs both and builds the comparison page.

import sys
from pathlib import Path
import numpy as np

CACHE = Path('/tmp/claude_tail')
TP_MAX = 22          # fitted pixels, pinned for all configs
LR = 1e1             # 5e1 makes tail='power' go noisy, not slow; gd returns one
                     # best-loss snapshot for all dates, so a diverged ensemble can
                     # still show a clean date 0.  See WORK.md 2026-07-27
NDATES = 50

CFGS = {
    'before': dict(label='before  (3,25)x200 c8',
                   size_r=(3, 25), cpoints=8),
    # 8 coarse shells complete the LOS integral out to 100 Re.  Junction sits at
    # TP_MAX, not at clim[1]: knots stop at 21 Re, but the mask still admits LOS
    # tangenting to 22, and discretization error concentrates at tangency
    'hybrid': dict(label='hybrid  (3,22)x200 +8 to 100 clim=(3,21) tail=power',
                   size_r=(3, 22), bins=200, outer_r=100, outer_bins=8,
                   cpoints=8, clim=(3, 21), tail='power'),
}


def retrieve(key):
    from glide.science.forward_sph import ForwardSph, geom2mask, ScienceGeomFast, Albedo
    from glide.science.model_sph import DefaultGrid, SphHarmSplineModel
    from glide.science.common import wipe_gpu
    from glide.calibration.column_density import solar_flux_to_g_factor as g
    from tomosphero import ZippedGeom
    from tomosphero.retrieval import gd, LogCallback
    from tomosphero.loss import HuberLoss, NegRegularizer
    import xarray as xr, torch as t

    wipe_gpu()
    d = {'device': 'cuda'}
    cfg = CFGS[key]

    dates = np.linspace(
        np.datetime64('2026-03-01').astype('datetime64[ns]').astype(float),
        np.datetime64('2026-04-01').astype('datetime64[ns]').astype(float),
        NDATES).astype('datetime64[ns]')

    from load import load
    nfi, wfi, dates = load(Path('/data-products'), dates)
    N = len(dates)
    ims = list(zip(nfi.images.values, wfi.images.values))

    rvg = ZippedGeom(
        sum([ScienceGeomFast(s, (100, 50), **d) for s in nfi.scraft.values]),
        sum([ScienceGeomFast(s, (100, 50), **d) for s in wfi.scraft.values]),
    )
    # rmask would otherwise follow rgrid.size.r and let the 100 Re grid ingest the
    # 24-40 Re pixels where the IPH floor dominates, changing two things at once
    mask = rvg.mask.to(d['device']) & geom2mask(rvg, 3, TP_MAX, r_shadow=3, device=d['device'])

    # bins + outer_bins is 200 for every config, so cost is held fixed
    rgrid = DefaultGrid((N, cfg.get('bins', 200), 45, 60), size_r=cfg['size_r'],
                        spacing='log', outer_r=cfg.get('outer_r'),
                        outer_bins=cfg.get('outer_bins', 0), t=dates, timeunit='ns')
    alb = xr.open_mfdataset(
        '/home/jackson/glide-sdc/glide/validation/radiative_transfer/pipeline_test/albedo_data_*.nc')
    f = ForwardSph(rgrid=rgrid, rvg=rvg, g_factor=g(11e11),
                   ralbedo=Albedo(alb, rgrid, **d)(), **d)
    f.op.regs = None
    t.cuda.empty_cache()

    # calibrate() applies f.rmask, which differs per grid; re-mask to the shared set
    meas = f.calibrate(ims, disable_noise=True) * mask

    mr = SphHarmSplineModel(rgrid, max_l=0, cpoints=cfg['cpoints'], spacing='log',
                            clim=cfg.get('clim'), tail=cfg.get('tail'), **d)

    logfile = f'/tmp/losses_tail_{key}.txt'
    open(logfile, 'w').close()
    coeffs, _, losses = gd(
        f, t.nan_to_num(meas), mr, lr=LR,
        loss_fns=[1 * HuberLoss(mask=mask), 1e5 * NegRegularizer()],
        num_iterations=1000, coeffs=t.zeros((N, *mr.coeffs_shape), **d),
        callbacks=[LogCallback(cfg['label'], logfile)],
    )

    with t.no_grad():
        coeffs = coeffs.detach()
        dens = mr(coeffs)                       # (N, r, e, a)
        m0 = t.nan_to_num(meas)
        resid = float(t.linalg.norm((f(dens) - m0) * mask)
                      / t.linalg.norm(m0 * mask) * 100)
        prof = dens[:, :, rgrid.shape.e // 2, rgrid.shape.a // 2].cpu().numpy()

    CACHE.mkdir(exist_ok=True)
    np.savez(CACHE / f'{key}.npz', prof=prof, r=rgrid.static.r.cpu().numpy(),
             coeffs=coeffs[:, 0].cpu().numpy(), cpoint_locs=mr.cpoint_locs,
             resid=resid, dates=dates.astype(str))
    print(f'{cfg["label"]}: resid {resid:.2f}%')


def compare():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from domrep import document, tags, itemgrid, caption, plot

    o = {k: dict(np.load(CACHE / f'{k}.npz')) for k in CFGS}
    R = np.geomspace(3.2, 25, 60)
    for k, v in o.items():
        v['P'] = np.stack([np.interp(R, v['r'], row) for row in v['prof']])
        v['label'] = CFGS[k]['label']
    N = len(o['before']['prof'])
    i20 = np.argmin(abs(R - 20))
    sc = lambda v: v['P'].std(0) / np.abs(v['P'].mean(0)) * 100

    def fig_overlay():
        fig, axes = plt.subplots(1, len(o), figsize=(6 * len(o), 4.5), dpi=150, sharey=True)
        for ax, v in zip(axes, o.values()):
            for row in v['P']:
                ax.loglog(R, row, color='C0', alpha=.25, lw=.8)
            ax.loglog(R, v['P'].mean(0), 'k-', lw=1.6)
            ax.set(xlabel='r (Re)', title=f'{v["label"]}\nresid {v["resid"]:.1f}%')
            ax.grid(alpha=.25, which='both')
        axes[0].set_ylabel('H density (atoms/cm³)')
        fig.suptitle(f'All {N} dates overlaid — spread is the temporal stability')
        fig.tight_layout(); return fig

    def fig_scatter():
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        for v in o.values():
            ax.semilogx(R, sc(v), '-', lw=1.8, label=v['label'])
        ax.set(xlabel='r (Re)', ylabel='date-to-date scatter (1σ, %)', ylim=(0, None),
               title='Temporal stability vs radius (lower is better)')
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.tight_layout(); return fig

    def fig_knots():
        fig, axes = plt.subplots(1, len(o), figsize=(6 * len(o), 4.5), dpi=150, sharey=True)
        for ax, v in zip(axes, o.values()):
            for j, kk in enumerate(v['cpoint_locs']):
                ax.semilogy(range(N), np.abs(v['coeffs'][:, j]) + 1e-12, '-', lw=1.2,
                            label=f'{kk:.1f} Re')
            ax.set(xlabel='date index', title=v['label'])
            ax.grid(alpha=.25, which='both')
        axes[0].set_ylabel('|A₀₀| control point value')
        axes[-1].legend(fontsize=6, ncol=2)
        fig.suptitle('Control points vs date — flat lines mean a stable retrieval')
        fig.tight_layout(); return fig

    def fig_mean():
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        for v in o.values():
            ax.loglog(R, v['P'].mean(0), '-', lw=1.8, label=v['label'])
        ax.set(xlabel='r (Re)', ylabel='H density (atoms/cm³)',
               title='Date-averaged profile')
        ax.grid(alpha=.25, which='both'); ax.legend(fontsize=8)
        fig.tight_layout(); return fig

    with document('1D far-field tail: before / after') as doc:
        tags.h1('Far-field tail fix — 1D retrieval')
        tags.p(f'{N} dates, March 2026.  Fitted pixels TP ≤ {TP_MAX} Re in both runs.  '
               + ' | '.join(f'{v["label"]}: resid {v["resid"]:.1f}%, '
                            f'scatter@20Re {sc(v)[i20]:.1f}%' for v in o.values()))
        with itemgrid(length=1):
            for cap, fn in [('All dates overlaid', fig_overlay),
                            ('Date-to-date scatter vs radius', fig_scatter),
                            ('Date-averaged profile', fig_mean),
                            ('Control points vs date', fig_knots)]:
                with caption(cap):
                    plot(fn(), width=900)
        tags.h1('Source Code')
        tags.code(tags.pre(open('recon_1D_tail.py').read()))

    outfile = Path('/www/storm/1D_tail_compare.html')
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(doc.render())
    print(f'Saved to {outfile}')

    for v in o.values():
        print(f'{v["label"]:<44} resid {float(v["resid"]):>6.2f}%  '
              f'scatter@20Re {sc(v)[i20]:>6.2f}%  '
              f'knots {np.array2string(v["coeffs"][0], precision=1)}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        retrieve(sys.argv[1])
    else:
        import subprocess
        for key in CFGS:
            subprocess.run([sys.executable, __file__, key], check=True)
        compare()
