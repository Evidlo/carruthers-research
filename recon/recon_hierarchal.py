#!/usr/bin/env python3

from glide.common_components.camera import CameraWFI, CameraNFI, CameraL1BWFI, CameraL1BNFI
from glide.common_components.cam import nadir_wfi_mode, nadir_nfi_mode
from glide.common_components.generate_view_geom import gen_mission
from glide.common_components.orbits import circular_orbit
from glide.science.forward_sph import *
from glide.science.model_sph import *
from glide.science.plotting import *
from glide.science.plotting_sph import carderr, cardplot, carderrmin, cardplotaxes
from glide.science.recon.loss_sph import *

from domrep import *

from pathlib import Path

from tomosphero.plotting import *
from tomosphero.retrieval import gd, LogCallback
from tomosphero.loss import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch as t
import inspect
from itertools import product


__file__ = 'recon_hierarchal.py'
code = open(__file__).read()

device = 'cuda'

# %% setup
# ----- Truth/Recon Models -----

sgrid = DefaultGrid((500, 45, 60), size_r=(3, 25), spacing='log')
rgrid = DefaultGrid((200, 45, 60), size_r=(3, 25), spacing='log')

truth_models = [
    Zoennchen24Model(grid=sgrid, device=device),
    # Pratik25Model(grid=sgrid, num_times=1, device=device),
    # TIMEGCMModel(grid=sgrid, device=device, offset=10, fill_value='nearest')
]

# hierarchical stages: progressively increase max_l, freezing previously-optimized coeffs
L_stages = [0, 1, 2, 3]
max_L = L_stages[-1]
cpoints = 12

# final model uses the highest L; earlier stages reuse this same model but with
# higher-l rows held at zero
recon_models = [
    SphHarmSplineModel(rgrid, max_l=max_L, device=device, cpoints=cpoints, spacing='log')
]

# ----- Measurement Generation -----

t_op = 360
num_obs=14; duration=14
cams = [CameraL1BNFI(nadir_nfi_mode(t_op=t_op)), CameraL1BWFI(nadir_wfi_mode(t_op=t_op))]
sc = gen_mission(num_obs=num_obs, duration=duration, start='2025-12-24', cams=cams)

f = ForwardSph(
    sc, sgrid=sgrid,
    rgrid=rgrid,
    rvg=sum([ScienceGeomFast(s, (100, 50)) for s in sc]),
    device=device
)


def num_harmonics(L):
    """Number of spherical harmonics for max order L (inclusive)."""
    return (L + 1) ** 2


def hierarchical_gd(f, y, model, L_stages, loss_fns, lr, num_iterations):
    """Coarse-to-fine gradient descent over increasing max_l.

    At each stage, optimize only the harmonic rows newly introduced by that
    stage's L. Rows from previous stages are frozen (gradient zeroed via
    callback); rows for future stages stay at zero.

    Returns:
        coeffs: final optimized coefficients of shape model.coeffs_shape
        y_result: forward-simulated measurement from final coeffs
        losses: dict[loss_fn -> list[float]] concatenated across all stages
    """
    # Clear losses log on new run
    open('/tmp/losses.txt', 'w').close()

    # Initialize to ones (matches gd's default) to avoid NaN in SphHarmL1Regularizer
    # which normalizes by the L=0 coefficient
    coeffs = t.ones(model.coeffs_shape, device=device, dtype=t.float64, requires_grad=True)
    losses = {loss_fn: [] for loss_fn in loss_fns}

    for stage_idx, L in enumerate(L_stages):
        curr_n = num_harmonics(L)
        prev_n = num_harmonics(L_stages[stage_idx - 1]) if stage_idx > 0 else 0
        print(f'  stage {stage_idx}: L={L}  training rows [{prev_n}:{curr_n}]')

        def freeze_cb(loc, prev_n=prev_n, curr_n=curr_n):
            # Zero gradients for frozen and future rows only
            g = loc['coeffs'].grad
            if g is not None:
                g[:prev_n] = 0
                g[curr_n:] = 0

        stage_coeffs, _, stage_losses = gd(
            f, y, model,
            coeffs=coeffs,
            loss_fns=loss_fns,
            num_iterations=num_iterations,
            lr=lr,
            callbacks=[freeze_cb, LogCallback(str(stage_idx))],
        )
        # carry optimized values forward to next stage
        # Keep frozen rows at their optimized values (they just don't get updated)
        coeffs = stage_coeffs.detach().clone().requires_grad_(True)

        # append this stage's loss history to the cumulative record
        for loss_fn, vals in stage_losses.items():
            losses[loss_fn].extend(vals)

    y_result = f(model(coeffs))
    return coeffs.detach(), y_result, losses


# %% recon

with document('Two Week Hierarchical Retrievals') as doc:

    for nt, mt in enumerate(truth_models):
        print('=============================================================')
        print(mt)
        print('=============================================================')

        truth = mt()
        meas = f.calibrate(f.simulate(truth))

        tags.h1(f'truth={mt}')
        with itemgrid(len(recon_models), flow='row'):

            for nr, mr in enumerate(recon_models):
                cshape = 'x'.join(map(str, mr.coeffs_shape))
                desc = f'hier_spline{cshape}L{mr.max_l}_{num_obs:02d}obs'
                print('---', desc, f'truth:{nt}/{len(truth_models)}  recon:{nr}/{len(recon_models)}', '---')

                t.cuda.empty_cache()

                loss_fns = [
                    1 * AbsLoss(projection_mask=f.proj_maskb),
                    1e4 * NegRegularizer(),
                    1e1 * SphHarmL1Regularizer(mr),
                    5e2 * DiffLoss(rgrid),
                    ReqErr(truth, mt.grid, mr.grid, interval=100),
                ]

                coeffs, retrieved_meas, losses = hierarchical_gd(
                    f, meas, mr,
                    L_stages=L_stages,
                    loss_fns=loss_fns,
                    lr=5e1,
                    num_iterations=8000,
                )

                retrieved = mr(coeffs)

                t.save(coeffs, f'/tmp/coeffs_{desc}.tr')

                figset = {'height': 200}

                if issubclass(type(mr), SphHarmModel):
                    sphharm = plot(sphharmplot(mr.sph_coeffs(coeffs), mr), **figset)
                else:
                    sphharm = ''

                caption(
                    f"recon={mr} (hierarchical L={L_stages})",
                    plot(carderr(retrieved.squeeze(), truth.squeeze(), rgrid, sgrid), **figset),
                    sphharm,
                    tags.br(),
                    tags.details(
                        tags.summary(),
                        plot(loss_plot(losses), **figset),
                        caption("Recon", plot(cardplot(retrieved.squeeze(), rgrid, norm='log'), **figset)),
                        caption("Truth", plot(cardplot(truth.squeeze(), sgrid, norm='log'), **figset)),
                        caption("Recon", plot(cardplotaxes(retrieved.squeeze(), rgrid, yscale='log'), **figset)),
                        caption("Truth", plot(cardplotaxes(truth.squeeze(), sgrid, yscale='log'), **figset)),
                    )
                )

    tags.h1("Source Code")
    tags.code(tags.pre(open('recon_hierarchal.py').read()))

# %% plot
vgshape = 'x'.join(map(str, f.rvg[0].shape))
outfile = Path(f'/www/sph/two_week_hierarchal_{vgshape}{f.rvg[0].spacing}.html')
outfile.write_text(doc.render())
print(f'Saved to {outfile}')

from datetime import datetime
outfile = Path(f'/www/sph/archive/{datetime.now().isoformat()}.html')
outfile.write_text(doc.render())
