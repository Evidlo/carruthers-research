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


__file__ = 'recon.py'
code = open(__file__).read()

d = {'device': 'cuda'}

# %% setup
# ----- Truth/Recon Models -----

sgrid = DefaultGrid((500, 45, 60), size_r=(3, 25), spacing='log')
rgrid = DefaultGrid((200, 45, 60), size_r=(3, 25), spacing='log')

truth_models = [
    Zoennchen24Model(grid=sgrid, **d),
    # Pratik25Model(grid=sgrid, num_times=1, **d),
    # TIMEGCMModel(grid=sgrid, **d, offset=10, fill_value='nearest')
]

L_opts = [3] # sph harm spline order
c_opts = [12] # sph harm spline control points
recon_models = [
    SphHarmSplineModel(rgrid, max_l=L, **d, cpoints=cpoints, spacing='log')
    for L, cpoints in product(L_opts, c_opts)
]

# stage-2 loss variants: current regularizer set, fidelity-only, and
# sensitivity-weighted L1 (Wiener-like, see recon_3D.py)
loss_opts = ['diff+l1', 'bare', 'diff+l1W']

# ----- Measurement Generation -----

t_op = 360
num_obs=14; duration=14
cams = [CameraL1BNFI(nadir_nfi_mode(t_op=t_op)), CameraL1BWFI(nadir_wfi_mode(t_op=t_op))]
sc = gen_mission(num_obs=num_obs, duration=duration, start='2025-12-24', cams=cams)

def safe_mask(npix):
    """Generate mask of sagged rows for image of size npix"""
    mask = np.full((npix,  npix), True)
    mask[npix//3:2*npix//3] = False
    return mask

# input_mask = (safe_mask(s.sensor.npix) for s in sc)

f = ForwardSph(
    sc, sgrid=sgrid, # calibrator=cal
    rgrid=rgrid,
    # rvg=sum([ScienceGeom(s, (100, 50)) for s in sc]),
    rvg=sum([ScienceGeomFast(s, (100, 50), **d) for s in sc]),
    # input_mask=input_mask,
    **d
)

# %% recon

with document('Two Week Retrievals') as doc:

    # iterate ground truths
    for nt, mt in enumerate(truth_models):
        print('=============================================================')
        print(mt)
        print('=============================================================')

        truth = mt()
        meas = f.calibrate(f.simulate(truth))

        truth_figs = []

        tags.h1(f'truth={mt}')
        with itemgrid(len(loss_opts), flow='row'):

            for nr, mr in enumerate(recon_models):
                t.cuda.empty_cache()

                # reconstruction model for fast initialization of A00
                mrinit = SphHarmSplineModel(
                    rgrid, max_l=0,
                    cpoints=mr.cpoints, spacing=mr.spacing,
                    **d
                )

                # sensitivity weighting (Wiener-like): penalize coefficients
                # ∝ 1/‖F·basis‖ (see recon_3D.py)
                sens = sensitivity(f, mr, mask=f.rmask)
                w = sens.median() / sens
                W = t.tensor(np.stack(
                    [np.interp(np.log(rgrid.r), np.log(mr.cpoint_locs), row) for row in w.cpu()]
                ), **d)

                initcoeffs = t.zeros(mr.coeffs_shape, **d)

                initcoeffs.data[0:1, :], _, _ = gd(
                    f, meas, mrinit, lr=1e1,
                    loss_fns=[1 * HuberLoss(mask=f.rmask), 1e5 * NegRegularizer(), 5e2 * DiffLoss(rgrid)],
                    num_iterations=2000,
                    callbacks=[LogCallback('L0init', '/tmp/losses_baseline.txt')],
                )

                for lname in loss_opts:
                    cshape = 'x'.join(map(str, mr.coeffs_shape))
                    desc = f'spline{cshape}L{mr.max_l}_{num_obs:02d}obs_{lname}'
                    print('---', desc, f'truth:{nt}/{len(truth_models)}  recon:{nr}/{len(recon_models)}', '---')

                    # Clear losses log on new run
                    open('/tmp/losses_baseline.txt', 'w').close()

                    loss_fns = {
                        'diff+l1': [
                            1 * HuberLoss(mask=f.rmask),
                            1e5 * NegRegularizer(),
                            5e2 * DiffLoss(rgrid),
                            1e1 * SphHarmL1Regularizer(mrinit),
                        ],
                        'bare': [
                            1 * HuberLoss(mask=f.rmask),
                            1e5 * NegRegularizer(),
                        ],
                        'diff+l1W': [
                            1 * HuberLoss(mask=f.rmask),
                            1e5 * NegRegularizer(),
                            5e2 * DiffLoss(rgrid),
                            1e1 * SphHarmL1Regularizer(mrinit, weights=W),
                        ],
                    }[lname]

                    # do full reconstruction
                    coeffs, retrieved_meas, losses = gd(
                        f, meas, mr, lr=1e0,
                        loss_fns=loss_fns, num_iterations=3000,
                        coeffs=initcoeffs.clone(),
                        callbacks=[LogCallback('fullL3', '/tmp/losses_baseline.txt')],
                    )

                    retrieved = mr(coeffs)  # (N, r, e, a)

                    t.save(coeffs, f'/tmp/coeffs_{desc}.tr')

                    # figure settings
                    figset = {'height': 200}

                    if issubclass(type(mr), SphHarmModel):
                        sphharm = plot(sphharmplot(mr.sph_coeffs(coeffs), mr), **figset)
                    else:
                        sphharm = ''

                    caption(
                        f"recon={mr} loss={lname}",
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
    tags.code(tags.pre(open('recon.py').read()))

# %% plot
# f = Path(f'/www/lara/direct_fit.html')
vgshape = 'x'.join(map(str, f.rvg[0].shape))
outfile = Path(f'/www/sph/two_week_{vgshape}{f.rvg[0].spacing}.html')
outfile.write_text(doc.render())
print(f'Saved to {outfile}')

from datetime import datetime
outfile = Path(f'/www/sph/archive/{datetime.now().isoformat()}.html')
outfile.write_text(doc.render())