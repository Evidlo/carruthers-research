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
from tomosphero.retrieval import *
from tomosphero.loss import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch as t
from itertools import product


device = 'cuda'

# %% setup
# ----- Truth/Recon Models -----

sgrid = DefaultGrid((500, 45, 60), size_r=(3, 25), spacing='log')
rgrid = DefaultGrid((200, 45, 60), size_r=(3, 25), spacing='log')

mt = Zoennchen24Model(grid=sgrid, device=device)

mr = SphHarmModel(
    rgrid,
    max_l=3, # sph harm maximum order
    device=device,
    # cpoints=12, # spline control points
    # spacing='log'
)

# ----- Measurement Generation -----

num_obs=10; duration=28
cams = [CameraL1BNFI(), CameraL1BWFI()]
sc = gen_mission(num_obs=num_obs, duration=duration, start='2025-12-24', cams=cams)

f = ForwardSph(
    sc, sgrid=sgrid, # calibrator=cal
    rgrid=rgrid,
    # rvg=sum([ScienceGeom(s, (100, 50)) for s in sc]),
    rvg=sum([ScienceGeomFast(s, (100, 50)) for s in sc]),
    device=device
)

truth = mt()
truth_resample = den_sph2sph(truth, sgrid, rgrid)
meas_truth = f(truth_resample)

meas_snr = 44 # dB SNR
noise_var = t.var(meas_truth) / 10**(meas_snr / 10)

meas = meas_truth + t.normal(0, t.ones_like(meas_truth) * t.sqrt(noise_var))

# %% recon

with document('Two Week Retrievals') as doc:

    truth_figs = []

    tags.h1(f'truth={mt}')
    t.cuda.empty_cache()

    # ----- Retrieval -----
    # choose loss functions and regularizers with weights
    loss_fns = [
        1 * AbsLoss(projection_mask=f.proj_maskb),
        1e4 * NegRegularizer(),
        1e1 * SphHarmL1Regularizer(mr),
        5e2 * DiffLoss(rgrid),
        ReqErr(truth, mt.grid, mr.grid, interval=100),
    ]

    # do a fast initialization reconstruction with L=0
    # mrinit = SphHarmSplineModel(
    mrinit = SphHarmModel(
        rgrid, max_l=0,
        # cpoints=mr.cpoints, spacing=mr.spacing,
        device=device,
    )
    initcoeffs = t.zeros(mr.coeffs_shape, device=device)

    # truth_resample = den_sph2sph(truth, mt.grid, mr.grid)
    # cheater_loss = CheaterLoss(truth_resample)
    # cheater_loss.kind = 'regularizer'
    initcoeffs.data[0:1, :], _, _ = gd(
        f, meas, mrinit, lr=5e1,
        loss_fns=loss_fns, num_iterations=1000,
    )
    # do full reconstruction
    coeffs, retrieved_meas, losses = gd(
        f, meas, mr, lr=5e0,
        loss_fns=loss_fns, num_iterations=1000,
        coeffs=initcoeffs,
    )

    retrieved = mr(coeffs)

    # figure settings
    figset = {'height': 200}

    if issubclass(type(mr), SphHarmModel):
        sphharm = plot(sphharmplot(mr.sph_coeffs(coeffs), mr), **figset)
    else:
        sphharm = ''

    caption(
        f"recon={mr}",
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
    tags.code(tags.pre(open('recon_standalone.py').read()))

# %% plot
# f = Path(f'/www/lara/direct_fit.html')
vgshape = 'x'.join(map(str, f.rvg[0].shape))
outfile = Path(f'/www/sph/two_week_{vgshape}{f.rvg[0].spacing}_notspline.html')
outfile.write_text(doc.render())
print(f'Saved to {outfile}')

from datetime import datetime
outfile = Path(f'/www/sph/archive/{datetime.now().isoformat()}.html')
outfile.write_text(doc.render())