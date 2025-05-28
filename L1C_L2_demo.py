#!/usr/bin/env python3

"""
Demo of production L1C to L2 optically thin retrieval

Includes extra code which will be provided in L1C to be removed
in production (marked with FIXME)
"""

from glide.science.forward_sph import ForwardSph, ScienceGeomFast
from glide.science.model_sph import DefaultGrid, SphHarmSplineModel
from glide.science.plotting import loss_plot, sphharmplot
from glide.science.plotting_sph import carderr, cardplot, carderrmin, cardplotaxes
from glide.science.recon.loss_sph import SphHarmL1Regularizer, ReqErr

from sph_raytracer.retrieval import gd
from sph_raytracer.loss import AbsLoss, NegRegularizer

device = 'cuda'

# FIXME: remove this section for production, `sc` and `meas_L1C` generated externally
# ----- Truth Density and Spacecraft Generation -----

from glide.common_components.camera import CameraWFI, CameraNFI, CameraL1BWFI, CameraL1BNFI
from glide.common_components.cam import nadir_wfi_mode, nadir_nfi_mode
from glide.common_components.generate_view_geom import gen_mission
from glide.science.model_sph import Zoennchen24Model, Pratik25Model, TIMEGCMModel

t_op = 360 # minutes
cams = [CameraL1BNFI(nadir_nfi_mode(t_op=t_op)), CameraL1BWFI(nadir_wfi_mode(t_op=t_op))]
sc = gen_mission(num_obs=1, duration=1, start='2025-12-24', cams=cams)


# choose a dataset to reconstruct
sgrid = DefaultGrid((500, 45, 60), size_r=(3, 25), spacing='log')
truth_model = (
    Zoennchen24Model(grid=sgrid, device=device)
    # Pratik25Model(grid=sgrid, num_times=1, device=device)
    # TIMEGCMModel(grid=sgrid, device=device, offset=10, fill_value='nearest')
)
# compute ground truth density
truth = truth_model()

# compute L1C measurements
f_truth = ForwardSph(sc, sgrid=sgrid, device=device)
meas_L1C = f_truth.simulate(truth)

# ----- Retrieval -----
# %% retrieval

# set up reconstruction model
recon_model = SphHarmSplineModel(
    DefaultGrid((500, 45, 60), size_r=(3, 25), spacing='log'),
    max_l=0, device=device, cpoints=16, spacing='log'
)

# set up forward operator with appropriate view geometry and grid
f = ForwardSph(
    sc, rgrid=recon_model.grid,
    # rvg=sum([ScienceGeom(s, (100, 50)) for s in sc]),
    rvg=sum([ScienceGeomFast(s, (100, 50)) for s in sc]),
    device=device
)

# calibrate and bin measurements to science pixel column densities
meas = f.calibrate(meas_L1C)

# choose loss functions and regularizers with weights
loss_fns = [
    1 * AbsLoss(projection_mask=f.proj_maskb),
    1e4 * NegRegularizer(),
    1e1 * SphHarmL1Regularizer(recon_model),
    # FIXME: remove this line for production, no access to ground truth
    ReqErr(truth, truth_model.grid, recon_model.grid, interval=100),
]

# reconstruction loop
coeffs, retrieved_meas, losses = gd(
    f, meas, recon_model, lr=5e0,
    loss_fns=loss_fns, num_iterations=3000,
)
retrieved = recon_model(coeffs)

# ----- Plotting -----
# %% plot

import matplotlib
# accelerate plot generation when in non-interactive mode
matplotlib.use('Agg')

figures = [
    # primary figures
    sphharmplot(recon_model.sph_coeffs(coeffs), recon_model),
    # extra debugging figures
    loss_plot(losses),
    cardplot(retrieved.squeeze(), recon_model.grid, norm='log'),
    cardplotaxes(retrieved.squeeze(), recon_model.grid, yscale='log'),
]

# FIXME: remove this section for production
# ----- Save plots to disk -----

from dominate_tags import *

# figure settings
figset = {'height': 200}
result = caption(
    f"recon={recon_model}",
    # FIXME: remove for production, no access to ground truth
    plot(carderr(retrieved.squeeze(), truth.squeeze(), recon_model.grid, sgrid), **figset),
    plot(figures[0], **figset),
    tags.br(),
    tags.details(
        tags.summary(),
        plot(figures[1], **figset),
        caption("Recon", plot(figures[2], **figset)),
        caption("Truth", plot(cardplot(truth.squeeze(), sgrid, norm='log'), **figset)),
        caption("Recon", plot(figures[3], **figset)),
        caption("Truth", plot(cardplotaxes(truth.squeeze(), sgrid, yscale='log'), **figset)),
    )
)

with open('/www/l1c_l2.html', 'w') as f:
    f.write(result.render())