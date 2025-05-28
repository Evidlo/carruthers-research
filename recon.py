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

from dominate_tags import *

from pathlib import Path
from sph_raytracer import *
from sph_raytracer.plotting import *
cn = color_negative
from sph_raytracer.retrieval import *
from sph_raytracer.loss import *
from sph_raytracer.model import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch as t
import inspect
from itertools import product


__file__ = 'recon.py'
code = open(__file__).read()

device = 'cuda'

# %% setup
# ----- Truth/Recon Models -----

sgrid = DefaultGrid((500, 45, 60), size_r=(3, 25), spacing='log')
rgrid = DefaultGrid((200, 45, 60), size_r=(3, 25), spacing='log')

truth_models = [
    Zoennchen24Model(grid=sgrid, device=device),
    Pratik25Model(grid=sgrid, num_times=1, device=device),
    TIMEGCMModel(grid=sgrid, device=device, offset=10, fill_value='nearest')
]

L_opts = [1, 2, 3] # sph harm spline order
c_opts = [6, 8, 12, 16] # sph harm spline control points
recon_models = [
    SphHarmSplineModel(rgrid, max_l=L, device=device, cpoints=cpoints, spacing='log')
    for L, cpoints in product(L_opts, c_opts)
]

# ----- Measurement Generation -----

t_op = 360
num_obs=14; duration=14
cams = [CameraL1BNFI(nadir_nfi_mode(t_op=t_op)), CameraL1BWFI(nadir_wfi_mode(t_op=t_op))]
sc = gen_mission(num_obs=num_obs, duration=duration, start='2025-12-24', cams=cams)

f = ForwardSph(
    sc, sgrid=sgrid, # calibrator=cal
    rgrid=rgrid,
    # rvg=sum([ScienceGeom(s, (100, 50)) for s in sc]),
    rvg=sum([ScienceGeomFast(s, (100, 50)) for s in sc]),
    device=device
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
        with itemgrid(len(c_opts), flow='row'):

            for nr, mr in enumerate(recon_models):
                cshape = 'x'.join(map(str, mr.coeffs_shape))
                desc = f'spline{cshape}L{mr.max_l}_{num_obs:02d}obs'
                print('---', desc, f'truth:{nt}/{len(truth_models)}  recon:{nr}/{len(recon_models)}', '---')

                t.cuda.empty_cache()

                # ----- Retrieval -----
                # choose loss functions and regularizers with weights
                loss_fns = [
                    1 * AbsLoss(projection_mask=f.proj_maskb),
                    1e4 * NegRegularizer(),
                    1e1 * SphHarmL1Regularizer(mr),
                    ReqErr(truth, mt.grid, mr.grid, interval=100),
                ]

                # do a fast initialization reconstruction with L=0
                mrinit = SphHarmSplineModel(
                    rgrid, max_l=0,
                    cpoints=mr.cpoints, spacing=mr.spacing,
                    device=device,
                )
                initcoeffs = t.zeros(mr.coeffs_shape, device=device)
                initcoeffs.data[0:1, :], _, _ = gd(
                    f, meas, mrinit, lr=5e0,
                    loss_fns=loss_fns, num_iterations=2500,
                )
                # do full reconstruction
                coeffs, retrieved_meas, losses = gd(
                    f, meas, mr, lr=5e0,
                    loss_fns=loss_fns, num_iterations=3000,
                    coeffs=initcoeffs,
                )


                retrieved = mr(coeffs)

                t.save(coeffs, f'/tmp/coeffs_{desc}.tr')

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
    tags.code(tags.pre(open('recon.py').read()))

# %% plot
# f = Path(f'/www/lara/direct_fit.html')
vgshape = 'x'.join(map(str, f.rvg[0].shape))
outfile = Path(f'/www/sph/two_week_{vgshape}{f.rvg[0].spacing}_L1.html')
outfile.write_text(doc.render())
print(f'Saved to {outfile}')