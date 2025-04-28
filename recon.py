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
    TIMEGCMModel(grid=sgrid, device=device, offset=10, fill_value=0)
]

L_opts = [2, 3] # sph harm spline order
c_opts = [12, 16] # sph harm spline control points
recon_models = [
    SphHarmSplineModel(rgrid, max_l=L, device=device, cpoints=cpoints, spacing='log')
    for L, cpoints in product(L_opts, c_opts)
]

# ----- Measurement Generation -----

t_op = 360
num_obs=10; duration=14
cams = [CameraL1BNFI(nadir_nfi_mode(t_op=t_op)), CameraL1BWFI(nadir_wfi_mode(t_op=t_op))]
sc = gen_mission(num_obs=num_obs, duration=duration, start='2025-12-24', cams=cams)

f = ForwardSph(
    sc, sgrid=sgrid, # calibrator=cal
    rgrid=rgrid,
    # rvg=sum([ScienceGeom(s, (100, 50)) for s in sc]),
    rvg=sum([ScienceGeomFast(s, (100, 50)) for s in sc]),
    use_albedo=(ual:=True), use_aniso=(uan:=True), use_noise=(uno:=True),
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
        meas = f.noise(truth)

        truth_figs = []

        tags.h1(f'truth={mt}')
        with itemgrid(len(c_opts), flow='row'):

            for nr, mr in enumerate(recon_models):
                cshape = 'x'.join(map(str, mr.coeffs_shape))
                desc = f'spline{cshape}L{mr.max_l}_{num_obs:02d}obs'
                print('---', desc, f'truth:{nt}/{len(truth_models)}  recon:{nr}/{len(recon_models)}', '---')

                t.cuda.empty_cache()

                # ----- Retrieval -----
                # %% retr

                # choose loss functions and regularizers with weights
                loss_fns = [
                    1 * AbsLoss(projection_mask=f.proj_maskb),
                    1e4 * NegRegularizer(),
                    ReqErr(truth, mt.grid, mr.grid, interval=100),
                    1e0 * SphHarmL1Regularizer(mr),
                    # 1e-5 * SphHarmL1Regularizer(mr),
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
                # FIXME - ugly, get rid of this eventually
                # zero out retrieval/truth below 3Re
                retrieved3 = retrieved.clone().detach()
                retrieved3[mr.grid.r < 3] = 0
                truth3 = truth.clone().detach()
                truth3[mt.grid.r < 3] = 0

                # desc = f'{win:02d}d_{num_obs:02d}obs_{{noise_type}}{t_op//60}hr_{differr.lam:.0e}_{season}_init_{differr._reg}'
                # %% plot

                t.save(coeffs, f'/tmp/coeffs_{desc}.tr')

                if issubclass(type(mr), SphHarmModel):
                    sphharm = plot(sphharmplot(mr.sph_coeffs(coeffs), mr), height=200)
                else:
                    sphharm = ''
                caption(
                    f"recon={mr}",
                    plot(carderr(retrieved3.squeeze(), truth3.squeeze(), rgrid, sgrid), height=200),
                    sphharm,
                    # plot(loss_plot(losses)),
                    # plot(cardplotaxes(recon.squeeze(), grid)),
                    # plot(cardplotaxes(truth.squeeze(), grid)),
                )

    tags.code(tags.pre(open('recon.py').read()))

# %% plot
# f = Path(f'/www/lara/direct_fit.html')
vgshape = 'x'.join(map(str, f.rvg[0].shape))
outfile = Path(f'/www/sph/two_week_{vgshape}{f.rvg[0].spacing}_L1.html')
outfile.write_text(doc.render())
print(f'Saved to {outfile}')