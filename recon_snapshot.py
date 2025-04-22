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
    Pratik25StormModel(grid=sgrid, num_times=1, device=device),
          TIMEGCMModel(grid=sgrid, num_times=1, device=device),
          TIMEGCMModel(grid=sgrid, num_times=1, device=device, offset=10),
             MSISModel(grid=sgrid, num_times=1, device=device),
]

# L_opts = (0, 1) # sph harm spline order
L_opts = (1,) # sph harm spline order
c_opts = (8, 12, 16) # sph harm spline control points
recon_models = [
    SphHarmSplineModel(rgrid, max_l=L, device=device, cpoints=cpoints, spacing='log')
    for L, cpoints in product(L_opts, c_opts)
]

# ----- Measurement Generation -----

t_op = 360
num_obs=1; duration=14
cams = [CameraL1BNFI(nadir_nfi_mode(t_op=t_op)), CameraL1BWFI(nadir_wfi_mode(t_op=t_op))]
sc = gen_mission(num_obs=num_obs, duration=duration, start='2025-12-24', cams=cams)

print('----- Measurement Forward Model -----')
f = ForwardSph(
    sc, sgrid=sgrid, # calibrator=cal
    rgrid=rgrid,
    use_albedo=(ual:=True), use_aniso=(uan:=True), use_noise=(uno:=True),
    device=device
)

# %% recon

with document('Snapshot Retrievals') as doc:

    # iterate ground truths
    for nt, mt in enumerate(truth_models):

        truth = mt()
        meas = f.noise(truth)

        truth_figs = []

        tags.h1(f'truth={mt}')
        with itemgrid(len(c_opts), flow='row'):

            for nr, mr in enumerate(recon_models):
                cshape = 'x'.join(map(str, mr.coeffs_shape))
                desc = f'spline_c{mr.cpoints}_L{mr.max_l}_{num_obs:02d}obs'
                print('-----------------------------')
                print(desc, f'truth:{nt}/{len(truth_models)}  recon:{nr}/{len(recon_models)}')
                print('-----------------------------')
                t.cuda.empty_cache()

                # ----- Retrieval -----

                # choose loss functions and regularizers with weights
                loss_fns = [
                    1 * AbsLoss(projection_mask=f.proj_maskb),
                    1e4 * NegRegularizer(),
                    ReqErr(truth, mt.grid, mr.grid, interval=100)
                ]

                # do full reconstruction
                coeffs, retrieved_meas, losses = gd(
                    f, meas, mr, lr=5e0,
                    loss_fns=loss_fns, num_iterations=3000,
                )

                retrieved = mr(coeffs)
                # FIXME - ugly, get rid of this eventually
                # zero out retrieval/truth below 3Re
                retrieved3 = retrieved.clone().detach()
                retrieved3[mr.grid.r < 3] = 0
                truth3 = truth.clone().detach()
                truth3[mt.grid.r < 3] = 0



                if issubclass(type(mr), SphHarmModel):
                    sphharm = plot(sphharmplot(mr.sph_coeffs(coeffs), mr), height=200)
                else:
                    sphharm = ''
                caption(
                    f"recon={mr}",
                    plot(carderr(retrieved3.squeeze(), truth3.squeeze(), rgrid, sgrid), height=200),
                    sphharm,
                    # FIXME
                    # plot(loss_plot(losses)),
                    # plot(cardplotaxes(recon.squeeze(), grid)),
                    # plot(cardplotaxes(truth.squeeze(), grid)),
                )

    tags.code(tags.pre(open('recon_snapshot.py').read()))

# %% plot
# f = Path(f'/www/lara/direct_fit.html')
outfile = Path(f'/www/sph/snapshot_debug.html')
outfile.write_text(doc.render())
print(f'Saved to {outfile}')