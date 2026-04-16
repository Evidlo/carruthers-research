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

from tomosphero.plotting import *
from tomosphero.retrieval import *
from tomosphero.loss import *

import torch as t


device = 'cuda'

# %% setup
# ----- Truth/Recon Models -----

sgrid = DefaultGrid((500, 45, 60), size_r=(3, 25), spacing='log')
rgrid = DefaultGrid((200, 45, 60), size_r=(3, 25), spacing='log')

mt = Zoennchen24Model(grid=sgrid, device=device)

# ----- Measurement Generation -----

t_op = 360
num_obs = 14 # number of measurement locations
duration = 14 # span of observation orbit
cams = [CameraL1BNFI(nadir_nfi_mode(t_op=t_op)), CameraL1BWFI(nadir_wfi_mode(t_op=t_op))]
sc = gen_mission(num_obs=num_obs, duration=duration, start='2025-12-24', cams=cams)

f = ForwardSph(
    sc, sgrid=sgrid, # calibrator=cal
    rgrid=rgrid,
    # rvg=sum([ScienceGeom(s, (100, 50)) for s in sc]),
    rvg=sum([ScienceGeomFast(s, (100, 50)) for s in sc]),
    device=device
)

truth = mt()
meas_real = f.calibrate(f.simulate(truth))
truth_resample = den_sph2sph(truth, sgrid, rgrid)
meas_truth = f(truth_resample)

# save problem inputs to be able to run retrievals standalone without Carruthers libraries
# np.savez(
#     'retrieval_inputs.npz',
#     {
#         'measurements': meas_truth,
#         ''
#     }
# )