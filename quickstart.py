#!/usr/bin/env python3

import torch as t
import numpy as np

from glide.common_components.view_geometry import *
from glide.common_components.generate_view_geom import *
from glide.common_components.camera import *
from glide.common_components.spacecraft import *
from glide.common_components.cam import *
from glide.common_components.science_pixel_binning import *
from glide.science.plotting import *
from glide.science.plotting_sph import *
from glide.science.forward import *
from glide.science.forward_sph import *
from glide.science.decorators import *
from glide.science.orbit import *
from glide.science.model import *
from glide.science.model_sph import *
from glide.science.recon import *
from glide.validation.scene import *
from glide.validation.instrument import *
from glide.calibration import *
from glide.common_components.cam import *
# from glide.validation.L1C_simulator import L1CSim
from sph_raytracer import *
import tomosipo as ts
from datetime import *
from astropy.time import Time


cams = [CameraL1BWFI(), CameraL1BNFI()]
sc = gen_mission(num_obs=1, cams=cams)
grid = DefaultGrid()
m = GonzaloModel(grid=grid)
g = grid = m.grid
g50 = DefaultGrid((50, 50, 50))
g500 = DefaultGrid((500, 50, 50))
mr = SphHarmModel(grid, max_l=3)
density = m()

# sim = L1CSim(disable_noise=True)
# x = sim(sc, [8000 * np.ones((512, 512)), 8000 * np.ones((1024, 1024))])

vol = default_vol(shape=100, size=50)
f = ForwardSph(sc, grid, use_aniso=True, use_albedo=False)

# small_vol = default_vol(shape=100)

# cm = CamMode('WFI', t_int=1)
# cs = CamSpec(cm)
# cam_modes, cam_specs, view_geoms = gen_mission(num_obs=1, cam_mode=cm, cam_spec=cs)

# import importlib.resources
# path = importlib.resources.files('glide') / 'validation/data_files/Date_12_29_2025/GLIDE_WFI_radiance_B.txt'
# I_exo = np.loadtxt(path).reshape((512, 512))
# cam = CameraWFI()
# view_geoms = carruthers_orbit(num_obs=1, cam=cam)
# cm, cs = cam.cam_mode, cam.cam_spec
# s = Scene(cm, cs, view_geoms[0], iph=True, I_exo=I_exo)
# i = Instrument(cm, cs, s)

# x = SnowmanModel(vol, device='cuda').density * 10
# m = SphHarmBasisModel(vol, device='cuda')
# x = m(t.rand(m.coeffs_shape, device='cuda'))
# f = Forward([cm], [cs], view_geoms, vol, use_noise=False, use_grad=True, use_albedo=False, use_aniso=False)
# y = f(x)[0]
