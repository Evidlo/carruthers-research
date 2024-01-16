#!/usr/bin/env python3

from glide.common_components.view_geometry import *
from glide.common_components.cam import *
from glide.science.plotting import *
from glide.science.forward import *
from glide.science.decorators import *
from glide.science.orbit import *
from glide.science.model import *
from glide.science.recon import *
from glide.validation.scene import *
from glide.validation.instrument import *
from glide.calibration import *
from glide.common_components.cam import *
import tomosipo as ts
from datetime import *
from astropy.time import Time

import importlib.resources
path = importlib.resources.files('glide') / 'validation/data_files/Date_12_29_2025/GLIDE_WFI_radiance_B.txt'
I_exo = np.loadtxt(path).reshape((512, 512))

sc = SpaceCraft('2025-06-01')
sc.add_sensor(CameraWFI())
sc.updatePositionFromFile(sc.date)
sc.pointAtEarth()

vol = default_vol(shape=100, size=50)
grid_cart = vol2cart(vol)
grid_sph = cart2sph(grid_cart)
cams=[CameraWFI()]
view_geoms = gen_mission(
    orbit=carruthers_orbit,
    num_obs=20, cams=cams,
    start='2025-07-01', duration=30*3,
)
# small_vol = default_vol(shape=100)

# cm = CamMode('WFI', t_int=1)
# cs = CamSpec(cm)
# cam_modes, cam_specs, view_geoms = gen_mission(num_obs=1, cam_mode=cm, cam_spec=cs)

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
