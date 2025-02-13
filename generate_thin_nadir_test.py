#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from glide.science.orbit import glide_orbit, sc2viewgeom
from glide.science.model import SnowmanModel, default_vol
from glide.science.forward import Forward
from glide.science.plotting import save_gif, preview3d, orbit_svg, imshow, color_negative
from glide.science.common import reg2ts, ts2reg, nptorch
from glide.validation.synth_noisy_data import generate_thin_nadir_images, generate_noisy_images
from glide.common_components.view_geometry import gen_mission
from glide.common_components.cam import CamMode, CamSpec

from glide.validation.imaging_mode import ImagingMode

date = '2025-12-29'
im = ImagingMode('nadir', 'WFI', date)

vol = default_vol()
x = SnowmanModel(vol, device='cuda').density()
# sc = glide_orbit()
cm = CamMode('WFI', t_int=60)
cs = CamSpec(cm, true_flat=True)
cam_modes, cam_specs, view_geoms = gen_mission(start=date, num_obs=1, cam_mode=cm, cam_spec=cs)
orbit_svg(vol, sc2viewgeom(view_geoms)).save('/srv/www/orbit.svg')

# --- Snowman Measurements ---

# f = Forward(view_geoms, vol, use_noise=False, use_grad=False, use_albedo=True, use_aniso=True)
# # scale measurement magnitude so it approximately matches GLIDE_WFI_radiance_B.txt
# y = f(x)[0].detach().cpu().numpy() * 20

# --- Exos Measurements ---

import importlib.resources
path = importlib.resources.files('glide') / 'validation/data_files/Date_12_29_2025/GLIDE_WFI_radiance_B.txt'
exos = np.loadtxt(path).reshape((512, 512))

# --- Apply Noise ---

# y[:256, :] = exos[:256, :]
y_noisy = generate_thin_nadir_images(exos, view_geoms[0], cam_modes[0], cam_specs[0])

# --- Plot ---
# %% plot

# plt.subplot(1, 2, 1)
# plt.hist(exos.flatten(), bins=50)
# plt.title('Clean Exos Intensity')

# plt.subplot(1, 2, 2)
# plt.hist(y_noisy.flatten(), bins=50)
# plt.title('Noisy Exos Intensity')

# plt.show()
