#!/usr/bin/env python3

from glide.science.model import zoennchen_model, default_vol
from glide.science.orbit import glide_orbit, sc2viewgeom
from glide.validation.photon_events import NoisyImage, gen_Lya_events, gen_OOB_events, gen_IPH_events
from glide.validation.gci_specs import GCI_specs
from glide.validation.noise_module import BackgroundNoise
from glide.validation.scene_model import gen_IPH_scene
from glide.common_components.view_geometry import gen_defaultSpaceCraft
from astropy.time import Time
from astropy import constants
import tomosipo as ts
import torch as t
import numpy as np
import matplotlib.pyplot as plt

vol = default_vol()

# phantom = t.tensor(zoennchen_model(vol), device='cuda', dtype=t.float32)
phantom = zoennchen_model(vol)

cam_id = 'WFI'
sc = glide_orbit(num_obs=1, start=Time('2025-09-01'))[0]
viewgeom = sc2viewgeom([sc])
forward_op = ts.operator(vol, viewgeom)

I_exo = forward_op(phantom).squeeze()

thlim = (0, 12.5)
wedge_mask = sc.getSensor(cam_id).wedgeMask(thlim)

# ----- camera settings and intrinsic noise -----
from glide.validation.imaging_mode import imaging_mode
cam_mode = imaging_mode('nadir', None, cam_id, fltr='LyaN', activation='lot')
t_int = cam_mode.fps * cam_mode.t_int*60 * cam_mode.t_frame

cam_specs = GCI_specs(cam_mode)

dark_noise = BackgroundNoise(
    cam_specs.dark_current,
    cam_mode.t_frame,
    cam_specs.pixel_size,
    events_name='mean_dark_counts'
)
rad_noise = BackgroundNoise(
    cam_specs.APS_radiation,
    cam_mode.t_frame,
    cam_specs.pixel_size,
    events_name='mean_rad_counts'
)
bkgd_noise = dark_noise + rad_noise

# ----- background noise -----
#Select camera
cam = sc.getSensor(cam_id)

#%%Calculate tangent points

uv  = cam.getPixels()
r_obs, SZA_sc, los_zenith, los_azimuth, los_TP_r = sc.calculateEarthTangentPoints(uv, camID=cam_id)

#Re-shape outputs
los_TP_r    = np.reshape(los_TP_r,(cam.npix, cam.npix))
los_zenith  = np.reshape(los_zenith,(cam.npix, cam.npix))
los_azimuth = np.reshape(los_azimuth,(cam.npix, cam.npix))
r_disk  = constants.R_earth.to_value('km')
r_inner = 1.5 * r_disk
r_max   = np.max(los_TP_r[int(cam.npix/2)-1,:])
mean_lya_inner = gen_Lya_events(I_exo, los_TP_r, cam_specs, (0, r_inner), cam_mode.t_frame, res=1, doSpectrum=True)
mean_lya_outer = gen_Lya_events(I_exo, los_TP_r, cam_specs, (r_inner, r_max), cam_mode.t_frame, res=1, doSpectrum=False)
mean_OOB = gen_OOB_events(I_exo, los_TP_r, cam_specs, (0, r_inner), cam_mode.t_frame)
I_IPH = gen_IPH_scene(sc, filename='Solar_Max_IPH_Map.nc', camID=cam_id)
mean_IPH = gen_IPH_events(I_IPH, los_TP_r, cam_specs, (r_disk, r_max), cam_mode.t_frame)

#Add events
mean_nadir_events  = mean_lya_outer.copy()
mean_nadir_events.add(mean_lya_inner)
mean_exo_events    = mean_nadir_events.copy()
mean_nadir_events.add(mean_OOB)
mean_photon_events = mean_nadir_events.copy()
mean_photon_events.add(mean_IPH)
mean_lya_outer.add(mean_lya_inner)

image_generator = NoisyImage(
    mean_photon_events,
    bkgd_noise, cam_specs, cam_mode,
    mask=np.ones(mean_photon_events.events.shape).astype(bool)
)
image_generator.makeNoisyImages(1)

noisy_image = image_generator.image_noisy_counts.reshape(I_exo.shape)

plt.close()
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(I_exo)
plt.colorbar()
plt.title('I_exo')
plt.savefig('/tmp/original.png')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image)
plt.colorbar()
plt.title('Noisy image')
plt.savefig('/tmp/noise.png')

plt.tight_layout()
plt.show()
