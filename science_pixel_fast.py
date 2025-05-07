#!/usr/bin/env python3

import numpy as np
from contexttimer import Timer

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
matplotlib.use('Agg')

from glide.science.model_sph import Zoennchen00Model, DefaultGrid
from glide.common_components.spacecraft import SpaceCraft
from glide.common_components.camera import CameraL1BWFI, CameraL1BNFI
from glide.science.forward_sph import NativeGeom, ScienceGeom, ScienceGeomFast
from sph_raytracer.plotting import *

from skimage.transform import warp_polar, warp

scishape = np.array((100, 50))

m = Zoennchen00Model(DefaultGrid())
sc = SpaceCraft('2025-12-24', sensors=[CameraL1BWFI()])
# sc = SpaceCraft('2025-12-24', sensors=[CameraL1BNFI()])
N = sc.sensor.npix
sgeom = ScienceGeom(sc, scishape)
sgeom_fast = ScienceGeomFast(sc, scishape)
ngeom = NativeGeom(sc)

meas_tru = m.analytic(sgeom).numpy()
meas = m.analytic(ngeom).numpy()

# load pregenerated measurements from file
d = np.load('science_pixel_fast.npy', allow_pickle=True)
meas_wfi, meas_nfi = d.item().get('wfi'), d.item().get('nfi')

shape = np.array((N, N))
center = shape / 2
coords = np.stack(np.meshgrid(*(np.arange(s) for s in shape), indexing='ij'), axis=-1)
# meas = np.zeros(shape)
# meas[6+N//2, :] = 1
# meas = np.linalg.norm(coords - center, axis=-1)
# meas = np.random.random(shape)
# add artificial feature for alignment debugging purposes
# meas[:N//4, :N//4] = 1
# meas[:N//3, N//4:N//3] = 1

# ----- Original Science Pixel Binning -----


with Timer(prefix='Original') as p:
    meas_sb = sgeom.bin(meas[None])


# ----- Fast Science Pixel Binning -----

# plt.imsave('/www/out.png', meas)

with Timer(prefix='Fast') as p:
    meas_warp = sgeom_fast.bin(meas[None]).squeeze()

# ----- Plotting -----

err = np.divide((meas_warp - meas_sb), meas_sb, where=meas_sb!=0) * 100
err = np.where(meas_sb>1e-5, err, 0)

err_warp = np.divide((meas_warp - meas_tru), meas_tru, where=meas_tru!=0) * 100
err_warp = np.where(meas_tru>1e-5, err_warp, 0)

err_sb = np.divide((meas_sb - meas_tru), meas_tru, where=meas_tru!=0) * 100
err_sb = np.where(meas_tru>1e-5, err_sb, 0)

plt.close('all')
cmap = plt.get_cmap('seismic')
cmap.set_bad(color='black')

plt.figure(dpi=200, figsize=(12, 12))

plt.subplot(2, 3, 1)
plt.title('Original/Truth % Error')
plt.imshow(err_sb, vmin=-2, vmax=2, cmap=cmap)
plt.colorbar()

plt.subplot(2, 3, 2)
plt.title('Fast/Truth % Error')
plt.imshow(err_warp, vmin=-2, vmax=2, cmap=cmap)
plt.colorbar()

plt.subplot(2, 3, 3)
plt.title('Truth')
plt.imshow(meas_tru, norm=LogNorm())
plt.colorbar()

plt.subplot(2, 3, 4)
plt.title('Original')
plt.imshow(meas_sb, norm=LogNorm())
plt.colorbar()

plt.subplot(2, 3, 5)
plt.title('Fast')
plt.imshow(meas_warp, norm=LogNorm())
plt.colorbar()

plt.subplot(2, 3, 6)
plt.title('Original/Fast % Error')
plt.imshow(err, vmin=-2, vmax=2, cmap=cmap)
plt.colorbar()

plt.tight_layout()
plt.savefig(f'/www/spb_{sc.sensor.mode.channel}.png')