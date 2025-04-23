#!/usr/bin/env python3

import numpy as np
from contexttimer import Timer

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
matplotlib.use('Agg')

from glide.common_components.science_pixel_binning import SciencePixelBinning
from glide.science.model_sph import Zoennchen00Model, DefaultGrid
from glide.common_components.spacecraft import SpaceCraft
from glide.common_components.camera import CameraL1BWFI, CameraL1BNFI
from glide.science.forward_sph import NativeGeom, ScienceGeom
from sph_raytracer.plotting import *

from skimage.transform import warp_polar, warp

scishape = np.array((100, 50))

m = Zoennchen00Model(DefaultGrid())
sc = SpaceCraft('2025-12-24', sensors=[CameraL1BWFI()])
# sc = SpaceCraft('2025-12-24', sensors=[CameraL1BNFI()])
N = sc.sensor.npix
sgeom = ScienceGeom(sc, scishape)
ngeom = NativeGeom(sc)

meas_tru = m.analytic(sgeom).numpy()
meas_nat = m.analytic(ngeom).numpy()

# load pregenerated measurements from file
d = np.load('science_pixel_fast.npy', allow_pickle=True)
meas_wfi, meas_nfi = d.item().get('wfi'), d.item().get('nfi')
# N = 1024; meas_data = meas_nfi
# N = 512; meas_data = meas_wfi

shape = np.array((N, N))
center = shape / 2
coords = np.stack(np.meshgrid(*(np.arange(s) for s in shape), indexing='ij'), axis=-1)
# meas = meas_data
meas = meas_nat
# meas = np.zeros(shape)
# meas[6+N//2, :] = 1
# meas = np.linalg.norm(coords - center, axis=-1)
# meas = np.random.random(shape)
# add artificial feature for alignment debugging purposes
# meas[:N//4, :N//4] = 1
# meas[:N//3, N//4:N//3] = 1

# ----- Original Science Pixel Binning -----


with Timer(prefix='Original') as p:
    # sb = SciencePixelBinning(N, scishape, rlim=(0, N/2), normalize=True)
    sb = sgeom.bin

    meas_sb = sb(meas[None])


# ----- Fast Science Pixel Binning -----

# plt.imsave('/www/out.png', meas)

with Timer(prefix='Fast') as p:
    oversample = 10
    """
    meas_warp = warp_polar(
        meas,
        output_shape=np.flip(scishape * oversample),
        radius=N//2 + .125
    ).T
    """

    def affine(x, xlim, ylim):
        xlim, ylim = np.asarray(xlim), np.asarray(ylim)
        normalized = (x - xlim[:, 0]) / (xlim[:, 1] - xlim[:, 0])
        return normalized * (ylim[:, 1] - ylim[:, 0]) + ylim[:, 0]

    rlim, thlim = sb.rlim, np.deg2rad(sb.thlim)
    rlim += np.array((+0.05, 0))
    outshape = scishape * oversample
    def map_func(output_coords):

        # warp(...) gives `output_coords` in (col, row) format for some reason
        theta, r = affine(
            output_coords + np.array((0.5, 0.5)),
            [(0, outshape[1]), (0, outshape[0])],
            [thlim, rlim]
        ).T

        xy = np.stack((r * np.cos(-theta), r * np.sin(-theta)), axis=-1)
        uv = xy + sb.ctr - 0.5
        # import ipdb
        # ipdb.set_trace()
        return uv

    meas_warp = warp(
        meas, map_func, output_shape=outshape, order=1
    )


    print(meas_warp.shape)
    # meas_warp = np.roll(meas_warp, oversample*scishape[1]//2 - 0, axis=1)
    # meas_warp = np.flip(meas_warp, axis=1)
    meas_warp = meas_warp.reshape((scishape[0], oversample, scishape[1], oversample))
    meas_warp = meas_warp.mean(axis=(-3, -1))
    # meas_warp = meas_warp / oversample**2
    print(meas_warp.shape)

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
plt.title('Science/Truth % Error')
plt.imshow(err_sb, vmin=-2, vmax=2, cmap=cmap)
plt.colorbar()

plt.subplot(2, 3, 2)
plt.title('Warp/Truth % Error')
plt.imshow(err_warp, vmin=-2, vmax=2, cmap=cmap)
plt.colorbar()

plt.subplot(2, 3, 3)
plt.title('Truth')
plt.imshow(meas_tru, norm=LogNorm())
plt.colorbar()

plt.subplot(2, 3, 4)
plt.title('Science')
plt.imshow(meas_sb, norm=LogNorm())
plt.colorbar()

plt.subplot(2, 3, 5)
plt.title('Warp')
plt.imshow(meas_warp, norm=LogNorm())
plt.colorbar()

plt.subplot(2, 3, 6)
plt.title('% Error')
plt.imshow(err, vmin=-2, vmax=2, cmap=cmap)
plt.colorbar()

plt.tight_layout()
plt.savefig('/www/out.png')