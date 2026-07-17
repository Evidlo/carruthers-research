#!/usr/bin/env python3
# %% load

import sys
sys.path.insert(0, '..')

from common import rob_bias
import xarray as xr
from domrep import document, slider, plot, caption, itemgrid, dropdown
from dominate import tags
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')
from scipy.ndimage import rotate

from multiml.multiml import *

ds = xr.open_dataset('/home/evan/nc/L1A/CARRUTHERS_GCI-NFI_L1A-STR_20260318_v1.0.nc')
# ds = xr.open_dataset('/tmp/str.nc')
# ds_dark = xr.open_dataset('/tmp/dark.nc')

ds = ds.where(ds.filter == 'SrF2', drop=True)

# --- L0 conversion ---
# %% flatten
# add bias back in
im = ds.images + ds.bias * ds.n_frames
im = im / ds.t_int
im = im[:17]
im = im - rob_bias(im, 150, 150)

# im_dark = ds_dark.images + ds_dark.bias * ds_dark.n_frames
# im_dark = im_dark / ds_dark.t_int
# im_dark = im_dark - rob_bias(im_dark, 150, 200)

# --- distortion ---
from glide.calibration.calibration_helpers import distort_image
from glide.validation.cam import load_lab_data
from glide.common_components.camera import CameraNFI

cam = CameraNFI()

cam.spec = load_lab_data(cam.spec)
# transform = cam.spec.get_distortion_transform()
# im_distort = distort_image(im, transform)

# im = np.clip(im, *np.percentile(im, (10, 70)))
# im = im[:, 200:512, 50:500]

# --- Registration ---
# %% register

# register images
im = np.stack([rotate(f, i * -0.113, reshape=False, order=1) for i, f in enumerate(im)])
# registered = shift(im, np.array((0.25, 6.75)), shift_method='roll')
registered = shift(im, np.array((0.3, -6.75)), shift_method='roll')

star1 = (..., slice(600, 800), slice(200, 400))
star2 = (..., slice(200, 400), slice(800, 1000))
# flat = cam.spec.flat[600:800, 200:400]

# registered_distort = shift(im_distort, np.array((-.3, -6.6)), shift_method='fourier')
# im_dark = im_dark[:, 200:512, 50:500]


# csums = correlate_and_sum(im)
# csums_dark = correlate_and_sum(im_dark)
# result, scaled = scale_and_sum(csums)

# %% plot

clim = -100, np.percentile(registered, 99.0)
clim = -20, 20
with document(title='Registration Experiment') as d:
    with itemgrid(length=2, flow='row'):
        # with plot():
        #     csums[0, 0, 0] = 0
        #     plt.imshow(csums[0, :100, :100])
        # with plot():
        #     csums_dark[0, 0, 0] = 0
        #     plt.imshow(csums_dark[0, :100, :100])
        with caption('Coadded Scene'):
            with plot():
                plt.imshow(registered.sum(axis=0))
                plt.colorbar()
                plt.clim(clim)
        with dropdown():
            with slider(interval=50, label='Nonflat'):
                for image, time in zip(registered, ds.time.data):
                    with plot(label=time):
                        plt.imshow(image)
                        plt.colorbar()
                        plt.clim(clim)
        with caption('Coadded Star 1'):
            with plot():
                plt.imshow(registered[star1].sum(axis=0))
                plt.colorbar()
                plt.clim(clim)
        with dropdown():
            with slider(interval=50, label='Nonflat'):
                for image, time in zip(registered[star1], ds.time.data):
                    with plot(label=time):
                        plt.imshow(image)
                        plt.colorbar()
                        plt.clim(clim)
        with caption('Coadded Star 2'):
            with plot():
                plt.imshow(registered[star2].sum(axis=0))
                plt.colorbar()
                plt.clim(clim)
        with dropdown():
            with slider(interval=50, label='Nonflat'):
                for image, time in zip(registered[star2], ds.time.data):
                    with plot(label=time):
                        plt.imshow(image)
                        plt.colorbar()
                        plt.clim(clim)
            # with slider(interval=50, label='Flat'):
            #     for image, time in zip(registered, ds.time.data):
            #         with plot(label=time):
            #             plt.imshow(0 * image / flat)
            #             plt.colorbar()
            #             plt.clim(clim)
            # with slider(interval=50, label='Undistorted'):
            #     for image, time in zip(registered_distort, ds.time.data):
            #         with plot(label=time):
            #             plt.imshow(image)
            #             plt.colorbar()
            #             plt.clim(clim)
                    # plt.clim(*np.percentile(registered, (10, 90.9)))
        # with slider():
        #     for image in im_dark:
        #         with plot(label='hello'):
        #             plt.imshow(image)

    tags.pre(open('register.py', 'r').read())

d.save(f:='/www/register.html')
print(f'Wrote to {f}')
