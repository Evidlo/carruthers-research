#!/usr/bin/env python3
# Test calibration: L1A images already have bias removed
# Just normalize and remove dark stripes
# Evan Widloski 2026-05-27

import numpy as np
import xarray as xr
from glide.science_data_processing.L1A import L1A
from glide.common_components.camera import CameraL1BNFI
from domrep import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import sys; sys.path.insert(0, '..')
from bias_wavelet import remove_dark_stripes

# ----- Load datasets -----
dataset1 = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-STR_20251215_v1.0.nc')).data
dataset1 = dataset1.isel(time=[15, 16, 18, 24, 25])

dataset2 = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-DRK_20251215_v1.0.nc')).data
dataset2 = dataset2.isel(time=[0])

dataset3 = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-SCI_20251221_v1.0.nc')).data
dataset3 = dataset3.isel(time=0)

dataset4 = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-OOB_20260316_v1.0.nc')).data
dataset4 = dataset4.isel(time=0)

data = xr.concat([dataset1, dataset2, dataset3, dataset4], dim='time')

print(f"Loaded {len(data.time)} images")

# ----- Calibrate -----
# L1A already has bias removed, just normalize
img_norm = data.images / data.n_frames

# remove dark stripes
img_nodark = remove_dark_stripes(img_norm, 'NFI')[0]

# ----- Plotting -----
from dominate.tags import pre

with document('Calibration Test') as doc:
    with itemgrid(flow='row', length=3):
        with caption("1. Raw (DN)"):
            with slider():
                for t, row in data.groupby('time'):
                    with plot(label=f'{row.filter.values[0]} ({t})'):
                        plt.imshow(data.images.sel(time=t).squeeze())
                        plt.colorbar()
                        plt.tight_layout()

        with caption("2. / n_frames"):
            with slider():
                for t, row in data.groupby('time'):
                    with plot(label=f'{row.filter.values[0]} ({t})'):
                        plt.imshow(img_norm.sel(time=t).squeeze(), vmin=-10, vmax=10)
                        plt.colorbar()
                        plt.tight_layout()

        with caption("3. - dark stripes"):
            with slider():
                for t, row in data.groupby('time'):
                    with plot(label=f'{row.filter.values[0]} ({t})'):
                        plt.imshow(img_nodark.sel(time=t).squeeze(), vmin=-10, vmax=10)
                        plt.colorbar()
                        plt.tight_layout()

    pre(open(__file__, 'r').read())

doc.save(f:='/www/calibrate_test.html')
print(f'Wrote {f}')
