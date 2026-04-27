#!/usr/bin/env python3
# Plot L0 images with naive subtraction via rob_bias

from common import rob_bias, load

import xarray as xr
from glide.science_data_processing.L1A import L1A

import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import numpy as np
import matplotlib
matplotlib.use('Agg')

# d_oob = L1A(xr.open_dataset('/home/evan/nc/L1A/CARRUTHERS_GCI-NFI_L1A-OOB_20260113_v1.0.nc'))
# d_str = L1A(xr.open_dataset('/home/evan/nc/L1A/CARRUTHERS_GCI-NFI_L1A-STR_20260113_v1.0.nc'))
# d_oob = L1A(xr.open_dataset('/home/evan/nc/L1A/CARRUTHERS_GCI-NFI_L1A-OOB_20260318_v1.0.nc'))
# d_str = L1A(xr.open_dataset('/home/evan/nc/L1A/CARRUTHERS_GCI-NFI_L1A-STR_20260318_v1.0.nc'))
# x_str, t_str = d_str.images[3], d_str.time[3]
# x_oob, t_oob = d_oob.images[0], d_oob.time[0]

x_str, t_str = load('images_20260318/star_nfi_l0.pkl'), '20260318'
x_oob, t_oob = load('images_20260318/oob_nfi_l0.pkl'), '20260318'

x_str -= rob_bias(x_str, 100, 200)
x_oob -= rob_bias(x_oob, 100, 200)

# clip = (10, 90) # percentile clip
clip = (5, 95) # percentile clip
# clip = (0, 100) # percentile clip

conf = {'norm': CenteredNorm()}

plt.close()
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(x_str, **conf)
plt.clim(np.nanpercentile(x_str, clip))
plt.colorbar()
plt.title(f"STR {t_str}")

plt.subplot(1, 2, 2)
plt.imshow(x_oob, **conf)
plt.clim(np.nanpercentile(x_oob, clip))
plt.colorbar()
plt.title(f"OOB {t_oob}")

plt.tight_layout()
plt.savefig('/www/out.png')