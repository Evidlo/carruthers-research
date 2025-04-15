#!/usr/bin/env python3

import xarray as xr

from astropy.constants import R_earth
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

# ----- Load Data -----

basedir = Path('/home/evan/tmp/data_2002_f107_210_f107a_210_validation_storm_2Hexo/output')
ds = xr.open_mfdataset(basedir / 'nH_3D_02_01_2002_00.nc').compute()
ds = ds.rename(theta='e', phi='a', nH='density')
# change the order of the dimensions to (r, e, a)
ds = ds.transpose('r', 'e', 'a')

# load density and coordinates
x = ds.density.values
r = ds.R.values / R_earth.to('cm').value
e = np.deg2rad(ds.LATGSE.values) + np.pi / 2
a = np.deg2rad(ds.LONGSE.values)

# Create lineplot at constant altitude circle in YZ plane
# get index of `rval` Re shell
r6_ind = np.searchsorted(r, rval:=6)
# get index of ±Y azimuth
aY_ind = np.searchsorted(a, np.pi/2)
aY_ind_neg = np.searchsorted(a, -np.pi/2)

# ----- Plotting -----

plt.close('all')
fig, ax = plt.subplots()
ax.plot(e, x[..., r6_ind, :, aY_ind].squeeze(), 'b')
ax.plot(-e, x[..., r6_ind, :, aY_ind_neg].squeeze(), 'b')
ax.set_xlabel('Elevation (rad)')
ax.set_ylabel('Density')
ax.set_title(f'Density in YZ plane at {rval} Re (nearest bin)')

fig.savefig('/www/discont.png')