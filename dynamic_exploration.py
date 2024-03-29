#!/usr/bin/env python3
# Evan Widloski - 2023-12-05
# Investigating dynamic data generated by Pratik

from pathlib import Path
from datetime import datetime
from astropy.constants import R_earth
import xarray as xr
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from contexttimer import Timer
from tqdm import tqdm


path = Path('data_2002_ap_3')

def add_time(ds):
    p = Path(ds.encoding['source'])
    date = datetime.strptime(p.stem, 'nH_3D_%m_%d_%Y')
    return ds.expand_dims(date=[date])


ds = xr.open_mfdataset((path / 'output').glob('*.nc'), preprocess=add_time)
r = ds.R[0].data.compute() / R_earth.to('cm').value
e = theta = np.deg2rad(ds.LATGSE[0].data.compute()) + np.pi / 2
a = phi = np.deg2rad(ds.LONGSE[0].data.compute())
ds = ds.assign_coords({
    'r': r,
    'theta': theta,
    'phi': phi,
})
# reorder dimensions as (r, theta, phi)
ds = ds.transpose('date', 'r', 'theta', 'phi')
# wrap first value to end in azimuth
# ds = xr.concat([ds, ds.isel(phi=0)], dim='phi')
# ds.phi.values[-1] += 2 * np.pi
r = ds.coords['r']
e = ds.coords['theta']
a = ds.coords['phi']

from glide.science.model import default_vol, vol2cart, den_sph2cart
from glide.science.common import cart2sph, sph2cart
from glide.science.plotting import save_gif, preview3d, color_negative, preview4d

vol = default_vol(shape=100, size=50)

print('sph2cart...')

d = den_sph2cart(ds.nH, r, e, a, vol)

print('previewing...')

save_gif('/srv/www/test3.gif', preview4d(d), rescale='sequence')

# %% plot

import dech
dech.Page([
    [
        dech.Figure("Unscaled 2002 Data", dech.Img(preview4d(d), animation=True, rescale='frame')),
        dech.Figure("Scaled 2002 Data", dech.Img(preview4d(d), animation=True, rescale='sequence')),
    ],
    [
        dech.Paragraph(f"{vol}")
    ]
]).save('/srv/www/display/dynamic_densities.html')