#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xarray as xr

"""
Want to create a table with 3 columns
+--------------+---------------+---------------+---------------+
|Date          |img_a          |img_b          |param          |
+--------------+---------------+---------------+---------------+
|2023-01-01T...|[[1.234, ...]  |None           |None           |
|              |               |               |               |
+--------------+---------------+---------------+---------------+
|...           |...            |...            |...            |
+--------------+---------------+---------------+---------------+
|2028-01-01T   |None           |[[1.234, ...]] |8.463          |
+--------------+---------------+---------------+---------------+

"""

# generate test data
dates_a = pd.date_range('2025-01-01', '2028-01-01', 10)
dates_b = pd.date_range('2025-02-01', '2028-01-03', 10)
dates_param = pd.date_range('2025-02-08', '2028-01-02', 10)

dates_param = dates_b = dates_a

img_a = list(np.random.random((len(dates_a), 5, 5)))
img_b = list(np.random.random((len(dates_b), 5, 5)))
param = list(np.random.random(len(dates_param)))

x = xr.DataArray(
    img_a,
    dims=['time', 'x', 'y'],
    name='img_a'
)
y = xr.DataArray(
    img_b,
    dims=['time', 'x', 'y'],
    name='img_b'
)
z = xr.DataArray(
    param,
    dims=['time'],
    name='param'
)

ds2 = xr.merge([x, y, z])

# date_bins = pd.date_range('2025-01-01', '2028-01-01', freq='MS')
# data_binned_means = data2.groupby_bins("time", date_bins).mean()
# print(data_binned_means.img_a.shape)
