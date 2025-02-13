#!/usr/bin/env python3
# Read a large dataset and perform some date binning tests
# See `build_xr.py` for building dataset
#
#   pip install xarray pandas numpy contexttimer

import xarray as xr
import numpy as np
import pandas as pd
from contexttimer import Timer

with Timer() as t:
    data = xr.open_mfdataset('xarray_test/*.nc', chunks={'date': 10})
print(f'Open dataset time: {t.elapsed}')

with Timer() as t:
    # generate arbitrary date bins and take means within each group
    # here I do monthly bins (MS)
    # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    date_bins = pd.date_range('2025-01-01', '2028-01-01', freq='MS')
    data_binned_means = data.groupby_bins("date", date_bins).mean()
    print(data_binned_means.img_a.shape)
print(f'Binned time: {t.elapsed}')

with Timer() as t:
    data_roll = data.rolling(date=10).mean()
print(f'Rolling time: {t.elapsed}')

"""
[evan@tikhonov databases] python read_xr.py
Open dataset time: 10.777960497885942
Resample time: 2.753757775761187
(19, 1000, 1000)
Binned time: 0.793106866069138
"""
