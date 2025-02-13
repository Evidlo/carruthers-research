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
dates = pd.date_range('2025-01-01', '2028-01-01', 10)
img_a = list(np.random.random((len(dates), 5, 5)))
img_b = list(np.random.random((len(dates), 5, 5)))
param = list(np.random.random(len(dates)))
# make some of the table entries None
# img_a[0] = None
# img_b[1] = None
# param[2] = None

# xarray Dataset - doesn't work
ds = xr.Dataset(
    {
        "img_a": (["date", "x", "y"], img_a),
        "img_b": (["date", "x", "y"], img_b),
        "param": (["date"], param)
    },
    coords={
        "date": dates,
    },
)

# pandas DataFrame - works
df = pd.DataFrame(
    {
        'img_a': img_a,
        'img_b': img_b,
        'param': param
    }
)
