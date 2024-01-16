#!/usr/bin/env python3
# Generate a large .nc dataset with random data
# See `read_xr.py` for reading dataset
#
#   pip install pandas numpy pqdm

import pandas as pd
import numpy as np
import xarray as xr
import datatree
from pathlib import Path
from pqdm.threads import pqdm

from shutil import rmtree
import os

# data output location
db = Path('database_test')
# delete/recreate data folder before test
rmtree(db, ignore_errors=True)
db.mkdir()
img_a_path = db.joinpath('img_a')
img_b_path = db.joinpath('img_b')
param_path = db.joinpath('param')
img_a_path.mkdir()
img_b_path.mkdir()
param_path.mkdir()

# build dataframe
# num_rows = 16000 # 240GB of data
# num_rows = 1600 # 24GB of data
num_rows = 160 # 2.4 GB of data
dates = pd.date_range('2025-01-01', '2028-01-01', num_rows)

# split dates list into chunks and generate data for each in parallel
chunk_len = 10 # .15GB of data
dates_chunks = []
for n in range(0, len(dates), chunk_len):
    dates_chunks.append(dates[n:n + chunk_len])

def generate_datafile(dates_chunk):
    # Generate a netCDF with some random images/parameters as database columns
    img_a_chunk = np.random.random((len(dates_chunk), 1000, 1000))
    img_a_dates_chunk = dates_chunk
    img_b_chunk = np.random.random((len(dates_chunk), 1000, 1000))
    img_b_dates_chunk = dates_chunk + pd.Timedelta('1d')
    param_chunk = np.random.random(len(dates_chunk))
    param_dates_chunk = dates_chunk + pd.Timedelta('2d')

    ds_img_a = xr.Dataset(
        {"img_a": (["date", "x", "y"], img_a_chunk)},
        coords={"date": img_a_dates_chunk},
    )
    ds_img_a.to_netcdf(img_a_path / img_a_dates_chunk[0].strftime('%Y-%m-%d.nc'))
    ds_img_b = xr.Dataset(
        {"img_b": (["date", "x", "y"], img_b_chunk)},
        coords={"date": img_b_dates_chunk},
    )
    ds_img_b.to_netcdf(img_b_path / img_b_dates_chunk[0].strftime('%Y-%m-%d.nc'))
    ds_param = xr.Dataset(
        {"param": (["date"], param_chunk)},
        coords={"date": param_dates_chunk},
    )
    ds_param.to_netcdf(param_path / param_dates_chunk[0].strftime('%Y-%m-%d.nc'))

# run parallel jobs
result = pqdm(dates_chunks, generate_datafile, n_jobs=32)
