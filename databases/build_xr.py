#!/usr/bin/env python3
# Generate a large .nc dataset with random data
# See `read_xr.py` for reading dataset
#
#   pip install pandas numpy pqdm

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from pqdm.processes import pqdm

from shutil import rmtree
import os

# data output location
db = Path('xarray_test')
# delete/recreate data folder before test
rmtree(db, ignore_errors=True)
db.mkdir()

# build dataframe
num_rows = 16000 # 240GB of data
# num_rows = 1600 # 24GB of data
# num_rows = 160 # 2.4 GB of data
dates = pd.date_range('2025-01-01', '2028-01-01', num_rows)

# split dates list into chunks and generate data for each in parallel
chunk_len = 10 # .15GB of data
dates_chunks = []
for n in range(0, len(dates), chunk_len):
    dates_chunks.append(dates[n:n + chunk_len])

def generate_datafile(dates_chunk):
    """Generate a netCDF with some random images/parameters for each date"""
    img_a_chunk = list(np.random.random((len(dates_chunk), 1000, 1000)))
    img_b_chunk = list(np.random.random((len(dates_chunk), 1000, 1000)))
    param_chunk = list(np.random.random(len(dates_chunk)))

    ds = xr.Dataset(
        {
            "img_a": (["date", "x", "y"], img_a_chunk),
            "img_b": (["date", "x", "y"], img_b_chunk),
            "param": (["date"], param_chunk)
        },
        coords={
            "date": dates_chunk,
        },
    )
    # use first date in this chunk for filename
    ds.to_netcdf(db.joinpath(f'{dates_chunk[0].isoformat()}.nc'))

# run parallel jobs
result = pqdm(dates_chunks, generate_datafile, n_jobs=16)
