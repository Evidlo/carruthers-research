#!/usr/bin/env python3

import pandas as pd
import xarray as xr

times = pd.date_range("2000-01-01 00:35", periods=8, freq="6H")
da = xr.DataArray(coords={"time": None}, dims=["time"])
ds = da.to_dataset(name="foo")
# ds2 = xr.Dataset(
#     {
#         "foo": None,
#     },
#     coords={
#         "time": None
#     },
# )


ZARR_PATH = "test.zarr"
ds.isel(time=[0]).to_zarr(ZARR_PATH, mode="w")
ds.isel(time=slice(1, None)).to_zarr(ZARR_PATH, append_dim="time")

ds_loaded = xr.open_dataset(ZARR_PATH, engine="zarr")
