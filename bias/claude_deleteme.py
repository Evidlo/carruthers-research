#!/usr/bin/env python3
import xarray as xr
import numpy as np

# Inspect the STR 0318 netcdf to understand filter values and image ordering
ds = xr.open_dataset('/home/evan/nc/L1A/CARRUTHERS_GCI-NFI_L1A-STR_20260318_v1.0.nc')
print("filter values:", ds['filter'].values)
print("n images:", len(ds.images))
print("image shape:", ds.images.shape)

# What register.py does: filter to LyaN only
ds_lyan = ds.where(ds['filter'] == 'LyaN', drop=True)
print("\nAfter LyaN filter:")
print("n images:", len(ds_lyan.images))
print("filter values:", ds_lyan['filter'].values)

# What generate_images.py does via L1A: check if L1A filters by filter type
from glide.science_data_processing.L1A import L1A
l = L1A(ds)
print("\nL1A images shape:", l.images.shape)
print("L1A im_modes:", l.im_modes)
# Check if there's a filter attribute
if hasattr(l, 'filter'):
    print("L1A filter:", l.filter)

# The key question: what is l.images[3] vs ds_lyan.images[0]?
print("\n--- Comparing images ---")
print("L1A images[3] (generate_images ind=3):", l.images[3].mean() if len(l.images) > 3 else "index 3 doesn't exist")
print("ds_lyan images[0] (register.py first LyaN):", ds_lyan.images[0].mean())
