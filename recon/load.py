#!/usr/bin/env python3
# Load wfi/nfi frames nearest to a set of sample dates from per-day L1C NetCDFs.

from pathlib import Path
import numpy as np
import xarray as xr

from glide.science_data_processing.L1 import get_spacecraft

def load(datapath, channel, dates, extra_scaling=1):
    """Open per-day L1C NetCDFs for `channel` ('NFI'|'WFI') under `datapath`,
    lazily concatenate along time, select frames nearest `dates`, and build
    SpaceCraft objects from the geometry fields.

    Returns an xarray Dataset with `l1c_ims` (the l1c images) and `scraft` (one
    SpaceCraft per time)."""
    paths = sorted(Path(datapath).glob(f'*{channel.upper()}*.nc'))
    ds = xr.open_mfdataset(
        paths, combine='by_coords',
        chunks={'time': 1}
    ).sel(time=dates, method='nearest')
    ds['scraft'] = ('time', get_spacecraft(ds))

    # convert to Rayleighs
    if channel == 'NFI':
        ds['l1c_ims'] *= 1.111e-5
    elif channel == 'WFI':
        ds['l1c_ims'] *= 1.451e-5
    else:
        raise ValueError(f"Invalid channel {channel}")

    ds['l1c_ims'] *= extra_scaling # FIXME: cross cal isue

    return ds


if __name__ == '__main__':

    # 100 uniformly spaced sample times between two dates
    dates = np.linspace(
        np.datetime64('2026-03-01').astype('datetime64[ns]').astype(float),
        np.datetime64('2026-04-01').astype('datetime64[ns]').astype(float),
        100,
    ).astype('datetime64[ns]')

    datapath = Path('/home/alex/carruthers/pseudo_l1c_data/')

    wfi = load(datapath, 'WFI', dates)
    nfi = load(datapath, 'NFI', dates)
