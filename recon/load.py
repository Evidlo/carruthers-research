#!/usr/bin/env python3
# Load wfi/nfi frames nearest to a set of sample dates from per-day L1C NetCDFs.

from pathlib import Path
import re
import numpy as np
import xarray as xr

from glide.science_data_processing.L1 import get_spacecraft

def datefilter(paths, dates):
    """Keep per-day files whose filename YYYYMMDD falls within `dates` ±1 day,
    at most one per day.  A day present at several versions (e.g. 20260510 as
    both v1.0 and v1.1) would otherwise be concatenated twice, giving a
    non-unique time index that .sel(method='nearest') cannot reindex."""
    lo, hi = dates.min().astype('datetime64[D]') - 1, dates.max().astype('datetime64[D]') + 1
    fdate = lambda p: np.datetime64('{}-{}-{}'.format(*re.search(r'_(\d{4})(\d{2})(\d{2})_', p.name).groups()))
    fver = lambda p: tuple(map(int, re.search(r'_v(\d+)\.(\d+)\.nc$', p.name).groups()))
    best = {}
    for p in sorted(paths, key=fver):  # ascending version, last write wins
        if lo <= fdate(p) <= hi:
            best[fdate(p)] = p
    return [best[k] for k in sorted(best)]

def load(datapath, dates, extra_scaling=1, tolerance=np.timedelta64(30, 'm')):
    """Open per-day NFI and WFI L1C NetCDFs under `datapath`/L1C/L1C, lazily
    concatenate along time, select frames nearest `dates`, and build SpaceCraft
    objects from the geometry fields.

    Returns `(nfi, wfi, dates)`, each an xarray Dataset with `images`
    (Rayleighs), `scraft` (registered L1C SpaceCraft per time) and `scraft_l1a`
    (pre-registration L1A-SCI SpaceCraft per time, for masking detector-oriented
    bad pixels), plus the surviving requested dates. Dates whose nearest NFI or
    WFI frame is further away than `tolerance` are dropped from all three."""
    datapath = Path(datapath)
    dates = np.asarray(dates, dtype='datetime64[ns]')

    out = {}
    for channel in ('NFI', 'WFI'):
        # nested/positional concat: per-day files are disjoint and filename-sorted,
        # so by_coords alignment (compat/join comparisons across files) is pure
        # overhead — ~4 s/file vs ~30 ms/file, identical result
        paths = datefilter(sorted((datapath / 'L1C/L1C').glob(f'*{channel}*.nc')), dates)
        ds = xr.open_mfdataset(
            paths, combine='nested', concat_dim='time', data_vars='all',
            coords='minimal', compat='override', join='override',
            chunks={"time": 1},
        ).sel(time=dates, method='nearest')
        ds['scraft'] = ('time', get_spacecraft(ds))

        # pre-registration spacecraft from the matching L1A-SCI frames
        paths = datefilter(sorted((datapath / 'L1A/L1A-SCI').glob(f'*{channel}*.nc')), dates)
        l1a = xr.open_mfdataset(
            paths, combine='nested', concat_dim='time', data_vars='all',
            coords='minimal', compat='override', join='override',
            chunks={"time": 1},
            drop_variables=['images', 'nominal_bias', 'residual_bias'],
        ).sel(time=ds.time, method='nearest')
        ds['scraft_l1a'] = ('time', get_spacecraft(l1a))

        ds['images'] *= extra_scaling # FIXME: cross cal isue
        out[channel] = ds

    nfi, wfi = out['NFI'], out['WFI']
    # nearest-selection can land far from the request when frames are missing
    keep = (np.abs(nfi.time.values - dates) <= tolerance) \
         & (np.abs(wfi.time.values - dates) <= tolerance)

    return nfi.isel(time=keep), wfi.isel(time=keep), dates[keep]


if __name__ == '__main__':

    # 100 uniformly spaced sample times between two dates
    dates = np.linspace(
        np.datetime64('2026-03-01').astype('datetime64[ns]').astype(float),
        np.datetime64('2026-04-01').astype('datetime64[ns]').astype(float),
        100,
    ).astype('datetime64[ns]')

    datapath = Path('/data-products')

    nfi, wfi, dates = load(datapath, dates)
