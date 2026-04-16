#!/usr/bin/env python3

from astropy.constants import R_earth
from cachalot import Cache
from collections.abc import Sequence
from collections import namedtuple
from datetime import datetime
from importlib.resources import files
import math
import netCDF4
import numpy as np
from numpy import newaxis as na
from pathlib import Path
from scipy.special import sph_harm, hyp2f1
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import shift, fourier_shift
from scipy.signal import resample
import torch as t
from tqdm import tqdm
import xarray as xr

from glide.science.model_sph import den_sph2sph

from glide.science.common import center2edge, affine
from glide.science.decorators import handle_torch
from sph_raytracer.model import Model
from sph_raytracer.geometry import SphericalGrid as SphericalGrid


class _DataModel(Model):
    """A parent class to be inherited from for models which load density from a file.
    """

    _path = None

    def __init__(self, grid=None, device=None, path=None, fill_value=float('nan'), method='linear', *args, **kwargs):

        """
        Args:
            grid (SphericalGrid or None): if given, interpolate loaded data onto to this grid
            device (None, str, or torch.device): device for returned array
            path (pathlib.Path): path to data dir/file
            fill_value (float or None): fill value for voxels outside of grid
                See `den_sph2sph`
            method (str): interpolation method to use when interpolating onto new grid
            *args (list): custom model args
            **kwargs (dict): custom model kwargs

        If `grid` is None, the grid is determined by the underlying data.  Otherwise the
        data will be interpolated onto the supplied grid"""

        # let subclasses define a default path
        if path is None:
            path = self._path

        # load original data and grid
        orig_density, orig_grid = self.load(path, *args, **kwargs)

        # use data as-is if no grid is given
        if grid is None:
            density, grid = orig_density, orig_grid

        # otherwise, interpolate to new grid shape
        else:
            # handle static grid on dynamic model by taking first time bin
            if orig_grid.dynamic and not grid.dynamic:
                orig_density = orig_density[0]
            density = den_sph2sph(orig_density, orig_grid, grid, fill_value=fill_value, method=method)

        # convert from numpy to torch
        density = t.from_numpy(density).to(device=device)
        orig_density = t.from_numpy(orig_density).to(device=device)

        self.orig_density, self.orig_grid = orig_density, orig_grid
        self.density, self.grid = density, grid
        self.path = path
        self.device = device

    @property
    def coeffs_shape(self):
        return []

    def __call__(self, *args, **kwargs):
        return self.density

    def load(self, path, *args, **kwargs):
        """Load the data from disk

        Args:
            path (pathlib.Path): path to data dir/file
            *args (list): custom model args
            **kwargs (dict): custom model kwargs

        Returns:
            density (ndarray): density array of shape `grid.shape`
            grid (SphericalGrid): data sample points
        """
        raise NotImplementedError

class DataDynamicModel(_DataModel):
    """Load a dynamic (4D) density dataset from disk
    """

    def load(self, path, num_times=None, freq=None, window=None, offset=0, num_chans=1, **kwargs):
        """Load the data from disk

        Args:
            path (pathlib.Path): path to data dir/file
            num_times (int): length of dynamic density time dimension (mutually exclusive with `freq`)
            freq (timedelta64): time between samples (mutually exclusive with `num_times`)
            window (int): observation window length (days)
            offset (int): window start offset (days from first timestamp)
            # FIXME: ugly?
            num_chans (int): number of channels
            *args (list): custom model args
            **kwargs (dict): custom model kwargs

        Returns:
            density (ndarray): density array of shape `grid.shape`
            grid (SphericalGrid): data sample points
        """

        self.ds = xr.open_dataset(path).density
        density = self.ds

        if any(x is not None for x in (window, freq, num_times, offset)):
            start_time = density.time[0].values + np.timedelta64(offset, 'D')
            if window is None:
                end_time = density.time[-1].values
            else:
                end_time = start_time + np.timedelta64(window, 'D')

            times = xr.date_range(start_time, end_time, periods=num_times, freq=freq)
            times = np.repeat(times, num_chans)

            density = density.interp(time=times, **kwargs)

        grid = SphericalGrid(
            shape=density.shape,
            t=density.time.values.astype('datetime64[s]').astype(int),
            timeunit='s',
            r_b=center2edge(density.r),
            e_b=center2edge(density.e),
            a_b=center2edge(density.a),
        )

        return density.data, grid

    def __repr__(self):
        if self.orig_grid.shape.t > 1:
            datestr = f'[{self.orig_grid.nptime[0]}...{self.orig_grid.nptime[-1]}]'
        else:
            datestr = f'[{self.orig_grid.nptime[0]}]'

        return f'{self.__class__.__name__}({self.grid.shape}, {datestr})'


class MSISModel(DataDynamicModel):
    """MATE NRLMSIS dataset"""

    _path = files('glide') / 'science/data_files/mate_MSIS_2014174-200.nc'


class TIMEGCMModel(DataDynamicModel):
    """MATE NRLMSIS dataset"""

    _path = files('glide') / 'science/data_files/mate_TIMEGCM_2008164-174.nc'

class Pratik25Model(DataDynamicModel):
    """New 2025 dataset that includes 6 weeks of quiet time before two weeks of the Spring
    Equinox Storm (Mar 15 to 28) and 6 weeks of quiet after storm recovery.

    Spans (Feb 1 to May 16, 2002) with 1-hour UT steps.
    """

    _path = '/home/evan/carr/recon_bug/pratik_storm_2002_f107a_210.nc'

def Pratik25StormModel(*args, offset=0, window=13, freq='1h', **kwargs):
    # storm starts in 2nd week
    storm_begin = 2 * 7
    return Pratik25Model(*args, offset=storm_begin + offset, window=window, freq=freq, **kwargs)