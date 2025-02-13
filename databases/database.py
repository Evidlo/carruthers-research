#!/usr/bin/env python3

import xarray as xr
import pandas as pd
from pathlib import Path
from collections import UserDict

class Database(UserDict):
    """A DIY implementation of a datatree"""

    def __init__(self, properties):
        """Initialize DataArbol class

        Args:
            properties (list[str]): list of columns in this database
        """
        self.properties = properties
        self.data = {}

    def open(self, root, chunks={'date': 10}):
        """Open hierarchal tree of netCDF files and load into database

        Args:
            root (pathlib.Path or str): path to database root dir
            chunks (dict): size of xarray chunks
        """
        root = Path(root)

        for prop in self.properties:
            self.data[prop] = xr.open_mfdataset((root / f'{prop}').glob('*.nc'), chunks=chunks)

    def save(self, root, chunks={'date': 'W'}, mode='a'):
        """Save netCDF files to hierarchal tree

        Args:
            root (pathlib.Path or str): path to database root dir
            groupby (str): time duration of each file
            chunks (dict): file chunking size.  default is weekly
        """
        root = Path(root)

        for prop, dataset in self.data.items():

            # make the folder for this property
            propdir = root / prop
            propdir.mkdir(parents=True)

            # split dataset into time chunks
            dates, datasets = zip(*dataset.resample(chunks))
            paths = [propdir / pd.to_datetime(d).strftime('%Y-%m-%d.nc') for d in dates]
            xr.save_mfdataset(datasets, paths, mode=mode)

    def __repr__(self):
        return f'Database({self.properties})'

if __name__ == "__main__":
    d = Database(['img_a', 'img_b', 'param'])
    d.open('database_test')
