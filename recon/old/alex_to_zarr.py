#!/usr/bin/env python3
# Convert Alex npz files to zarr store

from pathlib import Path
import pickle
import base64
import warnings
import numpy as np
from tqdm import tqdm

import zarr
import xarray as xr

def npz_to_zarr(paths, outpath):
    """Convert npz files to zarr store

    Args:
        paths (list[Path]): input npz files
        outpath (Path): zarr store location
    """
    for i, path in tqdm(enumerate(paths)):
        npz = np.load(path, allow_pickle=True)
        # pickle each SpaceCraft, then base64 so it stores as a variable-length
        # UTF-8 string (speced in zarr v3, no truncation on append). Decode +
        # unpickle on read: pickle.loads(base64.b64decode(s)).
        scraft = np.array(
            [base64.b64encode(pickle.dumps(s)).decode() for s in npz['scrafts']],
            dtype=object,
        )
        ds = xr.Dataset(
            data_vars={
                'scraft': (['time'], scraft),
                'im': (['time', 'x', 'y'], npz['l1c_ims']),
            },
            coords=dict(
                time=(['time'], npz['time']),
            ),
        )

        if i == 0:
            ds.to_zarr(outpath, mode='w', consolidated=False)
        else:
            ds.to_zarr(outpath, mode='a', append_dim='time', consolidated=False)

    # consolidate metadata once so default xr.open_dataset is fast
    zarr.consolidate_metadata(str(outpath))


datapath = Path('/home/alex/carruthers/pseudo_l1c_data/')
wfipaths = sorted(datapath.glob('*WFI*.npz'))
nfipaths = sorted(datapath.glob('*NFI*.npz'))

npz_to_zarr(wfipaths, Path('wfi.zarr'))
npz_to_zarr(nfipaths, Path('nfi.zarr'))