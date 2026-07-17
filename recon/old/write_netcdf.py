#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation as rot

from glide.science_data_processing.L1 import spacecrafts_to_L1, get_spacecraft


def write_block(outdir, time, scrafts, ims):
    """Write one block's frames to <outdir>/<first_timestamp>.nc.

    Args:
        outdir (Path): channel directory, e.g. Path('wfi')
        time (np.ndarray): shape (n,), datetime64 timestamps
        scrafts (list): length n, SpaceCraft objects
        ims (np.ndarray): shape (n, x, y), image stack
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        data_vars={'im': (['time', 'x', 'y'], ims)},
        coords=dict(time=(['time'], time)),
    )
    ds = spacecrafts_to_L1(ds, scrafts)

    fname = str(time[0].astype('datetime64[s]')) + '.nc'
    ds.to_netcdf(outdir / fname)


if __name__ == '__main__':
    from glide.common_components import spacecraft as sc
    from glide.common_components.camera import CameraWFI, CameraNFI

    root = Path('/tmp/pseudo_netcdf_demo')

    def make_scrafts(channel, time):
        cam_factory = CameraWFI if channel == 'wfi' else CameraNFI
        scrafts = []
        for t in time:
            cam = cam_factory()
            scraft = sc.SpaceCraft(date=t, sensors=[cam], ephem_file=None)
            scraft.position = np.array([1.5e8, 0, 0])
            scraft.orientation = rot.identity().as_matrix()
            scrafts.append(scraft)
        return scrafts

    frames_per_day = 48
    im_shape = (64, 64)

    for channel in ('wfi', 'nfi'):
        for day in range(3):
            t0 = np.datetime64('2025-01-01') + np.timedelta64(day, 'D')
            time = t0 + np.arange(frames_per_day) * np.timedelta64(30, 'm')
            scrafts = make_scrafts(channel, time)
            ims = np.zeros((frames_per_day, *im_shape), dtype=np.float32)

            # ALEX - your workers call this function
            write_block(root / channel, time, scrafts, ims)

    # all data is lazily loaded here
    ds = xr.open_mfdataset(
        str(root / 'wfi' / '*.nc'),
        combine='nested',
        concat_dim='time',
    )
    print(ds)
    scrafts = get_spacecraft(ds)
    print(f'round-tripped {len(scrafts)} scrafts; first date = {scrafts[0].date}')
