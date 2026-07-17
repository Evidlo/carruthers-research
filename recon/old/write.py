#!/usr/bin/env python3
# Per-process shard writer + final serial merge.
# Each worker writes a self-contained zarr shard for its block (e.g. one day).
# After all workers finish, merge_shards() concats them into the final store.

from pathlib import Path
import pickle
import base64
import numpy as np
import xarray as xr
import zarr


def write_shard(shard_path, time, scrafts, ims):
    """Write one block's worth of frames to its own zarr shard.

    Safe to call concurrently from different processes as long as each call
    targets a distinct shard_path.

    Args:
        shard_path (Path): output zarr directory for this block
        time (np.ndarray): shape (n,), datetime64 timestamps
        scrafts (list): length n, SpaceCraft objects (pickled + base64'd here)
        ims (np.ndarray): shape (n, x, y), image stack
    """
    scraft = np.array(
        [base64.b64encode(pickle.dumps(s)).decode() for s in scrafts],
        dtype=object,
    )
    ds = xr.Dataset(
        data_vars={
            'scraft': (['time'], scraft),
            'im': (['time', 'x', 'y'], ims),
        },
        coords=dict(time=(['time'], time)),
    )
    ds.to_zarr(shard_path, mode='w', consolidated=False)


def merge_shards(shard_paths, outpath):
    """Concat per-block shards into the final store along time. Run serially
    after all workers have finished."""
    shard_paths = sorted(shard_paths, key=lambda p: xr.open_zarr(p).time.values[0])
    ds = xr.open_mfdataset(
        shard_paths,
        engine='zarr',
        concat_dim='time',
        combine='nested',
    )
    ds.to_zarr(outpath, mode='w', consolidated=False)
    zarr.consolidate_metadata(str(outpath))


if __name__ == '__main__':
    # Demo: simulate three "days" of WFI data with fake scrafts and zero images.
    # In production each write_shard() call would run inside its own worker process.

    tmpdir = Path('/tmp/claude_write_demo')
    tmpdir.mkdir(exist_ok=True)
    shard_dir = tmpdir / 'shards'
    shard_dir.mkdir(exist_ok=True)

    class FakeSpaceCraft:
        def __init__(self, t):
            self.t = t

    frames_per_day = 48
    im_shape = (64, 64)

    shard_paths = []
    for day in range(3):
        t0 = np.datetime64('2025-01-01') + np.timedelta64(day, 'D')
        time = t0 + np.arange(frames_per_day) * np.timedelta64(30, 'm')
        scrafts = [FakeSpaceCraft(t) for t in time]
        ims = np.zeros((frames_per_day, *im_shape), dtype=np.float32)

        shard_path = shard_dir / f'wfi_day{day:03d}.zarr'
        write_shard(shard_path, time, scrafts, ims)
        shard_paths.append(shard_path)

    final = tmpdir / 'wfi.zarr'
    merge_shards(shard_paths, final)

    ds = xr.open_zarr(final)
    print(ds)
    print('first scraft round-trips:',
          pickle.loads(base64.b64decode(ds.scraft.values[0])).t)
