#!/usr/bin/env python3

from glide.science_data_processing.L1A import L1A
import xarray as xr
from pathlib import Path
import numpy as np

from common import mean_bias, save

# date and index of image from that day
# date = '2026[FIXME]'; ind = 3
date = '20260305'; ind = 3

input_dir = Path('/home/evan/mnt/carrdata/products/L1A')
output_dir = Path(f'images_{date}/')

output_dir.mkdir(exist_ok=True)


def export(l, mask=None):
    """Export L0 and L1A to single image files from L1A dataset

    Writes each image to e.g. `images_20260111/dark_nfi_l0.pkl`

    Args:
        l (L1A): dataset loaded from netcdf
        mask (ndarray(bool)): mask out pixels (e.g. hot pixels)

    Returns:
        im_l0 (ndarray)
        im_l1a (ndarray)
    """
    # try to get the desired index, otherwise get the first
    i = ind if ind < len(l.images) else 0

    # --- level 1A ---
    # scale to DN/s
    im_l1a = l.images[i] / l.t_int[i]
    path_l1a = (output_dir / f'{l.im_modes[i]}_{l.channel.lower()}').with_suffix('.pkl')
    # save in dictionary because `save` doesn't support masked arrays
    save(path_l1a, np.ma.array(im_l1a, mask=mask))

    print(f'  → {path_l1a}')

    # --- level 0 ---
    # add bias back in
    im_l0 = l.images[i] + l.bias[i] * l.n_frames[i, None, None]
    # scale to DN/s
    im_l0 = im_l0 / l.t_int[i]
    path_l0 = (output_dir / f'{l.im_modes[i]}_{l.channel.lower()}_l0').with_suffix('.pkl')
    # save in dictionary because `save` doesn't support masked arrays
    save(path_l0, np.ma.array(im_l0, mask=mask))

    print(f'  → {path_l0}')

    return im_l0, im_l1a

for chan in ('WFI', 'NFI'):
    dark = set(input_dir.glob(f'*{chan}*DRK*{date}*.nc'))
    images = set(input_dir.glob(f'*{chan}*{date}*.nc'))

    print(len(dark))
    #assert len(dark) == 1, "There is more than 1 dark!  What do?"
    path = list(dark)[0]
    print(path.stem)
    dark_dataset = L1A(xr.open_dataset(path))
    dark_l0, dark_l1a = export(dark_dataset)
    # find hot pixel mask
    hotmask = (dark_l0 - mean_bias(dark_l0)) > 20

    # iterate over all non-dark images
    for path in images - dark:
        print(path.stem)
        dataset = L1A(xr.open_dataset(path))

        export(dataset, hotmask)
