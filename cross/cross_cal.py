#!/usr/bin/env python3
# WFI/NFI cross-calibration check: WFI is resampled onto each native NFI pixel
# by averaging the k nearest WFI pixels in LOS direction (KD-tree on the GSE
# LOS unit vectors — nearest on the sky), so real exospheric structure divides
# out of the ratio and detector scale/roll/mapping need no explicit handling.
# Earlier azimuth-median / science-binned versions: cross_cal_old.py.

from glide.science.forward_sph import R_earth
from glide.science_data_processing.L1 import get_spacecraft

from domrep import *
from pathlib import Path
from scipy.spatial import cKDTree

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

datapath = Path('/home/alex/carruthers/pseudo_l1c_data/')

periods = {
    'january': ('2026-01-19', '2026-01-21'),
    # 'quiet': ('2026-03-01', '2026-03-15'),
    # 'march': ('2026-03-20', '2026-03-22'),
}

# Alex's counts→Rayleighs factors; only the relative NFI/WFI factor matters to the ratios
fac = {'NFI': 1.111e-5, 'WFI': 1.4515e-5}

# overlap band of NFI/WFI tangent point radii
band = (3.2, 6.0)


def load(datapath, channel, dates):
    """Per-day L1C NetCDFs for `channel` nearest `dates`, in Rayleighs, with
    SpaceCraft objects built from the geometry fields."""
    paths = sorted(Path(datapath).glob(f'*{channel.upper()}*.nc'))
    ds = xr.open_mfdataset(paths, combine='by_coords', chunks={'time': 1}) \
        .sel(time=dates, method='nearest')
    ds['scraft'] = ('time', get_spacecraft(ds))
    ds['l1c_ims'] *= fac[channel]
    return ds


def tp_radius(pos, rays):
    """Tangent point radius (Re) of LOS unit vectors.  rays (..., 3)"""
    proj = rays @ pos
    return np.linalg.norm(pos + rays * np.maximum(-proj, 0)[..., None], axis=-1)


def unit_los(sc, shape):
    """Native-pixel LOS unit vectors in GSE.  (npix, npix, 3)"""
    rays = np.moveaxis(sc.los().reshape(3, *shape), 0, 2)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def ratio_im(nfi_sc, wfi_sc, nfi_im, wfi_im, k=25):
    """interpolated-WFI / NFI per native NFI pixel in the overlap band, NaN
    outside.  WFI value = mean of the k nearest WFI pixels by LOS direction."""
    pos = nfi_sc.position_gse[:, 0] / R_earth.to('km').value
    nrays = unit_los(nfi_sc, nfi_im.shape)
    wrays = unit_los(wfi_sc, wfi_im.shape)
    wvals = np.where(wfi_sc.sensor.spec.mask_fov, wfi_im, np.nan).ravel()

    tp = tp_radius(pos, nrays)
    inband = nfi_sc.sensor.spec.mask_fov & np.isfinite(nfi_im) \
        & (tp >= band[0]) & (tp <= band[1])

    _, idx = cKDTree(wrays.reshape(-1, 3)).query(nrays[inband], k=k, workers=-1)
    im = np.full(nfi_im.shape, np.nan)
    im[inband] = np.nanmean(wvals[idx], axis=-1) / nfi_im[inband]
    return im


with document('NFI/WFI Cross Calibration (2D interpolation)') as doc:
    figset = {'height': 350}

    for name, (start, end) in periods.items():
        dates = np.linspace(
            np.datetime64(start).astype('datetime64[ns]').astype(float),
            np.datetime64(end).astype('datetime64[ns]').astype(float),
            30,
        ).astype('datetime64[ns]')
        nfi, wfi = load(datapath, 'NFI', dates), load(datapath, 'WFI', dates)

        # nearest-selection duplicates frames on short ranges
        _, keep = np.unique(nfi.time.values, return_index=True)

        ims2d, labels = [], []
        for i in sorted(keep):
            ims2d.append(ratio_im(
                nfi.scraft.values[i], wfi.scraft.values[i],
                nfi.l1c_ims.values[i], wfi.l1c_ims.values[i],
            ))
            labels.append(str(nfi.time.values[i])[:16])

        tags.h1(f'{name}: {start} … {end}')
        with itemgrid(length=3):
            plt.close('all')


            with caption('interpolated WFI / NFI pixels (2D sky interpolation)'):
                with slider(labels=labels):
                    for im, label in zip(ims2d, labels):
                        fig, ax = plt.subplots()
                        h = ax.imshow(im, clim=(1, 2))
                        fig.colorbar(h, label='interp WFI / NFI')
                        ax.set_title(label)
                        plot(fig, **figset)
                        plt.close(fig)

            with caption('Overlap-band median ratio vs date'):
                fig, ax = plt.subplots()
                ax.plot([np.datetime64(l) for l in labels], [np.nanmedian(im) for im in ims2d], 'o-')
                ax.axhline(1, color='k', lw=0.5)
                ax.set_ylabel('median WFI / NFI')
                fig.autofmt_xdate()
                plot(fig, **figset)

    tags.h1("Source Code")
    tags.code(tags.pre(open('cross_cal.py').read()))

outfile = Path('/www/cross/cross_cal.html')
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text(doc.render())
print(f'Saved to {outfile}')
