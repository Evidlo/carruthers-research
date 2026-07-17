#!/usr/bin/env python3
# WFI/NFI cross-calibration check: median binned radiance ratio at matched
# tangent-point radii in the overlap band, after the load.py correction factors.

from glide.science.forward_sph import ScienceGeomFast, R_earth
from glide.science_data_processing.L1 import get_spacecraft

from domrep import *
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

datapath = Path('/home/alex/carruthers/pseudo_l1c_data/')

periods = {
    'january': ('2026-01-19', '2026-01-21'),
    'quiet': ('2026-03-01', '2026-03-15'),
    'march': ('2026-03-20', '2026-03-22'),
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


def tp_radius(g):
    """Tangent point radius (Re) of each science pixel LOS.  (nrad, ntheta)"""
    pos, rays = g.pos.cpu().numpy(), g.rays.cpu().numpy()
    proj = rays @ pos
    return np.linalg.norm(pos[None, None] + rays * np.maximum(-proj, 0)[..., None], axis=-1)


def ratio_data(nfi_sc, wfi_sc, nfi_im, wfi_im):
    """Cross-cal comparison in the overlap band:
        (tp, ratio): WFI pixels over NFI azimuth-median interpolated to their TP radius
        im: interpolated-WFI / NFI per NFI science pixel  (nrad, ntheta)"""
    (ntp, nb, nm), (wtp, wb, wm) = [
        (tp_radius(g), np.where(g.mask.numpy(), g.bin(im), np.nan), g.mask.numpy())
        for s, im in [(nfi_sc, nfi_im), (wfi_sc, wfi_im)]
        for g in [ScienceGeomFast(s, (100, 50))]
    ]
    # azimuth-median radiance per ring vs ring TP radius, each camera
    (nrtp, nrmed), (wrtp, wrmed) = [
        (np.nanmedian(tp, axis=1), np.nanmedian(b, axis=1)) for tp, b in [(ntp, nb), (wtp, wb)]
    ]
    nk, wk = np.isfinite(nrmed), np.isfinite(wrmed)

    sel = wm & np.isfinite(wb) & (wtp >= max(band[0], nrtp[nk].min())) \
        & (wtp <= min(band[1], nrtp[nk].max()))
    scat = wtp[sel], wb[sel] / np.interp(wtp[sel], nrtp[nk], nrmed[nk])

    # rectangular: interpolated WFI curve over each native NFI pixel
    rays = nfi_sc.los().reshape(3, *nfi_im.shape)
    rays = np.moveaxis(rays, 0, 2)
    pos = nfi_sc.position_gse[:, 0] / R_earth.to('km').value
    proj = rays @ pos
    tp = np.linalg.norm(pos[None, None] + rays * np.maximum(-proj, 0)[..., None], axis=-1)
    inband = nfi_sc.sensor.spec.mask_fov & np.isfinite(nfi_im) \
        & (tp >= max(band[0], wrtp[wk].min())) & (tp <= min(band[1], wrtp[wk].max()))
    im = np.where(inband, np.interp(tp, wrtp[wk], wrmed[wk]) / nfi_im, np.nan)
    return scat, im


with document('NFI/WFI Cross Calibration') as doc:
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

        ratios, ims2d, labels = [], [], []
        for i in sorted(keep):
            scat, im = ratio_data(
                nfi.scraft.values[i], wfi.scraft.values[i],
                nfi.l1c_ims.values[i], wfi.l1c_ims.values[i],
            )
            ratios.append(scat)
            ims2d.append(im)
            labels.append(str(nfi.time.values[i])[:16])

        tags.h1(f'{name}: {start} … {end}')
        with itemgrid(length=3):
            plt.close('all')


            with caption('interpolated WFI azimuth-median / NFI pixels (2D)'):
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
                ax.plot([np.datetime64(l) for l in labels], [np.median(rat) for _, rat in ratios], 'o-')
                ax.axhline(1, color='k', lw=0.5)
                ax.set_ylabel('median WFI / NFI')
                fig.autofmt_xdate()
                plot(fig, **figset)

    tags.h1("Source Code")
    tags.code(tags.pre(open('cross_cal.py').read()))

outfile = Path('/www/cross/index.html')
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text(doc.render())
print(f'Saved to {outfile}')
