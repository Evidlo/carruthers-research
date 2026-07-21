#!/usr/bin/env python3
# WFI/NFI cross-calibration check: WFI is resampled onto each native NFI pixel
# by averaging the k nearest WFI pixels in LOS direction (KD-tree on the GSE
# LOS unit vectors — nearest on the sky), so real exospheric structure divides
# out of the ratio and detector scale/roll/mapping need no explicit handling.
# Earlier azimuth-median / science-binned versions: cross_cal_old.py.

from glide.science.forward_sph import R_earth
from load import load

from domrep import *
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial import cKDTree

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

datapath = Path('/data-products')

periods = {
    'january': ('2026-01-01', '2026-02-01'),
    'march': ('2026-03-01', '2026-04-01'),
}

# overlap band of NFI/WFI tangent point radii
band = (3.2, 6.0)


def tp_radius(pos, rays):
    """Tangent point radius (Re) of LOS unit vectors.  rays (..., 3)"""
    proj = rays @ pos
    return np.linalg.norm(pos + rays * np.maximum(-proj, 0)[..., None], axis=-1)


def unit_los(sc, shape):
    """Native-pixel LOS unit vectors in GSE.  (npix, npix, 3)"""
    rays = np.moveaxis(sc.los().reshape(3, *shape), 0, 2)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def ratio_im(nfi_sc, wfi_sc, nfi_im, wfi_im, k=25):
    """(ratio, interp-WFI, NFI) per native NFI pixel in the overlap band, NaN
    outside.  WFI value = mean of the k nearest WFI pixels by LOS direction."""
    pos = nfi_sc.position_gse[:, 0] / R_earth.to('km').value
    nrays = unit_los(nfi_sc, nfi_im.shape)
    wrays = unit_los(wfi_sc, wfi_im.shape)
    wvals = np.where(wfi_sc.sensor.spec.mask_fov, wfi_im, np.nan).ravel()

    tp = tp_radius(pos, nrays)
    inband = nfi_sc.sensor.spec.mask_fov & np.isfinite(nfi_im) \
        & (tp >= band[0]) & (tp <= band[1])

    _, idx = cKDTree(wrays.reshape(-1, 3)).query(nrays[inband], k=k, workers=4)
    im, num, den = (np.full(nfi_im.shape, np.nan) for _ in range(3))
    num[inband] = np.nanmean(wvals[idx], axis=-1)
    den[inband] = nfi_im[inband]
    im[inband] = num[inband] / den[inband]
    return im, num, den


def up_axes(nfi_sc, *scs):
    """Detector up-axis (displayed up, decreasing row) of each spacecraft in
    `scs` as (right, up) components in `nfi_sc`'s display frame.  Pass
    pre-registration (L1A) spacecraft to see true detector roll in the
    registered L1C frame."""
    rays = lambda sc: unit_los(sc, (sc.sensor.npix, sc.sensor.npix))
    nrays = rays(nfi_sc)
    c = nrays.shape[0] // 2
    ex = nrays[c, c + 1] - nrays[c, c - 1]
    ey = nrays[c - 1, c] - nrays[c + 1, c]
    ex, ey = ex / np.linalg.norm(ex), ey / np.linalg.norm(ey)

    def comp(r):
        m = r.shape[0] // 2
        u = r[m - 1, m] - r[m + 1, m]
        u = u / np.linalg.norm(u)
        return np.array([u @ ex, u @ ey])

    return [comp(rays(sc)) for sc in scs]


with document('NFI/WFI Cross Calibration (2D interpolation)') as doc:
    figset = {'height': 350}

    for name, (start, end) in periods.items():
        dates = np.arange(
            np.datetime64(start), np.datetime64(end), np.timedelta64(6, 'h'),
        ).astype('datetime64[ns]')
        nfi, wfi, dates = load(datapath, dates)

        # nearest-selection duplicates frames on short ranges
        _, keep = np.unique(nfi.time.values, return_index=True)

        # joblib/loky spawns fresh workers — fork-based pools (pqdm) OOM here
        # from copy-on-write duplication of the multi-GB parent
        ims2d, wims, nims = zip(*Parallel(n_jobs=16)(
            delayed(ratio_im)(
                nfi.scraft.values[i], wfi.scraft.values[i],
                nfi.images.values[i], wfi.images.values[i],
            ) for i in sorted(keep)
        ))
        # fixed dynamic range over the whole period, robust to hot pixels
        wclim = np.nanpercentile(np.stack(wims), (1, 99))
        nclim = np.nanpercentile(np.stack(nims), (1, 99))
        arrows = [up_axes(
            nfi.scraft.values[i],
            nfi.scraft_l1a.values[i], wfi.scraft_l1a.values[i],
        ) for i in sorted(keep)]
        labels = [str(nfi.time.values[i])[:16] for i in sorted(keep)]
        # actual WFI frame times — can differ from the NFI partner by hours
        wlabels = [str(wfi.time.values[i])[:16] for i in sorted(keep)]

        tags.h1(f'{name}: {start} … {end}')
        sliderlock(group=name)
        with itemgrid(length=4):
            plt.close('all')


            with caption('interpolated WFI / NFI pixels (2D sky interpolation)'):
                with slider(labels=labels, group=name):
                    for im, (nup, wup), label in zip(ims2d, arrows, labels):
                        fig, ax = plt.subplots()
                        h = ax.imshow(im, clim=(1, 2))
                        fig.colorbar(h, label='interp WFI / NFI')
                        # detector up-axes: base at center, short (display y is
                        # flipped, so up = -dy)
                        c, L = im.shape[0] / 2, 0.1 * im.shape[0]
                        for u, color in ((nup, 'tab:orange'), (wup, 'tab:green')):
                            ax.annotate(
                                '', xy=(c + L * u[0], c - L * u[1]), xytext=(c, c),
                                arrowprops=dict(color=color, arrowstyle='-|>'),
                            )
                        ax.legend(handles=[
                            plt.Line2D([], [], color='tab:orange', label='NFI detector up'),
                            plt.Line2D([], [], color='tab:green', label='WFI detector up'),
                        ], loc='upper right', fontsize=8)
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

            for ims, clim, titles, title in (
                    (wims, wclim, wlabels, 'interpolated WFI (annulus)'),
                    (nims, nclim, labels, 'NFI (annulus)')):
                with caption(title):
                    with slider(labels=labels, group=name):
                        for im, label in zip(ims, titles):
                            fig, ax = plt.subplots()
                            h = ax.imshow(im, clim=clim)
                            fig.colorbar(h, label='Rayleighs')
                            ax.set_title(label)
                            plot(fig, **figset)
                            plt.close(fig)

    tags.h1("Source Code")
    tags.code(tags.pre(open('cross_cal.py').read()))

outfile = Path('/www/cross/cross_cal.html')
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text(doc.render())
print(f'Saved to {outfile}')
