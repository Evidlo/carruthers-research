#!/usr/bin/env python3
# WFI/NFI cross-calibration check: WFI is resampled onto each native NFI pixel
# by averaging the k nearest WFI pixels in LOS direction (KD-tree on the GSE
# LOS unit vectors — nearest on the sky), so real exospheric structure divides
# out of the ratio and detector scale/roll/mapping need no explicit handling.
# Earlier azimuth-median / science-binned versions: cross_cal_old.py.

from geom import det_rays, ratio_im, unit_los, up_axes
from load import load

from domrep import *
from pathlib import Path
from tqdm import tqdm
import scienceplots


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('science')

datapath = Path('/data-products')

periods = {
    'january': ('2026-01-01', '2026-02-01'),
    'march': ('2026-03-01', '2026-04-01'),
}

with document('NFI/WFI Cross Calibration (2D interpolation)') as doc:
    figset = {'height': 350}

    for name, (start, end) in periods.items():
        dates = np.arange(
            np.datetime64(start), np.datetime64(end), np.timedelta64(6, 'h'),
        ).astype('datetime64[ns]')
        nfi, wfi, dates = load(datapath, dates)

        # nearest-selection duplicates frames on short ranges
        _, keep = np.unique(nfi.time.values, return_index=True)
        keep = sorted(keep)

        # `images` is dask-backed, so `.values` materializes the whole cube:
        # once here, not once per frame inside the loop below
        nscs, wscs = nfi.scraft.values, wfi.scraft.values
        nall, wall = nfi.images.values, wfi.images.values
        ndet, wdet = det_rays(nscs[0]), det_rays(wscs[0])

        # ~0.3 s/frame — the KD-tree is a small part of it, so no worker pool
        ims2d, wims, nims = zip(*[
            ratio_im(
                nscs[i], wscs[i],
                unit_los(nscs[i], ndet), unit_los(wscs[i], wdet),
                nall[i], wall[i],
            ) for i in tqdm(keep, desc=name)
        ])
        # fixed dynamic range over the whole period, robust to hot pixels
        wclim = np.nanpercentile(np.stack(wims), (1, 99))
        nclim = np.nanpercentile(np.stack(nims), (1, 99))
        nl1a, wl1a = nfi.scraft_l1a.values, wfi.scraft_l1a.values
        arrows = [up_axes(nscs[i], nl1a[i], wl1a[i]) for i in keep]
        labels = [str(nfi.time.values[i])[:16] for i in keep]
        # actual WFI frame times — can differ from the NFI partner by hours
        wlabels = [str(wfi.time.values[i])[:16] for i in keep]

        tags.h1(f'{name}: {start} … {end}')
        sliderlock(group=name)
        with itemgrid(length=4):
            plt.close('all')

            with caption('Overlap-band median ratio vs date'):
                fig, ax = plt.subplots()
                ax.plot([np.datetime64(l) for l in labels], [np.nanmedian(im) for im in ims2d], 'o-')
                ax.axhline(1, color='k', lw=0.5)
                ax.set_ylabel('median WFI / NFI')
                fig.autofmt_xdate()
                plot(fig, **figset)


            with caption('interpolated WFI / NFI pixels (2D sky interpolation)'):
                with slider(labels=labels, group=name):
                    for im, (nup, wup), label in tqdm(zip(ims2d, arrows, labels)):
                        fig, ax = plt.subplots()
                        h = ax.imshow(im, clim=(.75, 1.25), cmap=plt.get_cmap('seismic'))
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

            for ims, clim, titles, title in (
                    (wims, wclim, wlabels, 'interpolated WFI (annulus)'),
                    (nims, nclim, labels, 'NFI (annulus)')):
                with caption(title):
                    with slider(labels=labels, group=name):
                        for im, label in tqdm(zip(ims, titles)):
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
