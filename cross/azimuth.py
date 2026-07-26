#!/usr/bin/env python3
# Which channel owns the WFI/NFI cross-cal asymmetry?  The exosphere seen from
# L1 is nearly rotationally symmetric about the Earth direction, so each channel
# can be compared against ITSELF: divide by the azimuthal median at matched
# tangent radius.  Real asymmetry is common to both channels and cancels in the
# WFI/NFI ratio, so the ratio map equals (WFI residual / NFI residual) times a
# purely radial function — whichever residual map carries the gradient is the
# culprit.  Both are evaluated on the same NFI native pixels (same LOSs).

from geom import band, det_rays, ratio_im, tp_radius, unit_los, up_axes
from glide.science.forward_sph import R_earth
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
start, end = '2026-03-01', '2026-04-01'
group = 'march'
# WFI pixels averaged per NFI pixel.  WFI is 10x coarser, so k=1 is plain
# nearest-neighbor (10x10 constant blocks) — no noise penalty that the
# azimuthal medians don't already absorb.
k = 5


def azimuthal_norm(im, tp, nbins=40):
    """`im` divided by its own azimuthal median at matched tangent radius.

    im, tp (npix, npix) -> (npix, npix).  The median is taken in `nbins` radial
    bins over `band` and linearly interpolated between bin centers, so no ring
    edges are printed into the result."""
    edges = np.linspace(*band, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    which = np.digitize(np.where(np.isfinite(im), tp, np.nan), edges) - 1
    prof = np.array([
        np.nanmedian(np.where(which == i, im, np.nan)) for i in range(nbins)
    ])
    return im / np.interp(tp, centers, prof)


dates = np.arange(
    np.datetime64(start), np.datetime64(end), np.timedelta64(6, 'h'),
).astype('datetime64[ns]')
nfi, wfi, dates = load(datapath, dates)

# nearest-selection duplicates frames on short ranges
_, keep = np.unique(nfi.time.values, return_index=True)
keep = sorted(keep)

# `images` is dask-backed, so `.values` materializes the whole cube: once here,
# not once per frame inside the loop below
nscs, wscs = nfi.scraft.values, wfi.scraft.values
nall, wall = nfi.images.values, wfi.images.values
nl1a, wl1a = nfi.scraft_l1a.values, wfi.scraft_l1a.values
ndet, wdet = det_rays(nscs[0]), det_rays(wscs[0])

ratios, wims, nims, wress, nress, arrows = [], [], [], [], [], []
for i in tqdm(keep, desc=group):
    nrays, wrays = unit_los(nscs[i], ndet), unit_los(wscs[i], wdet)
    ratio, wim, nim = ratio_im(
        nscs[i], wscs[i], nrays, wrays, nall[i], wall[i], k=k,
    )
    tp = tp_radius(nscs[i].position_gse[:, 0] / R_earth.to('km').value, nrays)
    ratios.append(ratio)
    wims.append(wim)
    nims.append(nim)
    wress.append(azimuthal_norm(wim, tp))
    nress.append(azimuthal_norm(nim, tp))
    arrows.append(up_axes(nscs[i], nl1a[i], wl1a[i]))

labels = [str(nfi.time.values[i])[:16] for i in keep]
# actual WFI frame times — can differ from the NFI partner by hours
wlabels = [str(wfi.time.values[i])[:16] for i in keep]

with document('Cross Calibration: per-channel azimuthal symmetry') as doc:
    figset = {'height': 360}
    tags.h1(f'{group}: {start} … {end}')
    sliderlock(group=group)

    with itemgrid(length=3):
        plt.close('all')
        # shared dynamic range over the whole period, robust to hot pixels
        rclim = np.nanpercentile(np.stack(wims + nims), (1, 99))
        panels = (
            ('interp WFI / azimuthal median', wress, labels, (.75, 1.25),
             'seismic', 'WFI divided by its own azimuthal median'),
            ('NFI / azimuthal median', nress, labels, (.75, 1.25), 'seismic',
             'NFI divided by its own azimuthal median'),
            ('interp WFI / NFI', ratios, labels, (.75, 1.25), 'seismic',
             'Existing cross-cal ratio of the two channels below.'),
            ('interp WFI (Rayleighs)', wims, wlabels, rclim, 'viridis',
             'WFI resampled onto NFI LOS grid'),
            ('NFI (Rayleighs)', nims, labels, rclim, 'viridis', 'NFI'),
        )
        for label, ims, titles, clim, cmap, cap in panels:
            with caption(cap):
                with slider(labels=labels, group=group):
                    for im, (nup, wup), title in tqdm(zip(ims, arrows, titles)):
                        fig, ax = plt.subplots()
                        h = ax.imshow(im, clim=clim, cmap=plt.get_cmap(cmap))
                        fig.colorbar(h, label=label)
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
                        ax.set_title(title)
                        plot(fig, **figset)
                        plt.close(fig)

    # full frames with the frame center marked: the Earth disk should sit on the
    # dot if the boresight is Earth-aligned
    with itemgrid(length=3):
        for scs, alls, titles, label in ((wscs, wall, wlabels, 'WFI'),
                                         (nscs, nall, labels, 'NFI')):
            with caption(f'{label} full FOV, frame center marked'):
                with slider(labels=labels, group=group):
                    for i, title in tqdm(zip(keep, titles)):
                        im = np.where(scs[i].sensor.spec.mask_fov, alls[i], np.nan)
                        fig, ax = plt.subplots()
                        h = ax.imshow(im, clim=np.nanpercentile(im, (1, 100)))
                        fig.colorbar(h, label=f'{label} (Rayleighs)')
                        c = im.shape[0] / 2
                        ax.plot(c, c, 'o', color='tab:orange', ms=4)
                        ax.set_title(title)
                        plot(fig, **figset)
                        plt.close(fig)

    tags.h1("Source Code")
    tags.code(tags.pre(open('azimuth.py').read()))

outfile = Path('/www/cross/azimuth.html')
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text(doc.render())
print(f'Saved to {outfile}')
