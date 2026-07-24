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
import scienceplots

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('science')

datapath = Path('/data-products')
date = np.datetime64('2026-03-01T05:53')
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


nfi, wfi, _ = load(datapath, np.array([date], dtype='datetime64[ns]'))
nsc, wsc = nfi.scraft.values[0], wfi.scraft.values[0]
nrays, wrays = unit_los(nsc, det_rays(nsc)), unit_los(wsc, det_rays(wsc))

ratio, wim, nim = ratio_im(
    nsc, wsc, nrays, wrays, nfi.images.values[0], wfi.images.values[0], k=k,
)
tp = tp_radius(nsc.position_gse[:, 0] / R_earth.to('km').value, nrays)
nup, wup = up_axes(nsc, nfi.scraft_l1a.values[0], wfi.scraft_l1a.values[0])
wres, nres = azimuthal_norm(wim, tp), azimuthal_norm(nim, tp)

with document('Cross Calibration: per-channel azimuthal symmetry') as doc:
    figset = {'height': 360, 'matkwargs': {'dpi': 250}}
    tags.h1(str(nfi.time.values[0])[:16])

    with itemgrid(length=3):
        # shared dynamic range so the two channels' annuli are comparable
        rclim = np.nanpercentile(np.stack((wim, nim)), (1, 99))
        panels = (
            ('interp WFI / azimuthal median', wres, (.75, 1.25), 'seismic',
             'WFI divided by its own azimuthal median'),
            ('NFI / azimuthal median', nres, (.75, 1.25), 'seismic',
             'NFI divided by its own azimuthal median'),
            ('interp WFI / NFI', ratio, (.75, 1.25), 'seismic',
             'Existing cross-cal ratio of the two channels below.'),
            ('interp WFI (Rayleighs)', wim, rclim, 'viridis', 'WFI resampled onto NFI LOS grid'),
            ('NFI (Rayleighs)', nim, rclim, 'viridis', 'NFI'),
        )
        for label, im, clim, cmap, cap in panels:
            with caption(cap):
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
                plot(fig, **figset)
                plt.close(fig)

    # full frames with the frame center marked: the Earth disk should sit on the
    # dot if the boresight is Earth-aligned
    with itemgrid(length=3):
        for sc, im, label in ((wsc, wfi.images.values[0], 'WFI'),
                              (nsc, nfi.images.values[0], 'NFI')):
            im = np.where(sc.sensor.spec.mask_fov, im, np.nan)
            with caption(f'{label} full FOV, frame center marked'):
                fig, ax = plt.subplots()
                h = ax.imshow(im, clim=np.nanpercentile(im, (1, 100)))
                fig.colorbar(h, label=f'{label} (Rayleighs)')
                c = im.shape[0] / 2
                ax.plot(c, c, 'o', color='tab:orange', ms=4)
                plot(fig, **figset)
                plt.close(fig)

    tags.h1("Source Code")
    tags.code(tags.pre(open('azimuth.py').read()))

outfile = Path('/www/cross/azimuth.html')
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text(doc.render())
print(f'Saved to {outfile}')
