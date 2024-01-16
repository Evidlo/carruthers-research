#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
plt.close('all')

from glide.common_components.view_geometry import carruthers_orbit, CameraWFI, CameraNFI
from glide.science.orbit import viewgeom2ts
from glide.science.plotting import orbit_svg
from glide.science.model import default_vol
from glide.science.forward import viewgeom_endplane



def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


fig, axs = plt.subplots(2, 4, figsize=(10, 5), constrained_layout=True)

# ax_cam - plot aes for each camera
# cam - camera
# d - LOS downsample factor

interval = 15
vol = default_vol()
seasons = ('spring', 'summer', 'fall', 'winter')
cams = (CameraWFI(), CameraNFI())

for ax_cam, cam, d in zip(axs, cams, (16, 64)):

    for ax, season in zip(ax_cam, seasons):

        view_geoms = carruthers_orbit(start=season, duration=interval, cam=cam, num_obs=2)

        vgts = viewgeom2ts(view_geoms)
        endplane_positions = viewgeom_endplane(vgts)

        for c, sp, ep in zip(['b', 'r'], vgts.src_pos, endplane_positions):
            # plot every 10th pixel of top row of pixels
            for ep10 in ep[:, 0, ::d].T:
                ax.axline(sp[:2], ep10[:2], color=c)

            ax.set_aspect('equal')
            # ax.axis('equal')
            width = 25
            ax.set_xlim([-width, width])
            ax.set_ylim([-width, width])
            ax.grid(True
                    )
            ax.minorticks_on()

add_headers(
    fig,
    row_headers=list(map(lambda c: c.camID, cams)),
    col_headers=list(map(str.capitalize, seasons))
)
# plt.tight_layout()
plt.suptitle(f'Interval: {interval}d')
plt.savefig('/srv/www/out.png')
orbit_svg(vol, vgts).save('/srv/www/out.gif')