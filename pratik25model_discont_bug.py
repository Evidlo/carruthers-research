#!/usr/bin/env python3

from glide.science.model_sph import *
from glide.science.plotting_sph import cardplot, cardplotaxes

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from dominate_tags import *

morig = Pratik25Model(num_times=1)

r3_ind = t.searchsorted(morig.grid.r_b, 3)
r25_ind = t.searchsorted(morig.grid.r_b, 25) + 1
trim = DefaultGrid(r_b=morig.grid.r_b[r3_ind:r25_ind], e_b=morig.grid.e_b, a_b=morig.grid.a_b)
mtrim = Pratik25Model(trim, num_times=1, method='nearest')

grid = DefaultGrid()

mnear = Pratik25Model(grid, num_times=1, method='nearest')
# FIXME - nearc interpolation creates negative values
mnear.density = mnear.density.clip(min=1)

mline = Pratik25Model(grid, num_times=1, method='linear')
# FIXME - linec interpolation creates negative values
mline.density = mline.density.clip(min=1)

mcubi = Pratik25Model(grid, num_times=1, method='cubic')
# FIXME - cubic interpolation creates negative values
mcubi.density = mcubi.density.clip(min=1)

xorig = morig()
xtrim = mtrim()
xnear = mnear()
xline = mline()
xcubi = mcubi()

# set plot height
import functools
plot = functools.partial(plot, height="300px")

# ----- Plotting -----
# %% plot


def plot_YZ_circ(x, grid):
    """Create lineplot at 10Re circle in YZ plane

    Args:
        x (tensor): density data
        grid (SphericalGrid)
    """
    # get index of 10 Re shell
    r10_ind = t.searchsorted(grid.r, rval:=6)
    # get index of ±Y azimuth
    aY_ind = t.searchsorted(grid.a, np.pi/2)
    aY_ind_neg = t.searchsorted(grid.a, -np.pi/2)

    fig, ax = plt.subplots()
    ax.plot(grid.e, x[..., r10_ind, :, aY_ind].squeeze(), 'b')
    ax.plot(-grid.e, x[..., r10_ind, :, aY_ind_neg].squeeze(), 'b')
    ax.set_xlabel('Elevation (rad)')
    ax.set_ylabel('Density')
    ax.set_title(f'Density in YZ plane at {rval} Re (nearest bin)')

    return fig

with document('Pratik model bug') as d:

    util.raw("""Row Descriptions:<br>
    <ol>
    <li>Original data (out to ~400Re)</li>
    <li>Original data (trimmed to ~25Re)</li>
    <li>Interpolated data onto default grid (nearest neighbor)</li>
    <li>Interpolated data onto default grid (linear (current method))</li>
    <li>Interpolated data onto default grid (cubic)</li>
    </ol>


    Discontinuity is present in ground truth dataset due to +Z pole present in spherical grids.  This is why
    some researchers use alternative coordinate systems such as 'cubed-sphere', 'healpix', 'quadrilaterized spherical cube (QSC)', etc.
    <br>
    The density in the line plots differ because of different grid shapes.
    I'm just using the nearest bins to the YZ plane, not interpolating.
    """)


    caption(
        f'Unmodified Data - {morig.grid}',
        plot(cardplot(xorig, morig.grid, norm=LogNorm(), method='nearest')),
        plot(plot_YZ_circ(xorig, morig.grid)),
        plot(cardplotaxes(xorig.squeeze(), morig.grid, yscale='log', method='nearest'))
    )
    caption(
        f'Trimmed Data - {mtrim.grid}',
        plot(cardplot(xtrim, mtrim.grid, norm=LogNorm(), method='nearest')),
        plot(plot_YZ_circ(xtrim, mtrim.grid)),
        plot(cardplotaxes(xtrim.squeeze(), mtrim.grid, yscale='log', method='nearest'))
    )
    caption(
        f'Nearest Interpolated onto Default Grid - {mnear.grid}',
        plot(cardplot(xnear, mnear.grid, norm=LogNorm(), method='nearest')),
        plot(plot_YZ_circ(xnear, mnear.grid)),
        plot(cardplotaxes(xnear.squeeze(), mnear.grid, yscale='log', method='nearest'))
    )
    caption(
        f'Linearly Interpolated onto Default Grid - {mline.grid}',
        plot(cardplot(xline, mline.grid, norm=LogNorm(), method='nearest')),
        plot(plot_YZ_circ(xline, mline.grid)),
        plot(cardplotaxes(xline.squeeze(), mline.grid, yscale='log', method='nearest'))
    )
    caption(
        f'Cubic Interpolated onto Default Grid - {mcubi.grid}',
        plot(cardplot(xcubi, mcubi.grid, norm=LogNorm(), method='nearest')),
        plot(plot_YZ_circ(xcubi, mcubi.grid)),
        plot(cardplotaxes(xcubi.squeeze(), mcubi.grid, yscale='log', method='nearest'))
    )


# open('/www/lara/pratik25model_discont.html', 'w').write(d.render())