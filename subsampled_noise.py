#!/usr/bin/env python3

# comparing subsampled science geom vs binned science geom noise

from itertools import product
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# from glide.common_components.orbits import circular_orbit
from glide.common_components.generate_view_geom import gen_mission
from glide.common_components.camera import CameraL1BWFI, CameraL1BNFI
from glide.common_components.cam import nadir_wfi_mode, nadir_nfi_mode
from glide.science.forward_sph import *
from glide.science.model_sph import *

from sph_raytracer.plotting import image_stack


cams = [
    CameraL1BNFI(nadir_nfi_mode(t_op=360)),
    CameraL1BWFI(nadir_wfi_mode(t_op=360))
]
sc = gen_mission(num_obs=1, cams=cams)

model = lambda grid: Zoennchen00Model(grid)

# simulation grid, model, and view geometry (rectangular view geometry for simulator)
# NOTE: Jackson - see DefaultGrid.mask_rs for how to set radial mask inner/outer boundaries
mask_rs = {} # disable default 5 Re WFI mask
sgrid = DefaultGrid((500, 45, 60), mask_rs=mask_rs, spacing='log')
sm = model(sgrid)
svg = sum([NativeGeom(s) for s in sc])


# ----- Forward -----

items = product(
    # reconstruction grid
    [
        DefaultGrid((200, 45, 60), mask_rs=mask_rs, spacing='log'),
        # DefaultGrid((100, 45, 60), spacing='log'),
    ],
    # view geometry spacing
    ['lin'],
    # view geometry shape
    product(
        [25],
        [25],
        # [25 * r for r in range(2, 3)],
        # [25 * th for th in range(4, 5)]
    ))

data = []

for rgrid, rvg_spacing, rvg_shape in items:
    desc_str  = f'[viewgeom:{rvg_shape} {rvg_spacing}]'
    desc_str += f' [rgrid:{rgrid.shape} {rgrid.spacing}]'
    desc_str += f' [sgrid:{sgrid.shape} {sgrid.spacing}]'
    print(f'----- {desc_str} -----')

    # reconstruction model and view geometry (science view geometry)
    rm = model(rgrid)
    rvg = sum([ScienceGeom(s, rvg_shape, spacing=rvg_spacing) for s in sc])

    f_bin = ForwardSph(sc, rm.grid, sm.grid, rvg=rvg, svg=svg)

    # ----- Plotting -----
    # %% plot

    from meas_binerr import meas_binerr

    rvg_shape_str = "x".join(str(s).zfill(3) for s in rvg_shape)
    rgrid_shape_str = "x".join(str(s).zfill(3) for s in rgrid.shape)
    # fig = meas_binerr(f_bin, m, y_bin_nois, y_bin_less, y_bin_trac)
    fig = meas_binerr(f_bin, sm, rm)
    fig.text(0.1, .01, desc_str)
    fig.savefig(f:=f'/www/measbinexp/binned_{rvg_spacing}{rvg_shape_str}_{rgrid.spacing}{rgrid_shape_str}.png')
    print(f'Wrote {f}')

    # TODO: Jackson - experimental data for later plotting is computed here.
    # Variables inside `meas_binerr` are accessible through `fig.locals`.  You
    # should not need to modify that code.
    """
    example_result = fig.locals.err_nois_sq.max()
    data.append({
        'rvg_r_bins': rvg.shape[0],
        'example_result':example_result,
        ... more parameters and computed values here ...
    })
    """

# TODO: Jackson - learn a little bit about Pandas and Seaborn plotting
"""
# convert `data` to pandas dataframe which can easily be plotted
import pandas as pd
df = pd.DataFrame(data)
# save the results to disk to plot them later
df.to_pickle("experiment_results.pkl")
"""

"""
# load the saved results in your separate plotting script
import pandas as pd
df = pd.read_pickle("experiment_results.pkl")
# seaborn plotting from df goes here
import seaborn as sns
sns.lineplot(data=data, x='rvg_r_bins', y='example_result')

# also check out sns.PairGrid for plotting high dimensional data
# https://seaborn.pydata.org/generated/seaborn.PairGrid.html
"""