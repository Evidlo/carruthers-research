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


# ----- Forward -----

items = product(
    [
        DefaultGrid((200, 45, 60), spacing='log'),
        # DefaultGrid((100, 45, 60), spacing='log'),
    ],
    ['lin'],
    product(
        [25],
        [25, 50],
        # [25 * r for r in range(2, 3)],
        # [25 * th for th in range(4, 5)]
))
for g, spacing, shape in items:
    desc_str = f'[viewgeom:{shape} {spacing}] [grid:{g.shape} {g.spacing}]'
    print(f'----- {desc_str} -----')

    m = Zoennchen00Model(g)

    rvg = sum([NativeGeom(s) for s in sc])
    # science binned
    binvg = sum([ScienceGeom(s, shape, spacing=spacing) for s in sc])
    f_bin = ForwardSph(sc, m.grid, rvg=rvg, bvg=binvg)
    # science subsampled
    subvg = sum([SubsampledScienceGeom(r, shape, spacing=spacing) for r in rvg])
    f_sub = ForwardSph(sc, m.grid, rvg=rvg, bvg=subvg)

    d = m()
    y_bin_less = f_bin.noise(d, disable_noise=True)
    y_bin_nois = f_bin.noise(d)
    y_bin_trac = f_bin(d)

    # y_sub_less = f_sub.noise(d, disable_noise=True)
    # y_sub_nois = f_sub.noise(d)
    # y_sub_trac = f_sub(d)

    # ----- Plotting -----
    # %% plot

    from meas_binerr import meas_binerr

    vg_shape_str = "x".join(str(s).zfill(3) for s in shape)
    g_shape_str = "x".join(str(s).zfill(3) for s in g.shape)
    fig = meas_binerr(f_bin, m, y_bin_nois, y_bin_less, y_bin_trac)
    fig.text(0.1, .01, desc_str)
    fig.savefig(f'/www/measbinexp/binned_{spacing}{vg_shape_str}_{g.spacing}{g_shape_str}.png')
    # fig = meas_binerr(f_sub, m, y_sub_nois, y_sub_less, y_sub_trac)
    # fig.text(0.1, .01, desc_str)
    # fig.savefig(f'/www/measbinlim/{spacing}_subsamp_err_{shape_str}.png')
