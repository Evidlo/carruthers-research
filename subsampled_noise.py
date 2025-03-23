#!/usr/bin/env python3

# comparing subsampled science geom vs binned science geom noise

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
import torch_boiler as tb


cams = [
    CameraL1BNFI(nadir_nfi_mode(t_op=360)),
    CameraL1BWFI(nadir_wfi_mode(t_op=360))
]
sc = gen_mission(num_obs=1, cams=cams)

g = DefaultGrid()
m = Zoennchen00Model(g)

# ----- Forward -----

rvg = sum([NativeGeom(s) for s in sc])
# science binned
f_bin = ForwardSph(sc, m.grid, rvg=rvg)
# science subsampled
bvg = sum([SubsampledScienceGeom(r, (50, 100)) for r in rvg])
# f_sub = ForwardSph(sc, m.grid, rvg=rvg, bvg=bvg)

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

meas_binerr(f_bin, m, y_bin_nois, y_bin_less, y_bin_trac).savefig('/www/binnednan_err.png')
# meas_binerr(f_sub, m, y_sub_nois, y_sub_less, y_sub_trac).savefig('/www/subsamp_err.png')


# %% x

"""
# meas_binerr(f_sub, m).savefig('/www/subsampled_err.png')

# ----- Measurement -----
# %% test3

d = m()
y_bin = f_bin.noise(d, disable_noise=True)
y_sub = f_sub.noise(d, disable_noise=True)

# ----- Plotting -----
# %% plot

y_anal = m.analytic(bvg)

# err_bin = (y_bin - y_anal) / t.where(y_anal!=0, y_anal, float('inf'))
# err_sub = (y_sub - y_anal) / t.where(y_anal!=0, y_anal, float('inf'))
# err_bin = (y_bin - y_anal) / y_anal
# err_sub = (y_sub - y_anal) / y_anal; errtype = '%'
err_bin = (y_bin - y_anal)**2
err_sub = (y_sub - y_anal)**2; errtype = 'Sq'
# dont plot masked LOS
err_bin[f_bin.proj_maskb==False] = float('nan')
err_sub[f_sub.proj_maskb==False] = float('nan')

plt.close('all')

ax = plt.subplot(2, 2, 1, projection='polar')
image_stack(err_bin[0], f_bin.bvg[0], ax=ax, colorbar=True)
plt.title(f'Binned/Analytic {errtype} Err')

ax = plt.subplot(2, 2, 2, projection='polar')
image_stack(err_bin[1], f_bin.bvg[1], ax=ax, colorbar=True)
plt.title(f'Binned/Analytic {errtype} Err')

ax = plt.subplot(2, 2, 3, projection='polar')
image_stack(err_sub[0], f_sub.bvg[0], ax=ax, colorbar=True)
plt.title(f'Subsampled/Analytic {errtype} Err')

ax = plt.subplot(2, 2, 4, projection='polar')
image_stack(err_sub[1], f_sub.bvg[1], ax=ax, colorbar=True)
plt.title(f'Subsampled/Analytic {errtype} Err')

plt.tight_layout()
plt.savefig(f'/www/subsampled_nonoise_{errtype.lower()}.png')
"""