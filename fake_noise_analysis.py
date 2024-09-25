#!/usr/bin/env python3

from glide.common_components.camera import CameraWFI, CameraNFI, CameraL1BWFI, CameraL1BNFI
from glide.common_components.cam import nadir_wfi_mode, nadir_nfi_mode
from glide.common_components.generate_view_geom import gen_mission
from glide.common_components.orbits import circular_orbit
from glide.science.forward_sph import *
from glide.science.model_sph import *
from glide.science.plotting import *
from glide.science.plotting_sph import carderr, cardplot, carderrmin, coldenserr
from glide.science.recon.loss_sph import *

from sph_raytracer import *
from sph_raytracer.plotting import *
cn = color_negative
from sph_raytracer.retrieval import *
from sph_raytracer.loss import *
from sph_raytracer.model import *

from astropy.constants import R_earth

from pathlib import Path
from subprocess import run
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch as t
import inspect

code = open(__file__).read()

device = 'cuda'

# ----- Setup -----
# %% setup

from itertools import product
items = product(
    # [.033],
    [10], # num_obs
    [28], # window (days)
    ['spring'], # season
    [1e2], # difflam
    [360], # integration time
    [50], # grid shape
)

grid500 = DefaultGrid((500, 50, 50), spacing='log')
grid50 = DefaultGrid((500, 50, 50), spacing='log')

# for season, difflam, t_int in items:
for num_obs, win, season, difflam, t_op, gshp in items:
    t.cuda.empty_cache()

    # integrate for full time between each snapshot
    # t_int = win * 24 / num_obs

    cams = [
        CameraL1BNFI(nadir_nfi_mode(t_op=t_op)),
        CameraL1BWFI(nadir_wfi_mode(t_op=t_op))
    ]
    sc = gen_mission(num_obs=num_obs, duration=win, start=season, cams=cams)[1:2]

    f = ForwardSph(
        sc, grid500,
        use_albedo=(
            # False, ual:='_noal'
            True, ual:=''
        )[0],
        use_aniso=(
            # False, uno:='_noan'
            True, uno:=''
        )[0],
        use_noise=True,
        science_binning=True, scishape=(50, 100), device=device
    )
    shapestr = 'x'.join(map(str, f.scishape))

    # ----- Debug -----
    # %% debug2
    # m = Zoennchen24Model(grid=grid50, device=device); mstr = ''
    # m = ConstModel(grid=grid50, fill=1000, device=device); mstr='_const'
    m = Zoennchen00Model(grid=grid50, device=device); mstr='_zold'
    truth = m()
    # truth = f.albedo_mask
    vg_desc = f'{f.bin_funcs[0].spacing}{"x".join(map(str, f.vg.shape[1:]))}'
    # vg_desc = f'lin{"x".join(map(str, f.vg.shape[1:]))}'
    desc = f'{vg_desc}_{ual}{uno}{grid50.shape.r}r{mstr}_inner{grid50.mask_rs["WFI"][0]}Re'


    mask = t.ones(f.vg.shape, device=device)
    # mask[:, :6] = 0

    fake_nls = f.fake_noise(truth, disable_noise=True) * mask
    fakes = f.fake_noise(truth) * mask
    real_nls = f.noise(truth, disable_noise=True) * mask

    coldenserr(real_nls, fake_nls, fakes, outdir='/www/fake', outfile=f'coldens{ual}{uno}_{shapestr}')