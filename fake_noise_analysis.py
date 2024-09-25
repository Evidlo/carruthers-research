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

grid = DefaultGrid((500, 50, 50), spacing='log')

num_obs = 10
win = 28 # window (days)
season = 'spring'
t_op = 360 # integration time

t.cuda.empty_cache()

cams = [
    CameraL1BNFI(nadir_nfi_mode(t_op=t_op)),
    CameraL1BWFI(nadir_wfi_mode(t_op=t_op))
]
sc = gen_mission(num_obs=num_obs, duration=win, start=season, cams=cams)[1:2]

f = ForwardSph(
    sc, grid,
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
# %% forward


# m = Zoennchen24Model(grid=grid, device=device); mstr = ''
# m = ConstModel(grid=grid, fill=1000, device=device); mstr='_const'
m = Zoennchen00Model(grid=grid, device=device); mstr='_zold'
truth = m()

fake_nls = f.fake_noise(truth, disable_noise=True)
fakes = f.fake_noise(truth)
real_nls = f.noise(truth, disable_noise=True)

coldenserr(real_nls, fake_nls, fakes, outdir='/www/fake', outfile=f'coldens{ual}{uno}_{shapestr}')