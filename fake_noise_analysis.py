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
import numpy as np
import inspect

device = 'cpu'

# ----- Setup -----
# %% setup


# %% grid
"""
nr, ne, na = 200, 60, 80
r_b = np.geomspace(3, 25, nr+1)
# angle of midnight shadow cone in radians
cone = np.deg2rad(45)
halfcone = cone/2
ce = 30 # number of elevation bins in cone
e_b = np.concatenate((
    # (np.linspace(0, conerad, ce:=0, endpoint=False), np.linspace(conerad, np.pi, ne-ce+1))
    np.linspace(0, np.pi/2-halfcone, (ne-ce)//2+1)[:-1],
    np.linspace(np.pi/2-halfcone, np.pi/2+halfcone, ce+1),
    np.linspace(np.pi/2+halfcone, np.pi, (ne-ce)//2+1)[1:]
))
ca = 30 # number of azimuth bins in cone
a_b = np.concatenate((
    np.linspace(-np.pi, -np.pi+halfcone, ca//2 + 1),
    np.linspace(-np.pi+halfcone, np.pi-halfcone, na-ca+1)[1:-1],
    np.linspace(np.pi-halfcone, np.pi, ca//2 + 1),
))

r_b, e_b, a_b = map(t.from_numpy, (r_b, e_b, a_b))

grid = DefaultGrid(r_b=r_b, e_b=e_b, a_b=a_b)
grid.r = tr.sqrt(r_b[1:] * r_b[:-1])
"""

grid = BiResGrid((200, 60, 80), ce=30, ca=30, angle=45, spacing='log')
# grid = DefaultGrid((500, 50, 50), spacing='log')
gridrec = BiResGrid((200, 60, 80), ce=30, ca=30, angle=45, spacing='log')
# gridrec = DefaultGrid((50, 50, 50), spacing='log')

m = Zoennchen00Model(grid=grid, device=device)
a = albedo(grid) * m()
f = cardplot(a, grid, method='nearest')
f.suptitle('Albedo', fontsize=24)
plt.savefig('/www/acustom.png')

# %% other



num_obs = 10
win = 28 # window (days)
season = 'spring'
t_op = 360 # integration time

t.cuda.empty_cache()

cams = [
    CameraL1BNFI(nadir_nfi_mode(t_op=t_op)),
    CameraL1BWFI(nadir_wfi_mode(t_op=t_op))
]
sc = gen_mission(num_obs=num_obs, duration=win, start=season, cams=cams)[5:6]

freal = ForwardSph(
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
frec = ForwardSph(
    sc, gridrec,
    use_albedo=freal.use_albedo, use_aniso=freal.use_aniso,
    science_binning=freal.science_binning, scishape=freal.scishape, device=freal.device
)
vgshapestr = 'x'.join(map(str, freal.scishape))
gshapestr = 'x'.join(map(str, grid.shape))
grecshapestr = 'x'.join(map(str, gridrec.shape))

# ----- Debug -----
# %% forward


# m = Zoennchen24Model(grid=grid, device=device); mstr = ''
# m = ConstModel(grid=grid, fill=1000, device=device); mstr='_const'
m = Zoennchen00Model(grid=grid, device=device)
truth = m()
mrec = Zoennchen00Model(grid=gridrec, device=device)
truthrec = mrec()

fake_nls = frec.fake_noise(truthrec, disable_noise=True)
fakes = frec.fake_noise(truthrec)
real_nls = freal.noise(truth, disable_noise=True)

desc = f'coldens{ual}{uno}_{gshapestr}_{grecshapestr}'
coldenserr(real_nls, fake_nls, fakes, outdir='/www/fake', outfile=desc)