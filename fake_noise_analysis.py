#!/usr/bin/env python3

from glide.common_components.camera import CameraWFI, CameraNFI, CameraL1BWFI, CameraL1BNFI
from glide.common_components.cam import nadir_wfi_mode, nadir_nfi_mode
from glide.common_components.generate_view_geom import gen_mission
from glide.common_components.orbits import circular_orbit
from glide.science.forward_sph import *
from glide.science.model_sph import *
from glide.science.plotting import *
from glide.science.plotting_sph import carderr, cardplot, carderrmin
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

grid = DefaultGrid((500, 50, 50), spacing='log', mask_rs={'WFI':(5, 25)})

# for season, difflam, t_int in items:
for num_obs, win, season, difflam, t_int, gshp in items:
    t.cuda.empty_cache()

    # integrate for full time between each snapshot
    # t_int = win * 24 / num_obs

    cams = [
        CameraL1BNFI(nadir_nfi_mode(t_int=t_int)),
        CameraL1BWFI(nadir_wfi_mode(t_int=t_int))
    ]
    sc = gen_mission(num_obs=num_obs, duration=win, start=season, cams=cams)

    f = ForwardSph(sc, grid, use_albedo=False, use_aniso=False, use_noise=True, science_binning=True, device=device)

    # ----- Debug -----
    # %% debug2
    # m = Zoennchen24Model(grid=grid, device=device); mstr = ''
    # m = ConstModel(grid=grid, fill=1000, device=device); mstr='_const'
    m = Zoennchen00Model(grid=grid, device=device); mstr='_zold'
    truth = m()
    vg_desc = f'{f.bin_funcs[0].spacing}{"x".join(map(str, f.vg.shape[1:]))}'
    # vg_desc = f'lin{"x".join(map(str, f.vg.shape[1:]))}'
    desc = f'{vg_desc}_{grid.shape.r}r{mstr}_inner{grid.size.r[0]}Re'


    mask = t.ones(f.vg.shape, device=device)
    # mask[:, :6] = 0

    realmeasurements = f.noise(truth, disable_noise=True) * mask
    fakemeasurements = f.fake_noise(truth, disable_noise=True) * mask
    fakemeasurements2 = f.fake_noise(truth) * mask
    # %% debug
    # tp_r = tangent_points(f.vg.ray_starts, f.vg.rays).norm(dim=-1).to(device=device)
    # analmeasurements = 2 * t.sqrt(grid.size.r[1]**2 - tp_r**2) * constval * f.projection_masks
    analmeasurements = m.analytic(f.vg) * f.projection_masks / R_earth.to('cm').value



    for num_img in range(0, len(realmeasurements)):

        realmeas = realmeasurements[num_img]
        fakemeas = fakemeasurements[num_img]
        fakemeas2 = fakemeasurements2[num_img]
        analmeas = analmeasurements[num_img]


        M, N = 2, 2
        plt.close('all')
        plt.figure(figsize=(12, 5))

        # ax_realmeas_rect.set_title('Rect Real Noise')
        # ax_realmeas = plt.subplot(M, N, 2, projection='polar')
        # image_stack(realmeas, f.vg[num_img], ax=ax_realmeas)
        # ax_realmeas.set_title('Sci Real Noise')
        # ax_fakemeas = plt.subplot(M, N, 3, projection='polar')
        # image_stack(fakemeas, f.vg[num_img], ax=ax_fakemeas)
        # ax_fakemeas.set_title('Sci Fake Noise')

        # ax_realmeas = plt.subplot(M, N, 3)
        # ax_realmeas.imshow(realmeas.detach().cpu().numpy())
        # ax_realmeas.set_title('Sci Real Noise')
        # plt.gcf().colorbar(img, ax=ax_realmeas)

        # ax_fakemeas = plt.subplot(M, N, 6)
        # ax_fakemeas.imshow(fakemeas.detach().cpu().numpy())
        # ax_fakemeas.set_title('Sci Fake Noise')
        # plt.gcf().colorbar(img, ax=ax_fakemeas)

        #
        # changerelerr = 100 * ((fakemeas - fakemeas2) / fakemeas).detach().cpu().numpy()
        # changeerr = ((fakemeas - fakemeas2)**2).detach().cpu().numpy()

        # ax_err = plt.subplot(M, N, 3)
        # img = ax_err.imshow(changerelerr)
        # ax_err.set_title('Fake % Error Change')
        # plt.gcf().colorbar(img, ax=ax_err)

        # ax_err = plt.subplot(M, N, 4)
        # img = ax_err.imshow(changeerr)
        # ax_err.set_title('Fake Sq Error Change')
        # plt.gcf().colorbar(img, ax=ax_err)

        # relerr = 100 * ((realmeas - fakemeas) / fakemeas).detach().cpu().numpy()
        errnoise = ((fakemeas - fakemeas2)**2).detach().cpu().numpy()
        err = ((realmeas - fakemeas)**2).detach().cpu().numpy()

        ax_err = plt.subplot(M, N, 1)
        img = ax_err.imshow(errnoise)
        ax_err.set_title(f'Sq Error : Fake (noiseless) → Fake (noisy) ({num_img:02d}/{len(realmeasurements)})')
        plt.gcf().colorbar(img, ax=ax_err)


        ax_err = plt.subplot(M, N, 3)
        img = ax_err.imshow(err)
        ax_err.set_title(f'Sq Error : Real (noiseless) → Fake (noiseless) ({num_img:02d}/{len(realmeasurements)})')
        plt.gcf().colorbar(img, ax=ax_err)

        ax2 = plt.subplot(M, N, 2)
        fakeanalerr = ((fakemeas - analmeas)**2).detach().cpu().numpy()
        img = ax2.imshow(fakeanalerr)
        ax2.set_title(f'Sq Error : Fake (noiseless) → Analytic (noiseless) ({num_img:02d}/{len(realmeasurements)})')
        plt.gcf().colorbar(img, ax=ax2)

        ax4 = plt.subplot(M, N, 4)
        realanalerr = ((realmeas - analmeas)**2).detach().cpu().numpy()
        img = ax4.imshow(realanalerr)
        ax4.set_title(f'Sq Error : Real (noiseless) → Analytic (noiseless) ({num_img:02d}/{len(realmeasurements)})')
        plt.gcf().colorbar(img, ax=ax4)


        plt.tight_layout()
        outdir = Path('/www/tmp/')
        outimg = outdir / f'{desc}_{num_img:02d}.png'
        print('wrote', outimg)
        plt.savefig(outimg)

    cmd = f'convert -delay 30 -loop 0 {outdir / desc}_[0-9][0-9]*png {outdir / desc}.gif'
    print('running', cmd)
    run(cmd, shell=True)