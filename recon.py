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

from pathlib import Path
from sph_raytracer import *
from sph_raytracer.plotting import *
cn = color_negative
from sph_raytracer.retrieval import *
from sph_raytracer.loss import *
from sph_raytracer.model import *
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
# items = [['spring', 1e-1]]

# for season, difflam, t_int in items:
for num_obs, win, season, difflam, t_int, gshp in items:
    t.cuda.empty_cache()
    # desc = f'circ_{difflam:.0e}_projection'
    # desc = f'{season}_{mo}mo_{difflam:.0e}_fakenoise{t_int}_diffmean_grid{gshp}_lognew'
    desc = f'{win:02d}d_{num_obs:02d}obs_{{noise_type}}{t_int//60}hr_{difflam:.0e}_{season}'
    # desc = f'{season}_{mo}mo_{difflam:.0e}_realnoise{t_int}'
    # desc = f'{season}_{mo}mo_{difflam:.0e}_nonoise'
    # desc = f'{season}_{mo}mo_{difflam:.0e}'
    # desc = f'experiment'

    # integrate for full time between each snapshot
    # t_int = win * 24 / num_obs

    cams = [
        CameraL1BNFI(nadir_nfi_mode(t_int=t_int)),
        CameraL1BWFI(nadir_wfi_mode(t_int=t_int))
    ]
    sc = gen_mission(num_obs=num_obs, duration=win, start=season, cams=cams)
    # sc = gen_mission(num_obs=30, orbit=circular_orbit, cams=cams)

    # m = PratikModel(grid=None, date=season, rlim=(3, 25), device=device); mstr = 'pratik'
    # m = GonzaloModel(grid=None, device=device); mstr = 'gonzalo'
    m = Zoennchen24Model(grid=default_grid((500, 50, 50), spacing='log'), device=device); mstr = 'zoennchen'
    truth = m()


    f = ForwardSph(sc, m.grid, use_albedo=False, use_aniso=False, use_noise=True, science_binning=True, device=device)

    desc += f'_emask_noalan_tshp{m.grid.shape.r}_frac{"x".join(map(str, f.vg[0].shape))}_wfi5Re'

    # meas = f.noise(truth); noise_type = 'real'
    meas = f.noise(truth, disable_noise=True); noise_type = 'realdisabled'
    # meas = f(truth); noise_type = 'noiseless
    # meas = f.fake_noise(truth); noise_type='fake'
    desc = desc.format(noise_type=noise_type)

    # del f

    print('-----------------------------')
    print(desc)
    print('-----------------------------')

    # ----- Retrieval -----
    #%% recon
    # mr = FullyDenseModel(m.grid)
    # choose a model for retrieval
    # mr = SphHarmModel(default_grid(size_r=m.grid.size.r, shape=(49, 1, 3)), max_l=3, device=device)
    mr = SphHarmModel(default_grid(shape=(gshp,) * 3, spacing='log'), device=device, max_l=3)
    mr.proj = lambda coeffs: coeffs
    fr = ForwardSph(
        sc, mr.grid,
        use_albedo=f.use_albedo, use_aniso=f.use_aniso, use_noise=f.use_noise,
        science_binning=f.science_binning, device=device
    )
    # fr = f
    # choose loss functions and regularizers with weights
    loss_fns = [
        1 * SquareLoss(projection_mask=fr.pm),
        1e4 * NegRegularizer(),
        diff_err:=difflam * DiffLoss(mr.grid, mode='density', device=device, _req='sq_diff_log')
    ]
    #%% recon2
    # loss_fns = [CheaterLoss(truth)]
    loss_fns += [req_err := ReqErr(truth, m.grid, mr.grid, interval=100)]
    coeffs, retrieved_meas, losses = gd(fr, meas, mr, lr=5e0, loss_fns=loss_fns, num_iterations=90000)

    retrieved = mr(coeffs)
    # zero out retrieval/truth below 3Re
    retrieved3 = retrieved.clone().detach()
    retrieved3[mr.grid.r < 3] = 0
    truth3 = truth.clone().detach()
    truth3[m.grid.r < 3] = 0
    maxerr = float(req_err(None, None, retrieved3, None))

    # compute measurement errors
    meas = meas * fr.pm
    retrieved_meas = retrieved_meas * fr.pm
    nonzero = meas != 0
    retrieved_meas_error = t.zeros_like(meas)
    retrieved_meas_error[nonzero] = (retrieved_meas - meas)[nonzero] / meas[nonzero]

    # ----- Plotting -----
    # %% debug
    from glide.science.plotting_sph import coldenserr
    coldenserr(meas, f, truth, outdir='/www/recon')
    # %% plot
    from dech import *

    Img3 = lambda *a, **kw: Img(*a, height=300, *kw)

    # Img3 = lambda *a, **kw: Img(*a, *kw)
    fig_coeffs = []
    if isinstance(m, SphHarmModel):
        fig_coeffs += [Figure("Truth Coeffs", Img(sphharmplot(coeffs_truth, m), height=300))]
    if isinstance(mr, SphHarmModel):
        fig_coeffs += [Figure("Recon Coeffs", Img(sphharmplot(coeffs, mr), height=300))]

    display_dir = Path('/srv/www/sph')
    file = display_dir / f'{mstr}_{desc}.html'
    p = Page(
        [
            [
                Figure("Relative Err.", Img3(carderr(retrieved3, truth3, mr.grid, m.grid))),
                # HTML(' '.join([
                #     f'{maxerr=}',
                # ]))
                Figure("Loss", Img(loss_plot(losses))),
                HTML(fr.op.plot().to_jshtml())
            ],
            [
                # Figure("Relative Err.", Img3(carderrmin(retrieved3, truth3, mr.grid, m.grid))),
            ] + fig_coeffs,
            [
                Figure("Truth", Img(preview3d(cn(truth), m.grid), animation=True, rescale='sequence')),
                Figure("Recon", Img(preview3d(cn(retrieved), mr.grid), animation=True, rescale='sequence')),
            ],
            [Figure("Density", Img(cardplot(retrieved3, mr.grid)))],
            [
                # Figure("Truth Obs", Img(meas.cpu(), animation=True, format='gif')),
                # Figure("Recon Obs", Img(retrieved_meas.cpu(), animation=True, format='gif')),
                # Figure("Percent Err Obs", Img(retrieved_meas_error.cpu(), animation=True, format='gif')),
                # Figure("Truth Obs", HTML(image_stack(meas.cpu(), fr.vg).to_jshtml())),
                # Figure("Recon Obs", HTML(image_stack(retrieved_meas.cpu(), fr.vg).to_jshtml())),
                # Figure("Percent Err Obs", HTML(image_stack(retrieved_meas_error.cpu(), fr.vg).to_jshtml()))
            ],
            Code(code),
            Code(inspect.getsource(diff_err.compute))
        ],
        css='.animation {height: 300px};'
    )
    p.save(file)
    print(f"Wrote to {file}")
    plt.close('all')

    from datetime import datetime
    # archive error plot and all code
    p.save(display_dir / 'archive' / f'{datetime.now().isoformat()}.html')
