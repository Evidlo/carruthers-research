#!/usr/bin/env python3

from glide.common_components.camera import CameraWFI, CameraNFI, CameraL1BWFI, CameraL1BNFI
from glide.common_components.cam import nadir_wfi_mode, nadir_nfi_mode
from glide.common_components.generate_view_geom import gen_mission
from glide.common_components.orbits import circular_orbit
from glide.science.forward_sph import *
from glide.science.model_sph import *
from glide.science.plotting import *
from glide.science.plotting_sph import carderr, cardplot, carderrmin, cardplotaxes
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

mailstatus = 'failure'

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
    # [1e2], # difflam
    [16], # cpoints
    [360], # integration time
)
# items = [['spring', 1e-1]]

# for season, difflam, t_op in items:
# for num_obs, win, season, difflam, t_op, in items:
for num_obs, win, season, cpoints, t_op, in items:
    t.cuda.empty_cache()

    # integrate for full time between each snapshot
    # t_op = win * 24 / num_obs

    cams = [
        CameraL1BNFI(nadir_nfi_mode(t_op=t_op)),
        CameraL1BWFI(nadir_wfi_mode(t_op=t_op))
    ]
    sc = gen_mission(num_obs=num_obs, duration=win, start=season, cams=cams)
    # sc = gen_mission(num_obs=30, orbit=circular_orbit, cams=cams)

    # m = PratikModel(grid=None, date=season, rlim=(3, 25), device=device); mstr = 'pratik'
    # m = GonzaloModel(grid=None, device=device); mstr = 'gonzalo'
    grid = BiResGrid((200, 60, 80), ce=30, ca=30, angle=45, spacing='log')
    m = Zoennchen24Model(grid=grid, device=device); mstr = 'zoennchen'
    truth = m()

    # cal = Calibrator(..., scene=Scene(iph=False))

    f = ForwardSph(
        sc, m.grid, # calibrator=cal
        use_albedo=(ual:=True), use_aniso=(uan:=True), use_noise=(uno:=True),
        science_binning=(sb:=True), device=device
    )


    meas = f.noise(truth); noise_type = 'real'
    # meas = f.noise(truth, noise_engine='alex'); noise_type = 'alex'
    # meas = f.noise(truth, disable_noise=True); noise_type = 'realdisabled'
    # meas = f(truth); noise_type = 'noiseless
    # meas = f.fake_noise(truth); noise_type='fake'

    del f

    # ----- Retrieval -----
    #%% recon
    # mr = FullyDenseModel(m.grid)
    # choose a model for retrieval
    # mr = SphHarmModel(default_grid(size_r=m.grid.size.r, shape=(49, 1, 3)), max_l=3, device=device)
    # mr = SphHarmModel(grid, device=device, max_l=3)
    fr = ForwardSph(
        sc, grid,
        use_albedo=ual, use_aniso=uan, use_noise=uno,
        science_binning=sb, device=device
    )
    # %% debug
    mr = SphHarmSplineModel(grid, max_l=3, device=device, cpoints=cpoints, spacing='log')
    mr.proj = lambda coeffs: coeffs
    # fr = f
    # choose loss functions and regularizers with weights
    loss_fns = [
        # 1 * SquareLoss(projection_mask=fr.pm),
        1 * AbsLoss(projection_mask=fr.pm),
        # 1e2 * SquareRelLoss(projection_mask=fr.pm),
        1e4 * NegRegularizer(),
        # differr:=difflam * DiffLoss(mr.grid, mode='density', device=device, _reg='sq_diff_log')
        # differr:=1e-1 * DiffLoss(mr.grid, mode='density', device=device, _reg='gshp_diff_log')
        # differr:=difflam * DiffLoss(mr.grid, mode='density', device=device, _reg='log_corr_diff')
    ]
    # loss_fns = [CheaterLoss(truth)]
    loss_fns += [req_err := ReqErr(truth, m.grid, mr.grid, interval=100)]
    coeffs, retrieved_meas, losses = gd(
        fr, meas, mr, lr=5e0,
        loss_fns=loss_fns, num_iterations=120000,
    )

    # %% debug2

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

    # desc = f'{win:02d}d_{num_obs:02d}obs_{{noise_type}}{t_op//60}hr_{differr.lam:.0e}_{season}_init_{differr._reg}'
    cshape = 'x'.join(map(str, mr.coeffs_shape))
    errtype = type(loss_fns[0]).__name__.lower()
    # desc = f'spline{cshape}_{win:02d}d_{num_obs:02d}obs_{{noise_type}}{t_op//60}hr_{season}'
    # desc = desc.format(noise_type=noise_type)
    desc = f'spline{cshape}_{win:02d}d_{num_obs:02d}obs_{errtype}_{noise_type}{t_op//60}hr_{season}'

    print('-----------------------------')
    print(desc)
    print('-----------------------------')
    t.save(coeffs, f'/tmp/coeffs_{desc}.tr')

    # ----- Plotting -----
    # %% plot
    from dech import *

    Img3 = lambda *a, **kw: Img(*a, height=300, *kw)

    # Img3 = lambda *a, **kw: Img(*a, *kw)
    fig_coeffs = []
    if issubclass(type(m), SphHarmModel):
        fig_coeffs += [Figure("Truth Coeffs", Img(sphharmplot(m.sph_coeffs(coeffs_truth), m), height=300))]
    if issubclass(type(mr), SphHarmModel):
        fig_coeffs += [Figure("Recon Coeffs", Img(sphharmplot(mr.sph_coeffs(coeffs), mr), height=300))]

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
                Figure("Relative Err. (min. grid)", Img3(carderrmin(retrieved3, mr.grid, m.__class__))),
            ] + fig_coeffs,
            [
                Figure("Truth", Img(preview3d(cn(truth), m.grid), animation=True, rescale='sequence')),
                Figure("Recon", Img(preview3d(cn(retrieved), mr.grid), animation=True, rescale='sequence')),
            ],
            [
                Figure("Density", Img3(cardplot(retrieved3, mr.grid))),
                Figure("Density", Img3(cardplotaxes(retrieved3, mr.grid))),
            ],
            [
                # Figure("Truth Obs", Img(meas.cpu(), animation=True, format='gif')),
                # Figure("Recon Obs", Img(retrieved_meas.cpu(), animation=True, format='gif')),
                # Figure("Percent Err Obs", Img(retrieved_meas_error.cpu(), animation=True, format='gif')),
                # Figure("Truth Obs", HTML(image_stack(meas.cpu(), fr.vg).to_jshtml())),
                # Figure("Recon Obs", HTML(image_stack(retrieved_meas.cpu(), fr.vg).to_jshtml())),
                # Figure("Percent Err Obs", HTML(image_stack(retrieved_meas_error.cpu(), fr.vg).to_jshtml()))
            ],
            Code(code),
            # Code(inspect.getsource(differr.compute))
        ],
        css='.animation {height: 300px};'
    )
    p.save(file)
    print(f"Wrote to {file}")
    plt.close('all')

    from datetime import datetime
    # archive error plot and all code
    p.save(display_dir / 'archive' / f'{datetime.now().isoformat()}.html')

mailstatus = 'success'
