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

from astropy.time import Time

mailstatus = 'failure'

__file__ = 'recon_dynamic2static.py'
code = open(__file__).read()

device = 'cuda'

# ----- Setup -----
# %% setup

from itertools import product
items = product(
    # 4.033],
    [1], # num_obs
    [1], # window (days)
    # ['spring'], # season
    [Time('2025-12-24')], # 'best case' with new ephem
    # [1e2], # difflam
    [16], # cpoints
    [360], # integration time
    [3]
)
# items = [['spring', 1e-1]]

# for season, difflam, t_op in items:
# for num_obs, win, season, difflam, t_op, in items:
for num_obs, win, season, cpoints, t_op, L in items:
    t.cuda.empty_cache()

    # integrate for full time between each snapshot
    # t_op = win * 24 / num_obs

    cams = [
        CameraL1BNFI(nadir_nfi_mode(t_op=t_op)),
        CameraL1BWFI(nadir_wfi_mode(t_op=t_op))
    ]
    sc = gen_mission(num_obs=num_obs, duration=win, start=season, cams=cams)
    # sc = gen_mission(num_obs=num_obs, orbit=circular_orbit, cams=cams)

    # m = PratikModel(grid=None, date=season, rlim=(3, 25), device=device); mstr = 'pratik'
    # m = GonzaloModel(grid=None, device=device); mstr = 'gonzalo'
    sgrid = DefaultGrid((500, 45, 60), size_r=(3, 25), spacing='log')
    rgrid = DefaultGrid((200, 45, 60), size_r=(3, 25), spacing='log')
    # m = Zoennchen24Model(grid=sgrid, device=device); mstr = 'zoennchen'
    m = Pratik25Model(grid=sgrid, device=device); mstr = 'pratik'

    # m = Pratik25Model(window=14, freq='15.5h', device=device); mstr = 'pratik_dyn' # just 22 equally spaced observations over 14 days, starting march 1
    # m.resize(new_shape=(500,45,60), size_r=(3,25))
    # sgrid = m.grid
    # m = TIMEGCMModel(grid=grid, device=device, offset=np.datetime64(10, 'D')); mstr = 'TIMEGCM'
    truth = m()
    # import ipdb
    # ipdb.set_trace()
    # cal = Calibrator(..., scene=Scene(iph=False))

    print('----- Measurement Forward Model -----')
    f = ForwardSph(
        sc, sgrid=sgrid, # calibrator=cal
        rgrid=rgrid,
        use_albedo=(ual:=True), use_aniso=(uan:=True), use_noise=(uno:=True),
        device=device
    )

    #%% ding
    meas = f.noise(t.repeat_interleave(truth[None, ...], len(cams), dim=0)); noise_type = 'dynamic_grid'


    # ----- Retrieval -----
    #%% recon
    print('----- Recon. Forward Model -----')
    # %% debug
    mr = SphHarmSplineModel(rgrid, max_l=L, device=device, cpoints=cpoints, spacing='log')
    mr.proj = lambda coeffs: coeffs # FIXME: I think this is unnecessary now


    # choose loss functions and regularizers with weights
    loss_fns = [
        # 1 * SquareLoss(projection_mask=f.proj_maskb),
        1 * AbsLoss(projection_mask=f.proj_maskb),
        1e4 * NegRegularizer(),
    ]
    # loss_fns = [CheaterLoss(truth)]
    middle = len(truth) // 2
    if m.grid.dynamic:
        loss_fns += [req_err := ReqErr(truth[middle], m.grid, mr.grid, interval=100)]
    else:
        loss_fns += [req_err := ReqErr(truth, m.grid, mr.grid, interval=100)]

    if L > 0:
        # do a fast initialization reconstruction with L=0
        mrinit = SphHarmSplineModel(rgrid, max_l=0, device=device, cpoints=cpoints, spacing=mr.spacing)
        initcoeffs = t.zeros(mr.coeffs_shape, device=device)
        initcoeffs.data[0:1, :], _, _ = gd(
            f, meas, mrinit, lr=5e0,
            loss_fns=loss_fns, num_iterations=1000,
        )
        # %% debug2
        # do full reconstruction
        coeffs, retrieved_meas, losses = gd(
            f, meas, mr, lr=5e0,
            loss_fns=loss_fns, num_iterations=10000,
            coeffs=initcoeffs,
        )
    else:
        # A00 retrieval
        coeffs, retrieved_meas, losses = gd(
            f, meas, mr, lr=5e0,
            loss_fns=loss_fns, num_iterations=2500,
        )

    # %% debug3
    # truth = m()[m().shape[0]//2] # take centroid of data
    # m.grid= sgrid
    retrieved = mr(coeffs)
    # FIXME - ugly, get rid of this eventually
    # zero out retrieval/truth below 3Re
    retrieved3 = retrieved.clone().detach()
    retrieved3[mr.grid.r < 3] = 0
    truth3 = truth.clone().detach()
    truth3[..., m.grid.r < 3, :, :] = 0
    maxerr = float(req_err(None, None, retrieved3, None))

    # compute measurement errors
    meas = meas * f.proj_maskb
    retrieved_meas = retrieved_meas * f.proj_maskb
    nonzero = meas != 0
    retrieved_meas_error = t.zeros_like(meas)
    retrieved_meas_error[nonzero] = (retrieved_meas - meas)[nonzero] / meas[nonzero]

    # desc = f'{win:02d}d_{num_obs:02d}obs_{{noise_type}}{t_op//60}hr_{differr.lam:.0e}_{season}_init_{differr._reg}'
    cshape = 'x'.join(map(str, mr.coeffs_shape))
    errtype = type(loss_fns[0]).__name__.lower()
    # desc = f'spline{cshape}_{win:02d}d_{num_obs:02d}obs_{{noise_type}}{t_op//60}hr_{season}'
    # desc = desc.format(noise_type=noise_type)
    # desc = f'spline{cshape}_{win:02d}d_{num_obs:02d}obs_{errtype}_{noise_type}{t_op//60}hr_{season}'
    desc = f'spline{cshape}L{mr.max_l}_{num_obs:02d}obs'

    print('-----------------------------')
    print(desc)
    print('-----------------------------')
    #t.save(coeffs, f'/tmp/coeffs_{desc}.tr')

    # ----- Plotting -----
    # %% plot
    from dech import *

    Img3 = lambda *a, **kw: Img(*a, height=300, *kw)

    # Img3 = lambda *a, **kw: Img(*a, *kw)
    fig_coeffs = []
    # show special spherical harmonic coefficient plot if truth/retrieval model is sphharm
    # if issubclass(type(m), SphHarmModel):
    #     fig_coeffs += [Figure("Truth Coeffs", Img(sphharmplot(m.sph_coeffs(coeffs_truth), m), height=300))]
    if issubclass(type(mr), SphHarmModel):
        fig_coeffs += [Figure("Recon Coeffs", Img(sphharmplot(mr.sph_coeffs(coeffs), mr), height=300))]

    display_dir = Path('/srv/www/mate_testing')
    file = display_dir / f'{mstr}_{desc}.html'
    p = Page(
        [
            [
                Figure("Relative Err.", Img3(carderr(retrieved3, truth3, mr.grid, m.grid))), # took out middle index
                # HTML(' '.join([
                #     f'{maxerr=}',
                # ]))
                # Figure("Loss", Img(loss_plot(losses))),
                HTML(f.op.plot().to_jshtml())
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
                # Figure("Truth Obs", HTML(image_stack(meas.cpu(), f.vg).to_jshtml())),
                # Figure("Recon Obs", HTML(image_stack(retrieved_meas.cpu(), f.vg).to_jshtml())),
                # Figure("Percent Err Obs", HTML(image_stack(retrieved_meas_error.cpu(), f.vg).to_jshtml()))
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
