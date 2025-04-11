#!/usr/bin/env python3
# directly fit various reconstruction model to different datasets
# this means there is no projection operator.  we are just directly fitting densities
# which serves as a "best-case" guideline for checking whether reconstruction models
# fit a particular dataset

from itertools import product
import matplotlib
from matplotlib.colors import LogNorm
matplotlib.use('Agg')

from sph_raytracer.retrieval import gd
from sph_raytracer.loss import CheaterLoss
from sph_raytracer.model import FullyDenseModel

from glide.science.model_sph import *
from glide.science.plotting_sph import carderr, cardplot
from glide.science.plotting import sphharmplot
from glide.science.recon.loss_sph import L1Loss

from dominate_tags import *


device = 'cuda'
# grid = default_grid(spacing='log')
grid = DefaultGrid(size_r=(3, 15))

truth_models = (
    # PratikModel(grid, device=device, season='spring'),
    # ZoennchenModel(device=device),
    # GonzaloModel(grid, device=device),
    # MSISModel(grid=grid, fill_value=0, num_times=1, window=14, device=device),
    Zoennchen24Model(grid, device=device),
    TIMEGCMModel(grid, fill_value=0, num_times=1, offset=np.timedelta64(0, 'D'), device=device),
    TIMEGCMModel(grid, fill_value=0, num_times=1, offset=np.timedelta64(10, 'D'), device=device),
    Pratik25Model(grid, num_times=1, offset=np.timedelta64(0, 'W'), device=device),
    Pratik25Model(grid, num_times=1, offset=np.timedelta64(7, 'W'), device=device),
)

# truth_models = [
#     TIMEGCMModel(
#         grid=grid, fill_value=0, num_times=1, window=14, device=device,
#         offset=np.timedelta64(houroffset, 'h')
#     )
#     for houroffset in np.arange(0, 6, 6)
# ]


# recon_models = (
    # SphHarmModel(grid, max_l=2, device=device, monotonic=True),
    # SphHarmModel(grid, max_l=3, device=device, monotonic=True),
    # SphHarmModel(grid, max_l=4, device=device, monotonic=True),
    # SphHarmModel(grid, max_l=5, device=device),
    # SplineModel(grid, (10, 10, 10), device=device),
    # SplineModel(grid, (10, 10, 5), device=device),
    # SplineModel(grid, (20, 50, 50), device=device),
    # SplineModel(grid, (30, 50, 50), device=device),
# )

recon_models = []
for l, c in product(l_opts:=(0, 1, 2, 3, 4, 5), c_opts:=(8, 12, 16)):
    recon_models += [SphHarmSplineModel(grid, max_l=l, cpoints=c, device=device, spacing='log')]



from glide.debug import warning_exception
warning_exception()

with document('Direct Fit') as doc:
    for n_t, truth_model in enumerate(truth_models):
        truth = truth_model()
        truth[truth < 1] = 1
        grid = truth_model.grid

        # figures.append(
        #     caption(
        #         f"Truth {truth_model.orig_grid.nptime}",
        #         plot(cardplot(truth.squeeze(), grid, norm=LogNorm()), height=200)
        #     )
        # )

        tags.h1(f'truth={truth_model}')

        with itemgrid(len(c_opts), flow='row'):
            for n_r, recon_model in enumerate(recon_models):
                print(f'truth:{n_t}/{len(truth_models)}  recon:{n_r}/{len(recon_models)}')

                # create the loss and set to fidelity so it is minimized
                loss = CheaterLoss(truth)
                loss.kind = 'fidelity'
                coeffs, _, losses = gd(
                    lambda _: _, None,
                    model=recon_model,
                    num_iterations=300,
                    # coeffs=list(recon_model.spline.parameters())[0],
                    # lr=1e0,
                    lr=1e2,
                    # optimizer=(optimizer:=optim.Yogi),
                    loss_fns=[loss],
                    # coeffs=t.ones(recon_model.coeffs_shape, device=device, dtype=t.float32, requires_grad=True),
                    coeffs=t.ones(recon_model.coeffs_shape, device=device, dtype=t.float64, requires_grad=True),
                    device=device
                )
                recon = recon_model(coeffs)
                if issubclass(type(recon_model), SphHarmModel):
                    sphharm = plot(sphharmplot(recon_model.sph_coeffs(coeffs), recon_model), height=200)
                else:
                    sphharm = ''

                caption(
                    f"recon={recon_model}",
                    plot(carderr(recon.squeeze(), truth.squeeze(), grid, grid), height=200),
                    sphharm,
                )

    tags.code(tags.pre(open('direct_fit_sph.py').read()))


# %% plot
f = Path(f'/www/lara/direct_fit.html')
# f = Path(f'/www/out.html')
f.write_text(doc.render())
print(f'Saved to {f}')