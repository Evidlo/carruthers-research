#!/usr/bin/env python3
# directly fit various reconstruction model to different datasets
# this means there is no projection operator.  we are just directly fitting densities
# which serves as a "best-case" guideline for checking whether reconstruction models
# fit a particular dataset

from itertools import product
import matplotlib
from matplotlib.colors import LogNorm, SymLogNorm
matplotlib.use('Agg')

from sph_raytracer.retrieval import gd
from sph_raytracer.loss import CheaterLoss
from sph_raytracer.model import FullyDenseModel

from glide.science.model_sph import *
from glide.science.plotting_sph import carderr, cardplot, cardplotaxes
from glide.science.plotting import sphharmplot, loss_plot
from glide.science.recon.loss_sph import L1Loss

from dominate_tags import *


device = 'cuda'
# grid = default_grid(spacing='log')
grid = DefaultGrid((500, 50, 50), size_r=(3, 15))

truth_models = (
    # PratikModel(grid, device=device, season='spring'),
    # ZoennchenModel(device=device),
    # GonzaloModel(grid, device=device),
    # MSISModel(grid=grid, fill_value=0, num_times=1, window=14, device=device),

    Zoennchen24Model(grid, device=device),
    GonzaloModel(grid, device=device),
    Pratik25Model(grid, num_times=1, device=device),
    Pratik25StormModel(grid, num_times=1, offset=7, freq=None, device=device),
    TIMEGCMModel(grid, fill_value=0, num_times=1, device=device),
    TIMEGCMStormModel(grid, fill_value=0, num_times=1, device=device),
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
# for l, c in product(l_opts:=(0, 1, 2), c_opts:=(3, 4, 6, 8)):
# for l, c in product(l_opts:=(0, 1), c_opts:=(8, 12)):
# for l, c in product(l_opts:=[0], c_opts:=[2]):
for l, c in product(l_opts:=(0, 1, 2, 3, 4), c_opts:=(8, 12, 16)):
    recon_models += [SphHarmSplineModel(grid, max_l=l,
        cpoints=c, device=device, kind='bspline', spacing='log'
    )]
# for c in (c_opts:=(8, 12, 16)):
#     recon_models += [SphHarmSplineModel(
#         grid, lm=[(l, 0) for l in range(4)],
#         cpoints=c, device=device, kind='catmullrom', spacing='log'
#     )]


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
            for nr, mr in enumerate(recon_models):
                print(f'truth:{n_t+1}/{len(truth_models)}  recon:{nr+1}/{len(recon_models)}')

                # create the loss and set to fidelity so it is minimized
                loss = CheaterLoss(truth)
                loss.kind = 'fidelity'
                coeffs, _, losses = gd(
                    lambda _: _, None,
                    model=mr,
                    num_iterations=500,
                    # coeffs=list(mr.spline.parameters())[0],
                    # lr=1e0,
                    lr=1e2,
                    # optimizer=(optimizer:=optim.Yogi),
                    loss_fns=[loss],
                    # coeffs=t.ones(mr.coeffs_shape, device=device, dtype=t.float32, requires_grad=True),
                    coeffs=t.ones(mr.coeffs_shape, device=device, dtype=t.float64, requires_grad=True),
                    device=device
                )
                # FIXME: check for negative values in plotting functions
                recon = mr(coeffs).clamp(1)
                if issubclass(type(mr), SphHarmModel):
                    sphharm = plot(sphharmplot(mr.sph_coeffs(coeffs), mr), height=200)
                else:
                    sphharm = ''

                # figure settings
                figset = {'height': 200}

                # remove DC component
                c2 = coeffs.clone()
                c2[0, :] = 0
                recon2 = mr(c2)

                caption(
                    f"recon={mr}",
                    plot(carderr(recon.squeeze(), truth.squeeze(), grid, grid), height=200),
                    sphharm,

                    tags.details(
                        tags.summary(),
                        plot(loss_plot(losses), **figset),
                        caption("Recon", plot(cardplot(recon.squeeze(), grid, norm='log'), **figset)),
                        caption("Truth", plot(cardplot(truth.squeeze(), grid, norm='log'), **figset)),
                        caption("Recon", plot(cardplotaxes(recon.squeeze(), grid, yscale='log'), **figset)),
                        caption("Truth", plot(cardplotaxes(truth.squeeze(), grid, yscale='log'), **figset)),
                        caption("Recon (no DC)", plot(cardplot(recon2.squeeze(), grid, norm=SymLogNorm(10)), **figset)),
                    )
                )

    tags.code(tags.pre(open('direct_fit_sph.py').read()))


# %% plot
# f = Path(f'/www/lara/direct_fit.html')
f = Path(f'/www/sph/out_15_flip.html')
f.write_text(doc.render())
print(f'Saved to {f}')