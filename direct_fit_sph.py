#!/usr/bin/env python3
# directly fit various reconstruction model to different datasets
# this means there is no projection operator.  we are just directly fitting densities
# which serves as a "best-case" guideline for checking whether reconstruction models
# fit a particular dataset

from itertools import product
import matplotlib
matplotlib.use('Agg')

from sph_raytracer.retrieval import gd
from sph_raytracer.loss import CheaterLoss
from sph_raytracer.model import FullyDenseModel

from glide.science.model_sph import *
from glide.science.plotting_sph import carderr, cardplot
from glide.science.plotting import sphharmplot

from dominate_tags import Img
import dominate
from dominate.tags import style, div, figure, figcaption, code, pre

device = 'cuda'
# grid = default_grid(spacing='log')
grid = DefaultGrid(size_r=(3, 15))

truth_models = (
    # PratikModel(grid, device=device, season='spring'),
    # ZoennchenModel(device=device),
    # GonzaloModel(grid, device=device),
    # Zoennchen24Model(grid, device=device),
    # MSISModel(grid=grid, fill_value=0, num_times=1, window=14, device=device),
    TIMEGCMModel(grid=grid, fill_value=0, num_times=1, window=14, device=device),
)

truth_models = [
    TIMEGCMModel(
        grid=grid, fill_value=0, num_times=1, window=14, device=device,
        offset=np.timedelta64(houroffset, 'h')
    )
    for houroffset in np.arange(0, 10*24, 6)
]


recon_models = (
    # SphHarmModel(grid, max_l=2, device=device, monotonic=True),
    # SphHarmModel(grid, max_l=3, device=device, monotonic=True),
    # SphHarmModel(grid, max_l=4, device=device, monotonic=True),
    # SphHarmModel(grid, max_l=5, device=device),
    # SplineModel(grid, (10, 10, 10), device=device),
    # SplineModel(grid, (10, 10, 5), device=device),
    # SplineModel(grid, (20, 50, 50), device=device),
    # SplineModel(grid, (30, 50, 50), device=device),

    SphHarmSplineModel(grid, max_l=0, device=device, cpoints=16, spacing='log'),
    SphHarmSplineModel(grid, max_l=1, device=device, cpoints=16, spacing='log'),
    SphHarmSplineModel(grid, max_l=2, device=device, cpoints=16, spacing='log'),
    SphHarmSplineModel(grid, max_l=3, device=device, cpoints=16, spacing='log'),
    SphHarmSplineModel(grid, max_l=4, device=device, cpoints=16, spacing='log'),
    SphHarmSplineModel(grid, max_l=5, device=device, cpoints=16, spacing='log'),
)

figures = []


from glide.debug import warning_exception
warning_exception()

for n_t, truth_model in enumerate(truth_models):
    truth = truth_model()
    truth[truth < 1] = 1
    grid = truth_model.grid


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
            sphharm = Img(sphharmplot(recon_model.sph_coeffs(coeffs), recon_model), height=200)
        else:
            sphharm = ''
        figures.append(
            figure(
                figcaption(f"truth={truth_model}, recon={recon_model}, {truth_model.orig_grid.nptime}"),
                div(
                    Img(carderr(recon.squeeze(), truth.squeeze(), grid, grid), height=200),
                    sphharm,
                    cls="imgcontainer"
                )
            )
        )

    figures.append(
        figure(
            figcaption(f"Truth {truth_model.orig_grid.nptime}"),
            Img(cardplot(truth.squeeze(), grid))
        )
    )


# %% plot

doc = dominate.document('Direct Fit Sph')

with doc.head:
    # col_fmt = '1fr ' * 2 * len(models)
    col_fmt = 'min-content ' * len(recon_models)
    style(f"""
        .gridcontainer {{
            display: grid;
            grid-template-columns: {col_fmt};
            grid-auto-flow: row;
        }}
        .inline {{
            display: inline
        }}
        .imgcontainer {{
            display: flex;
            flex-direction: row;
        }}
        figure {{
            margin: 0;
            font-size: 8pt;
        }}
    """)

    # style(f'.container {{display: grid; grid-auto-columns: min-content;}}')

with doc:
    div(figures, cls='gridcontainer')
    code(pre(open(__file__).read()))

f = Path(f'/srv/www/direct_fit.html')
f.write_text(doc.render())
print(f'Saved to {f}')