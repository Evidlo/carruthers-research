#!/usr/bin/env python3
# directly fit sph harm model to data

# import torch_optimizer as optim
from itertools import product
import matplotlib
matplotlib.use('Agg')

from sph_raytracer.retrieval import gd
from sph_raytracer.loss import CheaterLoss
from sph_raytracer.model import FullyDenseModel

from glide.science.model_sph import *
from glide.science.plotting_sph import carderr, cardplot
from glide.science.plotting import sphharmplot

import dominate
from dominate.tags import style, div, figure, figcaption

device = 'cuda'
# grid = default_grid(spacing='log')
grid = default_grid()

truth_models = (
    # PratikModel(grid, device=device, season='spring'),
    # ZoennchenModel(device=device),
    # GonzaloModel(grid, device=device),
    Zoennchen24Model(grid, device=device),
)

recon_models = (
    # SphHarmModel(grid, max_l=2, device=device, monotonic=True),
    # SphHarmModel(grid, max_l=3, device=device, monotonic=True),
    # SphHarmModel(grid, max_l=4, device=device, monotonic=True),
    # SphHarmModel(grid, max_l=5, device=device),
    SplineModel(grid, (10, 50, 50), device=device),
    SplineModel(grid, (20, 50, 50), device=device),
    SplineModel(grid, (30, 50, 50), device=device),
)

figures = []

def Img(content, class_="", title=None, width=None, height=None, animation=False,
        duration=3, rescale='frame', format=None):
    from dominate.util import raw
    from io import BytesIO
    import imageio
    import base64
    # generate styling
    styles = []
    if type(width) is int:
        width = f"{width}px"
    if width is not None:
        styles.append(f"width:{width}")
    if type(height) is int:
        height = f"{height}px"
    if height is not None:
        styles.append(f"height:{height}")

    # if given path to image
    if type(content) is str:
        src = content

    # if given matplotlib figure
    elif type(content).__name__ == 'Figure':
        buff = BytesIO()
        format = format or 'png'
        content.savefig(buff, format=format)
        src = 'data:image/{};base64,{}'.format(
            format,
            base64.b64encode(buff.getvalue()).decode()
        )

    # if given numpy array
    elif type(content).__name__ == 'ndarray':
        buff = BytesIO()
        styles.append("image-rendering:crisp-edges")
        if animation:
            format = format or 'gif'
            buff = gif(content, duration=duration, rescale=rescale, format=format)
            src = 'data:image/{};base64,{}'.format(
                format,
                base64.b64encode(buff.getvalue()).decode()
            )
        else:
            format = format or 'png'
            imageio.imsave(buff, content.astype('uint8'), format=format)
            src = 'data:image/{};base64,{}'.format(
                format,
                base64.b64encode(buff.getvalue()).decode()
            )

    elif content is None:
        src = ''

    else:
        raise TypeError(f"Unsupported object {type(content)}")

    style = ";".join(styles)
    return raw(f'<img class="{class_}" style="{style}" src="{src}"/>')

from glide.debug import warning_exception
warning_exception()

for truth_model in truth_models:
    # c = model.default_coeffs()
    # c = t.zeros(model.coeffs_shape, device=device)
    # c[1, 0] = -1
    # c[1, 1] = 1
    # c[5, 1] = .0975
    # c[clim:] = 0
    # c[5] = 0
    truth = truth_model()
    truth[truth < 1] = 1
    grid = truth_model.grid


    for recon_model in recon_models:


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
            loss_fns=[100 * loss],
            coeffs=t.ones(recon_model.coeffs_shape, device=device, dtype=t.float32, requires_grad=True),
            device=device
        )
        recon = recon_model(coeffs)
        if type(recon_model) is SphHarmModel:
            sphharm = Img(sphharmplot(coeffs, recon_model), height=200)
        else:
            sphharm = ''
        figures.append(
            figure(
                figcaption(f"truth={truth_model}, recon={recon_model}"),
                div(
                    Img(carderr(recon, truth, grid, grid), height=200),
                    sphharm,
                    cls="imgcontainer"
                )
            )
        )


# %% plot
f = Path(f'/srv/www/sph/direct_fit.html')

doc = dominate.document('Direct Fit Sph')

with doc.head:
    # col_fmt = '1fr ' * 2 * len(models)
    col_fmt = 'min-content ' * len(recon_models)
    style(f"""
        .gridcontainer {{
            display: grid;
            grid-template-rows: {col_fmt};
            grid-auto-flow: column;
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

f.write_text(doc.render())
print(f'Saved to {f}')