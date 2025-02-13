#!/usr/bin/env python3
# directly fit sph harm model to data

import torch_optimizer as optim
from itertools import product

from glide.science.model import ZoennchenModel, SphHarmBasisModel, default_vol, density2xr, vol2cart, PratikModel
from glide.science.recon.loss import *
from glide.science.recon.gd import gd
from glide.science.plotting import carderr, carderraxes, cardplot, cardplotaxes, sphharmplot

from dech import *

desc = ''

s = 50
vol = default_vol(shape=s, size=50)

m = PratikModel(vol, device='cuda', rinner=3); desc = '_pratik'
# m = ZoennchenModel(vol, device='cuda'); desc = '_zoennchen'

density_truth = m()

figs_a = []
figs_b = []
for max_l in (2, 3, 4):
# for max_l in (4,):
    figrow_a = []
    figrow_b = []

    for shell in (10, 20, 30):
    # for shell in (30,):

        s = SphHarmBasisModel(vol, num_shells=shell, max_l=max_l, axis=(1, 0, 0), device='cuda')

        _, _, coeffs, recon = gd(
            lambda _: _, None,
            model=s,
            num_iterations=1000,
            lr=1e2,
            optimizer=(optimizer:=optim.Yogi),
            loss_fn=cheater_loss_gen(density_truth),
            loss_history=True,
        )
        figrow_a.append(
            Figure(
                f"shell={shell}, L<={max_l}",
                Img(carderr(recon, density_truth, vol), width=400)
                # Img(carderraxes(recon, density_truth, vol), width=400)
                # Img(cardplotaxes(recon, density_truth, vol), width=400)
            )
        )
        figrow_b.append(
            Figure(
                f"shell={shell}, L<={max_l}",
                Img(sphharmplot(coeffs, s), width=400)
            )
        )

    figs_a.append(figrow_a)
    figs_b.append(figrow_b)


# %% plot

f = f'/srv/www/display/direct_fit{desc}.html'
Page(figs_a).save(f)
Page(figs_b).save(f'/srv/www/display/direct_fit_coeffs{desc}.html')
print(f'Saved to {f}')