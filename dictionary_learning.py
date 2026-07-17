#!/usr/bin/env python3

"""
Try to learn a dictionary which fits a dataset and which has good
conditioning for a particular forward operator

cost = sum_i  ||xᵢ - DWcᵢ||₂² + λ ||FDW||_f²
"""

from glide.science.forward_sph import ForwardSph, ScienceGeom, NativeGeom
from glide.common_components.camera import CameraL1BWFI, CameraL1BNFI
from glide.common_components.cam import nadir_wfi_mode, nadir_nfi_mode
from glide.common_components.generate_view_geom import gen_mission
from glide.science.model_sph import *
from glide.science.recon.loss_sph import ReqErr

from glide.science.plotting_sph import cardplot, carderr
from glide.science.plotting import sphharmplot
from sph_raytracer.retrieval import gd
from sph_raytracer.plotting import image_stack

import torch as t
import math

# ----- Problem Setup -----

spec = {'device':'cuda', 'dtype':t.float64}
device = 'cuda'

cams = [
    CameraL1BNFI(nadir_nfi_mode(t_op=360)),
    CameraL1BWFI(nadir_wfi_mode(t_op=360))
]
sc = gen_mission(num_obs=1, cams=cams)

rgrid = DefaultGrid((200, 45, 60), size_r=(3, 15), spacing='log')
sgrid = DefaultGrid((500, 45, 60), size_r=(3, 15), spacing='log')

# load data model
# FIXME: this is ugly
# dm = TIMEGCMModel(device=device)
model = TIMEGCMModel
# model = Pratik25StormModel
dm = model(device=device)
dyn_grid = DefaultGrid(
    t=dm.grid.t,
    r_b=rgrid.r_b, e_b=rgrid.e_b, a_b=rgrid.a_b,
)
sdyn_grid = DefaultGrid(
    t=dm.grid.t,
    r_b=sgrid.r_b, e_b=sgrid.e_b, a_b=sgrid.a_b,
)
dm = model(grid=dyn_grid, device=device)
dataset = dm()
# sdm = model(grid=sdyn_grid, device=device)
sdm = TIMEGCMModel(grid=sdyn_grid, device=device)
sdataset = sdm()

f = ForwardSph(sc, rgrid, sgrid=sgrid, device=device, use_albedo=True, use_aniso=True, use_noise=True)
# ----- Learning Setup -----
# %% setup

num_atoms = 8
m = SphHarmSplineModel(rgrid, max_l=2, cpoints=16, device=device, extra_channels=[num_atoms])

# random init method
weights = t.rand(m.coeffs_shape, **spec)

# ones rows init method
# weights = t.zeros(m.coeffs_shape, **spec)
# initialize weights with weights[l==atom_num] = 1
# for atom in range(num_atoms):
#     weights[atom, m.l == atom, :] = 1

# SVD init method
"""
def ident(dims):
    mesh = t.meshgrid(*(t.arange(d, device=device) for d in dims), indexing='xy')
    ident = t.zeros((*dims, *dims), **spec)
    ident[*mesh, *mesh] = 1
    return ident

# LOS for all separate bases
# flatten LOS (num_vant, num_x, num_y)
# flatten bases (num_harmonics, cpoints)
y_sph = f(m(ident((9, 16))))
y_sph_flat = y_sph.flatten(-3).flatten(0, 1).mT
u, s, v = t.svd(y_sph_flat)
# weights = s[None, None, :num_atoms] * v.reshape((len(m.l), m.cpoints, -1))[:, :, :num_atoms]
weights =  v.reshape((len(m.l), m.cpoints, -1))
weights = t.moveaxis(weights, -1, 0)

# y_weights = u.T[:num_atoms].reshape((-1, *f.rvg.shape))
# verified: this is the same as f(mr())
y_weights2 = u.T.reshape((-1, *f.rvg.shape))
"""



weights.requires_grad_()
mr = LearnedDictionaryModel(m, weights)

y_weights = f(mr())


# ----- Learning -----
# %% recon


from dictionary_learning_losses import *
# loss_fns = [
#     1 * CheaterLoss(dataset),
#     # 1 * ConditionLoss(mr),
# ]
# loss_fns[0].kind = 'fidelity'

# coeffs, retrieved_meas, losses = gd(
#     f, None, mr, lr=5e0,
#     loss_fns=loss_fns, num_iterations=500,
#     coeffs=coeffs,
#     optim_vars=[coeffs, weights]
# )

# %% recon2

t.cuda.empty_cache()


coeffs = t.ones((dm.grid.shape.t, mr.coeffs_shape), requires_grad=True, **spec)
num_iterations = 5000
progress_bar = True
from sph_raytracer.retrieval import detach_loss

best_loss = float('inf')
best_coeffs = None

optim = t.optim.Adam([coeffs, weights], lr=1e-1)
# perform requested number of iterations
# f_meas = f(dataset)
try:
    from tqdm import tqdm
    for _ in (pbar := tqdm(range(num_iterations), disable=not progress_bar)):
        optim.zero_grad()

        # --- fidelity term ---
        # (y - FDWc)_2^2
        # f_result = t.mean((f_meas - f(mr(coeffs)))**2)
        # (y - FDWc)_f^2
        # f_result = t.linalg.norm(f_meas.flatten(-3) - f(mr(coeffs)).flatten(-3), ord='fro')**2
        # f_loss = 1 * f_result / (math.prod(f.range_shape) * mr.num_atoms)

        # orthogonal weights
        # a, b = (0, 1, 2), (2, 1, 0)
        # q, r = t.linalg.qr(weights.moveaxis(a, b))
        # orth_weights = (q @ t.tril(r)).moveaxis(b, a)
        orth_weights = weights

        f_result = t.mean((dataset - mr.m(t.einsum('w...,tw...->t...', orth_weights, coeffs)))**2)
        # f_result = t.max(t.mean((dataset - mr.m(t.einsum('w...,tw...->t...', orth_weights, coeffs)))**2, dim=(1, 2, 3)))
        # FIXME: t.mean already normalizes out the shape of dm.grid
        f_loss = 1 * f_result / math.prod(dm.grid.shape)


        # --- regularization term ---
        # r_meas = f(mr()).flatten(start_dim=-3)
        # r_result = t.linalg.norm(r_meas, ord='fro')**2
        # normalize loss by number of LOS, dict
        # r_loss = 1e-2 * r_result / math.prod(f.range_shape) * mr.num_atoms

        r_loss = 0
        r_loss = 1e2 * t.mean(-coeffs.clip(max=0))

        # --- sparsity term ---
        # s_loss = 0
        # ||W||_1
        # s_result = t.abs(weights).sum()
        # encourage coefficients to be on same atom
        # s_loss = 1e-6 * s_result / math.prod(weights.shape)

        # ||sum_(radius) DW||_0
        # s_result = m.sph_coeffs(weights).sum(dim=-1).abs()
        # s_loss = 1e1 * s_result.sum() / math.prod(s_result.shape)

        # W = weights.abs().sum(dim=-1)
        # s_loss = 1e-9 * ((W @ W.T) - t.eye(W.shape[0], device=W.device)).pow(2).sum()
        # gram = W @ W.T
        # s_loss = 1e-9 * (gram.triu(1).pow(2).sum() + gram.tril(-1).pow(2).sum())
        s_result = t.linalg.norm(f(mr()).flatten(1).T, ord='fro')
        s_loss = 1e-9 * s_result / math.prod(f.rvg.shape)

        tot_loss = f_loss + r_loss + s_loss

        pbar.set_description(f'F:{f_loss:.1e} R:{r_loss:.1e} S:{s_loss:.1e}')

        # save the reconstruction with the lowest loss
        if tot_loss < best_loss:
            best_coeffs = coeffs

        tot_loss.backward(retain_graph=True)
        optim.step()

# allow user to stop iterations
except KeyboardInterrupt:
    pass

# ----- Recon -----
# %% recon
from dominate_tags import *

from sph_raytracer.loss import *
from glide.science.recon.loss_sph import *

fig_recons = []
for truth_ind in [0, 50, 100, 150, 200, 250]:
    loss_fns = [
        1 * AbsLoss(projection_mask=f.proj_maskb),
        1e4 * NegRegularizer(),
        ReqErr(dataset[truth_ind], m.grid, mr.grid, interval=100)
    ]

    f_noisy = f.noise(sdataset[truth_ind])
    # f_noisy = f(dataset[truth_ind])
    retr_coeffs, retr_meas, losses = gd(
        f, f_noisy, mr, lr=5e-2,
        loss_fns=loss_fns, num_iterations=500,
    )
    retrieved = mr(retr_coeffs)
    fig_recons.append(
        caption(
            f"truth={dm}, recon={mr}, {dm.grid.nptime[truth_ind]}",
            plot(cardplot(retrieved, mr.grid), height="300px"),
            plot(sphharmplot(m.sph_coeffs(mr.l_coeffs(retr_coeffs)), m), height="300px"),
            plot(carderr(retrieved, dataset[truth_ind], mr.grid, dm.grid), height="300px")
        )
    )

# ----- Plotting -----
# %% plot

import matplotlib.pyplot as plt
import matplotlib
plt.close('all')
matplotlib.use('Agg')


with document('Dictionary Learning') as d:
    util.raw(f"""
    This is a single-vantage reconstruction of samples from the {type(dm).__name__} dataset
    using a dictionary learned from {type(dm).__name__}.
    """)
    l_bases = mr() # learned 3D bases
    with itemgrid(len(l_bases), flow='column'):
        for n, (weight, l_basis) in enumerate(zip(orth_weights, l_bases)):
            print(n)
            plt.close()
            caption(
                f"Figure {n}",
                plot(cardplot(l_basis, mr.grid), height="300px"),
                plot(sphharmplot(m.sph_coeffs(weight), m), height="300px"),
                plot(image_stack(y_weights[n, 0], f.rvg[0]))
            )
    tags.h1("Reconstructions")
    tags.hr()
    tags.div(fig_recons)


    # n = 0 # look at nth result in stack
    # recon = mr(coeffs)[n]
    # recon_coeffs = t.einsum('w...,tw->t...', weights, coeffs)[n]
    # caption(
    #     "Sum",
    #     plot(cardplot(recon, mr.grid), height="300px"),
    #     plot(sphharmplot(m.sph_coeffs(recon_coeffs), m), height="300px"),
    # )
    tags.pre(open('dictionary_learning.py', 'r').read())

open(outfile:=f'/www/dictionary/{type(dm).__name__}_{type(sdm).__name__}.html', 'w').write(d.render())
print(f'Wrote {outfile}')
from datetime import datetime
open(outfile:=f'/www/dictionary_archive/{datetime.now().isoformat()}.html', 'w').write(d.render())
print(f'Wrote {outfile}')