#!/usr/bin/env python3
# https://robertdyro.com/sensitivity_torch/tour/

import torch as t
from torch import optim
from torch.autograd import grad
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch_bspline_uncertainty_quantification import BSpline

time = t.linspace(0, 1, 1000 + 1)[:-1]
# x_truth = t.sin(4 * t.pi * time)
x_truth = t.e**(-time * 5)


# sensing matrix
sens = [t.e**(-((time - offset) / .05)**2) for offset in t.linspace(0, 1, 20)]
sens = t.stack(sens)

def f(x):
    return sens @ x

sigma2 = 20  # Noise variance
def f_noise(x):
    y_truth = f(x)
    return y_truth + math.sqrt(sigma2) * t.randn_like(y_truth)

# Inputs
y_truth = f(x_truth)
y = f_noise(x_truth)

M = BSpline(time, 4) # Number of splines
lambda_reg = 1e-3  # Regularization parameter


# Suppose x_hat solves: ∇_x L(x, y) = 0

def L(x, y):
    return t.norm(f(M(x)) - y)**2 # + lambda_reg * R(x)


# Define the estimator (e.g., via an optimizer)
def solve_inverse_problem(y):
    global hist
    # Implement or unroll optimizer to find x_hat
    x_hat = t.rand(M.num_cpoints, requires_grad=True)
    # optimizer = optim.Adam([x_hat], lr=1e-4, weight_decay=1e8)
    # optimizer = optim.Adam([x_hat], lr=1e-4, betas=(.2, .8))
    optimizer = optim.AdamW([x_hat], lr=1e-4, weight_decay=1e-0); iterations=80000
    # optimizer = optim.AdamW([x_hat], lr=1e-3, weight_decay=1e-0); iterations=30000
    # optimizer = optim.SGD([x_hat], lr=1e-12, momentum=0)

    for _ in (bar:=tqdm(range(iterations), desc='iterations')):
        loss = L(x_hat, y)
        loss.backward()
        bar.set_description(f'{loss:.1e}')
        optimizer.step()

        hist.append((x_hat.detach().clone(), f'{loss:.1e}'))

    return x_hat  # shape: [n_x]

hist = []
# Step 1: Solve for c_hat (e.g., by gradient descent or closed form)
try:
    x_hat = solve_inverse_problem(y)
except KeyboardInterrupt:
    pass

# %% uncert

# Step 2: Define FOC: F(x, y) = ∇_x L(x, y)
def F(x, y):
    grad_x, = grad(L(x, y), x, create_graph=True)
    return grad_x  # shape: [n_x]

# Step 3: Compute Jv = d(M x_hat)/dy using implicit diff

def implicit_jvp(F, x_hat, y, v):
    # Compute ∂F/∂x (Jacobian wrt x)
    def Fx(x): return F(x, y)
    # J_xv, = t.autograd.functional.jvp(Fx, (x_hat,), (t.eye(x_hat.numel()).to(x_hat.device),))
    # J_xv, = t.autograd.functional.jvp(Fx, (x_hat,), (v,))
    # J_x = J_xv  # shape: [n_x, n_x] (dense for now)
    J_x, = t.autograd.functional.jacobian(Fx, (x_hat,))

    # Compute ∂F/∂y @ v
    def Fy(y_): return F(x_hat, y_)
    _, J_yv = t.autograd.functional.jvp(Fy, (y,), (v,))

    # Solve J_x dx = -J_yv  ⇒ dx/dy @ v = - J_x^{-1} @ J_yv
    dx_dy_v = t.linalg.solve(J_x, -J_yv)

    # Compute M dx/dy @ v
    return M(dx_dy_v)

result = implicit_jvp(F, x_hat, y, t.randn_like(y, requires_grad=True))

# ----- Plotting -----
# %% plot
import matplotlib
matplotlib.use('Agg')
plt.close('all')

# c = t.zeros_like(c)
# c[::2] = 1

def plot_knots(M):
    cpoint_locations = M.knot_vector[3:-3]
    for c in cpoint_locations:
        plt.axvline(c, color='lightgray', zorder=0)

plt.close()
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(time, x_truth)
plt.title('x truth')

plt.subplot(2, 3, 4)
plt.plot(time, sens.T)
plt.title('sensing')

plt.subplot(2, 3, 2)
plt.plot(y_truth, '-*')
plt.title('y truth')

plt.subplot(2, 3, 5)
plt.plot(y, '-*')
plt.title('y noise')

plt.subplot(2, 3, 3)
plt.plot(time, x_retr:=M(x_hat.detach()))
plt.title('x retrieved')
plot_knots(M)

plt.subplot(2, 3, 6)
plt.plot(time, (x_retr - x_truth) / x_truth, '-*')
plt.title('x err')

plt.savefig('/tmp/out2.png')


# %% anim
plt.close('all')

from matplotlib.animation import ArtistAnimation
plots = []
fig, ax = plt.subplots(1)
for x_hist, title in hist[::len(hist)//100]:
    ax.plot(time, x_truth, color='blue')
    a = ax.plot(time, M(x_hist), color='red')
    ax.set_title(title)
    plots.append(a)
ArtistAnimation(fig, plots, interval=20).save('/tmp/out.gif')