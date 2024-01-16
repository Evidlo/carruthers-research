import numpy as np
import torch
import tomosipo as ts
import tomosipo.torch_support
import matplotlib.pyplot as plt


def plot_imgs(height=3, cmap="gray", clim=(None, None), **kwargs):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(kwargs),
        figsize=(height * len(kwargs), height)
    )
    fig.patch.set_alpha(1.0)
    if len(kwargs) == 1:
        axes = [axes]
    for ax, (k, v) in zip(axes, kwargs.items()):
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        pcm = ax.imshow(v.squeeze(), cmap=cmap, clim=clim)
        fig.colorbar(pcm, ax=ax)
        ax.set_title(k)
    fig.tight_layout()


def operator_norm(A, num_iter=10):
    x = torch.randn(A.domain_shape)
    for i in range(num_iter):
        x = A.T(A(x))
        x /= torch.norm(x) # L2 vector-norm
    return (torch.norm(A.T(A(x))) / torch.norm(x)).item()

#  ------------------ forward -----------------
# %% forward


# Use GPU
dev = torch.device("cuda")

# Create tomosipo geometries
vg = ts.volume(shape=(1, 256, 256), size=(1/256, 1, 1))
pg = ts.parallel(angles=384, shape=(1, 384), size=(1/256, 1.5))
A = ts.operator(vg, pg)

phantom = torch.zeros(A.domain_shape, device=dev)
phantom[:, 32:224, 32:224] = 1.0  # box
phantom[:, 64:192, 64:192] = 0.0  # and hollow

y = A(phantom)
# Add 10% Gaussian noise
y += 0.1 * y.mean() * torch.randn(*y.shape, device=dev)

plot_imgs(
    phantom=phantom,
    sino=y.squeeze().transpose(0, 1)
)

plt.show()

# ------------------- adjoint -------------------
# %% adjoint

adj = A.T(y)
plot_imgs(
    adjoint=adj.squeeze()
)

plt.show()

# ------------------- recon -------------------
# %% recon

L = operator_norm(A, num_iter=100)
t = 1.0 / L                     # τ
s = 1.0 / L                     # σ
theta = 1                       # θ
N = 500                         # Compute 500 iterations

u = torch.zeros(A.domain_shape, device=dev)
p = torch.zeros(A.range_shape, device=dev)
u_avg = torch.clone(u)

for _ in range(N):
    p = (p + s * (A(u_avg) - y)) / (1 + s)
    u_new = torch.clamp(u - t * A.T(p), min=0, max=None)
    u_avg = u_new + theta * (u_new - u)
    u = u_new

rec = u.cpu()                   # move final reconstruction to CPU
