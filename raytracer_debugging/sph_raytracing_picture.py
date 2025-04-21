#!/usr/bin/env python3

from contexttimer import Timer
from sph_raytracing import *
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('Agg')
plt.close('all')

import torch as t

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2000"
t.cuda.empty_cache()
t.set_num_threads(48)

spec, ispec = SPEC, ISPEC

spec['device'] = ispec['device'] = 'cuda'
# spec['device'] = ispec['device'] = 'cpu'

# vol = SphericalVol(shape=(30, 60, 100))
# vol = SphericalVol(shape=(50, 50, 50))
from glide.science.model import *
glidevol = default_vol(shape=100, size=50)
m = PratikModel(glidevol, method='linear')
thetas = np.concatenate((m.a.to_numpy(), [np.pi]))
vol = SphericalVol(rs=m.r.to_numpy(), phis=m.e.to_numpy(), thetas=thetas)

xs = np.array([-100, 0, 0])

rays = np.array([1, 0, 0])
shift1 = np.array([0, 0, .01])
shift2 = np.array([0, .01, 0])
num_lines = 512
rays = (
    rays[None, None, :]
    + shift1[None, None, :] * np.linspace(10, -10, num_lines)[:, None, None]
    + shift2[None, None, :] * np.linspace(-10, 10, num_lines)[None, :, None]
).reshape((num_lines, num_lines, 3))
# rays = np.array([[[1, 0, 0]]])
xs = xs[None, None, :].repeat(num_lines, 0).repeat(num_lines, 1)


regs, lens = trace_indices(vol, xs, rays, spec, ispec, invalid=False, debug=True)

# %% fancy
# r_c, e_c, a_c = r.to('cuda'), e.to('cuda'), a.to('cuda')
# lens = lens.to('cuda')
# x = t.rand(vol.shape, dtype=tr.float32, device='cuda')
# import imageio
# img = imageio.imread('/tmp/out.png')[:100, 20:80, 0]
# img[img == 255] = 0

# img = np.asarray(img, dtype='float64')
# img = t.from_numpy(img)
# x[-1, :, :] = img.T**2

# --- multiple geoms ---
# fig, axes = plt.subplots(1, 3)
# xs = t.zeros((3, *vol.shape), **spec)
# s = min(vol.shape) - min(vol.shape) // 4
# xs[0, s, :, :] = 1
# xs[1, :, s, :] = 1
# xs[2, :, :, s] = 1
# axes[0].set_title('Single Shell')
# axes[1].set_title('Single Cone')
# axes[2].set_title('Single Wedge')

# for ax, x in zip(axes, xs):
#     with Timer(prefix='integrate'):
#         for _ in range(10):
#             result = (x[r, e, a] * lens).sum(axis=-1)

#     ax.imshow(result.detach().cpu())

# --- checkerboard ---
# x = t.zeros(vol.shape, **spec)
# s = 12
# o = t.zeros((s, s))
# o[:s//2, :s//2] = 1
# o[s//2:, s//2:] = 1
# o = t.tile(o, (vol.shape[1] // s + 1, vol.shape[2] // s + 1))[:vol.shape[1], :vol.shape[2]]
# x[-3:, :, :] = o
# result = (x[r, e, a] * lens).sum(axis=-1)
# plt.imshow(result.detach().cpu())
# plt.title('Checkerboard')
# plt.colorbar()

# --- nested shells ---
# x = t.zeros(vol.shape, **spec)
# # plt.title('Nested Shells')
# # x[-1, :, :] = 1
# # x[-10, :, :] += 1
# plt.title('Nested Solid Spheres')
# x[:-1, :, :] = 1
# x[:-10, :, :] += 1

# result = (x[r, e, a] * lens).sum(axis=-1)
# plt.imshow(result.detach().cpu())
# plt.colorbar()

# --- simple sphere ---
# x = t.ones(vol.shape, **spec)
# result = (x[r, e, a] * lens).sum(axis=-1)
# plt.imshow(result.detach().cpu())
# plt.colorbar()
# plt.title('Sphere')

# --- pratik ---
xs = t.asarray(m.ds.nH.to_numpy(), **spec)
@t.jit.script
def foo(xs, regs, num_lines):
    results = t.empty((xs.shape[0], num_lines, num_lines))
    for i, x in enumerate(xs):
        print(i)
        # results.append((x[regs] * lens).sum(axis=-1))
        results[i] = (x[regs] * lens).sum(axis=-1)
        return results
results = foo(xs, regs, num_lines)
from glide.science.plotting import *
save_gif('/srv/www/out.gif', results, rescale='sequence')
# show_gif(results, rescale='sequence')
# plt.imshow(result.detach().cpu())
# plt.colorbar()
# plt.title('Pratik')


plt.tight_layout()
plt.savefig('/srv/www/out2.png')