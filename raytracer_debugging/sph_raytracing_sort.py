#!/usr/bin/env python3

import numpy as np
import torch as t
import matplotlib.pyplot as plt
from sph_raytracing import *

from glide.science.plotting import *

# xs = np.array([(-100, 1, 0)])
xs = np.array([
    (-100, 1, 0),
    (-100, 3, 0),
    (-100, 5, 0),
    (-100, 7, 0),
    (-100, 9, 0),
    (-100, -9, 0),
    (-100, -7, 0),
    (-100, -5, 0),
    (-100, -3, 0),
    (-100, -1, 0),
    # (-100, 0, 0),
])
rays = np.array([(1, 0, 0)] * len(xs))


"""
limits (tuple[float] or None): if ``radii`` is int, this is the
    upper/lower radial limit of logarithmically spaced spheres
"""

# if isinstance(radii, int):
#     radii = np.logspace(np.log10(limits[0]), np.log10(limits[1]), radii)
# if isinstance(radii, int):
#     limits = tr.asarray(limits, **spec)
#     radii = tr.logspace(tr.log10(limits[0]), tr.log10(limits[1]), radii, **spec)
# if isinstance(theta, int):
#     # theta = np.linspace(0, 2 * np.pi, theta, endpoint=False)
#     theta = tr.linspace(0, 2 * np.pi, theta + 1, **spec)[:-1]
# if isinstance(phi, int):
#     theta = np.linspace(0, np.pi/2, phi, endpoint=True)
# if isinstance(phi, int):
#     theta = tr.linspace(0, np.pi/2, phi, **spec)

# vol = (2, 1, 1)
# vol = (10, 1, 1)
vol = (50, 1, 50)
rs = np.linspace(0, 9, vol[0])
# rs = [9]
phis = np.linspace(0, np.pi, vol[1], endpoint=True)
# thetas = np.linspace(0, 2 * np.pi, vol[2], endpoint=False)
thetas = np.linspace(-np.pi, np.pi, vol[2], endpoint=False)

# r_t, r_points, r_regs, r_inds, r_ns = r(xs, rays, rs)
# e_t, e_points, e_regs, r_inds, e_ns = e(xs, rays, phis)
# a_t, a_points, a_regs, r_inds, a_ns = a(xs, rays, thetas)
r_t, r_points, r_regs, r_inds, r_ns = r_torch(xs, rays, rs)
# FIXME:
r_t, r_points, r_regs, r_inds, r_ns = r_t[:, :-1], r_points[:, :-1], r_regs[:, :-1], r_inds[:, :-1], r_ns[:, :-1]
e_t, e_points, e_regs, e_inds, e_ns = e_torch(xs, rays, phis)
a_t, a_points, a_regs, a_inds, a_ns = a_torch(xs, rays, thetas)
r_kinds = 0 * t.ones_like(r_t, dtype=int)
e_kinds = 1 * t.ones_like(e_t, dtype=int)
a_kinds = 2 * t.ones_like(a_t, dtype=int)

max_r = rs.max()
filter_r_torch(r_t, r_points, max_r)
filter_r_torch(e_t, e_points, max_r)
filter_r_torch(a_t, a_points, max_r)

# concatenate intersection distances/points from all geometry kinds
all_ts = t.cat((r_t, e_t, a_t), dim=1)
all_points = t.cat((r_points, e_points, a_points), dim=1)
# keep track of which kind the intersection point is and the geometry index
all_kinds = t.cat((r_kinds, e_kinds, a_kinds), dim=1)
all_regs = t.cat((r_regs, e_regs, a_regs), dim=1)
all_inds = t.cat((r_inds, e_inds, a_inds), dim=1)
all_ns = t.cat((r_ns, e_ns, a_ns), dim=1)

# sort points by distance
# https://discuss.pytorch.org/t/sorting-and-rearranging-multi-dimensional-tensors/148340
all_ts_s, s = all_ts.sort(dim=1)
all_kinds_s = all_kinds.gather(1, s)
all_regs_s = all_regs.gather(1, s)
all_inds_s = all_inds.gather(1, s)
all_ns_s = all_ns.gather(1, s)
# s_expanded = all_regs[:, :, None].repeat_interleave(3, dim=2)
s_expanded = s[:, :, None].repeat_interleave(3, dim=2)
all_points_s = all_points.gather(1, s_expanded)

xxx = tr.zeros([np.prod(all_kinds_s.shape)] + list(vol))
last = 0
i = 0
kmap = {0:"r", 1:"e", 2:"a"}
for kinds, regs, inds, ns, points, ts in zip(all_kinds_s, all_regs_s, all_inds_s, all_ns_s, all_points_s, all_ts_s):
    start_reg = find_starts(points[0], rs, phis, thetas)
    c = list(start_reg.numpy())
    print('start_reg:', c)
    for kind, reg, ind, n, p, t_ in zip(kinds, regs, inds, ns, points, ts):
        # xxx[c[0], c[1], c[2]] = i
        # xxx[c[0], c[1], c[2]] = kind + 1
        # xxx[i, c[0], c[1], c[2]] = kind + 1
        xxx[i] += last
        c[kind] = int(reg)
        p = np.array(p)
        print(
            kmap[int(kind)],
            f'r:{reg:<2}',
            f'i:{ind:<2}',
            f'n:{n:<2}',
            f'c:[{c[0]:>2},{c[1]:>2},{c[2]:>2}]',
            f'p:[{p[0]:>4.1f},{p[1]:>4.1f},{p[2]:>4.1f}]',
            f't:{t_:>.1f}'
        )
            # f'{c[0]:>2}/{vol[0]}',
            # f'{c[1]:>2}/{vol[1]}',
            # f'{c[2]:>2}/{vol[2]}',
        if not np.isinf(t_) and not np.isnan(t_):
            xxx[i, c[0], c[1], c[2]] += 1
        last = xxx[i]
        i += 1

# %% plot
# plt.subplot(1, 2, 1)
# plt.imshow(xxx[20].sum(axis=1))
# plt.colorbar()

# sum over elevation dimension and scale image size up
result = xxx.sum(axis=2)
result = np.repeat(result, 5, axis=1)
result = np.repeat(result, 5, axis=2)
save_gif('/srv/www/test.gif', result, rescale='sequence', duration=5)

plt.close()
