#!/usr/bin/env python3

# verifying sph raytracing implementation visually

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sph_raytracing import *

# %% rays

xs = np.array((-3, 0, 0))
rays = np.array((1, 0, 1))

# xs = np.array((-3, 0, -1))
# rays = np.array((1, 1, 0))

# xs = np.array((-3, 0, 0))
# rays = np.array((1, 0, 0.01))

shift1 = np.cross(rays, (0, 0, 1))
shift2 = np.cross(shift1, rays)
num_lines = 100
xs = (
    xs[None, None, :]
    + shift1[None, None, :] * np.linspace(-5, 5, num_lines)[:, None, None]
    + shift2[None, None, :] * np.linspace(-5, 5, num_lines)[None, :, None]
).reshape((num_lines**2, 3))
rays = rays[None, :].repeat(num_lines**2, 0)

# %% plot

# xs = np.asarray(xs, dtype='float')
# rays = np.asarray(rays, dtype='float')

r_t, r_points = r(xs, rays, np.linspace(0, 4, 5))
e_t, e_points = e(xs, rays, 8)
a_t, a_points = a(xs, rays, 5)

filter_r(r_points, 4)
filter_r(e_points, 4)
filter_r(a_points, 4)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, subplot_kw=dict(projection='3d'))
# plt.setp(axes, aspect='equal')
# plt.setp(axes, axis='equal')
# plt.setp(axes, grid=True)
r_ax, e_ax, a_ax = axes

# r_ax.scatter(r_points[:, :, :, 0].flatten(), r_points[:, :, :, 1].flatten())
r_ax.scatter(r_points[:, :, :, 0].flatten(), r_points[:, :, :, 1].flatten(), r_points[:, :, :, 2].flatten(), alpha=.1)
r_ax.set_aspect('equal')
r_ax.axis('equal')
# r_ax.set_xlabel('X');r_ax.set_ylabel('Y')
r_ax.set_xlabel('X'); r_ax.set_ylabel('Y'); r_ax.set_zlabel('Z')
r_ax.grid(True)
# r_ax.view_init(elev=0, azim=-90, roll=0)
r_ax.set_title('Radial Shells')

# e_ax.scatter(e_points[:, :, :, 0].flatten(), e_points[:, :, :, 2].flatten())
e_ax.scatter(e_points[:, :, :, 0].flatten(), e_points[:, :, :, 1].flatten(), e_points[:, :, :, 2].flatten(), alpha=.1)
e_ax.set_aspect('equal')
e_ax.axis('equal')
# e_ax.set_xlabel('X'); e_ax.set_ylabel('Z')
e_ax.set_xlabel('X'); e_ax.set_ylabel('Y'); e_ax.set_zlabel('Z')
e_ax.grid(True)
e_ax.view_init(elev=0, azim=-90, roll=0)
e_ax.set_title('Elevation Cones')

# a_ax.scatter(a_points[:, :, 0].flatten(), a_points[:, :, 1].flatten())
a_ax.scatter(a_points[:, :, 0].flatten(), a_points[:, :, 1].flatten(), a_points[:, :, 2].flatten(), alpha=.1)
a_ax.set_aspect('equal')
a_ax.axis('equal')
# a_ax.set_xlabel('X'); a_ax.set_ylabel('Y')
a_ax.set_xlabel('X'); a_ax.set_ylabel('Y'); a_ax.set_zlabel('Z')
a_ax.grid(True)
# a_ax.view_init(elev=40, azim=-45, roll=0)
a_ax.set_title('Azimuth Planes')

# plt.tight_layout()
plt.savefig('/srv/www/out.png')