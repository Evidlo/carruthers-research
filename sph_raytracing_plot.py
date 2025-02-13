#!/usr/bin/env python3

# verifying sph raytracing implementation visually

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sph_raytracing import *

xs = np.array([(-3, 0, 0)])
rays = np.array([(1, 0.1, 0.1)])

num_lines = 50
xs = xs + np.array((0, 1, 1))[None, :] * np.linspace(-5, 5, num_lines)[:, None]
rays = rays.repeat(num_lines, 0)


xs = np.asarray(xs, dtype='float')
rays = np.asarray(rays, dtype='float')

r_t, r_points = r(xs, rays, np.linspace(0, 4, 10))
e_t, e_points = e(xs, rays, 11)
a_t, a_points = a(xs, rays, 11)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# plt.setp(axes, aspect='equal')
# plt.setp(axes, axis='equal')
# plt.setp(axes, grid=True)
r_ax, e_ax, a_ax = axes

r_ax.scatter(r_points[:, :, :, 0].flatten(), r_points[:, :, :, 1].flatten())
r_ax.set_aspect('equal')
r_ax.axis('equal')
r_ax.set_xlabel('X');r_ax.set_ylabel('Y')
r_ax.grid(True)
r_ax.set_title('Radial Shells')

e_ax.scatter(e_points[:, :, :, 0].flatten(), e_points[:, :, :, 2].flatten())
e_ax.set_aspect('equal')
e_ax.axis('equal')
e_ax.set_xlabel('X'); e_ax.set_ylabel('Z')
e_ax.grid(True)
e_ax.set_title('Elevation Cones')

a_ax.scatter(a_points[:, :, 0].flatten(), a_points[:, :, 1].flatten())
a_ax.set_aspect('equal')
a_ax.axis('equal')
a_ax.set_xlabel('X'); a_ax.set_ylabel('Y')
a_ax.grid(True)
a_ax.set_title('Azimuth Planes')

plt.tight_layout()
plt.savefig('/srv/www/out.png')