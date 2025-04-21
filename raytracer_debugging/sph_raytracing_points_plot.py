#!/usr/bin/env python3

# verifying sph raytracing intersection points visually

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sph_raytracing import *


def filter_r_torch(t, points, max_r):
    """Sort points by distance.  Then filter out invalid intersections (nan t values)
    and points which lie outside radius `max_r` (inplace)

    Args:
        t (tensor): 2D distances array of shape (...)
        points (tensor): 3D points array of shape (..., 3)
        max_r (float): replace points outside this radius with (nan, nan, nan)
    """

    # find voxels greater than radius
    greater = tr.linalg.norm(points, axis=-1) > max_r
    t[greater] =  float('inf')
    greater = tr.repeat_interleave(greater.unsqueeze(-1), 3, dim=-1)
    points[greater] = float('nan')



# %% rays

spec = {'dtype': 'float32'}

# choose a central LOS


# towards +X
# xs = np.array((-3, 0, 0), **spec)
# rays = np.array((1, 0, 0), **spec)

# towards +Z
# xs = np.array((0, 0, -3), **spec)
# rays = np.array((0, 0, 1), **spec)

# towards +XY
# xs = np.array((-3, -3, 0), **spec)
# rays = np.array((1, 1, 0), **spec)

# towards +XYZ
xs = np.array((3, 3, 3), **spec)
rays = np.array((-1, -1, -1), **spec)

# xs = np.array((-3, 0, 0), **spec)
# rays = np.array((1, 0, 1), **spec)
# xs = np.array((0, 0, 3), **spec)
# rays = np.array((0.1, 0, -1), **spec)
# xs = np.array((-3, 0, -1), **spec)
# rays = np.array((1, 1, 0), **spec)

# build grid of LOS centered around LOS defined above
# this is our 'detector plane'.  make it perp. to central LOS
# FIXME: this is so fugly
np.random.seed(1)
rrr = np.random.random(3)
shift1 = np.cross(rays, rrr / np.linalg.norm(rrr))
shift2 = np.cross(shift1, rays)
num_lines = 200
xs = (
    xs[None, None, :]
    + shift1[None, None, :] * np.linspace(-10, 10, num_lines)[:, None, None]
    + shift2[None, None, :] * np.linspace(-10, 10, num_lines)[None, :, None]
)
# ).reshape((num_lines**2, 3))
rays = rays[None, None, :].repeat(num_lines, 0).repeat(num_lines, 1)

# %% plot

# define spherical grid
rs = np.linspace(0, 4, 5)
phis = np.linspace(0, np.pi, 9, endpoint=True)
thetas = np.linspace(0, 2 * np.pi, 5, endpoint=False)
# thetas = np.linspace(-np.pi, np.pi, 5, endpoint=False)
# thetas = np.deg2rad([90, 120, 150, 180, 210, 240, 270, 300])
# thetas = np.deg2rad([1, 15])
phis = np.deg2rad((10, 20, 30, 90, 170))
thetas = np.deg2rad((0, 72, 144, 216, 288))

# r_t, r_points = r(xs, rays, rs)[:2]
# e_t, e_points = e(xs, rays, phis)[:2]
# a_t, a_points = a(xs, rays, thetas)[:2]
# filter_r(r_t, r_points, 4)
# filter_r(e_t, e_points, 4)
# filter_r(a_t, a_points, 4)

r_t, r_points = r_torch(xs, rays, rs)[:2]
e_t, e_points = e_torch(xs, rays, phis)[:2]
a_t, a_points = a_torch(xs, rays, thetas)[:2]
filter_r_torch(r_t, r_points, 4)
filter_r_torch(e_t, e_points, 4)
filter_r_torch(a_t, a_points, 4)
r_t, e_t, a_t = r_t.reshape(40000, -1), e_t.reshape(40000, -1), a_t.reshape(40000, -1)
r_points, e_points, a_points = r_points.reshape(40000, -1, 3), e_points.reshape(40000, -1, 3), a_points.reshape(40000, -1, 3)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='3d'))
# plt.setp(axes, aspect='equal')
# plt.setp(axes, axis='equal')
# plt.setp(axes, grid=True)
r_ax, e_ax, a_ax = axes

r_ax.scatter(r_points[:, :, 0].flatten(), r_points[:, :, 1].flatten(), r_points[:, :, 2].flatten(), alpha=.1)
r_ax.set_aspect('equal')
r_ax.axis('equal')
# r_ax.set_xlabel('X');r_ax.set_ylabel('Y')
r_ax.set_xlabel('X'); r_ax.set_ylabel('Y'); r_ax.set_zlabel('Z')
r_ax.grid(True)
# r_ax.view_init(elev=0, azim=-90, roll=0)
r_ax.set_title('Radial Shells')

# e_ax.scatter(e_points[:, :, :, 0].flatten(), e_points[:, :, :, 2].flatten())
e_ax.scatter(e_points[:, :, 0].flatten(), e_points[:, :, 1].flatten(), e_points[:, :, 2].flatten(), alpha=.1)
e_ax.set_aspect('equal')
e_ax.axis('equal')
# e_ax.set_xlabel('X'); e_ax.set_ylabel('Z')
e_ax.set_xlabel('X'); e_ax.set_ylabel('Y'); e_ax.set_zlabel('Z')
e_ax.grid(True)
e_ax.view_init(elev=30, azim=-90, roll=0)
e_ax.set_title('Elevation Cones')

# a_ax.scatter(a_points[:, :, 0].flatten(), a_points[:, :, 1].flatten())
a_ax.scatter(a_points[:, :, 0].flatten(), a_points[:, :, 1].flatten(), a_points[:, :, 2].flatten(), alpha=.1)
a_ax.set_aspect('equal')
a_ax.axis('equal')
# a_ax.set_xlabel('X'); a_ax.set_ylabel('Y')
a_ax.set_xlabel('X'); a_ax.set_ylabel('Y'); a_ax.set_zlabel('Z')
a_ax.grid(True)
a_ax.view_init(elev=80, azim=-45, roll=0)
a_ax.set_title('Azimuth Planes')

plt.tight_layout()
plt.savefig('/srv/www/out.png')