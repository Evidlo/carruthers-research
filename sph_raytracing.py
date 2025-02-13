#!/usr/bin/env python3
# doing some simple parallel computations with jax

import numpy as np
from numpy import newaxis as na
import torch as t

def filter_r(points, max_r):
    points[np.linalg.norm(points, axis=-1) > max_r] = float('nan')

def filter_r_torch(points, max_r):
    # find voxels greater than radius
    greater = t.linalg.norm(points, axis=-1) > max_r
    greater = t.repeat_interleave(greater.unsqueeze(-1), 3, dim=-1)
    points[greater] = float('nan')
    return points

def a(xs, rays, num_planes):
    """Compute intersections of rays with azimuth planes

    Args:
        xs (tuple): starting points of rays (num_rays, 3)
        rays (tuple): directions of rays (num_rays, 3)
        num_planes (int): Number of azimuthal planes

    Returns:
        t (ndarray): distance of each point from x along ray
            (num_rays, num_planes).  Can be negative
        points (ndarray): intersection points of rays with planes
            (num_rays, num_planes, 3)
    """
    xs, rays, num_planes = map(np.asarray, (xs, rays, num_planes))
    theta = np.linspace(0, 2 * np.pi, num_planes, endpoint=False)
    planes = np.stack((np.cos(theta), np.sin(theta), np.zeros(num_planes)), axis=-1)

    dotproduct = lambda a, b: np.einsum('abc,ijc->ib', a, b)
    # distance along ray
    t = (
        -dotproduct(planes[None, :, :], xs[:, None, :]) /
        dotproduct(planes[None, :, :], rays[:, None, :])
    )
    points = xs[:, None, :] + t[:, :, None] * rays[:, None, :]

    return t, points

def e(xs, rays, theta):
    """Compute intersections of rays with elevation cones

    Args:
        xs (tuple): starting points of rays (num_rays, 3)
        rays (tuple): directions of rays (num_rays, 3)
        theta (int or list[float]): Number of elevation cones or cone elevations

    Returns:
        t (ndarray): distance of each point from x along ray
            (num_rays, num_planes, 2).  Can be negative
        points (ndarray): intersection points of rays with planes
            (num_rays, num_planes, 2, 3)

    Reference: http://lousodrome.net/blog/light/2017/01/03/intersection-of-a-ray-and-a-cone/
    Reference: "Intersection of a Line and a Cone", David Eberly, Geometric Tools
    """
    if type(theta) is int:
        theta = np.linspace(0, np.pi/2, theta, endpoint=True)
    xs = np.asarray(xs, dtype='float')
    rays = np.asarray(rays, dtype='float')

    rays /= np.linalg.norm(rays, axis=1)[:, None] # (num_rays, 3)

    # (num_rays, num_cones)

    v = np.array((0, 0, 1))

    dotproduct = lambda a, b: np.einsum('ij,ij->i', a, b)
    a = rays[:, 2:]**2 - (np.cos(theta)**2)[None, :]
    b = 2 * (rays[:, 2:] * xs[:, 2:] - dotproduct(rays, xs)[:, None] * (np.cos(theta)**2)[None, :])
    c = xs[:, 2:]**2 - (np.linalg.norm(xs, axis=1)**2)[:, None] * (np.cos(theta)**2)[None, :]

    # a = dotproduct(rays, v)[:, None] - (np.cos(theta)**2)[None, :]
    # b = 2 * (dotproduct(rays, v) *)

    # ray parallel to cone
    # t_parallel = np.empty((len(rays), num_cones, 2))
    # t_parallel[:, :, 0] = -c / b
    # t_parallel[:, :, 1] = float('inf')

    # ray not parallel to cone
    delta = b**2 - 4*a*c

    # compute single or double intersection
    is_single = np.isclose(delta, 0)

    # ignore warnings about sqrt(-1) and /0
    with np.errstate(invalid='ignore'):
        t1 = (-b + np.sqrt(delta)) / (2 * a)
        t2 = (-b - np.sqrt(delta)) / (2 * a)

    t_normal = np.empty((len(rays), len(theta), 2))
    t_normal[:, :, 0] = np.where(is_single, -2*c / b, t1)
    t_normal[:, :, 1] = np.where(is_single, float('inf'), t2)

    # FIXME: we don't check if ray is parallel to cone
    # is_parallel = np.isclose(a, 0) and not np.isclose(b, 0)
    # is_parallel = np.logical_and(
    #     np.isclose(a, 0),
    #     np.logical_not(np.isclose(b, 0))
    # )
    # t = np.choose(is_parallel, (t_normal, t_parallel))
    t = t_normal

    # ignore warnings about nan, inf multiplication
    with np.errstate(invalid='ignore'):
        points = rays[:, na, na, :] * t[:, :, :, na] + xs[:, na, na, :]

    return t, points


def e_single(x, ray, theta=np.pi/4):
    x = np.asarray(x, dtype='float')
    ray = np.asarray(ray, dtype='float')

    v = np.array((0, 0, 1))

    ray /= np.linalg.norm(ray)

    a = np.dot(ray, v)**2 - np.cos(theta)**2
    b = 2 * (np.dot(ray, v) * np.dot(x, v) - np.dot(ray, x) * np.cos(theta)**2)
    c = np.dot(x, v)**2 - np.dot(x, x) * np.cos(theta)**2
    delta = b**2 - 4*a*c

    if np.isclose(delta, 0):
        t = -c / b, float('inf')
    else:
        t = (-b + np.sqrt(delta)) / (2*a), (-b - np.sqrt(delta)) / (2*a)

    return a, b, c, t

def e_single2(x, ray):
    theta = np.pi / 4
    x = np.asarray(x, dtype='float')
    ray = np.asarray(ray, dtype='float')

    ray /= np.linalg.norm(ray)

    a = ray[1]**2 - np.cos(theta)**2
    b = 2 * (ray[1] * x[1] - np.dot(ray, x) * np.cos(theta)**2)
    c = x[1]**2 - np.linalg.norm(x)**2 * np.cos(theta)**2

    if np.isclose(a, 0) and not np.isclose(b, 0):
        t = -c / b, float('inf')
    else:
        delta = b**2 - 4*a*c
        t = (-b + np.sqrt(delta)) / (2*a), (-b - np.sqrt(delta)) / (2*a)

    return t

def r(xs, rays, radii, limits=None):
    """Compute intersections of ray with concentric spheres

    Args:
        xs (tuple): starting points of rays (num_rays, 3)
        rays (tuple): directions of rays (num_rays, 3)
        radii (int or list): number of spheres, or radius of each sphere
        limits (tuple[float] or None): if ``radii`` is int, this is the
            upper/lower radial limit of logarithmically spaced spheres

    Returns:
        t (ndarray): distance of each point from x along ray
            (num_rays, num_spheres, 2).  Can be negative
        points (ndarray): intersection points of rays with spheres
            (num_rays, num_spheres, 2, 3)

    Reference: https://kylehalladay.com/blog/tutorial/math/2013/12/24/Ray-Sphere-Intersection.html
    """
    if isinstance(radii, int):
        radii = np.logspace(np.log10(limits[0]), np.log10(limits[1]), radii)
    elif isinstance(radii, (list, np.ndarray)):
        pass
    else:
        raise ValueError("radii must be int or list")

    xs = np.asarray(xs, dtype='float')
    rays = np.asarray(rays, dtype='float')

    rays /= np.linalg.norm(rays, axis=1)[:, None] # (num_rays, 3)

    dotproduct = lambda a, b: np.einsum('ij,ij->i', a, b)

    tc = dotproduct(-xs, rays) # (num_rays)
    d = np.sqrt(dotproduct(xs, xs) - tc**2) # (num_rays)

    # ignore sqrt(-1) warning
    with np.errstate(invalid='ignore'):
        t1c = np.sqrt(radii[None, :]**2 - d[:, None]**2) # (num_rays, num_spheres)

    t = np.empty((len(rays), len(radii), 2))
    t[:, :, 0], t[:, :, 1] = tc[:, None] - t1c, tc[:, None] + t1c

    points = rays[:, na, na, :] * t[:, :, :, na] + xs[:, na, na, :]

    return t, points

if __name__ == "__main__":
    # x = np.array((-3, 3, 0))
    # ray = np.array((1, 0, 0))
    xs = np.array([(-3, 0, 0), (-3, 0, 0)])
    rays = np.array([(1, 0, 0), (1, 0, 0)])
    radii = np.array([0, 1, 2])


    xs = np.asarray(xs, dtype='float')
    rays = np.asarray(rays, dtype='float')

    radii = np.array([0, 1, 2])

    rays /= np.linalg.norm(rays, axis=1)[:, None] # (num_rays, 3)

    dotproduct = lambda a, b: np.einsum('ij,ij->i', a, b)

    tc = dotproduct(-xs, rays) # (num_rays)
    d = np.sqrt(dotproduct(xs, xs) - tc**2) # (num_rays)

    t1c = np.sqrt(radii[None, :]**2 - d[:, None]**2) # (num_rays, num_spheres)

    result = tc[:, None] - t1c, tc[:, None] + t1c # (2, num_rays, num_spheres)
    print('p1s:', result[0])
    print('p2s:', result[1])

    print(r(xs, rays, radii))
