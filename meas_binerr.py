#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch as t

from sph_raytracer.plotting import image_stack

def meas_binerr(f, sm, rm, y_nois=None, y_less=None, y_rec=None):
    """Visualize binning error present in measurements.

    Compare Squared&Relative Err of Noisy vs Noiseless case.
    We expect the error of the noisy case to be at least 100x the noiseless case.
    If this is not true, then significant non-noise errors (e.g. binning) are present.

    Args:
        f (ForwardSph): forward operator
        sm (Model): simulator model on high resolution grid
        rm (Model): reconstruction model on reconstruction grid

        y_nois (tensor, optional): cached noisy+binned measurements
        y_less (tensor, optioanl): cached noiseless+binned measurements
        y_rec (tensor, optional): cached directly raytraced noiseless measurements

    Returns:
        figure
    """

    if y_nois is None and y_less is None and y_rec is None:
        # generate noiseless, noisy, and recon (science pixel) measurements
        y_nois = f.noise(sm())
        y_less = f.noise(sm(), disable_noise=True)
        y_rec = f(rm())

    # compute percent and squared err
    # set err=0 where y_rec=0
    divisor = t.where(y_rec == 0, float('inf'), y_rec)
    err_nois_sq = (y_nois - y_rec)**2
    err_less_sq = (y_less - y_rec)**2
    # err_nois_pc = (y_nois - y_rec) / divisor * 100
    # err_less_pc = (y_less - y_rec) / divisor * 100
    err_nois_abs = t.abs(y_nois - y_rec)
    err_less_abs = t.abs(y_less - y_rec)

    # dont plot masked LOS
    err_nois_sq[f.proj_maskb==False] = float('nan')
    err_less_sq[f.proj_maskb==False] = float('nan')
    err_nois_abs[f.proj_maskb==False] = float('nan')
    err_less_abs[f.proj_maskb==False] = float('nan')

    rows = len(f.rvg)
    plt.close('all')
    fig = plt.figure(figsize=(18, 2.5 * rows), dpi=200)
    for n in range(0, rows):
        # number of plots
        p = 6

        ax = plt.subplot(rows, p, p * n + 1, projection='polar')
        image_stack(err_nois_sq[n], f.rvg[n], ax=ax, colorbar=True)
        plt.title(f'NoisyBinned/Traced Sq Err')

        ax = plt.subplot(rows, p, p * n + 2, projection='polar')
        image_stack(err_less_sq[n], f.rvg[n], ax=ax, colorbar=True)
        plt.title(f'NoiselessBinned/Traced Sq Err')

        ax = plt.subplot(rows, p, p * n + 3, projection='polar')
        image_stack(err_nois_abs[n], f.rvg[n], ax=ax, colorbar=True)
        plt.title(f'NoisyBinned/Traced Abs Err')

        ax = plt.subplot(rows, p, p * n + 4, projection='polar')
        image_stack(err_less_abs[n], f.rvg[n], ax=ax, colorbar=True)
        plt.title(f'NoislessBinned/Traced Abs Err')

        ax = plt.subplot(rows, p, p * n + 5, projection='3d')
        # f.op.plot(geom=f.rvg[n], ax=ax)
        f.rvg[n].plot(ax=ax)
        plt.title(f'Viewing Geometry')

        ax = plt.subplot(rows, p, p * n + 6)
        plt.plot(y_rec[n, :, 0], label='Traced')
        plt.plot(y_less[n, :, 0], label='Noiseless')
        plt.legend()
        plt.xlabel('Radial Bin')
        plt.ylabel('Col. Dens')
        plt.title('Radial Profile')

    fig.tight_layout()

    # export local variables for later inspection
    from types import SimpleNamespace
    fig.locals = SimpleNamespace(**locals())

    return fig