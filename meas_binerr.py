#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch as t

from sph_raytracer.plotting import image_stack

def meas_binerr(f, m, y_nois=None, y_less=None, y_trac=None):
    """Visualize binning error present in measurements.

    Compare Squared&Relative Err of Noisy vs Noiseless case.
    We expect the error of the noisy case to be at least 100x the noiseless case.
    If this is not true, then significant non-noise errors (e.g. binning) are present.

    Args:
        f (ForwardSph): forward operator
        m (Model): model for analytically computing measurements.
            E.g. Zoennchen00Model.
        y_nois (tensor, optional): cached noisy+binned measurements
        y_less (tensor, optioanl): cached noiseless+binned measurements
        y_trac (tensor, optional): cached directly raytraced noiseless measurements

    Returns:
        figure
    """
    assert m is not None, "Model m must be provided"

    if y_nois is None and y_less is None and y_trac is None:
        # generate noiseless, noisy, and analytic measurements
        d = m()
        y_nois = f.noise(d, disable_noise=True)
        y_less = f.noise(d)
        y_trac = f(d)

    # y_anal = m.analytic(f.bvg)

    # divisor = t.where(t.tensor(y_anal != 0), y_anal, float('inf'))
    # err_nois_sq = (y_nois - y_anal)**2
    # err_nois_pc = (y_nois - y_anal) / divisor * 100
    # err_less_sq = (y_less - y_anal)**2
    # err_less_pc = (y_less - y_anal) / divisor * 100

    divisor = t.where(t.tensor(y_trac != 0), y_trac, float('inf'))
    err_nois_sq = (y_nois - y_trac)**2
    err_nois_pc = (y_nois - y_trac) / divisor * 100
    err_less_sq = (y_less - y_trac)**2
    err_less_pc = (y_less - y_trac) / divisor * 100

    # dont plot masked LOS
    err_nois_sq[f.proj_maskb==False] = float('nan')
    err_nois_pc[f.proj_maskb==False] = float('nan')
    err_less_sq[f.proj_maskb==False] = float('nan')
    err_less_pc[f.proj_maskb==False] = float('nan')

    rows = len(f.bvg)
    plt.close('all')
    fig = plt.figure(figsize=(18, 2.5 * rows), dpi=200)
    for n in range(0, rows):
        # number of plots
        p = 6

        ax = plt.subplot(rows, p, p * n + 1, projection='polar')
        image_stack(err_nois_sq[n], f.bvg[n], ax=ax, colorbar=True)
        plt.title(f'NoisyBinned/Traced Sq Err')

        ax = plt.subplot(rows, p, p * n + 2, projection='polar')
        image_stack(err_less_sq[n], f.bvg[n], ax=ax, colorbar=True)
        plt.title(f'NoiselessBinned/Traced Sq Err')

        ax = plt.subplot(rows, p, p * n + 3, projection='polar')
        image_stack(err_nois_pc[n], f.bvg[n], ax=ax, colorbar=True)
        plt.title(f'NoisyBinned/Traced % Err')

        ax = plt.subplot(rows, p, p * n + 4, projection='polar')
        image_stack(err_less_pc[n], f.bvg[n], ax=ax, colorbar=True)
        plt.title(f'NoislessBinned/Traced % Err')

        ax = plt.subplot(rows, p, p * n + 5, projection='3d')
        # f.op.plot(geom=f.bvg[n], ax=ax)
        f.bvg[n].plot(ax=ax)
        plt.title(f'Viewing Geometry')

        ax = plt.subplot(rows, p, p * n + 6)
        plt.plot(y_trac[n, :, 0], label='Traced')
        plt.plot(y_less[n, :, 0], label='Noiseless')
        plt.legend()
        plt.xlabel('Radial Bin')
        plt.ylabel('Col. Dens')
        plt.title('Radial Profile')

    fig.tight_layout()
    return fig