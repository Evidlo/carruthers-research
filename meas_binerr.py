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

    # FIXME: this is so bad!  why is this happening?
    # err_nois_sq = np.nan_to_num(err_nois_sq, posinf=0, neginf=0)
    # err_nois_pc = np.nan_to_num(err_nois_pc, posinf=0, neginf=0)
    # err_less_sq = np.nan_to_num(err_less_sq, posinf=0, neginf=0)
    # err_less_pc = np.nan_to_num(err_less_pc, posinf=0, neginf=0)

    # dont plot masked LOS
    err_nois_sq[f.proj_maskb==False] = float('nan')
    err_nois_pc[f.proj_maskb==False] = float('nan')
    err_less_sq[f.proj_maskb==False] = float('nan')
    err_less_pc[f.proj_maskb==False] = float('nan')

    rows = len(f.bvg)
    plt.close('all')
    fig = plt.figure(figsize=(12, 2.5 * rows), dpi=200)
    for n in range(0, rows):
        ax = plt.subplot(rows, 4, 4 * n + 1, projection='polar')
        image_stack(err_nois_sq[n], f.bvg[n], ax=ax, colorbar=True)
        plt.title(f'NoisyBinned/Traced Sq Err')

        ax = plt.subplot(rows, 4, 4 * n + 2, projection='polar')
        image_stack(err_less_sq[n], f.bvg[n], ax=ax, colorbar=True)
        plt.title(f'NoiselessBinned/Traced Sq Err')

        ax = plt.subplot(rows, 4, 4 * n + 3, projection='polar')
        image_stack(err_nois_pc[n], f.bvg[n], ax=ax, colorbar=True)
        plt.title(f'NoisyBinned/Traced % Err')

        ax = plt.subplot(rows, 4, 4 * n + 4, projection='polar')
        image_stack(err_less_pc[n], f.bvg[n], ax=ax, colorbar=True)
        plt.title(f'NoislessBinned/Traced % Err')

    fig.tight_layout()
    return fig