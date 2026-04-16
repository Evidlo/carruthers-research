#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import iplot as iplt

def detach(x):
    """Detach and convert to numpy

    Args:
        x (tensor): input tensor, possibly on GPU

    Returns:
        ndarray: numpy array on CPU
    """
    if isinstance(x, t.Tensor):
        return x.clone().detach().cpu().numpy()
    else:
        return x

def plot_profile(s, y, p):
    """Plot PWL fit to data

    Args:
        s (tensor): row statistic
        y (tensor): pixel intensity
        p (PWL): piecewise linear function
    """
    s, y, fit = detach(s), detach(y), detach(p(s))
    plt.close()
    fig = plt.figure(figsize=(10, 10))
    s_sweep = np.linspace(s.min(), s.max(), 500)[:, None]
    for i in range(plots:=min(y.shape[1], 3)):
        plt.subplot(plots, 1, i+1)
        plt.plot(s, y[:, i], 'ro', label='data')
        plt.plot(s, fit[:, i], 'b', markersize=1, label='PWL')
        for xpos in detach(p.breakpoints[0]):
            plt.axvline(xpos, color='blue')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlim([0, 0.01])
        # plt.ylim([0, 0.51])
    plt.legend()
    plt.tight_layout()
    return fig


def plot_profile_linear(s, s_opp, y, mapping, col=0):
    """Plot thresholded linear fit as a function of s and s_opp.

    Args:
        s (tensor):       row statistic (N, 1)
        s_opp (tensor):   opposite row statistic (N, 1)
        y (tensor):       pixel intensities (N, C)
        mapping (callable): mapping(s, s_opp) -> correction (N, C)
        col (int):        which column to visualize
    """
    sn     = detach(s.squeeze())
    sopp_n = detach(s_opp.squeeze())
    yn     = detach(y[:, col])

    g1, g2 = np.meshgrid(
        np.linspace(sn.min(), sn.max(), 30),
        np.linspace(sopp_n.min(), sopp_n.max(), 30),
        indexing='ij',
    )
    s_grid    = t.from_numpy(g1.ravel()).to(s)
    sopp_grid = t.from_numpy(g2.ravel()).to(s)
    with t.no_grad():
        fit = detach(mapping(s_grid.unsqueeze(1), sopp_grid.unsqueeze(1))[:, col])

    iplt.close()
    iplt.scatter3d(sn, sopp_n, yn, label='data', c='blue')
    iplt.scatter3d(g1.ravel(), g2.ravel(), fit, label='fit', s=3, c='red', opacity=0.5)
    iplt.xlabel('s')
    iplt.ylabel('s_opp')
    iplt.zlabel('y')
    iplt.legend()
    return iplt.gcf()


def plot_profile_echo(s1, s2, y, p, col=0, lims=None):
    """Plot 2D PWL fit as a function of two row statistics using scatter3d.

    Data points and a grid-evaluated model surface are overlaid.

    Args:
        s1 (tensor): first row statistic (N,)
        s2 (tensor): second row statistic (N,)
        y (tensor): pixel intensities (N, C)
        p (FixedPWL): 2D piecewise linear model
        col (int): which output channel to plot
        lims (tuple(tuple(float))): limits of each axis, as percentile
    """
    s1n, s2n = detach(s1.squeeze()), detach(s2.squeeze())
    yn = detach(y[:, col])

    # Evaluate model on a grid to show the fitted surface
    g1, g2 = np.meshgrid(
        np.linspace(s1n.min(), s1n.max(), 60),
        np.linspace(s2n.min(), s2n.max(), 60),
        indexing='ij',
    )
    device = p.biases.device
    grid = t.stack([t.from_numpy(g1.ravel()).float(),
                    t.from_numpy(g2.ravel()).float()], dim=1).to(device)
    with t.no_grad():
        fit = detach(p(grid)[:, col])

    iplt.close()
    iplt.scatter3d(s2n, s1n, yn, label='data', c='red')
    iplt.scatter3d(g2.ravel(), g1.ravel(), fit, label='model', s=.001, c='blue')
    iplt.legend()
    if lims is not None:
        iplt.xlim((
            np.percentile(s1n, float(lims[0][0])),
            np.percentile(s1n, float(lims[0][1])),
        ))
        iplt.ylim((
            np.percentile(s2n, float(lims[0][0])),
            np.percentile(s2n, float(lims[0][1])),
        ))
        iplt.zlim((
            np.percentile(yn, float(lims[0][0])),
            np.percentile(yn, float(lims[0][1])),
        ))
    iplt.xlabel('s2')
    iplt.ylabel('s1')
    iplt.zlabel('y')
    return iplt.gcf()