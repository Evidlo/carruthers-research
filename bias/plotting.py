#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch as t

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