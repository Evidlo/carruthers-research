#!/usr/bin/env python3
import torch as t
import numpy as np
import pickle

def tnp(x):
    """Return pytorch or numpy, based on input array"""
    if isinstance(x, t.Tensor):
        return t
    else:
        return np

def save(f, arr):
    """Save (masked) arrays to disk

    Args:
        f (str): path to save location
        arr (ndarray): possibly masked array
    """
    pickle.dump(arr, open(f, 'wb'))


def load(f):
    """Load arrays from disk. Masked arrays are unwrapped to plain ndarrays
    with masked entries replaced by NaN.

    Args:
        f (str): path to save location

    Returns:
        ndarray
    """
    return np.ma.filled(pickle.load(open(f, 'rb')), np.nan)


def rescale(x, lim=None):
    """Rescale (min(x), max(x)) → (lim[0], lim[1])"""

    if isinstance(lim, int):
        lim = [0, lim]

    return (x - lim[0]) / max(x) * (lim[1] - lim[0]) + lim[0]


def mean_bias(x, clip_out=0, clip_in=0):
    """Compute dumb top/bottom bias using mean

    Args:
        x (ndarray): input image
        clip_out (int): number of outside rows to clip
        clip_in (int): number of inside rows to clip

    Returns:
        result (ndarray): same shape as x


    +---------------+
    |   outside     |
    |               |
    |   inside      |
    +---------------+
    |   inside      |
    |               |
    |   outside     |
    +---------------+
    """
    result = tnp(x).empty_like(x)

    half = x.shape[0] // 2
    result[:half] = x[clip_out:half-clip_in].mean(axis=0, keepdims=True)
    result[half:] = x[half+clip_in:-clip_out or None].mean(axis=0, keepdims=True)

    return result


def rob_bias(x, clip_out=0, clip_in=0, percent=20):
    """Compute top/bottom bias using middle quantile

    Args:
        x (ndarray): input image
        clip_out (int): number of outside rows to clip
        clip_in (int): number of inside rows to clip

    Returns:
        result (ndarray): same shape as x


    +---------------+
    |   outside     |
    |               |
    |   inside      |
    +---------------+
    |   inside      |
    |               |
    |   outside     |
    +---------------+
    """
    result = np.empty_like(x)

    half = x.shape[0] // 2

    m = x[clip_out:half-clip_in]
    mask = np.logical_and(
        m > np.nanpercentile(m, 40, axis=0, keepdims=True),
        m < np.nanpercentile(m, 60, axis=0, keepdims=True)
    )
    result[:half] = np.ma.array(m, mask=~mask).mean(axis=0)

    m = x[half+clip_in:-clip_out or None]
    mask = np.logical_and(
        m > np.nanpercentile(m, 40, axis=0, keepdims=True),
        m < np.nanpercentile(m, 60, axis=0, keepdims=True)
    )
    result[half:] = np.ma.array(m, mask=~mask).mean(axis=0)

    return result