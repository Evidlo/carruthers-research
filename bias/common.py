#!/usr/bin/env python3
import numpy as np
import pickle


def save(f, arr):
    """Save (masked) arrays to disk

    Args:
        f (str): path to save location
        arr (ndarray): possibly masked array
    """
    pickle.dump(arr, open(f, 'wb'))


def load(f):
    """Load (masked) arrays from disk

    Args:
        f (str): path to save location

    Returns:
        ndarray
    """
    return pickle.load(open(f, 'rb'))


def rescale(x, lim=None):
    """Rescale (min(x), max(x)) → (lim[0], lim[1])"""

    if isinstance(lim, int):
        lim = [0, lim]

    return (x - lim[0]) / max(x) * (lim[1] - lim[0]) + lim[0]