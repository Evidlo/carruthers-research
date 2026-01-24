#!/usr/bin/env python3

def rescale(x, lim=None):
    """Rescale (min(x), max(x)) → (lim[0], lim[1])"""

    if isinstance(lim, int):
        lim = [0, lim]

    return (x - lim[0]) / max(x) * (lim[1] - lim[0]) + lim[0]