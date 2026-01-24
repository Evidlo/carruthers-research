#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from common import rescale

w, h = 1024, 1024

x = np.zeros((h, w))

grid = np.mgrid[:w, :h].T

# --- Earth disk ---

def gaussian(loc, size=1):
    """Generate gaussian at `loc` of variance `size`"""
    dist = np.linalg.norm(grid - loc, axis=-1)
    return 1 / (np.sqrt(2) * np.pi) * np.e**(-(dist**2) / size)

center = np.array((w//2, h//2))
disk = 10 * gaussian(center, 2000)

x += disk

# --- Stars ---
nstars = 10
ampstar = 3
star_locs = np.stack((
    np.random.randint(h, size=nstars),
    np.random.randint(w, size=nstars),
), axis=-1)

for loc in star_locs:
    x += np.random.random() * ampstar * gaussian(loc, 10)

# --- Bias ---
upper = np.random.random(w)
lower = np.random.random(w)
bias = np.zeros((h, w))
bias[:h//2, :] = upper
bias[h//2:, :] = lower

x += .125 * bias

# --- Sag ---
a = np.sum(x**2, axis=1)[:, None]

# f1 = lambda x, a: x * np.e**(-a / 10)
f1 = lambda x, a: x ** rescale(a, 4)
x = f1(x, a)
# x *= (a[:, None] / 10)


# --- Plotting ---

fig = plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
# hist mapping plot here

plt.subplot(1, 2, 2)
plt.imshow(x)
plt.show()