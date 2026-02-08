#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
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
# aggregate statistic for a row
a = np.sum(x**2, axis=1)[:, None]

# Save x before mapping for visualization
x_before = x.copy()

f1 = lambda x, a: x * np.e**(-a / 2)
# f1 = lambda x, a: x ** 2
# f1 = lambda x, a: x ** rescale(a, 4)
x = f1(x, a)
# x *= (a[:, None] / 10)


# --- Plotting ---

fig = plt.figure(figsize=(16, 6))

# Create custom grid for subplot with marginal histograms
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 3, 3, 0.1], height_ratios=[1, 3],
                       hspace=0.05, wspace=0.05, left=0.05, right=0.52, top=0.95, bottom=0.1)

# Top histogram (before mapping)
ax_top = fig.add_subplot(gs[0, 1])
ax_top.hist(x_before.flatten(), bins=60, color='blue', alpha=0.7)
ax_top.set_ylabel('Count')
ax_top.set_title('Before/After Mapping')
ax_top.tick_params(labelbottom=False)
ax_top.grid(True, alpha=0.3)

# Center 3D scatter plot (mapping function)
ax_center = fig.add_subplot(gs[1, 1], projection='3d')

# Sample every 100th point for performance
indices = np.arange(0, x_before.size, 1000)

# Convert flat indices to row, col positions
rows = indices // w
cols = indices % w

# Get corresponding values
x_before_samples = x_before.flatten()[indices]
a_samples = a.flatten()[rows]  # a is indexed by row only
x_samples = x.flatten()[indices]

# Create 3D scatter plot
scatter = ax_center.scatter(x_before_samples, a_samples, x_samples,
                            c=x_samples, cmap='viridis', alpha=0.5, s=1, rasterized=True)
ax_center.set_xlabel('x_before')
ax_center.set_ylabel('a')
ax_center.set_zlabel('x')

# Right histogram (after mapping)
ax_right = fig.add_subplot(gs[1, 2])
ax_right.hist(x.flatten(), bins=60, orientation='horizontal', color='red', alpha=0.7)
ax_right.set_xlabel('Count')
ax_right.tick_params(labelleft=False)
ax_right.grid(True, alpha=0.3)

# Image subplot
ax_img = fig.add_subplot(1, 2, 2)
ax_img.imshow(x)
ax_img.set_title('Final Image')

plt.tight_layout()
plt.show()