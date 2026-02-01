#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# x = np.load('images/oob_nfi.npy') / 300
# clip_level = 25
# mask = np.load('images/mask_nfi.npy')

# x = np.load('images/sci_nfi.npy') / 300
# clip_level = 500
# mask = np.load('images/mask_nfi.npy')

# x = np.load('images/sci_wfi.npy') / 300
# clip_level = 5000
# x += x.mean(axis=1, keepdims=True) / 30
# mask = np.load('images/mask_wfi.npy')

x = np.load('images/oob_wfi.npy') / 300
clip_level = 25
mask = np.load('images/mask_wfi.npy')
x += x.mean(axis=1, keepdims=True) / 80

# plt.imshow(np.clip(x, 0, 100))
# plt.savefig('/www/oob_nfi.png', dpi=300)

clip = lambda x: np.clip(x, 0, clip_level)

h, w = x.shape

# --- Squish Cols ---
squish1 = np.stack(np.split(x, 8, axis=1), axis=-1)
squish = np.median(squish1, axis=1)

# group columns
cols = [[0, 1,], [2, 3, 4, 5], [6, 7]]
# cols = [[0, 1, ], [1, 2, 3], [4], [5, 6], [7]]
squish = np.stack([squish[:, c].mean(axis=-1) for c in cols], axis=-1)

plt.close()
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
# plt.plot(clip(squish))
for i, group in enumerate(clip(squish).T):
    plt.plot(group, label=i)
plt.legend()

plt.subplot(1, 2, 2)
plt.imshow(clip(x))

j = 0
for c in cols[:-1]:
    j += len(c)
    plt.axvline(squish1.shape[1] * j, color='red')
plt.savefig('/www/out.png', dpi=500)

# --- Aggregate ---
agg = np.sum(x, axis=-1)
# agg = np.sum(x**2, axis=-1)

# result[result == 0] = np.nan

# plt.close()
# plt.plot(result, linewidth=0.4, alpha=1)
# plt.savefig('/www/out.png', dpi=300)