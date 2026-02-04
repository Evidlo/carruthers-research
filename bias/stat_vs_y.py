#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# CLAUDE: These are one fieldset labeled "Image"
file = 'oob_nfi_l0' # CLAUDE: dropdown (other options below)
# file = 'oob_nfi'
# file = 'sci_wfi'
# file = 'sci_nfi'
x = np.load(f'images/{file}.npy') / 300
clip_level = 50

# CLAUDE: these are one fieldset labeled "Clipping"

# --- Squish Cols ---

selected_col = 300 # CLAUDE: determined by hovering over subplot 2

h, w = x.shape

squish1 = np.stack(np.split(x, 8, axis=1), axis=-1)
col_width = squish1.shape[1]

# group columns
col_groups = [[.25, .40,], [.60, .75]] # CLAUDE: this is a textbox input

nrows, ncols = x.shape
groups = []
for frac_lo, frac_hi in col_groups:
    c0 = int(round(frac_lo * ncols))
    c1 = int(round(frac_hi * ncols))
    groups.append(np.median(x[:, c0:c1], axis=1))
squish = np.stack(groups, axis=-1)

stat = np.sum(x, axis=1, keepdims=True)

# --- Plotting ---

plt.close()
clip = lambda x: np.clip(x, 0, clip_level)

# plot
plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.plot(x[:, selected_col], color='gray', label='selected')
for i, group in enumerate(clip(squish).T):
    plt.plot(group, label=i)
plt.legend()

plt.subplot(2, 2, 2)
plt.imshow(clip(x))

# plot vertical lines showing column groupings
for c in col_groups:
    plt.axvline(c[0] * w, color='red')
    plt.axvline(c[1] * w, color='red')

plt.subplot(2, 2, 3)
plt.scatter(stat, x[:, selected_col], s=.1)
# CLAUDE: set the xlim/ylim to the 10/90% percentile for the overall stat/x
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Statistic')
plt.ylabel('Y')

plt.savefig('/www/out.png')