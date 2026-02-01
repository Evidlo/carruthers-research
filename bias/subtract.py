#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# CLAUDE: These are one fieldset labeled "Image"
file = 'oob_wfi' # CLAUDE: dropdown (other options below)
# file = 'oob_nfi'
# file = 'sci_wfi'
# file = 'sci_nfi'
x = np.load(f'images/{file}.npy') / 300

# CLAUDE: these are one fieldset labeled "Clipping"
clip_level = 25 # CLAUDE: slider from start to stop
clip_level_start = 0 # CLAUDE: numeric input
clip_level_stop = 5000 # CLAUDE: numeric input

# CLAUDE: these are one fieldset labeled "Subtraction"
func = 'x += x.mean(axis=1, keepdims=True) / a' # CLAUDE: text input labeled "Function"
a = 80 # CLAUDE: slider from start to stop
a_start = 0 # CLAUDE: numeric input
a_stop = 500 # CLAUDE: numeric input
b = 1 # CLAUDE: slider from start to stop
b_start = 0 # CLAUDE: numeric input
b_stop = 1 # CLAUDE: numeric input
c = 1 # CLAUDE: slider from start to stop
c_start = 0 # CLAUDE: numeric input
c_stop = 1 # CLAUDE: numeric input
exec(func)


# --- Squish Cols ---

h, w = x.shape

squish1 = np.stack(np.split(x, 8, axis=1), axis=-1)
col_width = squish1.shape[1]

# group columns
# col_groups = [[0, 1,], [2, 3, 4, 5], [6, 7]]
col_ranges = [[0, .25,], [.25, .75], [.75, 1]]
# cols = [[0, 1, ], [1, 2, 3], [4], [5, 6], [7]]
squish = np.stack([
    np.median(squish1[:, int(c[0]*w):int(c[1]*w)], axis=1) for c in col_ranges
], axis=-1)

# --- Plotting ---

plt.close()
clip = lambda x: np.clip(x, 0, clip_level)

# plot
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
for i, group in enumerate(clip(squish).T):
    plt.plot(group, label=i)
plt.legend()

plt.subplot(1, 2, 2)
plt.imshow(clip(x))

# plot vertical lines showing column groupings
for c in col_ranges[:-1]:
    plt.axvline(c[1] * w, color='red')
