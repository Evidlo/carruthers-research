#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')


file = 'oob_nfi_l0'; dfile = 'drk_nfi_l0'
# file = 'oob_nfi'
# file = 'sci_wfi'
# file = 'sci_nfi'
x = np.load(f'images/{file}.npy') / 300
d = np.load(f'images/{dfile}.npy') / 300
# d = x[100:200].mean(axis=0, keepdims=True)
x -= d
# clip rows
x = x[100:512]; d = d[100:512]
clip_level = 5000


selected_cols = [127, 128, 129]
selected_cols = [128]

h, w = x.shape

stat = np.sum(x, axis=1, keepdims=True)

# --- Plotting ---

plt.close()
clip = lambda x: np.clip(x, 0, clip_level)

def get_col(selected_col):
    # clip to percentile
    # y = x[:, selected_col] - np.median(x[:, selected_col])
    y = x[:, selected_col]
    a, b = np.percentile(y, 1), np.percentile(y, 99)
    clip = lambda x: np.clip(x, a, b)
    return clip(y)

# plot
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
ax1 = plt.gca()
for selected_col in selected_cols:
    ax1.plot(get_col(selected_col), color='blue', label='selected')
ax1.set_ylabel('Selected Column', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(stat, color='red', label='stat')
ax2.set_ylabel('Statistic', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)

plt.subplot(2, 1, 2)
for selected_col in selected_cols:
    print(f'{selected_col}:', d[:, selected_col].mean())
    plt.scatter(stat, get_col(selected_col), s=1, label=selected_col)
plt.legend()
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Statistic')
plt.ylabel(f'Selected Y')

plt.tight_layout()
plt.show()
# plt.savefig('/www/out.png')