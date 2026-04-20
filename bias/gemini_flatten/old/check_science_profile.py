
import numpy as np
from common import load
import json

img = load('images_20260111/science_nfi_l0.pkl')
half = img.shape[0] // 2
bias_top = np.median(img[150:363], axis=0)
bias_bot = np.median(img[662:875], axis=0)

orig_x = np.zeros_like(img)
orig_x[:half] = img[:half] - bias_top
orig_x[half:] = img[half:] - bias_bot

# Check for signal in "empty" columns
empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
col_medians = np.median(orig_x, axis=0)
col_stds = np.std(orig_x, axis=0)

print(f"Empty cols median range: {col_medians[empty_cols].min():.2f} to {col_medians[empty_cols].max():.2f}")
print(f"Empty cols std range: {col_stds[empty_cols].min():.2f} to {col_stds[empty_cols].max():.2f}")

# Find which columns in the "empty" range have high signal
high_signal_cols = empty_cols[np.abs(col_medians[empty_cols]) > 5]
if len(high_signal_cols) > 0:
    print(f"Warning: {len(high_signal_cols)} 'empty' columns have median signal > 5: {high_signal_cols[:10]}...")
else:
    print("Empty columns look reasonably empty in terms of median.")

# Check row medians in the bias estimation regions
top_bias_rows_median = np.median(orig_x[150:363])
bot_bias_rows_median = np.median(orig_x[662:875])
print(f"Top bias rows global median: {top_bias_rows_median:.4f}")
print(f"Bot bias rows global median: {bot_bias_rows_median:.4f}")

# Check sagged vs unsagged rows in the empty columns
S = np.sum(img, axis=1)
with open('params.json', 'r') as f:
    params = json.load(f)
alpha = params[2]

sagged_rows = np.where(S > alpha)[0]
unsagged_rows = np.where(S <= alpha)[0]

print(f"Number of sagged rows: {len(sagged_rows)}")
if len(sagged_rows) > 0:
    sagged_median = np.median(orig_x[np.ix_(sagged_rows, empty_cols)])
    unsagged_median = np.median(orig_x[np.ix_(unsagged_rows, empty_cols)])
    print(f"Sagged rows median (empty cols): {sagged_median:.4f}")
    print(f"Unsagged rows median (empty cols): {unsagged_median:.4f}")
    print(f"Difference: {sagged_median - unsagged_median:.4f}")
