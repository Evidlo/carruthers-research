
import numpy as np
from common import load, rob_bias
import json
import matplotlib.pyplot as plt

img = load('images_20260111/oob_nfi_l0.pkl')
half = img.shape[0] // 2

# 1. Static rob_bias (using pre-defined rows)
# The static ranges are roughly 150:363 and 662:875
b_static_top = np.median(img[150:363], axis=0)
b_static_bot = np.median(img[662:875], axis=0)

# 2. Dynamic bias (using S < alpha)
with open('params.json', 'r') as f:
    params = json.load(f)
alpha = params[2]

S = np.sum(img, axis=1)
unsag_top = np.where(S[:half] <= alpha)[0]
unsag_bot = np.where(S[half:] <= alpha)[0]

b_dyn_top = np.median(img[:half][unsag_top], axis=0)
b_dyn_bot = np.median(img[half:][unsag_bot], axis=0)

# Plot comparison
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(b_dyn_top - b_static_top, label='Top (Dynamic - Static)')
plt.title('Difference in Column Bias ($b_j$) Estimates')
plt.ylabel('Delta (counts)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(b_dyn_bot - b_static_bot, label='Bot (Dynamic - Static)')
plt.ylabel('Delta (counts)')
plt.xlabel('Column Index')
plt.legend()

plt.tight_layout()
plt.savefig('/www/gemini/bias_comparison.png')

diff_top = np.abs(b_dyn_top - b_static_top).mean()
diff_bot = np.abs(b_dyn_bot - b_static_bot).mean()
print(f"Mean absolute difference in b_j: Top={diff_top:.4f}, Bot={diff_bot:.4f}")
print(f"Number of rows used: Static=213, Dynamic Top={len(unsag_top)}, Dynamic Bot={len(unsag_bot)}")
