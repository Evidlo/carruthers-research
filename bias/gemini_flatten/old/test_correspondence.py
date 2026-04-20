
import numpy as np
from common import load
import matplotlib.pyplot as plt

img = load('images_20260111/oob_nfi_l0.pkl')
half = img.shape[0] // 2

# Row medians of empty columns
empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
row_medians = np.median(img[:, empty_cols], axis=1)

top_med = row_medians[:half]
bot_med = row_medians[half:]

# Test different correspondences
# 1. i -> i + 512 (parallel)
# 2. i -> 1023 - i (mirrored)
# 3. i -> 511 - i ? (unlikely)

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(top_med, label='Top')
plt.plot(bot_med, label='Bot (parallel)')
plt.title('Parallel Comparison')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(top_med, label='Top')
plt.plot(bot_med[::-1], label='Bot (mirrored)')
plt.title('Mirrored Comparison (1023-i)')
plt.legend()

# Correlation
corr_parallel = np.corrcoef(top_med, bot_med)[0, 1]
corr_mirrored = np.corrcoef(top_med, bot_med[::-1])[0, 1]

print(f"Parallel Correlation: {corr_parallel:.4f}")
print(f"Mirrored Correlation: {corr_mirrored:.4f}")

plt.tight_layout()
plt.savefig('/www/gemini/correspondence_test.png')
