
import numpy as np
from common import load
import json

def apply_correction(img, params):
    sigma, beta, alpha, sigma_p, alpha_p = params
    half = img.shape[0] // 2
    S_obs = np.sum(img, axis=1)
    res = np.zeros_like(img)
    s, s_p = S_obs[:half], S_obs[half:]
    mask, mask_p = (s > alpha).astype(float), (s_p > alpha_p).astype(float)
    denom = np.maximum(1 - sigma * s[:, None] * mask[:, None] - sigma_p * s_p[:, None] * mask_p[:, None], 0.1)
    res[:half] = (img[:half] + beta * mask[:, None]) / denom
    s, s_p = S_obs[half:], S_obs[:half]
    mask, mask_p = (s > alpha).astype(float), (s_p > alpha_p).astype(float)
    denom = np.maximum(1 - sigma * s[:, None] * mask[:, None] - sigma_p * s_p[:, None] * mask_p[:, None], 0.1)
    res[half:] = (img[half:] + beta * mask[:, None]) / denom
    return res

files = ['images_20260111/oob_nfi_l0.pkl', 'images_20260115/oob_nfi_l0.pkl', 
         'images_20260117/oob_nfi_l0.pkl', 'images_20260119/oob_nfi_l0.pkl']
with open('params.json', 'r') as f:
    params = json.load(f)

residuals = []
for f in files:
    img = load(f)
    corr = apply_correction(img, params)
    bias_top = np.median(img[150:363], axis=0)
    bias_bot = np.median(img[662:875], axis=0)
    x = np.zeros_like(corr)
    x[:512] = corr[:512] - bias_top
    x[512:] = corr[512:] - bias_bot
    residuals.append(x)

avg_res = np.mean(residuals, axis=0)
c_j = np.mean(avg_res, axis=0) # shape (1024,)

empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
c_j_empty = c_j[empty_cols]

print(f"c_j stats (empty cols): min={c_j_empty.min():.4f}, max={c_j_empty.max():.4f}, std={c_j_empty.std():.4f}")

# Save c_j
np.save('cj.npy', c_j)
