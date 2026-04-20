
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

with open('params.json', 'r') as f:
    params = json.load(f)

img = load('images_20260111/science_nfi_l0.pkl')
corr = apply_correction(img, params)

bias_top = np.median(img[150:363], axis=0)
bias_bot = np.median(img[662:875], axis=0)

orig_x = np.zeros_like(img)
orig_x[:512] = img[:512] - bias_top
orig_x[512:] = img[512:] - bias_bot

corr_x = np.zeros_like(corr)
corr_x[:512] = corr[:512] - bias_top
corr_x[512:] = corr[512:] - bias_bot

empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
S = np.sum(img, axis=1)
alpha = params[2]
sagged_rows = np.where(S > alpha)[0]

if len(sagged_rows) > 0:
    loss_orig = np.mean(orig_x[np.ix_(sagged_rows, empty_cols)]**2)
    loss_corr = np.mean(corr_x[np.ix_(sagged_rows, empty_cols)]**2)
    print(f"Science Image Loss Improvement (sagged rows only): {(loss_orig-loss_corr)/loss_orig*100:.2f}%")
else:
    print("No sagged rows found in science image.")
