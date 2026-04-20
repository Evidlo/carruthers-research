
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import matplotlib.pyplot as plt

def apply_correction_per_col(img, params_npz):
    betas = params_npz['betas']
    sigmas = params_npz['sigmas']
    sigmas_p = params_npz['sigmas_p']
    alpha = params_npz['alpha']
    alpha_p = params_npz['alpha_p']
    half = 512
    S_obs = np.sum(img, axis=1)
    min_top_idx = np.argsort(S_obs[:half])[:50]
    bt = np.median(img[:half][min_top_idx], axis=0)
    min_bot_idx = np.argsort(S_obs[half:])[:50]
    bb = np.median(img[half:][min_bot_idx], axis=0)
    sag_top = (S_obs[:half] > alpha)[:, None] * (betas[None, :] + sigmas[None, :] * (S_obs[:half] - alpha)[:, None]) + \
              (S_obs[half:] > alpha_p)[:, None] * (sigmas_p[None, :] * (S_obs[half:] - alpha_p)[:, None])
    sag_bot = (S_obs[half:] > alpha)[:, None] * (betas[None, :] + sigmas[None, :] * (S_obs[half:] - alpha)[:, None]) + \
              (S_obs[:half] > alpha_p)[:, None] * (sigmas_p[None, :] * (S_obs[:half] - alpha_p)[:, None])
    sag_min_top = np.median(sag_top[min_top_idx], axis=0)
    sag_min_bot = np.median(sag_bot[min_bot_idx], axis=0)
    res = np.zeros_like(img)
    res[:half] = (img[:half] - bt[None, :]) + (sag_top - sag_min_top[None, :])
    res[half:] = (img[half:] - bb[None, :]) + (sag_bot - sag_min_bot[None, :])
    return res

if __name__ == "__main__":
    params_per_col = np.load('gemini_flatten/params_per_col.npz')
    files = ['images_20260111/oob_nfi_l0.pkl', 'images_20260115/oob_nfi_l0.pkl']
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    
    all_S = []
    all_res = []
    
    for f in files:
        img = np.asarray(load(f))
        S = np.sum(img, axis=1)
        x = apply_correction_per_col(img, params_per_col)
        # Median residual per row
        row_res = np.median(x[:, empty_cols], axis=1)
        all_S.append(S)
        all_res.append(row_res)
        
    S_flat = np.concatenate(all_S)
    res_flat = np.concatenate(all_res)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(S_flat, res_flat, alpha=0.2, s=1)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Row Sum (S)')
    plt.ylabel('Corrected Residual (counts)')
    plt.title('Residuals after Per-Column Ramp Correction')
    plt.grid(True, alpha=0.3)
    plt.savefig('/www/gemini/final_residuals_scatter.png')
    print("Residual scatter plot saved.")
