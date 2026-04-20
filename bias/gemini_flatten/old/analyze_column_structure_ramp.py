
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import json
import matplotlib.pyplot as plt

def apply_correction_per_col(img, params_npz):
    betas = params_npz['betas']
    sigmas = params_npz['sigmas']
    sigmas_p = params_npz['sigmas_p']
    alpha = params_npz['alpha']
    alpha_p = params_npz['alpha_p']
    
    half = img.shape[0] // 2
    S_obs = np.sum(img, axis=1)
    
    # Estimate bias on cleanest ORIGINAL rows
    min_top_idx = np.argsort(S_obs[:half])[:50]
    bt = np.median(img[:half][min_top_idx], axis=0)
    min_bot_idx = np.argsort(S_obs[half:])[:50]
    bb = np.median(img[half:][min_bot_idx], axis=0)
    
    sag_top = (S_obs[:half] > alpha)[:, None] * (betas[None, :] + sigmas[None, :] * (S_obs[:half] - alpha)[:, None]) + \
              (S_obs[half:] > alpha_p)[:, None] * (sigmas_p[None, :] * (S_obs[half:] - alpha_p)[:, None])
              
    sag_bot = (S_obs[half:] > alpha)[:, None] * (betas[None, :] + sigmas[None, :] * (S_obs[half:] - alpha)[:, None]) + \
              (S_obs[:half] > alpha_p)[:, None] * (sigmas_p[None, :] * (S_obs[:half] - alpha_p)[:, None])

    # Sag in the bias rows
    sag_min_top = np.median(sag_top[min_top_idx], axis=0)
    sag_min_bot = np.median(sag_bot[min_bot_idx], axis=0)
    
    res = np.zeros_like(img)
    res[:half] = (img[:half] - bt[None, :]) + (sag_top - sag_min_top[None, :])
    res[half:] = (img[half:] - bb[None, :]) + (sag_bot - sag_min_bot[None, :])
    
    return res

if __name__ == "__main__":
    params_per_col = np.load('gemini_flatten/params_per_col.npz')
    alpha = params_per_col['alpha']
    
    # Load hot pixels
    hot_mask = np.zeros((1024, 1024), dtype=bool)
    if os.path.exists('gemini_flatten/hot_pixels.npy'):
        hot_coords = np.load('gemini_flatten/hot_pixels.npy')
        hot_mask[hot_coords[:, 0], hot_coords[:, 1]] = True

    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260113/oob_nfi_l0.pkl',
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl', 
        'images_20260119/oob_nfi_l0.pkl'
    ]
    
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    all_corrected_empty = []
    
    for f in files:
        img = np.asarray(load(f))
        x = apply_correction_per_col(img, params_per_col)
        all_corrected_empty.append(x[:, empty_cols])
    
    avg_x_empty = np.mean(all_corrected_empty, axis=0)
    m_avg_x = np.ma.array(avg_x_empty, mask=hot_mask[:, empty_cols])
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.imshow(m_avg_x, aspect='auto', vmin=-5, vmax=5, cmap='RdBu_r')
    plt.colorbar(label='Residual counts')
    plt.title('Average Residual in Empty Columns (Corrected - Per-Column Ramp)')
    plt.xlabel('Empty Column index')
    plt.ylabel('Row Index')
    
    plt.subplot(2, 1, 2)
    row_profile = np.ma.median(m_avg_x, axis=1)
    plt.plot(row_profile)
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Median Row Profile of Empty Columns (Per-Column Ramp)')
    plt.ylabel('Median Counts')
    plt.xlabel('Row Index')
    
    plt.tight_layout()
    plt.savefig('/www/gemini/column_structure_analysis_per_col.png')
    
    var_row = np.var(row_profile)
    var_col = np.var(np.ma.median(m_avg_x, axis=0))
    print(f"Per-Column Ramp Row-wise median variation: {var_row:.4f}")
    print(f"Per-Column Ramp Column-wise median variation: {var_col:.4f}")
