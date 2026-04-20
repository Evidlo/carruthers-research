
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import json
import matplotlib.pyplot as plt

def apply_correction_final(img, p):
    half = 512
    S_obs = np.sum(img, axis=1)
    
    # Bias on cleanest ORIGINAL
    min_top_idx = np.argsort(S_obs[:half])[:50]
    bt = np.median(img[:half][min_top_idx], axis=0)
    min_bot_idx = np.argsort(S_obs[half:])[:50]
    bb = np.median(img[half:][min_bot_idx], axis=0)
    
    # Primary Sag
    sag_top = (S_obs[:half] > p['alpha_top'])[:, None] * (p['betas_top'][None, :] + p['sigmas_top'][None, :] * (S_obs[:half] - p['alpha_top'])[:, None])
    sag_bot = (S_obs[half:] > p['alpha_bot'])[:, None] * (p['betas_bot'][None, :] + p['sigmas_bot'][None, :] * (S_obs[half:] - p['alpha_bot'])[:, None])
    
    # Echo Sag
    sag_echo_top = (S_obs[half:] > p['alpha_bot'])[:, None] * (p['echo_sigma_top'] * p['sigmas_top'][None, :] * (S_obs[half:] - p['alpha_bot'])[:, None])
    sag_echo_bot = (S_obs[:half] > p['alpha_top'])[:, None] * (p['echo_sigma_bot'] * p['sigmas_bot'][None, :] * (S_obs[:half] - p['alpha_top'])[:, None])
    
    # Total Sag
    total_sag_top = sag_top + sag_echo_top
    total_sag_bot = sag_bot + sag_echo_bot

    sag_min_top = np.median(total_sag_top[min_top_idx], axis=0)
    sag_min_bot = np.median(total_sag_bot[min_bot_idx], axis=0)
    
    res = np.zeros_like(img)
    res[:half] = (img[:half] - bt[None, :]) + (total_sag_top - sag_min_top[None, :])
    res[half:] = (img[half:] - bb[None, :]) + (total_sag_bot - sag_min_bot[None, :])
    
    return res

if __name__ == "__main__":
    p = np.load('gemini_flatten/params_final_multi.npz')
    
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
        x = apply_correction_final(img, p)
        all_corrected_empty.append(x[:, empty_cols])
    
    avg_x_empty = np.mean(all_corrected_empty, axis=0)
    m_avg_x = np.ma.array(avg_x_empty, mask=hot_mask[:, empty_cols])
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(m_avg_x, aspect='auto', vmin=-5, vmax=5, cmap='RdBu_r')
    plt.colorbar(label='Residual counts')
    plt.title('Residuals (Final Per-Column Multi-Component Model)')
    
    plt.subplot(2, 1, 2)
    row_profile = np.ma.median(m_avg_x, axis=1)
    plt.plot(row_profile)
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Median Row Profile (Final Model)')
    
    plt.tight_layout()
    plt.savefig('/www/gemini/column_structure_analysis_final.png')
    
    var_row = np.var(row_profile)
    print(f"Final Model Row-wise variation: {var_row:.4f}")
