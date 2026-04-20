
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import json
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    with open('gemini_flatten/params.json', 'r') as f:
        params = json.load(f)
    
    # Load hot pixels
    hot_mask = np.zeros((1024, 1024), dtype=bool)
    if os.path.exists('gemini_flatten/hot_pixels.npy'):
        hot_coords = np.load('gemini_flatten/hot_pixels.npy')
        hot_mask[hot_coords[:, 0], hot_coords[:, 1]] = True

    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl', 
        'images_20260119/oob_nfi_l0.pkl'
    ]
    
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    all_corrected_empty = []
    
    alpha = params[2]
    
    for f in files:
        img = np.asarray(load(f))
        corr = apply_correction(img, params)
        S = np.sum(img, axis=1)
        half = 512
        
        # Robust bias
        ut = np.where(S[:half] <= alpha)[0]
        ub = np.where(S[half:] <= alpha)[0]
        if len(ut)==0: ut=np.arange(half)
        if len(ub)==0: ub=np.arange(half)
        
        bt = np.median(img[:half][ut], axis=0)
        bb = np.median(img[half:][ub], axis=0)
        
        x = np.zeros_like(corr)
        x[:half] = corr[:half] - bt
        x[half:] = corr[half:] - bb
        
        all_corrected_empty.append(x[:, empty_cols])
    
    avg_x_empty = np.mean(all_corrected_empty, axis=0)
    # Mask hot pixels in the average
    m_avg_x = np.ma.array(avg_x_empty, mask=hot_mask[:, empty_cols])
    
    plt.figure(figsize=(12, 10))
    
    # 1. 2D Map of residuals in empty columns
    plt.subplot(2, 1, 1)
    plt.imshow(m_avg_x, aspect='auto', vmin=-5, vmax=5, cmap='RdBu_r')
    plt.colorbar(label='Residual counts')
    plt.title('Average Residual in Empty Columns (Corrected)')
    plt.xlabel('Empty Column index')
    plt.ylabel('Row Index')
    
    # 2. Average row-profile in empty columns
    plt.subplot(2, 1, 2)
    row_profile = np.ma.median(m_avg_x, axis=1)
    plt.plot(row_profile)
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Median Row Profile of Empty Columns')
    plt.ylabel('Median Counts')
    plt.xlabel('Row Index')
    
    plt.tight_layout()
    plt.savefig('/www/gemini/column_structure_analysis.png')
    
    # Check variation
    var_row = np.var(row_profile)
    var_col = np.var(np.ma.median(m_avg_x, axis=0))
    print(f"Row-wise median variation: {var_row:.4f}")
    print(f"Column-wise median variation: {var_col:.4f}")
