
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import json
import matplotlib.pyplot as plt

def apply_correction_per_col_pooled(img, p_binned, coeffs):
    half = 512
    S_obs = np.sum(img, axis=1)
    
    min_top_idx = np.argsort(S_obs[:half])[:50]
    bt = np.median(img[:half][min_top_idx], axis=0)
    min_bot_idx = np.argsort(S_obs[half:])[:50]
    bb = np.median(img[half:][min_bot_idx], axis=0)
    
    def get_sag_lut(S, side_params):
        knots = np.array(side_params['knots'])
        values = np.array(side_params['values'])
        return np.interp(S, knots, values, left=0)

    # Primary Sag: use per-column coefficients
    sag_top = coeffs['coeffs_top'][None, :] * get_sag_lut(S_obs[:half], p_binned['top'])[:, None]
    sag_bot = coeffs['coeffs_bot'][None, :] * get_sag_lut(S_obs[half:], p_binned['bot'])[:, None]
    
    # Echo Sag (let's keep 0.1 global for now as a first pass)
    echo_top = 0.1 * coeffs['coeffs_top'][None, :] * get_sag_lut(S_obs[half:], p_binned['bot'])[:, None]
    echo_bot = 0.1 * coeffs['coeffs_bot'][None, :] * get_sag_lut(S_obs[:half], p_binned['top'])[:, None]
    
    ts_top = sag_top + echo_top
    ts_bot = sag_bot + echo_bot

    sm_top = np.median(ts_top[min_top_idx], axis=0)
    sm_bot = np.median(ts_bot[min_bot_idx], axis=0)
    
    res = np.zeros_like(img)
    res[:half] = (img[:half] - bt[None, :]) + (ts_top - sm_top[None, :])
    res[half:] = (img[half:] - bb[None, :]) + (ts_bot - sm_bot[None, :])
    
    return res

if __name__ == "__main__":
    with open('gemini_flatten/params_binned.json', 'r') as f:
        p_binned = json.load(f)
    coeffs = np.load('gemini_flatten/params_per_col_pooled.npz')
            
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
        x = apply_correction_per_col_pooled(img, p_binned, coeffs)
        all_corrected_empty.append(x[:, empty_cols])
    
    avg_x_empty = np.mean(all_corrected_empty, axis=0)
    m_avg_x = np.ma.array(avg_x_empty, mask=hot_mask[:, empty_cols])
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(m_avg_x, aspect='auto', vmin=-5, vmax=5, cmap='RdBu_r')
    plt.colorbar(label='Residual counts')
    plt.title('Residuals (Per-Column Pooled Model)')
    
    plt.subplot(2, 1, 2)
    row_profile = np.ma.median(m_avg_x, axis=1)
    plt.plot(row_profile)
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Median Row Profile (Per-Column Pooled Model)')
    
    plt.tight_layout()
    plt.savefig('/www/gemini/column_structure_analysis_per_col_pooled.png')
    
    var_row = np.var(row_profile)
    sigma_row = np.sqrt(var_row)
    col_meds = np.ma.median(m_avg_x, axis=0)
    sigma_col = np.std(col_meds)
    top_edge = np.ma.median(m_avg_x[512-20:512])
    bot_edge = np.ma.median(m_avg_x[512:512+20])
    jump = np.abs(top_edge - bot_edge)
    
    with open('gemini_flatten/noise_targets.json', 'r') as f:
        t = json.load(f)

    print("\n--- Correction Grade Card (Per-Column Pooled) ---")
    print(f"Row Flatness (sigma): {sigma_row:.4f} (Target: {t['target_sigma_row']:.4f})")
    print(f"Col Flatness (sigma): {sigma_col:.4f} (Target: {t['target_sigma_col']:.4f})")
    print(f"Half-Half Jump:       {jump:.4f} (Target: < 0.5)")
    
    if sigma_row <= t['target_sigma_row'] and sigma_col <= t['target_sigma_col'] and jump < 0.5:
        print("RESULT: SUCCESS")
    else:
        print("RESULT: FAILURE")
