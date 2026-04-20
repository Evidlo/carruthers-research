
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import json
import matplotlib.pyplot as plt

def apply_spline_correction(img, p):
    half = 512
    S_obs = np.sum(img, axis=1)
    
    def get_corr(S, knots, coeffs_grid):
        # S: (N,)
        # knots: (K,)
        # coeffs_grid: (1024, K)
        # returns (N, 1024)
        N = len(S)
        res = np.zeros((N, 1024))
        # This is a per-column interpolation
        # Using a loop over columns is slow, so we use a faster approach
        # Since knots are shared, we can calculate weights once
        # val = interp(S, knots, values)
        # We'll use a vectorized approach:
        for c in range(1024):
            res[:, c] = np.interp(S, knots, coeffs_grid[c])
        return res

    sag_top = get_corr(S_obs[:half], p['knots_top'], p['top'])
    sag_bot = get_corr(S_obs[half:], p['knots_bot'], p['bot'])
    
    # 0.1 echo for now
    echo_top = 0.1 * get_corr(S_obs[half:], p['knots_bot'], p['top'])
    echo_bot = 0.1 * get_corr(S_obs[:half], p['knots_top'], p['bot'])
    
    z_top = img[:half] + sag_top + echo_top
    z_bot = img[half:] + sag_bot + echo_bot
    
    min_top_idx = np.argsort(S_obs[:half])[:50]
    min_bot_idx = np.argsort(S_obs[half:])[:50]
    bt = np.median(z_top[min_top_idx], axis=0)
    bb = np.median(z_bot[min_bot_idx], axis=0)
    
    return np.concatenate([z_top - bt[None, :], z_bot - bb[None, :]], axis=0)

if __name__ == "__main__":
    p = np.load('gemini_flatten/params_analytical_spline.npz')
    hot_mask = np.zeros((1024, 1024), dtype=bool)
    if os.path.exists('gemini_flatten/hot_pixels.npy'):
        hc = np.load('gemini_flatten/hot_pixels.npy')
        hot_mask[hc[:, 0], hc[:, 1]] = True

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
        x = apply_spline_correction(img, p)
        all_corrected_empty.append(x[:, empty_cols])
        
        # Save FULL flattened image
        stem = os.path.basename(f).replace('.pkl', '')
        date = os.path.dirname(f).replace('images_', '')
        plt.figure(figsize=(10, 10))
        x_disp = np.ma.array(x, mask=hot_mask)
        plt.imshow(x_disp, vmin=-5, vmax=50, cmap='gray')
        plt.title(f'Flattened OOB (Spline) - {date}/{stem}')
        plt.axis('off')
        plt.savefig(f'/www/gemini/oob_{date}_{stem}_flattened_spline.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    
    avg_x_empty = np.mean(all_corrected_empty, axis=0)
    m_avg_x = np.ma.array(avg_x_empty, mask=hot_mask[:, empty_cols])
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(m_avg_x, aspect='auto', vmin=-5, vmax=5, cmap='RdBu_r')
    plt.colorbar(label='Residual counts')
    plt.title('Residuals (Analytical Spline Model)')
    
    plt.subplot(2, 1, 2)
    row_profile = np.ma.median(m_avg_x, axis=1)
    plt.plot(row_profile)
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Median Row Profile (Spline Model)')
    
    plt.tight_layout()
    plt.savefig('/www/gemini/column_structure_analysis_spline.png')
    
    var_row = np.var(row_profile)
    sigma_row = np.sqrt(var_row)
    col_meds = np.ma.median(m_avg_x, axis=0)
    sigma_col = np.std(col_meds)
    top_edge = np.ma.median(m_avg_x[512-20:512])
    bot_edge = np.ma.median(m_avg_x[512:512+20])
    jump = np.abs(top_edge - bot_edge)
    
    with open('gemini_flatten/noise_targets.json', 'r') as f:
        t = json.load(f)

    print("\n--- Correction Grade Card (Spline) ---")
    print(f"Row Flatness (sigma): {sigma_row:.4f} (Target: {t['target_sigma_row']:.4f})")
    print(f"Col Flatness (sigma): {sigma_col:.4f} (Target: {t['target_sigma_col']:.4f})")
    print(f"Half-Half Jump:       {jump:.4f} (Target: < 0.5)")
    
    if sigma_row <= t['target_sigma_row']*2 and sigma_col <= t['target_sigma_col']*2 and jump < 1.0:
        print("RESULT: PASS (Within 2x noise floor)")
    else:
        print("RESULT: FAIL")
