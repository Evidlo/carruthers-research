
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import json
import matplotlib.pyplot as plt

def apply_piecewise_model(S, knots, coeffs_grid):
    # knots: [k1, k2, k3]
    # coeffs_grid: (1024, 4) -> c1, c2, c3, st
    # S: (N_rows,)
    # returns (N_rows, 1024) correction array
    k1, k2, k3 = knots
    c1 = coeffs_grid[:, 0]
    c2 = coeffs_grid[:, 1]
    c3 = coeffs_grid[:, 2]
    st = coeffs_grid[:, 3]
    
    res = np.zeros((len(S), 1024))
    
    # Region 1: < k1
    m1 = S <= k1
    res[m1] = c1[None, :]
    
    # Region 2: k1 to k2
    m2 = (S > k1) & (S <= k2)
    if np.any(m2):
        # Linear interp per row:
        # val = c1 + (S - k1) / (k2 - k1) * (c2 - c1)
        w = (S[m2] - k1) / (k2 - k1)
        res[m2] = (1-w)[:, None] * c1[None, :] + w[:, None] * c2[None, :]
        
    # Region 3: k2 to k3
    m3 = (S > k2) & (S <= k3)
    if np.any(m3):
        w = (S[m3] - k2) / (k3 - k2)
        res[m3] = (1-w)[:, None] * c2[None, :] + w[:, None] * c3[None, :]
        
    # Region 4: > k3
    m4 = S > k3
    if np.any(m4):
        res[m4] = c3[None, :] + st[None, :] * (S[m4] - k3)[:, None]
        
    return res

def apply_analytical_correction(img, p):
    half = 512
    S_obs = np.sum(img, axis=1)
    
    # Precompute Primary Sag for both halves
    # Correction z = y + sag_primary + sag_echo
    sag_top = apply_piecewise_model(S_obs[:half], p['knots_top'], p['top'])
    sag_bot = apply_piecewise_model(S_obs[half:], p['knots_bot'], p['bot'])
    
    # Echo Sag (Using small coefficients as placeholders, we can optimize later)
    # Assume echo uses the same per-column profile but scaled down
    echo_top = 0.08 * apply_piecewise_model(S_obs[half:], p['knots_bot'], p['top'])
    echo_bot = 0.01 * apply_piecewise_model(S_obs[:half], p['knots_top'], p['bot'])
    
    z_top = img[:half] + sag_top + echo_top
    z_bot = img[half:] + sag_bot + echo_bot
    
    # Consistent bias estimation from corrected data z
    min_top_idx = np.argsort(S_obs[:half])[:50]
    min_bot_idx = np.argsort(S_obs[half:])[:50]
    
    bt = np.median(z_top[min_top_idx], axis=0)
    bb = np.median(z_bot[min_bot_idx], axis=0)
    
    x_top = z_top - bt[None, :]
    x_bot = z_bot - bb[None, :]
    
    return np.concatenate([x_top, x_bot], axis=0)

if __name__ == "__main__":
    p = np.load('gemini_flatten/params_analytical_piecewise.npz')
    
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
        x = apply_analytical_correction(img, p)
        all_corrected_empty.append(x[:, empty_cols])
        
        # Save FULL flattened image
        stem = os.path.basename(f).replace('.pkl', '')
        date = os.path.dirname(f).replace('images_', '')
        plt.figure(figsize=(10, 10))
        x_disp = np.ma.array(x, mask=hot_mask)
        plt.imshow(x_disp, vmin=-5, vmax=50, cmap='gray')
        plt.title(f'Flattened OOB (Analytical) - {date}/{stem}')
        plt.axis('off')
        plt.savefig(f'/www/gemini/oob_{date}_{stem}_flattened_analytical.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    
    avg_x_empty = np.mean(all_corrected_empty, axis=0)
    m_avg_x = np.ma.array(avg_x_empty, mask=hot_mask[:, empty_cols])
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(m_avg_x, aspect='auto', vmin=-5, vmax=5, cmap='RdBu_r')
    plt.colorbar(label='Residual counts')
    plt.title('Residuals (Analytical Piecewise Model)')
    
    plt.subplot(2, 1, 2)
    row_profile = np.ma.median(m_avg_x, axis=1)
    plt.plot(row_profile)
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Median Row Profile (Analytical Model)')
    
    plt.tight_layout()
    plt.savefig('/www/gemini/column_structure_analysis_analytical.png')
    
    var_row = np.var(row_profile)
    sigma_row = np.sqrt(var_row)
    col_meds = np.ma.median(m_avg_x, axis=0)
    sigma_col = np.std(col_meds)
    top_edge = np.ma.median(m_avg_x[512-20:512])
    bot_edge = np.ma.median(m_avg_x[512:512+20])
    jump = np.abs(top_edge - bot_edge)
    
    with open('gemini_flatten/noise_targets.json', 'r') as f:
        t = json.load(f)

    print("\n--- Correction Grade Card (Analytical Piecewise) ---")
    print(f"Row Flatness (sigma): {sigma_row:.4f} (Target: {t['target_sigma_row']:.4f})")
    print(f"Col Flatness (sigma): {sigma_col:.4f} (Target: {t['target_sigma_col']:.4f})")
    print(f"Half-Half Jump:       {jump:.4f} (Target: < 0.5)")
    
    if sigma_row <= t['target_sigma_row']*2 and sigma_col <= t['target_sigma_col']*2 and jump < 1.0:
        print("RESULT: PASS (Within 2x noise floor)")
    else:
        print("RESULT: FAIL")
