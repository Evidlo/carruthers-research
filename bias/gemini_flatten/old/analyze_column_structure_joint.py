
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import json
import matplotlib.pyplot as plt

def apply_multiplicative_correction(img, p):
    half = 512
    S_obs = np.sum(img, axis=1)
    
    def get_sag_val(S, knots, values):
        return np.interp(S, knots, values, left=values[0], right=values[-1])

    # 1. Precompute P curves
    st, sb = S_obs[:half], S_obs[half:]
    pt, pb = p['top'], p['bot']
    
    ps_t = get_sag_val(st, pt['knots'], pt['values'])
    ps_b = get_sag_val(sb, pb['knots'], pb['values'])
    
    # 2. Correction including echo
    dt = np.maximum(1 - ps_t[:, None] - pt['echo_scale'] * ps_b[:, None], 0.1)
    z_t = (img[:half] + pt['beta']) / dt
    
    db = np.maximum(1 - ps_b[:, None] - pb['echo_scale'] * ps_t[:, None], 0.1)
    z_b = (img[half:] + pb['beta']) / db
    
    # 3. Robust bias estimation from corrected data z (Rows 150-362)
    bt = np.median(z_t[150:363], axis=0)
    bb = np.median(z_b[150:363], axis=0)
    
    x_t = z_t - bt[None, :]
    x_b = z_b - bb[None, :]
    
    return np.concatenate([x_t, x_b], axis=0)

if __name__ == "__main__":
    with open('gemini_flatten/params_multiplicative.json', 'r') as f:
        p = json.load(f)
    
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
        x = apply_multiplicative_correction(img, p)
        all_corrected_empty.append(x[:, empty_cols])
    
    avg_x_empty = np.mean(all_corrected_empty, axis=0)
    m_avg_x = np.ma.array(avg_x_empty, mask=hot_mask[:, empty_cols])
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(m_avg_x, aspect='auto', vmin=-3, vmax=3, cmap='RdBu_r')
    plt.colorbar(label='Residual counts')
    plt.title('Residuals (Final Joint Multiplicative Model)')
    plt.xlabel('Empty Column index')
    plt.ylabel('Row Index')
    
    plt.subplot(2, 1, 2)
    row_profile = np.ma.median(m_avg_x, axis=1)
    plt.plot(row_profile)
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Median Row Profile (Final Model)')
    plt.ylabel('Median Counts')
    plt.xlabel('Row Index')
    
    plt.tight_layout()
    plt.savefig('/www/gemini/column_structure_analysis_final_joint.png')
    
    var_row = np.var(row_profile)
    print(f"Final Joint Model Row-wise variation: {var_row:.4f}")
