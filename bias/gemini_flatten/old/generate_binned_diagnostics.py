
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import json
import matplotlib.pyplot as plt

def apply_correction_binned(img, p_binned):
    half = 512
    S_obs = np.sum(img, axis=1)
    
    # Bias on cleanest ORIGINAL
    min_top_idx = np.argsort(S_obs[:half])[:50]
    bt = np.median(img[:half][min_top_idx], axis=0)
    min_bot_idx = np.argsort(S_obs[half:])[:50]
    bb = np.median(img[half:][min_bot_idx], axis=0)
    
    def get_sag_lut(S, side_params):
        knots = np.array(side_params['knots'])
        values = np.array(side_params['values'])
        return np.interp(S, knots, values, left=0)

    sag_top = get_sag_lut(S_obs[:half], p_binned['top'])[:, None]
    sag_bot = get_sag_lut(S_obs[half:], p_binned['bot'])[:, None]
    
    # Asymmetric Echo (Optimized)
    echo_top = 0.52 * get_sag_lut(S_obs[half:], p_binned['bot'])[:, None]
    echo_bot = 0.09 * get_sag_lut(S_obs[:half], p_binned['top'])[:, None]
    
    total_sag_top = sag_top + echo_top
    total_sag_bot = sag_bot + echo_bot

    sag_min_top = np.median(total_sag_top[min_top_idx], axis=0)
    sag_min_bot = np.median(total_sag_bot[min_bot_idx], axis=0)
    
    res = np.zeros_like(img)
    res[:half] = (img[:half] - bt[None, :]) + (total_sag_top - sag_min_top[None, :])
    res[half:] = (img[half:] - bb[None, :]) + (total_sag_bot - sag_min_bot[None, :])
    
    return res

if __name__ == "__main__":
    with open('gemini_flatten/params_binned.json', 'r') as f:
        p_binned = json.load(f)
    
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
    prefixes = ['oob_20260111_binned', 'oob_20260113_binned', 'oob_20260115_binned', 'oob_20260117_binned', 'oob_20260119_binned']
    
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])

    for f, pref in zip(files, prefixes):
        img = np.asarray(load(f))
        x = apply_correction_binned(img, p_binned)
        S = np.sum(img, axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes[0, 0].plot(S)
        axes[0, 0].set_title(f'Row Sums - {os.path.basename(f)}')
        
        from common import rob_bias
        orig_x = img - rob_bias(img)
        axes[0, 1].imshow(orig_x, vmin=-10, vmax=50, cmap='gray')
        axes[0, 1].set_title('Original rob_bias')
        
        axes[1, 0].imshow(x, vmin=-10, vmax=50, cmap='gray')
        axes[1, 0].set_title('Binned Empirical Correction')
        
        m_orig = np.ma.array(orig_x[:, empty_cols], mask=hot_mask[:, empty_cols])
        m_curr = np.ma.array(x[:, empty_cols], mask=hot_mask[:, empty_cols])
        axes[1, 1].plot(np.ma.median(m_orig, axis=0), alpha=0.3, label='Orig (Col Med)')
        axes[1, 1].plot(np.ma.median(m_curr, axis=1), alpha=0.5, label='Corrected (Row Med)')
        axes[1, 1].axhline(0, color='k', linestyle='--')
        axes[1, 1].legend()
        axes[1, 1].set_title('Flatness Evaluation (Empty Cols, Hot Masked)')
        
        plt.tight_layout()
        plt.savefig(f'/www/gemini/{pref}_diagnostic.png')
        plt.close()
        print(f"Saved {pref}")
