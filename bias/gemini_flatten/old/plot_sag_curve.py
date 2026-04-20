
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open('gemini_flatten/params.json', 'r') as f:
        params = json.load(f)
    alpha = params[2]
    
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
    
    all_S_top = []
    all_S_bot = []
    all_res_top = []
    all_res_bot = []
    
    for f in files:
        img = np.asarray(load(f))
        S = np.sum(img, axis=1)
        half = 512
        
        # Use absolute minimum for bias to avoid bimodal bias-corruption
        min_top_idx = np.argsort(S[:half])[:50]
        bt = np.median(img[:half][min_top_idx], axis=0)
        min_bot_idx = np.argsort(S[half:])[:50]
        bb = np.median(img[half:][min_bot_idx], axis=0)
        
        res_top = (img[:half] - bt[None, :])[:, empty_cols]
        res_bot = (img[half:] - bb[None, :])[:, empty_cols]
        
        m_top = hot_mask[:half, empty_cols]
        m_bot = hot_mask[half:, empty_cols]
        
        row_res_top = np.ma.median(np.ma.array(res_top, mask=m_top), axis=1)
        row_res_bot = np.ma.median(np.ma.array(res_bot, mask=m_bot), axis=1)
        
        all_res_top.append(row_res_top)
        all_res_bot.append(row_res_bot)
        all_S_top.append(S[:half])
        all_S_bot.append(S[half:])

    S_top = np.concatenate(all_S_top)
    S_bot = np.concatenate(all_S_bot)
    res_top = np.concatenate(all_res_top)
    res_bot = np.concatenate(all_res_bot)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(S_top, res_top, alpha=0.1, s=1, color='blue')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Row Sum (S)')
    plt.ylabel('Median Residual (y - b)')
    plt.title('Top Half Sag Curve')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(S_bot, res_bot, alpha=0.1, s=1, color='green')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Row Sum (S)')
    plt.ylabel('Median Residual (y - b)')
    plt.title('Bottom Half Sag Curve')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/www/gemini/sag_curve_halves.png')
    
    # Save separate data
    np.savez('gemini_flatten/sag_curve_halves_data.npz', 
             S_top=S_top, res_top=res_top.filled(np.nan),
             S_bot=S_bot, res_bot=res_bot.filled(np.nan))
    print("Split sag curve plot saved.")
