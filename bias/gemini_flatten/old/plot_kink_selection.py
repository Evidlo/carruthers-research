
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__ == "__main__":
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
        
        # Bias from absolute cleanest rows
        min_top_idx = np.argsort(S[:half])[:50]
        bt = np.median(img[:half][min_top_idx], axis=0)
        min_bot_idx = np.argsort(S[half:])[:50]
        bb = np.median(img[half:][min_bot_idx], axis=0)
        
        res_top_full = (img[:half] - bt[None, :])[:, empty_cols]
        res_bot_full = (img[half:] - bb[None, :])[:, empty_cols]
        
        # TRIM ROWS TO REMOVE ECHO
        # Top half: Keep 150 to 511
        # Bot half: Keep 512 to 873 (indices 0 to 361 relative to half)
        res_top = res_top_full[150:]
        res_bot = res_bot_full[:362]
        
        s_top = S[150:512]
        s_bot = S[512:874]
        
        # Median residuals per row
        all_res_top.append(np.median(res_top, axis=1))
        all_res_bot.append(np.median(res_bot, axis=1))
        all_S_top.append(s_top)
        all_S_bot.append(s_bot)

    S_top = np.concatenate(all_S_top)
    S_bot = np.concatenate(all_S_bot)
    res_top = np.concatenate(all_res_top)
    res_bot = np.concatenate(all_res_bot)
    
    plt.figure(figsize=(15, 12))
    
    for i, (S_data, res_data, side) in enumerate([(S_top, res_top, 'top'), (S_bot, res_bot, 'bot')]):
        # Focus on the transition region
        s_min = S_data.min()
        s_focus = s_min + 0.3e6
        m_region = S_data < s_focus
        
        # High res binning
        bins = np.linspace(s_min, s_focus, 150)
        bin_idx = np.digitize(S_data, bins)
        
        binned_s = []
        binned_res = []
        for b in range(1, len(bins)):
            m = bin_idx == b
            if np.any(m):
                binned_s.append(np.mean(S_data[m]))
                binned_res.append(np.median(res_data[m]))
        
        ax = plt.subplot(2, 1, i+1)
        plt.scatter(S_data[m_region], res_data[m_region], alpha=0.1, s=1, color='gray')
        plt.plot(binned_s, binned_res, 'r-o', markersize=4, linewidth=2, label='Binned Median (Echo Trimmed)')
        
        # Grid settings
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.02e6))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        plt.grid(True, which='both', alpha=0.5)
        
        plt.title(f'{side.capitalize()} Half - Transition Analysis (Trimmed Rows 150-873)')
        plt.xlabel('Row Sum (S)')
        plt.ylabel('Residual (counts)')
        plt.axhline(0, color='k', linewidth=1)
        plt.legend()

    plt.tight_layout()
    plt.savefig('/www/gemini/kink_selection_aid_trimmed.png')
    print("Trimmed kink selection aid plot saved.")
