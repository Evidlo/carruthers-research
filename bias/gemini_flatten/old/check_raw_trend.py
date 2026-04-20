
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img = np.asarray(load('images_20260111/oob_nfi_l0.pkl'))
    S = np.sum(img, axis=1)
    half = 512
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    
    plt.figure(figsize=(10, 8))
    
    for i, side in enumerate(['top', 'bot']):
        sub = img[:half] if side == 'top' else img[half:]
        s_sub = S[:half] if side == 'top' else S[half:]
        
        # Robust row meds
        row_meds = np.median(sub[:, empty_cols], axis=1)
        # Bias subtract from absolute min
        min_idx = np.argsort(s_sub)[:50]
        row_meds -= np.median(row_meds[min_idx])
        
        plt.subplot(2, 1, i+1)
        plt.scatter(s_sub, row_meds, alpha=0.2, s=2)
        
        # Bin it
        bins = np.linspace(s_sub.min(), s_sub.max(), 100)
        idx = np.digitize(s_sub, bins)
        bs = [np.mean(s_sub[idx==b]) for b in range(1, len(bins)) if np.any(idx==b)]
        br = [np.median(row_meds[idx==b]) for b in range(1, len(bins)) if np.any(idx==b)]
        plt.plot(bs, br, 'r-o', markersize=3)
        
        plt.title(f'{side.capitalize()} Half - Raw Median vs S')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('Row Sum (S)')
        plt.ylabel('Residual (counts)')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/www/gemini/raw_unsagged_trend.png')
    print("Raw trend plot saved.")
