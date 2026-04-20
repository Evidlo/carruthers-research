
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.load('gemini_flatten/sag_curve_halves_data.npz')
    
    plt.figure(figsize=(12, 8))
    
    for i, side in enumerate(['top', 'bot']):
        S = data[f'S_{side}']
        res = data[f'res_{side}']
        mask = ~np.isnan(res)
        S, res = S[mask], res[mask]
        
        # Binning
        bins = np.linspace(S.min(), S.max(), 100)
        bin_idx = np.digitize(S, bins)
        
        binned_s = []
        binned_res = []
        for b in range(1, len(bins)):
            m = bin_idx == b
            if np.any(m):
                binned_s.append(np.mean(S[m]))
                # Use median to ignore stars/hot pixels
                binned_res.append(np.median(res[m]))
        
        plt.subplot(2, 1, i+1)
        plt.scatter(S, res, alpha=0.05, s=1, color='gray')
        plt.plot(binned_s, binned_res, 'r-o', markersize=3, label='Binned Median')
        plt.axhline(0, color='k', linestyle='--')
        plt.title(f'{side.capitalize()} Half - Binned Median Analysis')
        plt.xlabel('Row Sum (S)')
        plt.ylabel('Residual (y - b)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Find transition index
        # Looking for the first significant drop
        diffs = np.diff(binned_res)
        # Find where it drops and stays low
        threshold_guess = binned_s[0]
        for j in range(len(diffs)):
            if diffs[j] < -1.0: # Significant drop
                threshold_guess = binned_s[j]
                break
        print(f"{side} detected sharp drop near S={threshold_guess:.2f}")

    plt.tight_layout()
    plt.savefig('/www/gemini/sag_binned_analysis.png')
    print("Binned analysis plot saved.")
