
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.load('gemini_flatten/sag_curve_halves_data.npz')
    
    plt.figure(figsize=(12, 10))
    
    for i, side in enumerate(['top', 'bot']):
        S = data[f'S_{side}']
        res = data[f'res_{side}']
        mask = ~np.isnan(res)
        S, res = S[mask], res[mask]
        
        # High res binning
        bins = np.linspace(S.min(), S.max(), 300)
        bin_idx = np.digitize(S, bins)
        
        binned_s = []
        binned_res = []
        for b in range(1, len(bins)):
            m = bin_idx == b
            if np.any(m):
                binned_s.append(np.mean(S[m]))
                binned_res.append(np.median(res[m]))
        
        plt.subplot(2, 1, i+1)
        # Use hexbin for density visualization
        plt.hexbin(S, res, gridsize=100, cmap='Greys', bins='log', alpha=0.3)
        plt.plot(binned_s, binned_res, 'r-', linewidth=2, label='Binned Median')
        plt.axhline(0, color='k', linestyle='--')
        plt.title(f'{side.capitalize()} Half - High Res Binned Analysis')
        plt.xlabel('Row Sum (S)')
        plt.ylabel('Residual (y - b)')
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('/www/gemini/sag_highres_binned.png')
    print("High-res binned plot saved.")
