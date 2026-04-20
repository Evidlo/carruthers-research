
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    data = np.load('gemini_flatten/sag_curve_halves_data.npz')
    
    # Load parameters
    with open('gemini_flatten/params_halves_kink.json', 'r') as f:
        pk = json.load(f)
    # Get global linear trend from the per-col fits average? Or just fit again globally.
    
    plt.figure(figsize=(14, 10))
    
    for i, side in enumerate(['top', 'bot']):
        S = data[f'S_{side}']
        res = data[f'res_{side}']
        mask = ~np.isnan(res)
        S, res = S[mask], res[mask]
        
        bins = np.linspace(S.min(), S.max(), 100)
        bin_idx = np.digitize(S, bins)
        binned_s = [np.mean(S[bin_idx == b]) for b in range(1, len(bins)) if np.any(bin_idx == b)]
        binned_res = [np.median(res[bin_idx == b]) for b in range(1, len(bins)) if np.any(bin_idx == b)]
        
        plt.subplot(2, 1, i+1)
        plt.scatter(S, res, alpha=0.03, s=1, color='gray')
        plt.plot(binned_s, binned_res, 'ko', markersize=2, label='Binned Data')
        
        S_test = np.linspace(S.min(), S.max(), 1000)
        
        # Kink Model
        ak, bk, sk = pk[side]
        plt.plot(S_test, - (bk * (S_test > ak) + sk * np.maximum(0, S_test - ak)), 'r-', label='Kink Fit')
        plt.axvline(ak, color='r', linestyle='--', alpha=0.5)
        
        # Linear Model (Fit again globally for plot)
        A = np.vstack([np.ones(len(S)), S]).T
        sol, _, _, _ = np.linalg.lstsq(A, res, rcond=None)
        plt.plot(S_test, sol[0] + sol[1]*S_test, 'g--', label='Linear Fit')
        
        plt.title(f'{side.capitalize()} Half - Model Comparison')
        plt.xlabel('Row Sum (S)')
        plt.ylabel('Residual (y - b)')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/www/gemini/model_comparison_binned.png')
    print("Comparison plot saved.")
