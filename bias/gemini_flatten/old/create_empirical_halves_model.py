
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import json

if __name__ == "__main__":
    data = np.load('gemini_flatten/sag_curve_halves_data.npz')
    
    final_params = {}
    plt.figure(figsize=(12, 10))
    
    for i, side in enumerate(['top', 'bot']):
        S = data[f'S_{side}']
        res = data[f'res_{side}']
        mask = ~np.isnan(res)
        S, res = S[mask], res[mask]
        
        # High-res binning for each half independently
        bins = np.linspace(S.min(), S.max(), 300)
        bin_idx = np.digitize(S, bins)
        target_s = []
        target_res = []
        for b in range(1, len(bins)):
            m = bin_idx == b
            if np.any(m):
                target_s.append(np.mean(S[m]))
                target_res.append(np.median(res[m]))
        
        target_s = np.array(target_s)
        target_res = np.array(target_res)
        
        # Zero out the floor based on first few clean rows
        floor_res = np.median(target_res[:10])
        target_res -= floor_res
        
        final_params[side] = {
            'knots': target_s.tolist(),
            'values': (-target_res).tolist() 
        }
        
        plt.subplot(2, 1, i+1)
        plt.scatter(S, res - floor_res, alpha=0.03, s=1, color='gray')
        plt.plot(target_s, target_res, 'r-', linewidth=2, label=f'{side.capitalize()} Empirical Model')
        plt.axhline(0, color='k', linestyle='--')
        plt.title(f'{side.capitalize()} Half Empirical Model')
        plt.xlabel('Row Sum (S)')
        plt.ylabel('Residual (y - b)')
        plt.legend()

    plt.tight_layout()
    plt.savefig('/www/gemini/sag_empirical_halves.png')
    
    with open('gemini_flatten/params_binned.json', 'w') as f:
        json.dump(final_params, f, indent=4)
    print("Independent empirical models saved.")
