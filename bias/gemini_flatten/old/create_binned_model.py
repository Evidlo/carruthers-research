
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
    
    results = {}
    plt.figure(figsize=(12, 10))
    
    for i, side in enumerate(['top', 'bot']):
        S = data[f'S_{side}']
        res = data[f'res_{side}']
        mask = ~np.isnan(res)
        S, res = S[mask], res[mask]
        
        # 1. Get binned median for a clean target
        bins = np.linspace(S.min(), S.max(), 200)
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
        
        # 2. Use the binned median as the SAG MODEL directly (look-up table)
        # We need to extend it to cover all possible S values
        # Baseline should be 0 below the floor
        
        results[side] = {
            'knots': target_s.tolist(),
            'values': (-target_res).tolist() # Sag is positive suppression
        }
        
        plt.subplot(2, 1, i+1)
        plt.scatter(S, res, alpha=0.03, s=1, color='gray')
        plt.plot(target_s, target_res, 'r-', linewidth=2, label='Binned Model')
        plt.axhline(0, color='k', linestyle='--')
        plt.title(f'{side.capitalize()} Half - Binned Model')
        plt.legend()

    plt.tight_layout()
    plt.savefig('/www/gemini/sag_binned_model.png')
    with open('gemini_flatten/params_binned.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Binned model saved.")
