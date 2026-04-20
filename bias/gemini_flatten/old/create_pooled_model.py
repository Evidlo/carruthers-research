
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
    
    # Pool data from both halves to force a single physical model
    S_pooled = np.concatenate([data['S_top'], data['S_bot']])
    res_pooled = np.concatenate([data['res_top'], data['res_bot']])
    
    mask = ~np.isnan(res_pooled)
    S_pooled, res_pooled = S_pooled[mask], res_pooled[mask]
    
    # 1. Get binned median
    bins = np.linspace(S_pooled.min(), S_pooled.max(), 300)
    bin_idx = np.digitize(S_pooled, bins)
    target_s = []
    target_res = []
    for b in range(1, len(bins)):
        m = bin_idx == b
        if np.any(m):
            target_s.append(np.mean(S_pooled[m]))
            target_res.append(np.median(res_pooled[m]))
    
    target_s = np.array(target_s)
    target_res = np.array(target_res)
    
    # Force Sag to be 0 at the minimum observed S to avoid global offsets
    # Use the median of the first few bins as the floor
    floor_res = np.median(target_res[:5])
    target_res -= floor_res
    
    results = {
        'knots': target_s.tolist(),
        'values': (-target_res).tolist() # Sag is positive suppression
    }
    
    plt.figure(figsize=(10, 6))
    plt.scatter(S_pooled, res_pooled - floor_res, alpha=0.03, s=1, color='gray')
    plt.plot(target_s, target_res, 'r-', linewidth=2, label='Pooled Model')
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Pooled Binned Model (Shared across halves)')
    plt.xlabel('Row Sum (S)')
    plt.ylabel('Residual (y - b)')
    plt.savefig('/www/gemini/sag_pooled_model.png')
    
    # Apply to both top and bot in the params file
    final_params = {'top': results, 'bot': results}
    with open('gemini_flatten/params_binned.json', 'w') as f:
        json.dump(final_params, f, indent=4)
    print("Pooled binned model saved.")
