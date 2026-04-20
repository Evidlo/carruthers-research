
import numpy as np
import os
import sys

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import json
import matplotlib.pyplot as plt
from model_pwl import apply_piecewise_linear

if __name__ == "__main__":
    img_path = 'images_20260115/oob_nfi_l0.pkl'
    img = np.asarray(load(img_path))
    S_full = np.sum(img, axis=1)
    half = 512
    
    with open('gemini_flatten/params_multiplicative.json', 'r') as f:
        params = json.load(f)
    
    fig, axes = plt.subplots(6, 1, figsize=(12, 20), sharex=True)
    cols = [100, 200, 300]
    
    for i, side in enumerate(['top', 'bot']):
        p = params[side]
        op = params['bot' if side == 'top' else 'top']
        
        S_self = S_full[:half] if side == 'top' else S_full[half:]
        S_other = S_full[half:] if side == 'top' else S_full[:half]
        data_side = img[:half] if side == 'top' else img[half:]
        
        ps_self = apply_piecewise_linear(S_self, p) / 2500.0
        ps_other = apply_piecewise_linear(S_other, op) / 2500.0
        mask_s = (S_self > p['knots'][0]).astype(float)
        mask_o = (S_other > op['knots'][0]).astype(float)
        
        # Consistent z calculation (multiplicative)
        denom = np.maximum(1 - ps_self * mask_s - p['echo_scale'] * ps_other * mask_o, 0.1)
        z_side = (data_side + p['beta'] * mask_s[:, None]) / denom[:, None]
        
        # Bias from z in nonsagged region
        is_unsag = (S_self <= p['knots'][0]) & (S_other <= op['knots'][0])
        if np.any(is_unsag):
            b_side = np.median(z_side[is_unsag], axis=0)
        else:
            b_side = np.median(z_side[150:363], axis=0)

        for j, c in enumerate(cols):
            ax = axes[i*3 + j]
            y = data_side[:, c]
            b_j = b_side[c]
            
            dy_obs = y - b_j
            
            # Primary model prediction only for line
            sort_idx = np.argsort(S_self)
            S_sorted = S_self[sort_idx]
            dy_pred_primary = - (b_j * (apply_piecewise_linear(S_sorted, p) / 2500.0) * (S_sorted > p['knots'][0]).astype(float) + p['beta'] * (S_sorted > p['knots'][0]).astype(float))
            
            ax.scatter(S_self, dy_obs, alpha=0.3, s=2, color='blue', label='Data (y - b)')
            ax.plot(S_sorted, dy_pred_primary, 'r-', linewidth=2, label='Primary Model')
            
            for knot in p['knots']:
                ax.axvline(knot, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_title(f'{side.upper()} Half - Column {c} (b_j = {b_j:.1f})')
            ax.set_ylabel('Residual (counts)')
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.2)
            if i*3 + j == 0: ax.legend()

    axes[-1].set_xlabel('Row Sum (S)')
    plt.tight_layout()
    plt.savefig('/www/gemini/model_vs_cols_v2.png')
    print("Corrected 1D plots saved.")
