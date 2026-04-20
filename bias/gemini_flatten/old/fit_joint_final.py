
import numpy as np
import os
import sys

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import scipy.optimize as opt
import json

def get_sag_lut(S, side_params):
    knots = np.array(side_params['knots'])
    values = np.array(side_params['values'])
    return np.interp(S, knots, values, left=0)

if __name__ == "__main__":
    with open('gemini_flatten/params_binned.json', 'r') as f:
        p_binned = json.load(f)
    
    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260113/oob_nfi_l0.pkl',
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl', 
        'images_20260119/oob_nfi_l0.pkl'
    ]
    
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    
    # Pre-calculate pooled model values for all rows
    img_data = []
    for f in files:
        img = np.asarray(load(f))
        S = np.sum(img, axis=1)
        half = 512
        m_top = get_sag_lut(S[:half], p_binned['top'])
        m_bot = get_sag_lut(S[half:], p_binned['bot'])
        # Simplified echo logic: 0.1 global
        e_top = 0.1 * get_sag_lut(S[half:], p_binned['bot'])
        e_bot = 0.1 * get_sag_lut(S[:half], p_binned['top'])
        
        img_data.append({
            'img': img,
            'S': S,
            'model_top': m_top + e_top,
            'model_bot': m_bot + e_bot
        })

    results = {'top': np.zeros(1024), 'bot': np.zeros(1024)}
    
    print("Fitting per-column scaling with joint bias estimation...")
    for side in ['top', 'bot']:
        for c in empty_cols:
            all_dy = []
            all_m = []
            
            for d in img_data:
                half = 512
                sub = d['img'][:half] if side == 'top' else d['img'][half:]
                S_sub = d['S'][:half] if side == 'top' else d['S'][half:]
                m = d['model_top'] if side == 'top' else d['model_bot']
                
                # min_idx identifies rows where we expect zero charge
                min_idx = np.argsort(S_sub)[:50]
                
                # Target: (y + coeff * m) - b = 0
                # b = median(y[min] + coeff * m[min])
                # y - median(y[min]) = -coeff * (m - median(m[min]))
                
                y_norm = sub[:, c] - np.median(sub[min_idx, c])
                m_norm = m - np.median(m[min_idx])
                
                all_dy.extend(list(y_norm))
                all_m.extend(list(m_norm))
                
            all_dy = np.array(all_dy)
            all_m = np.array(all_m)
            
            # Solve all_dy = -coeff * all_m
            sol, _, _, _ = np.linalg.lstsq(all_m[:, None], all_dy, rcond=None)
            results[side][c] = -sol[0]

        # Interpolate
        full_idx = np.arange(1024)
        results[side] = np.interp(full_idx, empty_cols, results[side][empty_cols])
    
    np.savez('gemini_flatten/params_joint_final.npz', 
             coeffs_top=results['top'], 
             coeffs_bot=results['bot'])
    print("Joint parameters saved.")
