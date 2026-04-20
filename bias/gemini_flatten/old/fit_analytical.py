
import numpy as np
import os
import sys

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import scipy.optimize as opt
import json

def apply_piecewise_model(S, knots, coeffs):
    # knots: [k1, k2, k3]
    # coeffs: [c1, c2, c3, sigma_tail]
    # This represents the correction to ADD to y to get z
    
    # Below k1: use c1 (or linear to c1? let's assume floor c1)
    # Between k1-k2: linear interp
    # Between k2-k3: linear interp
    # Above k3: c3 + sigma_tail * (S - k3)
    
    k1, k2, k3 = knots
    c1, c2, c3, st = coeffs
    
    res = np.zeros_like(S)
    
    # Region 1: < k1
    m1 = S <= k1
    res[m1] = c1
    
    # Region 2: k1 to k2
    m2 = (S > k1) & (S <= k2)
    res[m2] = np.interp(S[m2], [k1, k2], [c1, c2])
    
    # Region 3: k2 to k3
    m3 = (S > k2) & (S <= k3)
    res[m3] = np.interp(S[m3], [k2, k3], [c2, c3])
    
    # Region 4: > k3
    m4 = S > k3
    res[m4] = c3 + st * (S[m4] - k3)
    
    return res

if __name__ == "__main__":
    with open('gemini_flatten/knot_coords.json', 'r') as f:
        knot_data = json.load(f)
    
    # Knots are fixed S values
    knots_top = [pt[0] for pt in knot_data['top']]
    knots_bot = [pt[0] for pt in knot_data['bot']]
    
    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260113/oob_nfi_l0.pkl',
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl', 
        'images_20260119/oob_nfi_l0.pkl'
    ]
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    
    img_list = [np.asarray(load(f)) for f in files]
    
    # Global fit of (c1, c2, c3, sigma) per column
    results = {
        'top': np.zeros((1024, 4)),
        'bot': np.zeros((1024, 4))
    }
    
    print("Fitting per-column analytical piecewise models...")
    for side in ['top', 'bot']:
        knots = knots_top if side == 'top' else knots_bot
        half = 512
        
        for c in empty_cols:
            all_dy = []
            all_S = []
            for img in img_list:
                S = np.sum(img, axis=1)
                sub = img[:half] if side == 'top' else img[half:]
                S_sub = S[:half] if side == 'top' else S[half:]
                
                # Bias from absolute min
                min_idx = np.argsort(S_sub)[:50]
                # Target: (y + Model(S)) - median(y[min] + Model(S[min])) = 0
                # Simplification: Model(S) = -dy
                dy = sub[:, c] - np.median(sub[min_idx, c])
                all_dy.extend(list(dy))
                all_S.extend(list(S_sub))
            
            all_dy = np.array(all_dy)
            all_S = np.array(all_S)
            
            def obj(p):
                # Correction to ADD: c1, c2, c3, st
                pred_corr = apply_piecewise_model(all_S, knots, p)
                # Corrected y: z = y + pred_corr
                # We want z - median(z[min_idx_approx]) to be flat
                return np.mean((all_dy + pred_corr)**2)
            
            # Use a slightly larger range for sigma tail
            res = opt.minimize(obj, [0, 0, 0, 1e-6], bounds=[(-50, 50), (-50, 50), (-50, 50), (0, 1e-4)])
            results[side][c] = res.x

        # Interpolate
        full_idx = np.arange(1024)
        for i in range(4):
            results[side][:, i] = np.interp(full_idx, empty_cols, results[side][empty_cols, i])
            
    np.savez('gemini_flatten/params_analytical_piecewise.npz', 
             top=results['top'], bot=results['bot'],
             knots_top=knots_top, knots_bot=knots_bot)
    print("Analytical parameters saved.")
