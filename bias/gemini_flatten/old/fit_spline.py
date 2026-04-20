
import numpy as np
import os
import sys

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import scipy.optimize as opt
import json

def apply_spline_model(S, knots, values):
    # Piecewise linear interpolation between fixed knots
    # knots: [k1, k2, ..., kN]
    # values: [v1, v2, ..., vN]
    # Above last knot: vN + sigma_tail * (S - kN)
    # We will treat sigma_tail as an implicit parameter between the last two points
    return np.interp(S, knots, values, left=values[0], right=values[-1] + (values[-1]-values[-2])/(knots[-1]-knots[-2])*(S-knots[-1]) if len(knots)>1 else values[-1])

if __name__ == "__main__":
    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260113/oob_nfi_l0.pkl',
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl', 
        'images_20260119/oob_nfi_l0.pkl'
    ]
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    img_list = [np.asarray(load(f)) for f in files]
    
    # Define 6 knots to cover the transition and linear tail
    def get_auto_knots(S_min, S_max):
        # Focus knots around the onset of sag
        return [S_min, S_min+0.01e6, S_min+0.05e6, S_min+0.1e6, S_min+0.2e6, S_max]

    knots_t = get_auto_knots(2.82e6, 3.7e6)
    knots_b = get_auto_knots(2.38e6, 3.2e6)
    
    results = {
        'top': np.zeros((1024, 6)),
        'bot': np.zeros((1024, 6))
    }
    
    print("Fitting per-column 6-knot spline models...")
    for side in ['top', 'bot']:
        knots = knots_t if side == 'top' else knots_b
        half = 512
        for c in empty_cols:
            all_dy, all_S = [], []
            for img in img_list:
                S = np.sum(img, axis=1)
                sub = img[:half] if side == 'top' else img[half:]
                S_sub = S[:half] if side == 'top' else S[half:]
                min_idx = np.argsort(S_sub)[:50]
                dy = sub[:, c] - np.median(sub[min_idx, c])
                all_dy.extend(list(dy)); all_S.extend(list(S_sub))
            
            all_dy, all_S = np.array(all_dy), np.array(all_S)
            def obj(p):
                # Correction to ADD
                pred = np.interp(all_S, knots, p)
                return np.mean((all_dy + pred)**2)
            
            res = opt.minimize(obj, np.zeros(6), bounds=[(-50, 50)]*6)
            results[side][c] = res.x

        full_idx = np.arange(1024)
        for i in range(6):
            results[side][:, i] = np.interp(full_idx, empty_cols, results[side][empty_cols, i])
            
    np.savez('gemini_flatten/params_analytical_spline.npz', 
             top=results['top'], bot=results['bot'],
             knots_top=knots_t, knots_bot=knots_b)
    print("Spline parameters saved.")
