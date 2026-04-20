
import numpy as np
import os
import sys

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import scipy.optimize as opt
import json
from model_pwl import apply_piecewise_linear

if __name__ == "__main__":
    with open('gemini_flatten/knot_coords.json', 'r') as f:
        knot_data = json.load(f)
    k_t = [pt[0] for pt in knot_data['top']]
    k_b = [pt[0] for pt in knot_data['bot']]
    
    files = ['images_20260111/oob_nfi_l0.pkl', 'images_20260115/oob_nfi_l0.pkl', 'images_20260117/oob_nfi_l0.pkl']
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    
    training_data = []
    hot_mask = np.zeros((1024, 1024), dtype=bool)
    if os.path.exists('gemini_flatten/hot_pixels.npy'):
        hc = np.load('gemini_flatten/hot_pixels.npy')
        hot_mask[hc[:, 0], hc[:, 1]] = True

    for f in files:
        img = np.asarray(load(f))
        S = np.sum(img, axis=1)
        half = 512
        for side in ['top', 'bot']:
            sub = img[:half] if side == 'top' else img[half:]
            m = hot_mask[:half, empty_cols] if side == 'top' else hot_mask[half:, empty_cols]
            rt = np.ma.median(np.ma.array(sub[:, empty_cols], mask=m), axis=1).filled(np.nan)
            training_data.append({
                'side': side,
                'S_self': S[:half] if side == 'top' else S[half:],
                'S_other': S[half:] if side == 'top' else S[:half],
                'rt': rt
            })

    def objective(params):
        # [bt, vt1..vt3, st, est,  bb, vb1..vb3, sb, esb]
        bt, vt1, vt2, vt3, st, est,  bb, vb1, vb2, vb3, sb, esb = params
        
        total_err = 0
        for d in training_data:
            side = d['side']
            if side == 'top':
                p_dict = {'knots': k_t, 'values': [vt1, vt2, vt3], 'sigma_tail': st}
                op_dict = {'knots': k_b, 'values': [vb1, vb2, vb3], 'sigma_tail': sb}
                beta, es = bt, est
            else:
                p_dict = {'knots': k_b, 'values': [vb1, vb2, vb3], 'sigma_tail': sb}
                op_dict = {'knots': k_t, 'values': [vt1, vt2, vt3], 'sigma_tail': st}
                beta, es = bb, esb
            
            ps = apply_piecewise_linear(d['S_self'], p_dict) / 2500.0
            po = apply_piecewise_linear(d['S_other'], op_dict) / 2500.0
            
            mask_s = (d['S_self'] > p_dict['knots'][0]).astype(float)
            mask_o = (d['S_other'] > op_dict['knots'][0]).astype(float)
            
            denom = np.maximum(1 - ps * mask_s - es * po * mask_o, 0.1)
            z = (d['rt'] + beta * mask_s) / denom
            
            # Use current bias logic
            min_idx = d['S_self'] <= p_dict['knots'][0]
            if np.any(min_idx):
                b_avg = np.median(z[min_idx])
            else:
                b_avg = np.median(z)
            
            total_err += np.nanmean((z - b_avg)**2)
        return total_err / len(training_data)

    print("Fitting robust joint multiplicative model (Centralized)...")
    bnds_half = [(0, 15), (-5, 15), (-5, 15), (-5, 15), (0, 1e-4), (0, 1.0)]
    res = opt.differential_evolution(objective, bnds_half * 2, popsize=15, maxiter=80)
    
    p = res.x
    final_params = {
        'top': {'alpha': k_t[0], 'beta': p[0], 'knots': k_t, 'values': p[1:4].tolist(), 'sigma_tail': p[4], 'echo_scale': p[5]},
        'bot': {'alpha': k_b[0], 'beta': p[6], 'knots': k_b, 'values': p[7:10].tolist(), 'sigma_tail': p[10], 'echo_scale': p[11]}
    }
    with open('gemini_flatten/params_multiplicative.json', 'w') as f:
        json.dump(final_params, f, indent=4)
    print("Optimization complete.")
