
import numpy as np
import os
import sys

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import scipy.optimize as opt
import json

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
    col_data = {c: {'top': {'S': [], 'Sp': [], 'dy': []}, 'bot': {'S': [], 'Sp': [], 'dy': []}} for c in empty_cols}
    
    def get_sag_lut(S, side_params):
        knots = np.array(side_params['knots'])
        values = np.array(side_params['values'])
        return np.interp(S, knots, values, left=0)

    for f in files:
        img = np.asarray(load(f))
        S = np.sum(img, axis=1)
        half = 512
        min_top_idx = np.argsort(S[:half])[:50]
        bt = np.median(img[:half][min_top_idx], axis=0)
        min_bot_idx = np.argsort(S[half:])[:50]
        bb = np.median(img[half:][min_bot_idx], axis=0)
        
        dy_top = img[:half] - bt[None, :]
        dy_bot = img[half:] - bb[None, :]
        
        for c in empty_cols:
            col_data[c]['top']['S'].extend(list(S[:half]))
            col_data[c]['top']['Sp'].extend(list(S[half:]))
            col_data[c]['top']['dy'].extend(list(dy_top[:, c]))
            col_data[c]['bot']['S'].extend(list(S[half:]))
            col_data[c]['bot']['Sp'].extend(list(S[:half]))
            col_data[c]['bot']['dy'].extend(list(dy_bot[:, c]))
            
    # Model: x = (y - b) + (beta_j * model(S) + sigma_j * model(S'))
    # Wait, the physics says y = x - Sag(S, x+b).
    # Simple multiplicative model: y = z * (1 - coeff_j * model(S))
    # z = y / (1 - coeff_j * model(S))
    # Let's use a simple scaling of the binned model per column.
    
    results = {
        'top': {'coeffs': np.zeros(1024)},
        'bot': {'coeffs': np.zeros(1024)}
    }
    
    print("Fitting per-column scaling of pooled model...")
    model_top = get_sag_lut(np.array(col_data[empty_cols[0]]['top']['S']), p_binned['top'])
    model_bot = get_sag_lut(np.array(col_data[empty_cols[0]]['bot']['S']), p_binned['bot'])

    for side in ['top', 'bot']:
        m = model_top if side == 'top' else model_bot
        for c in empty_cols:
            dy = np.array(col_data[c][side]['dy'])
            # We want dy = - coeff * m  => coeff = -dy / m
            # Linear regression:
            sol, _, _, _ = np.linalg.lstsq(m[:, None], dy, rcond=None)
            results[side]['coeffs'][c] = -sol[0]

        # Interpolate
        full_idx = np.arange(1024)
        results[side]['coeffs'] = np.interp(full_idx, empty_cols, results[side]['coeffs'][empty_cols])
    
    # Save
    np.savez('gemini_flatten/params_per_col_pooled.npz', 
             coeffs_top=results['top']['coeffs'], 
             coeffs_bot=results['bot']['coeffs'])
    print("Per-column pooled parameters saved.")
