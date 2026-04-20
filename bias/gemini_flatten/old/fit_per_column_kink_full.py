
import numpy as np
import os
import sys

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import scipy.optimize as opt
import json

if __name__ == "__main__":
    with open('gemini_flatten/params_halves_kink.json', 'r') as f:
        p_halves = json.load(f)
    
    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260113/oob_nfi_l0.pkl',
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl', 
        'images_20260119/oob_nfi_l0.pkl'
    ]
    
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    col_data = {c: {'top': {'S': [], 'dy': []}, 'bot': {'S': [], 'dy': []}} for c in empty_cols}
    
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
            col_data[c]['top']['dy'].extend(list(dy_top[:, c]))
            col_data[c]['bot']['S'].extend(list(S[half:]))
            col_data[c]['bot']['dy'].extend(list(dy_bot[:, c]))
            
    results = {
        'top': {'alphas': np.zeros(1024), 'betas': np.zeros(1024), 'sigmas': np.zeros(1024)},
        'bot': {'alphas': np.zeros(1024), 'betas': np.zeros(1024), 'sigmas': np.zeros(1024)}
    }
    
    print("Fitting per-column Kink parameters (Full DE)...")
    for side in ['top', 'bot']:
        s_min = np.min(col_data[empty_cols[0]][side]['S'])
        s_max = np.max(col_data[empty_cols[0]][side]['S'])
        
        for c in empty_cols:
            S_arr = np.array(col_data[c][side]['S'])
            dy_arr = np.array(col_data[c][side]['dy'])
            
            def obj(params):
                a, b, s = params
                mask = S_arr > a
                pred = - (b + s*(S_arr - a))*mask
                return np.mean((dy_arr - pred)**2)
            
            # Limited iterations for speed
            res = opt.differential_evolution(obj, [(s_min, s_max), (0, 30), (0, 5e-5)], maxiter=10, popsize=8)
            results[side]['alphas'][c], results[side]['betas'][c], results[side]['sigmas'][c] = res.x

        # Interpolate
        full_idx = np.arange(1024)
        for key in ['alphas', 'betas', 'sigmas']:
            results[side][key] = np.interp(full_idx, empty_cols, results[side][key][empty_cols])
    
    # Save
    np.savez('gemini_flatten/params_per_col_kink_full.npz', 
             alphas_top=results['top']['alphas'], betas_top=results['top']['betas'], sigmas_top=results['top']['sigmas'],
             alphas_bot=results['bot']['alphas'], betas_bot=results['bot']['betas'], sigmas_bot=results['bot']['sigmas'])
    print("Full per-column kink parameters saved.")
