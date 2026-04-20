
import numpy as np
import os
import sys

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import scipy.optimize as opt
import json

if __name__ == "__main__":
    with open('gemini_flatten/params_halves.json', 'r') as f:
        p_halves = json.load(f)
    
    alpha_top = p_halves['top'][0]
    alpha_bot = p_halves['bot'][0]
    
    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260113/oob_nfi_l0.pkl',
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl', 
        'images_20260119/oob_nfi_l0.pkl'
    ]
    
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    
    # Store per-half data for each column
    col_data = {c: {'top': {'S': [], 'dy': []}, 'bot': {'S': [], 'dy': []}} for c in empty_cols}
    
    for f in files:
        img = np.asarray(load(f))
        S = np.sum(img, axis=1)
        half = 512
        
        # Bias from absolute cleanest rows
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
            
    # Fit per-column, per-half
    results = {
        'top': {'betas': np.zeros(1024), 'sigmas': np.zeros(1024)},
        'bot': {'betas': np.zeros(1024), 'sigmas': np.zeros(1024)}
    }
    
    print("Fitting per-column, per-half parameters...")
    for side in ['top', 'bot']:
        alpha = alpha_top if side == 'top' else alpha_bot
        for c in empty_cols:
            S_arr = np.array(col_data[c][side]['S'])
            dy_arr = np.array(col_data[c][side]['dy'])
            mask = S_arr > alpha
            
            def col_model(params):
                b, s = params
                pred = - (b + s*(S_arr - alpha))*mask
                return np.mean((dy_arr - pred)**2)
            
            # Initial guess from robust fit
            init_b = p_halves[side][1]
            init_s = p_halves[side][2]
            
            res = opt.minimize(col_model, [init_b, init_s], bounds=[(0, 50), (0, 5e-5)])
            results[side]['betas'][c], results[side]['sigmas'][c] = res.x

        # Interpolate
        full_idx = np.arange(1024)
        results[side]['betas'] = np.interp(full_idx, empty_cols, results[side]['betas'][empty_cols])
        results[side]['sigmas'] = np.interp(full_idx, empty_cols, results[side]['sigmas'][empty_cols])
    
    # Save
    np.savez('gemini_flatten/params_per_col_halves.npz', 
             betas_top=results['top']['betas'], sigmas_top=results['top']['sigmas'], alpha_top=alpha_top,
             betas_bot=results['bot']['betas'], sigmas_bot=results['bot']['sigmas'], alpha_bot=alpha_bot)
    print("Per-column, per-half parameters saved.")
