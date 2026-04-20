
import numpy as np
import os
import sys

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import scipy.optimize as opt
import json

def get_clean_x_ramp(img, params, empty_cols, hot_mask):
    # params: [sigma_global, beta_global, alpha, sigma_p_global, alpha_p]
    # We want to extend this to per-column eventually, but let's first refine globals
    sigma, beta, alpha, sigma_p, alpha_p = params
    half = 512
    S = np.sum(img, axis=1)
    
    # Robust bias
    ut = np.where(S[:half] <= alpha)[0]
    ub = np.where(S[half:] <= alpha)[0]
    if len(ut) == 0: ut = np.arange(half)
    if len(ub) == 0: ub = np.arange(half)
    bt = np.median(img[:half][ut], axis=0)
    bb = np.median(img[half:][ub], axis=0)
    
    # Corrected x
    S_top, S_bot = S[:half], S[half:]
    mask_top = (S_top > alpha).astype(float)
    mask_bot = (S_bot > alpha).astype(float)
    mask_p_top = (S_bot > alpha_p).astype(float)
    mask_p_bot = (S_top > alpha_p).astype(float)
    
    x = np.zeros_like(img)
    x[:half] = img[:half] + mask_top[:, None] * (beta + sigma * (S_top - alpha))[:, None] + (mask_p_top[:, None] * sigma_p * (S_bot - alpha_p)[:, None]) - bt
    x[half:] = img[half:] + mask_bot[:, None] * (beta + sigma * (S_bot - alpha))[:, None] + (mask_p_bot[:, None] * sigma_p * (S_top - alpha_p)[:, None]) - bb
    
    return x

if __name__ == "__main__":
    # Let's perform a per-column fit of beta and sigma on the empty columns!
    # For each empty column j:
    # (y_ij - b_j) = - (beta_j * 1(S_i > alpha) + sigma_j * (S_i - alpha) * 1(S_i > alpha) + ...)
    
    with open('gemini_flatten/params_ramp.json', 'r') as f:
        p_glob = json.load(f)
    
    alpha = p_glob[2]
    alpha_p = p_glob[4]
    
    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260113/oob_nfi_l0.pkl',
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl', 
        'images_20260119/oob_nfi_l0.pkl'
    ]
    
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    
    # Collect all (S, S_prime, delta_y) for EACH column
    col_data = {c: {'S': [], 'Sp': [], 'dy': []} for c in empty_cols}
    
    for f in files:
        img = np.asarray(load(f))
        S = np.sum(img, axis=1)
        half = 512
        
        # Find rows with MINIMUM S for cleanest bias
        min_top_idx = np.argsort(S[:half])[:50] 
        bt = np.median(img[:half][min_top_idx], axis=0)
        min_bot_idx = np.argsort(S[half:])[:50]
        bb = np.median(img[half:][min_bot_idx], axis=0)
        
        # dy = (y - b)
        dy_top = img[:half] - bt[None, :]
        dy_bot = img[half:] - bb[None, :]
        
        for c in empty_cols:
            col_data[c]['S'].extend(list(np.concatenate([S[:half], S[half:]])))
            col_data[c]['Sp'].extend(list(np.concatenate([S[half:], S[:half]])))
            col_data[c]['dy'].extend(list(np.concatenate([dy_top[:, c], dy_bot[:, c]])))
            
    # Fit per-column
    betas = np.zeros(1024)
    sigmas = np.zeros(1024)
    sigmas_p = np.zeros(1024)
    
    print("Fitting per-column parameters...")
    for c in empty_cols:
        S_arr = np.array(col_data[c]['S'])
        Sp_arr = np.array(col_data[c]['Sp'])
        dy_arr = np.array(col_data[c]['dy'])
        
        mask = S_arr > alpha
        mask_p = Sp_arr > alpha_p
        
        # dy = - (beta + sigma*(S-alpha))*mask - sigma_p*(Sp-alpha_p)*mask_p
        def col_model(params):
            b, s, sp = params
            pred = - (b + s*(S_arr - alpha))*mask - sp*(Sp_arr - alpha_p)*mask_p
            return np.mean((dy_arr - pred)**2)
        
        res = opt.minimize(col_model, [p_glob[1], p_glob[0], p_glob[3]], bounds=[(0, 50), (0, 1e-4), (0, 1e-4)])
        betas[c], sigmas[c], sigmas_p[c] = res.x

    # Interpolate for the central columns (rough guess)
    full_idx = np.arange(1024)
    betas = np.interp(full_idx, empty_cols, betas[empty_cols])
    sigmas = np.interp(full_idx, empty_cols, sigmas[empty_cols])
    sigmas_p = np.interp(full_idx, empty_cols, sigmas_p[empty_cols])
    
    # Save per-column params
    np.savez('gemini_flatten/params_per_col.npz', betas=betas, sigmas=sigmas, sigmas_p=sigmas_p, alpha=alpha, alpha_p=alpha_p)
    print("Per-column parameters saved.")
