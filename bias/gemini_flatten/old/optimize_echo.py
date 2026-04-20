
import numpy as np
import os
import sys

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import scipy.optimize as opt
import json

def get_corrected_row_medians(img_list, p_halves, echo_sigmas, empty_cols, hot_mask):
    # Jointly calculate row medians after correction with current echo_sigmas
    es_top, es_bot = echo_sigmas
    all_res = []
    
    half = 512
    for img in img_list:
        S = np.sum(img, axis=1)
        min_top_idx = np.argsort(S[:half])[:50]
        bt = np.median(img[:half][min_top_idx], axis=0)
        min_bot_idx = np.argsort(S[half:])[:50]
        bb = np.median(img[half:][min_bot_idx], axis=0)
        
        sag_top = (S[:half] > p_halves['alpha_top'])[:, None] * (p_halves['betas_top'][None, :] + p_halves['sigmas_top'][None, :] * (S[:half] - p_halves['alpha_top'])[:, None])
        sag_bot = (S[half:] > p_halves['alpha_bot'])[:, None] * (p_halves['betas_bot'][None, :] + p_halves['sigmas_bot'][None, :] * (S[half:] - p_halves['alpha_bot'])[:, None])
        
        # Echo
        sag_echo_top = (S[half:] > p_halves['alpha_bot'])[:, None] * (es_top * p_halves['sigmas_top'][None, :] * (S[half:] - p_halves['alpha_bot'])[:, None])
        sag_echo_bot = (S[:half] > p_halves['alpha_top'])[:, None] * (es_bot * p_halves['sigmas_bot'][None, :] * (S[:half] - p_halves['alpha_top'])[:, None])
        
        ts_top = sag_top + sag_echo_top
        ts_bot = sag_bot + sag_echo_bot
        
        sm_top = np.median(ts_top[min_top_idx], axis=0)
        sm_bot = np.median(ts_bot[min_bot_idx], axis=0)
        
        x = np.zeros_like(img)
        x[:half] = (img[:half] - bt[None, :]) + (ts_top - sm_top[None, :])
        x[half:] = (img[half:] - bb[None, :]) + (ts_bot - sm_bot[None, :])
        
        # Median across empty cols
        row_res = np.ma.median(np.ma.array(x[:, empty_cols], mask=hot_mask[:, empty_cols]), axis=1)
        all_res.append(row_res)
        
    return np.mean(all_res, axis=0)

if __name__ == "__main__":
    p = np.load('gemini_flatten/params_per_col_halves.npz')
    
    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl'
    ]
    imgs = [np.asarray(load(f)) for f in files]
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    
    hot_mask = np.zeros((1024, 1024), dtype=bool)
    if os.path.exists('gemini_flatten/hot_pixels.npy'):
        hc = np.load('gemini_flatten/hot_pixels.npy')
        hot_mask[hc[:, 0], hc[:, 1]] = True

    def objective(echo_params):
        res = get_corrected_row_medians(imgs, p, echo_params, empty_cols, hot_mask)
        return np.var(res)

    print("Optimizing echo sigmas...")
    res = opt.differential_evolution(objective, [(0, 1.0), (0, 1.0)], popsize=10, maxiter=20)
    
    es_top, es_bot = res.x
    print(f"Optimized Echo Sigmas: Top={es_top:.4f}, Bot={es_bot:.4f}")
    
    # Save final multi-component params
    np.savez('gemini_flatten/params_final_multi.npz', 
             betas_top=p['betas_top'], sigmas_top=p['sigmas_top'], alpha_top=p['alpha_top'], echo_sigma_top=es_top,
             betas_bot=p['betas_bot'], sigmas_bot=p['sigmas_bot'], alpha_bot=p['alpha_bot'], echo_sigma_bot=es_bot)
    print("Final multi-component parameters saved.")
