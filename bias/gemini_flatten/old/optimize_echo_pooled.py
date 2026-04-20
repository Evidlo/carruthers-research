
import numpy as np
import os
import sys

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import scipy.optimize as opt
import json

def get_corrected_row_medians(img_list, p_binned, echo_sigma, empty_cols, hot_mask):
    all_res = []
    half = 512
    
    def get_sag_lut(S, side_params):
        knots = np.array(side_params['knots'])
        values = np.array(side_params['values'])
        return np.interp(S, knots, values, left=0)

    for img in img_list:
        S = np.sum(img, axis=1)
        min_top_idx = np.argsort(S[:half])[:50]
        bt = np.median(img[:half][min_top_idx], axis=0)
        min_bot_idx = np.argsort(S[half:])[:50]
        bb = np.median(img[half:][min_bot_idx], axis=0)
        
        sag_top = get_sag_lut(S[:half], p_binned['top'])[:, None]
        sag_bot = get_sag_lut(S[half:], p_binned['bot'])[:, None]
        
        # Echo
        echo_top = echo_sigma * get_sag_lut(S[half:], p_binned['bot'])[:, None]
        echo_bot = echo_sigma * get_sag_lut(S[:half], p_binned['top'])[:, None]
        
        ts_top = sag_top + echo_top
        ts_bot = sag_bot + echo_bot
        
        sm_top = np.median(ts_top[min_top_idx], axis=0)
        sm_bot = np.median(ts_bot[min_bot_idx], axis=0)
        
        x = np.zeros_like(img)
        x[:half] = (img[:half] - bt[None, :]) + (ts_top - sm_top[None, :])
        x[half:] = (img[half:] - bb[None, :]) + (ts_bot - sm_bot[None, :])
        
        row_res = np.ma.median(np.ma.array(x[:, empty_cols], mask=hot_mask[:, empty_cols]), axis=1)
        all_res.append(row_res)
        
    return np.mean(all_res, axis=0)

if __name__ == "__main__":
    with open('gemini_flatten/params_binned.json', 'r') as f:
        p = json.load(f)
    
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

    def objective(echo_param):
        res = get_corrected_row_medians(imgs, p, echo_param[0], empty_cols, hot_mask)
        # Minimize global variance AND the jump specifically
        var = np.var(res)
        top_edge = np.mean(res[512-20:512])
        bot_edge = np.mean(res[512:512+20])
        jump_penalty = (top_edge - bot_edge)**2
        return var + 10 * jump_penalty # Heavy penalty on jump

    print("Optimizing shared echo sigma...")
    res = opt.differential_evolution(objective, [(0, 1.0)], popsize=15, maxiter=30)
    
    es = res.x[0]
    print(f"Optimized Shared Echo Sigma: {es:.4f}")
    
    # Save 
    with open('gemini_flatten/echo_sigma.json', 'w') as f:
        json.dump({"echo_sigma": es}, f)
