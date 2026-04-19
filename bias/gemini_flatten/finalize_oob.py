
import numpy as np
from common import load
import json

def apply_correction(img, params):
    sigma, beta, alpha, sigma_p, alpha_p = params
    half = img.shape[0] // 2
    S_obs = np.sum(img, axis=1)
    res = np.zeros_like(img)
    s, s_p = S_obs[:half], S_obs[half:]
    mask, mask_p = (s > alpha).astype(float), (s_p > alpha_p).astype(float)
    denom = np.maximum(1 - sigma * s[:, None] * mask[:, None] - sigma_p * s_p[:, None] * mask_p[:, None], 0.1)
    res[:half] = (img[:half] + beta * mask[:, None]) / denom
    s, s_p = S_obs[half:], S_obs[:half]
    mask, mask_p = (s > alpha).astype(float), (s_p > alpha_p).astype(float)
    denom = np.maximum(1 - sigma * s[:, None] * mask[:, None] - sigma_p * s_p[:, None] * mask_p[:, None], 0.1)
    res[half:] = (img[half:] + beta * mask[:, None]) / denom
    return res

if __name__ == "__main__":
    with open('params.json', 'r') as f:
        params = json.load(f)
    
    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260113/oob_nfi_l0.pkl',
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl', 
        'images_20260119/oob_nfi_l0.pkl'
    ]
    
    all_residuals = []
    alpha = params[2]
    
    for f in files:
        img = load(f)
        corr = apply_correction(img, params)
        S_obs = np.sum(img, axis=1)
        half = img.shape[0] // 2
        unsagged_top = np.where(S_obs[:half] <= alpha)[0]
        unsagged_bot = np.where(S_obs[half:] <= alpha)[0]
        if len(unsagged_top) == 0: unsagged_top = np.arange(half)
        if len(unsagged_bot) == 0: unsagged_bot = np.arange(half)
        
        bias_top = np.median(img[:half][unsagged_top], axis=0)
        bias_bot = np.median(img[half:][unsagged_bot], axis=0)
        
        res = np.zeros_like(corr)
        res[:half] = corr[:half] - bias_top
        res[half:] = corr[half:] - bias_bot
        all_residuals.append(res)
    
    avg_cj = np.mean(all_residuals, axis=0)
    final_cj_all = np.mean(avg_cj, axis=0)
    
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    final_cj = np.zeros_like(final_cj_all)
    final_cj[empty_cols] = final_cj_all[empty_cols]
    
    np.save('final_cj_oob.npy', final_cj)
    print(f"Final OOB c_j (empty cols only) saved. Mean (on empty)={np.mean(final_cj[empty_cols]):.4f}, Std (on empty)={np.std(final_cj[empty_cols]):.4f}")
    
    # Save a human readable summary
    summary = {
        "params": {
            "sigma": params[0],
            "beta": params[1],
            "alpha": params[2],
            "sigma_prime": params[3],
            "alpha_prime": params[4]
        },
        "cj_file": "final_cj_oob.npy",
        "description": "Parameters optimized on all OOB NFI L0 images with parallel correspondence (i <-> i+512) and robust bias estimation."
    }
    with open('model_summary_oob.json', 'w') as f:
        json.dump(summary, f, indent=4)
