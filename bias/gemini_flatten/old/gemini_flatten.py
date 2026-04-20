
import numpy as np
from common import load
import scipy.optimize as opt
import json
import os

def get_params_multi(img_list, empty_cols, nonsag_top_range, nonsag_bot_range):
    all_img_data = []
    all_s_max = []
    all_s_min_guess = []
    
    for img in img_list:
        S_obs = np.sum(img, axis=1)
        half = img.shape[0] // 2
        S_top, S_bot = S_obs[:half], S_obs[half:]
        
        idx_top = np.argsort(S_top)
        idx_bot = np.argsort(S_bot)
        
        y_top_empty = img[:half, empty_cols]
        y_bot_empty = img[half:, empty_cols]
        
        all_img_data.append({
            'S_top': S_top,
            'S_bot': S_bot,
            'y_top_empty': y_top_empty,
            'y_bot_empty': y_bot_empty,
            'S_top_sorted': S_top[idx_top],
            'S_bot_sorted': S_bot[idx_bot],
            'y_top_empty_sorted': img[:half][idx_top][:, empty_cols],
            'y_bot_empty_sorted': img[half:][idx_bot][:, empty_cols]
        })
        
        all_s_max.append(S_obs.max())
        s_nonsag_init = np.concatenate([S_obs[nonsag_top_range[0]:nonsag_top_range[1]], 
                                        S_obs[nonsag_bot_range[0]:nonsag_bot_range[1]]])
        all_s_min_guess.append(np.max(s_nonsag_init))

    # Load hot pixels if they exist
    hot_mask = None
    if os.path.exists('hot_pixels.npy'):
        hot_coords = np.load('hot_pixels.npy')
        hot_mask = np.zeros((1024, 1024), dtype=bool)
        hot_mask[hot_coords[:, 0], hot_coords[:, 1]] = True
        print(f"Loaded {len(hot_coords)} hot pixels for masking.")

    def objective(params):
        sigma, beta, alpha, sigma_p, alpha_p = params
        total_loss = 0
        for d in all_img_data:
            n_top = np.searchsorted(d['S_top_sorted'], alpha)
            n_bot = np.searchsorted(d['S_bot_sorted'], alpha)
            
            if n_top == 0: n_top = len(d['S_top_sorted'])
            if n_bot == 0: n_bot = len(d['S_bot_sorted'])
            
            b_top_empty = np.median(d['y_top_empty_sorted'][:n_top], axis=0)
            b_bot_empty = np.median(d['y_bot_empty_sorted'][:n_bot], axis=0)
            
            # Top
            s, s_p = d['S_top'], d['S_bot']
            mask_sag, mask_p = (s > alpha).astype(float), (s_p > alpha_p).astype(float)
            denom = np.maximum(1 - sigma * s[:, None] * mask_sag[:, None] - sigma_p * s_p[:, None] * mask_p[:, None], 0.1)
            z_top = (d['y_top_empty'] + beta * mask_sag[:, None]) / denom
            x_top = z_top - b_top_empty[None, :]
            
            # Bot
            s, s_p = d['S_bot'], d['S_top']
            mask_sag, mask_p = (s > alpha).astype(float), (s_p > alpha_p).astype(float)
            denom = np.maximum(1 - sigma * s[:, None] * mask_sag[:, None] - sigma_p * s_p[:, None] * mask_p[:, None], 0.1)
            z_bot = (d['y_bot_empty'] + beta * mask_sag[:, None]) / denom
            x_bot = z_bot - b_bot_empty[None, :]
            
            if hot_mask is not None:
                m_top = hot_mask[:512, empty_cols]
                m_bot = hot_mask[512:, empty_cols]
                total_loss += np.mean(x_top[~m_top]**2) + np.mean(x_bot[~m_bot]**2)
            else:
                total_loss += np.mean(x_top**2) + np.mean(x_bot**2)
                
        return total_loss / len(all_img_data)

    alpha_min = np.min(all_s_min_guess)
    alpha_max = np.max(all_s_max)
    print(f"Alpha bounds: {alpha_min:.2f} to {alpha_max:.2f}")

    bounds = [
        (0, 1e-6), (0, 10), (alpha_min, alpha_max),
        (0, 1e-6), (alpha_min, alpha_max)
    ]
    
    res = opt.differential_evolution(objective, bounds, maxiter=10, popsize=12)
    return res.x

def apply_correction(img, params):
    sigma, beta, alpha, sigma_p, alpha_p = params
    half = img.shape[0] // 2
    S_obs = np.sum(img, axis=1)
    
    res = np.zeros_like(img)
    
    # Top
    s = S_obs[:half]
    s_p = S_obs[half:]
    mask = (s > alpha).astype(float)
    mask_p = (s_p > alpha_p).astype(float)
    denom = np.maximum(1 - sigma * s[:, None] * mask[:, None] - sigma_p * s_p[:, None] * mask_p[:, None], 0.1)
    res[:half] = (img[:half] + beta * mask[:, None]) / denom
    
    # Bot
    s = S_obs[half:]
    s_p = S_obs[:half]
    mask = (s > alpha).astype(float)
    mask_p = (s_p > alpha_p).astype(float)
    denom = np.maximum(1 - sigma * s[:, None] * mask[:, None] - sigma_p * s_p[:, None] * mask_p[:, None], 0.1)
    res[half:] = (img[half:] + beta * mask[:, None]) / denom
    
    return res

if __name__ == "__main__":
    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260113/oob_nfi_l0.pkl',
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl', 
        'images_20260119/oob_nfi_l0.pkl'
    ]
    imgs = [load(f) for f in files]
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    
    params = get_params_multi(imgs, empty_cols, (150, 363), (662, 875))
    print(f"OOB-only optimized params: sigma={params[0]:.2e}, beta={params[1]:.4f}, alpha={params[2]:.1f}, sigma_p={params[3]:.2e}, alpha_p={params[4]:.1f}")
    
    with open('params.json', 'w') as f:
        json.dump(list(params), f)

    for i, img in enumerate(imgs):
        corrected_z = apply_correction(img, params)
        S_obs = np.sum(img, axis=1)
        half = img.shape[0] // 2
        
        alpha = params[2]
        unsagged_top = np.where(S_obs[:half] <= alpha)[0]
        unsagged_bot = np.where(S_obs[half:] <= alpha)[0]
        if len(unsagged_top) == 0: unsagged_top = np.arange(half)
        if len(unsagged_bot) == 0: unsagged_bot = np.arange(half)
        
        bias_top = np.median(img[:half][unsagged_top], axis=0)
        bias_bot = np.median(img[half:][unsagged_bot], axis=0)
        
        corrected_x = np.zeros_like(corrected_z)
        corrected_x[:half] = corrected_z[:half] - bias_top
        corrected_x[half:] = corrected_z[half:] - bias_bot
        
        orig_x = np.zeros_like(img)
        orig_x[:half] = img[:half] - bias_top
        orig_x[half:] = img[half:] - bias_bot
        
        loss_orig = np.mean(orig_x[:, empty_cols]**2)
        loss_corr = np.mean(corrected_x[:, empty_cols]**2)
        print(f"Image {i} ({files[i]}): Original Loss={loss_orig:.4f}, Corrected Loss={loss_corr:.4f}, Improvement={(loss_orig-loss_corr)/loss_orig*100:.2f}%")
