
import numpy as np
import os
import sys

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import scipy.optimize as opt
import json

def apply_correction_ramp(img, params):
    sigma, beta, alpha, sigma_p, alpha_p = params
    half = img.shape[0] // 2
    S_obs = np.sum(img, axis=1)
    res = np.zeros_like(img)
    
    # Top
    s, s_p = S_obs[:half], S_obs[half:]
    mask = (s > alpha).astype(float)
    mask_p = (s_p > alpha_p).astype(float)
    
    # Ramp model: Sag = beta * 1(S>alpha) + sigma * (S-alpha) * 1(S>alpha)
    # y = z - Sag * z => z = y / (1 - Sag)
    sag_p = mask * (beta + sigma * (s - alpha))
    sag_e = mask_p * (sigma_p * (s_p - alpha_p)) # No floor for echo?
    
    # We must normalize sag to counts/pixel. 
    # But wait, sigma in fit_sag_model.py was relative to pixel counts directly.
    # So sag is in counts.
    # z = y + sag
    # Actually, if it's proportional to pixel value: z = y / (1 - sag_coeff)
    # Let's use the additive version first as it's more stable for these small values
    # x = (y + beta*mask + sigma*(s-alpha)*mask + sigma_p*(s_p-alpha_p)*mask_p) - b
    
    res[:half] = img[:half] + mask[:, None] * (beta + sigma * (s - alpha))[:, None] + (mask_p[:, None] * sigma_p * (s_p - alpha_p)[:, None])
    
    # Bot
    s, s_p = S_obs[half:], S_obs[:half]
    mask = (s > alpha).astype(float)
    mask_p = (s_p > alpha_p).astype(float)
    res[half:] = img[half:] + mask[:, None] * (beta + sigma * (s - alpha))[:, None] + (mask_p[:, None] * sigma_p * (s_p - alpha_p)[:, None])
    
    return res

def get_params_multi(img_list, empty_cols, nonsag_top_range, nonsag_bot_range, hot_mask=None):
    all_img_data = []
    all_s_max = []
    all_s_min_guess = []
    
    for img in img_list:
        S_obs = np.sum(img, axis=1)
        half = img.shape[0] // 2
        S_top, S_bot = S_obs[:half], S_obs[half:]
        idx_top = np.argsort(S_top)
        idx_bot = np.argsort(S_bot)
        
        all_img_data.append({
            'S_top': S_top,
            'S_bot': S_bot,
            'y_top_empty': img[:half, empty_cols],
            'y_bot_empty': img[half:, empty_cols],
            'S_top_sorted': S_top[idx_top],
            'S_bot_sorted': S_bot[idx_bot],
            'y_top_empty_sorted': img[:half][idx_top][:, empty_cols],
            'y_bot_empty_sorted': img[half:][idx_bot][:, empty_cols]
        })
        all_s_max.append(S_obs.max())
        s_nonsag_init = np.concatenate([S_obs[nonsag_top_range[0]:nonsag_top_range[1]], 
                                        S_obs[nonsag_bot_range[0]:nonsag_bot_range[1]]])
        all_s_min_guess.append(np.max(s_nonsag_init))

    def objective(params):
        sigma, beta, alpha, sigma_p, alpha_p = params
        total_loss = 0
        for d in all_img_data:
            # Consistent bias estimation from unsagged rows (using alpha)
            n_top = np.searchsorted(d['S_top_sorted'], alpha)
            n_bot = np.searchsorted(d['S_bot_sorted'], alpha)
            if n_top == 0: n_top = len(d['S_top_sorted'])
            if n_bot == 0: n_bot = len(d['S_bot_sorted'])
            
            b_top_empty = np.median(d['y_top_empty_sorted'][:n_top], axis=0)
            b_bot_empty = np.median(d['y_bot_empty_sorted'][:n_bot], axis=0)
            
            # Apply ramp correction
            s, s_p = d['S_top'], d['S_bot']
            mask, mask_p = (s > alpha).astype(float), (s_p > alpha_p).astype(float)
            
            x_top = d['y_top_empty'] + mask[:, None] * (beta + sigma * (s - alpha))[:, None] + (mask_p[:, None] * sigma_p * (s_p - alpha_p)[:, None]) - b_top_empty[None, :]
            
            s, s_p = d['S_bot'], d['S_top']
            mask, mask_p = (s > alpha).astype(float), (s_p > alpha_p).astype(float)
            x_bot = d['y_bot_empty'] + mask[:, None] * (beta + sigma * (s - alpha))[:, None] + (mask_p[:, None] * sigma_p * (s_p - alpha_p)[:, None]) - b_bot_empty[None, :]
            
            if hot_mask is not None:
                m_top = hot_mask[:512, empty_cols]
                m_bot = hot_mask[512:, empty_cols]
                total_loss += np.mean(x_top[~m_top]**2) + np.mean(x_bot[~m_bot]**2)
            else:
                total_loss += np.mean(x_top**2) + np.mean(x_bot**2)
                
        return total_loss / len(img_list)

    alpha_min = np.min(all_s_min_guess)
    alpha_max = np.max(all_s_max)
    bounds = [
        (0, 1e-4), (0, 20), (alpha_min, alpha_max),
        (0, 1e-4), (alpha_min, alpha_max)
    ]
    res = opt.differential_evolution(objective, bounds, maxiter=15, popsize=12)
    return res.x

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
    
    # Load hot pixels
    hot_mask = np.zeros((1024, 1024), dtype=bool)
    if os.path.exists('gemini_flatten/hot_pixels.npy'):
        hot_coords = np.load('gemini_flatten/hot_pixels.npy')
        hot_mask[hot_coords[:, 0], hot_coords[:, 1]] = True

    params = get_params_multi(imgs, empty_cols, (150, 363), (662, 875), hot_mask)
    print(f"Ramp Model optimized params: sigma={params[0]:.2e}, beta={params[1]:.4f}, alpha={params[2]:.1f}, sigma_p={params[3]:.2e}, alpha_p={params[4]:.1f}")
    
    with open('gemini_flatten/params_ramp.json', 'w') as f:
        json.dump(list(params), f)

    for i, img in enumerate(imgs):
        corrected_x_full = apply_correction_ramp(img, params)
        S_obs = np.sum(img, axis=1)
        half = 512
        alpha = params[2]
        ut = np.where(S_obs[:half] <= alpha)[0]
        ub = np.where(S_obs[half:] <= alpha)[0]
        if len(ut) == 0: ut = np.arange(half)
        if len(ub) == 0: ub = np.arange(half)
        bt = np.median(img[:half][ut], axis=0)
        bb = np.median(img[half:][ub], axis=0)
        
        final_x = np.zeros_like(img)
        final_x[:half] = corrected_x_full[:half] - bt
        final_x[half:] = corrected_x_full[half:] - bb
        
        # Loss on empty cols, masked
        m = hot_mask[:, empty_cols]
        curr_x_empty = final_x[:, empty_cols]
        loss_corr = np.mean(curr_x_empty[~m]**2)
        
        orig_x = np.zeros_like(img)
        orig_x[:half] = img[:half] - bt
        orig_x[half:] = img[half:] - bb
        loss_orig = np.mean(orig_x[:, empty_cols][~m]**2)
        
        print(f"Image {i} ({files[i]}): Original Loss={loss_orig:.4f}, Corrected Loss={loss_corr:.4f}, Improvement={(loss_orig-loss_corr)/loss_orig*100:.2f}%")
