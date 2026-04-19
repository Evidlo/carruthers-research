
import numpy as np
from common import load
import json
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter

def apply_correction(img, params):
    sigma, beta, alpha, sigma_p, alpha_p = params
    half = img.shape[0] // 2
    S_obs = np.sum(img, axis=1)
    res = np.zeros_like(img)
    
    # Top
    s, s_p = S_obs[:half], S_obs[half:]
    mask, mask_p = (s > alpha).astype(float), (s_p > alpha_p).astype(float)
    denom = np.maximum(1 - sigma * s[:, None] * mask[:, None] - sigma_p * s_p[:, None] * mask_p[:, None], 0.1)
    res[:half] = (img[:half] + beta * mask[:, None]) / denom
    
    # Bot
    s, s_p = S_obs[half:], S_obs[:half]
    mask, mask_p = (s > alpha).astype(float), (s_p > alpha_p).astype(float)
    denom = np.maximum(1 - sigma * s[:, None] * mask[:, None] - sigma_p * s_p[:, None] * mask_p[:, None], 0.1)
    res[half:] = (img[half:] + beta * mask[:, None]) / denom
    return res

def get_clean_x(img, params):
    # For dark images, sag is likely zero, but we apply for consistency
    corr = apply_correction(img, params)
    S_obs = np.sum(img, axis=1)
    half = img.shape[0] // 2
    alpha = params[2]
    
    # We use a robust bias for darks too
    unsag_top = np.where(S_obs[:half] <= alpha)[0]
    unsag_bot = np.where(S_obs[half:] <= alpha)[0]
    if len(unsag_top) == 0: unsag_top = np.arange(half)
    if len(unsag_bot) == 0: unsag_bot = np.arange(half)
    
    b_top = np.median(img[:half][unsag_top], axis=0)
    b_bot = np.median(img[half:][unsag_bot], axis=0)
    
    x = np.zeros_like(corr)
    x[:half] = corr[:half] - b_top
    x[half:] = corr[half:] - b_bot
    return x

def save_hot_plot(x, hot_mask, f, prefix):
    stem = os.path.basename(f).replace('.pkl', '')
    date_folder = os.path.dirname(f).replace('images_', '')
    
    v_min, v_max = -5, 50
    disp = np.clip(x, v_min, v_max)
    disp_norm = (disp - v_min) / (v_max - v_min)
    
    # Dilate for visibility
    from scipy.ndimage import binary_dilation
    visible_mask = binary_dilation(hot_mask, structure=np.ones((3,3)))
    
    rgb = np.stack([disp_norm]*3, axis=-1)
    rgb[visible_mask, 0] = 1.0
    rgb[visible_mask, 1] = 0.0
    rgb[visible_mask, 2] = 0.0
    
    plt.figure(figsize=(12, 12))
    plt.imshow(rgb, interpolation='nearest')
    plt.title(f"Hot Pixels (Red) - {prefix} {date_folder}/{stem}")
    plt.axis('off')
    plt.savefig(f'/www/gemini/{prefix}_{date_folder}_{stem}_hot.png', bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    dark_files = [
        'images_20251117/dark_nfi_l0.pkl',
        'images_20260111/dark_nfi_l0.pkl',
        'images_20260113/dark_nfi_l0.pkl',
        'images_20260115/dark_nfi_l0.pkl',
        'images_20260117/dark_nfi_l0.pkl',
        'images_20260119/dark_nfi_l0.pkl'
    ]
    
    oob_files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260113/oob_nfi_l0.pkl',
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl', 
        'images_20260119/oob_nfi_l0.pkl'
    ]
    
    with open('params.json', 'r') as f:
        params = json.load(f)
        
    print("Processing dark images for hot pixel identification...")
    all_dark_x = []
    for f in dark_files:
        img = np.asarray(load(f))
        all_dark_x.append(get_clean_x(img, params))
        
    # Stack darks and take median
    stack = np.stack(all_dark_x, axis=0)
    median_dark = np.median(stack, axis=0)
    
    # Identifying hot pixels from darks is much cleaner
    # No Earth signal, so simple threshold works well
    # Threshold 25 counts over bias floor (was 50.0)
    hot_threshold = 25.0 
    hot_mask = median_dark > hot_threshold
    
    # Save the coordinates
    hot_coords = np.argwhere(hot_mask)
    np.save('hot_pixels.npy', hot_coords)
    print(f"Found {len(hot_coords)} stable hot pixels from dark frames ({len(hot_coords)/1024**2*100:.4f}% of detector).")
    
    # Produce visualization on OOB images
    if not os.path.exists('/www/gemini/'):
        os.makedirs('/www/gemini/')
        
    for f in oob_files:
        print(f"Generating hot pixel map for {f}...")
        img = np.asarray(load(f))
        x = get_clean_x(img, params)
        save_hot_plot(x, hot_mask, f, "oob")

    # Produce visualization on 3 dark images
    for f in dark_files[:3]:
        print(f"Generating hot pixel map for dark {f}...")
        img = np.asarray(load(f))
        x = get_clean_x(img, params)
        save_hot_plot(x, hot_mask, f, "dark")
    
    print("Hot pixel identification and all plotting complete.")

def save_hot_plot(x, hot_mask, f, prefix):
    stem = os.path.basename(f).replace('.pkl', '')
    date_folder = os.path.dirname(f).replace('images_', '')
    
    v_min, v_max = -5, 50
    disp = np.clip(x, v_min, v_max)
    disp_norm = (disp - v_min) / (v_max - v_min)
    
    # Dilate for visibility
    from scipy.ndimage import binary_dilation
    visible_mask = binary_dilation(hot_mask, structure=np.ones((3,3)))
    
    rgb = np.stack([disp_norm]*3, axis=-1)
    rgb[visible_mask, 0] = 1.0
    rgb[visible_mask, 1] = 0.0
    rgb[visible_mask, 2] = 0.0
    
    plt.figure(figsize=(12, 12))
    plt.imshow(rgb, interpolation='nearest')
    plt.title(f"Hot Pixels (Red) - {prefix} {date_folder}/{stem}")
    plt.axis('off')
    plt.savefig(f'/www/gemini/{prefix}_{date_folder}_{stem}_hot.png', bbox_inches='tight', pad_inches=0)
    plt.close()
