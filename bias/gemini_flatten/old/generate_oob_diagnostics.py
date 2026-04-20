
import numpy as np
from common import load
import json
import matplotlib.pyplot as plt
import os

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

def save_diagnostic_plots(img_path, params, output_prefix):
    img = load(img_path)
    corr = apply_correction(img, params)
    
    S_obs = np.sum(img, axis=1)
    half = img.shape[0] // 2
    alpha = params[2]
    unsagged_top = np.where(S_obs[:half] <= alpha)[0]
    unsagged_bot = np.where(S_obs[half:] <= alpha)[0]
    if len(unsagged_top) == 0: unsagged_top = np.arange(half)
    if len(unsagged_bot) == 0: unsagged_bot = np.arange(half)
    
    bias_top = np.median(img[:half][unsagged_top], axis=0)
    bias_bot = np.median(img[half:][unsagged_bot], axis=0)
    
    orig_x = np.zeros_like(img)
    orig_x[:half] = img[:half] - bias_top
    orig_x[half:] = img[half:] - bias_bot
    
    corr_x = np.zeros_like(corr)
    corr_x[:half] = corr[:half] - bias_top
    corr_x[half:] = corr[half:] - bias_bot
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].plot(S_obs)
    axes[0, 0].axhline(params[2], color='r', linestyle='--', label='Alpha')
    axes[0, 0].set_title(f'Row Sums - {os.path.basename(img_path)}')
    
    im1 = axes[0, 1].imshow(orig_x, vmin=-10, vmax=50, cmap='gray')
    axes[0, 1].set_title('Original Bias-Subtracted')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow(corr_x, vmin=-10, vmax=50, cmap='gray')
    axes[1, 0].set_title('Corrected Bias-Subtracted')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Load hot pixels if they exist
    hot_mask_sub = None
    if os.path.exists('hot_pixels.npy'):
        hot_coords = np.load('hot_pixels.npy')
        hot_mask = np.zeros((1024, 1024), dtype=bool)
        hot_mask[hot_coords[:, 0], hot_coords[:, 1]] = True
        hot_mask_sub = hot_mask
        
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    
    if hot_mask_sub is not None:
        # Use masked arrays for medians to ignore hot pixels
        m_orig = np.ma.array(orig_x[:, empty_cols], mask=hot_mask_sub[:, empty_cols])
        m_corr = np.ma.array(corr_x[:, empty_cols], mask=hot_mask_sub[:, empty_cols])
        axes[1, 1].plot(np.ma.median(m_orig, axis=0), alpha=0.5, label='Original')
        axes[1, 1].plot(np.ma.median(m_corr, axis=0), alpha=0.5, label='Corrected')
    else:
        axes[1, 1].plot(np.median(orig_x[:, empty_cols], axis=0), alpha=0.5, label='Original')
        axes[1, 1].plot(np.median(corr_x[:, empty_cols], axis=0), alpha=0.5, label='Corrected')
    
    axes[1, 1].set_title('Column Medians (Empty Cols, Hot Masked)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'/www/gemini/{output_prefix}_diagnostic.png')
    plt.close()

if __name__ == "__main__":
    with open('params.json', 'r') as f:
        params = json.load(f)
    
    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260113/oob_nfi_l0.pkl',
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260305/oob_nfi_l0.pkl'
    ]
    prefixes = ['oob_20260111', 'oob_20260113', 'oob_20260115', 'oob_20260305']
    
    for f, p in zip(files, prefixes):
        save_diagnostic_plots(f, params, p)
    print("OOB diagnostic plots saved to /www/gemini/")
