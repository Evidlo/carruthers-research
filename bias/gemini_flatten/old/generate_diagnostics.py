
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
    
    half = img.shape[0] // 2
    bias_top = np.median(img[150:363], axis=0)
    bias_bot = np.median(img[662:875], axis=0)
    
    orig_x = np.zeros_like(img)
    orig_x[:half] = img[:half] - bias_top
    orig_x[half:] = img[half:] - bias_bot
    
    corr_x = np.zeros_like(corr)
    corr_x[:half] = corr[:half] - bias_top
    corr_x[half:] = corr[half:] - bias_bot
    
    S = np.sum(img, axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Row Sums
    axes[0, 0].plot(S)
    axes[0, 0].axhline(params[2], color='r', linestyle='--', label='Alpha')
    axes[0, 0].axhline(params[4], color='g', linestyle='--', label='Alpha Prime')
    axes[0, 0].set_title(f'Row Sums - {os.path.basename(img_path)}')
    axes[0, 0].legend()
    
    # Original Image (subset)
    im1 = axes[0, 1].imshow(orig_x, vmin=-10, vmax=50, cmap='gray')
    axes[0, 1].set_title('Original Bias-Subtracted')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Corrected Image (subset)
    im2 = axes[1, 0].imshow(corr_x, vmin=-10, vmax=50, cmap='gray')
    axes[1, 0].set_title('Corrected Bias-Subtracted')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Column medians (to see the flatness)
    axes[1, 1].plot(np.median(orig_x, axis=0), alpha=0.5, label='Original')
    axes[1, 1].plot(np.median(corr_x, axis=0), alpha=0.5, label='Corrected')
    axes[1, 1].set_title('Column Medians')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'/www/gemini/{output_prefix}_diagnostic.png')
    plt.close()

if __name__ == "__main__":
    if not os.path.exists('/www/gemini/'):
        os.makedirs('/www/gemini/')
        
    with open('params.json', 'r') as f:
        params = json.load(f)
        
    save_diagnostic_plots('images_20260111/oob_nfi_l0.pkl', params, 'oob')
    save_diagnostic_plots('images_20260111/science_nfi_l0.pkl', params, 'science')
    print("Diagnostic plots saved to /www/gemini/")
