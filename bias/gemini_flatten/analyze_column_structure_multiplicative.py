
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import json
import matplotlib.pyplot as plt

def apply_piecewise_linear(S, p_dict):
    knots = p_dict['knots']
    values = p_dict['values']
    sigma_tail = p_dict['sigma_tail']
    
    k1, k2, k3 = knots
    v1, v2, v3 = values
    res = np.zeros_like(S)
    
    m1 = S <= k1
    res[m1] = v1
    m2 = (S > k1) & (S <= k2)
    res[m2] = np.interp(S[m2], [k1, k2], [v1, v2])
    m3 = (S > k2) & (S <= k3)
    res[m3] = np.interp(S[m3], [k2, k3], [v2, v3])
    m4 = S > k3
    res[m4] = v3 + sigma_tail * (S[m4] - k3)
    return res

def apply_multiplicative_correction(img, p):
    half = 512
    S_obs = np.sum(img, axis=1)
    st, sb = S_obs[:half], S_obs[half:]
    
    # Unitless P = model(counts) / 2500
    ps_t = apply_piecewise_linear(st, p['top']) / 2500.0
    ps_b = apply_piecewise_linear(sb, p['bot']) / 2500.0
    
    mask_t = (st > p['top']['knots'][0]).astype(float)
    mask_b = (sb > p['bot']['knots'][0]).astype(float)
    
    dt = np.maximum(1 - ps_t[:, None] * mask_t[:, None] - p['top']['echo_scale'] * ps_b[:, None] * mask_b[:, None], 0.1)
    db = np.maximum(1 - ps_b[:, None] * mask_b[:, None] - p['bot']['echo_scale'] * ps_t[:, None] * mask_t[:, None], 0.1)
    
    zt = (img[:half] + p['top']['beta'] * mask_t[:, None]) / dt
    zb = (img[half:] + p['bot']['beta'] * mask_b[:, None]) / db
    
    # Robust bias
    min_t, min_b = (st <= p['top']['knots'][0]), (sb <= p['bot']['knots'][0])
    bt = np.median(zt[min_t], axis=0) if np.any(min_t) else np.median(zt[150:363], axis=0)
    bb = np.median(zb[min_b], axis=0) if np.any(min_b) else np.median(zb[150:363], axis=0)
    
    return np.concatenate([zt - bt[None, :], zb - bb[None, :]], axis=0)

if __name__ == "__main__":
    with open('gemini_flatten/params_multiplicative.json', 'r') as f:
        p = json.load(f)
    with open('gemini_flatten/noise_targets.json', 'r') as f:
        targets = json.load(f)
    
    hot_mask = np.zeros((1024, 1024), dtype=bool)
    if os.path.exists('gemini_flatten/hot_pixels.npy'):
        hc = np.load('gemini_flatten/hot_pixels.npy')
        hot_mask[hc[:, 0], hc[:, 1]] = True
    
    files = ['images_20260111/oob_nfi_l0.pkl', 'images_20260115/oob_nfi_l0.pkl']
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    
    tr, tc = targets['target_sigma_row'], targets['target_sigma_col']

    for f in files:
        img = np.asarray(load(f))
        x = apply_multiplicative_correction(img, p)
        m_x = np.ma.array(x, mask=hot_mask)
        
        S = np.sum(img, axis=1)
        is_unsag = (S[:512] <= p['top']['knots'][0]) & (S[512:] <= p['bot']['knots'][0])
        is_unsag_full = np.concatenate([is_unsag, is_unsag])
        is_pure_sag = np.zeros(1024, dtype=bool)
        is_pure_sag[512-75:512] = True; is_pure_sag[512:512+75] = True
        
        def get_stats(mask_rows):
            if not np.any(mask_rows): return 0, 0
            sub = m_x[mask_rows][:, empty_cols]
            return np.std(np.ma.median(sub, axis=1)), np.std(np.ma.median(sub, axis=0))

        sr_u, sc_u = get_stats(is_unsag_full)
        sr_s, sc_s = get_stats(is_pure_sag)
        jump = np.abs(np.ma.median(m_x[512-20:512, empty_cols]) - np.ma.median(m_x[512:512+20, empty_cols]))

        print(f"\nImage: {f}")
        print(f"  UNSAG    | Row: {sr_u:.4f} (Target: {tr:.4f}) | Col: {sc_u:.4f} (Target: {tc:.4f})")
        print(f"  PURE SAG | Row: {sr_s:.4f} (Target: {tr:.4f}) | Col: {sc_s:.4f} (Target: {tc:.4f})")
        print(f"  Jump:    {jump:.4f} (Target: < 0.5)")
        
        # SAVE IMAGE
        stem = os.path.basename(f).replace('.pkl', '')
        date = os.path.dirname(f).replace('images_', '')
        plt.figure(figsize=(10, 10))
        plt.imshow(np.ma.array(x, mask=hot_mask), vmin=-5, vmax=50, cmap='gray')
        plt.title(f'Flattened OOB (Analytical PWL) - {date}/{stem}')
        plt.axis('off')
        plt.savefig(f'/www/gemini/oob_{date}_{stem}_flattened_final.png', bbox_inches='tight', pad_inches=0)
        plt.close()
