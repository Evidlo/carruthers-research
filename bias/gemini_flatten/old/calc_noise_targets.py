
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load

if __name__ == "__main__":
    # Use 20260111 as the reference for "clean" stats
    img = np.asarray(load('images_20260111/oob_nfi_l0.pkl'))
    
    # Safe Patches (as per AGENT.md)
    # Rows: 150-361 and 662-873
    # Columns: 50-350 (avoiding edges and Earth)
    
    top_patch = img[150:362, 50:350]
    bot_patch = img[662:874, 50:350]
    
    # We must subtract bias per-column for each patch to see the natural variance
    top_patch_clean = top_patch - np.median(top_patch, axis=0)
    bot_patch_clean = bot_patch - np.median(bot_patch, axis=0)
    
    def get_noise_metrics(patch):
        # 1. Row-wise variation target (standard deviation of row medians)
        row_meds = np.median(patch, axis=1)
        sigma_row = np.std(row_meds)
        
        # 2. Column-wise variation target (standard deviation of column medians)
        # Note: we do this BEFORE bias subtraction to see the natural bj variation we should target
        col_meds = np.median(patch, axis=0)
        sigma_col = np.std(col_meds)
        
        # 3. Peak-to-peak row variation (to catch steps)
        ptp_row = np.max(row_meds) - np.min(row_meds)
        
        return sigma_row, sigma_col, ptp_row

    sr_t, _, ptp_t = get_noise_metrics(top_patch_clean)
    sr_b, _, ptp_b = get_noise_metrics(bot_patch_clean)

    # For column targets, use the residuals from a standard rob_bias across the whole image
    from common import rob_bias
    full_res = img - rob_bias(img)
    
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    # Use the same rows as the patches
    col_meds_t = np.median(full_res[150:362, empty_cols], axis=0)
    col_meds_b = np.median(full_res[662:874, empty_cols], axis=0)
    
    target_sc = (np.std(col_meds_t) + np.std(col_meds_b)) / 2

    targets = {
        "target_sigma_row": (sr_t + sr_b) / 2,
        "target_sigma_col": target_sc,
        "target_ptp_row": (ptp_t + ptp_b) / 2
    }
    
    import json
    with open('gemini_flatten/noise_targets.json', 'w') as f:
        json.dump(targets, f, indent=4)
        
    print("--- Noise Targets (Detector Floor) ---")
    for k, v in targets.items():
        print(f"{k}: {v:.4f}")
