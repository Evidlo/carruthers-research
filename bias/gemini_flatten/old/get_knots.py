
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load

if __name__ == "__main__":
    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260113/oob_nfi_l0.pkl',
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl', 
        'images_20260119/oob_nfi_l0.pkl'
    ]
    empty_cols = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
    
    all_S_top, all_S_bot = [], []
    all_res_top, all_res_bot = [], []
    
    for f in files:
        img = np.asarray(load(f))
        S = np.sum(img, axis=1)
        half = 512
        min_top_idx = np.argsort(S[:half])[:50]
        bt = np.median(img[:half][min_top_idx], axis=0)
        min_bot_idx = np.argsort(S[half:])[:50]
        bb = np.median(img[half:][min_bot_idx], axis=0)
        
        # Consistent with the "Selection Aid" script (rows 150-873)
        rt = np.median((img[:half] - bt[None, :])[150:, empty_cols], axis=1)
        rb = np.median((img[half:] - bb[None, :])[:362, empty_cols], axis=1)
        
        all_res_top.append(rt); all_res_bot.append(rb)
        all_S_top.append(S[150:512]); all_S_bot.append(S[512:874])

    results = {}
    for side, S_pool, res_pool in [('top', np.concatenate(all_S_top), np.concatenate(all_res_top)),
                                   ('bot', np.concatenate(all_S_bot), np.concatenate(all_res_bot))]:
        s_min = S_pool.min()
        bins = np.linspace(s_min, s_min + 0.3e6, 150)
        bin_idx = np.digitize(S_pool, bins)
        bs = [np.mean(S_pool[bin_idx == b]) for b in range(1, len(bins)) if np.any(bin_idx == b)]
        br = [np.median(res_pool[bin_idx == b]) for b in range(1, len(bins)) if np.any(bin_idx == b)]
        
        idx = [0, 2, 7] if side == 'top' else [0, 3, 8]
        results[side] = [(bs[i], br[i]) for i in idx]
        print(f"{side.capitalize()} Knots:")
        for i, pt in zip(idx, results[side]):
            print(f"  Point {i+1}: S={pt[0]:.0f}, res={pt[1]:.2f}")

    import json
    with open('gemini_flatten/knot_coords.json', 'w') as f:
        json.dump(results, f)
