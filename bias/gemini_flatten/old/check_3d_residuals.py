
import numpy as np
import os
import sys
import json

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load

def get_sag_val(S, knots, values):
    return np.interp(S, knots, values, left=0)

if __name__ == "__main__":
    img_path = 'images_20260115/oob_nfi_l0.pkl'
    img = np.asarray(load(img_path))
    S_full = np.sum(img, axis=1)
    half = 512
    
    with open('gemini_flatten/params_multiplicative.json', 'r') as f:
        params = json.load(f)
    pt = params['top']
    pb = params['bot']
    
    col_start, n_cols = 100, 10
    cols = np.arange(col_start, col_start + n_cols)
    
    s_top = S_full[:half]
    s_bot = S_full[half:]
    min_top = np.argsort(s_top)[:50]
    b_top = np.median(img[:half][min_top], axis=0)[cols]
    
    p_self = get_sag_val(s_top, pt['knots'], pt['values'])
    p_other = get_sag_val(s_bot, pb['knots'], pb['values'])
    mask = (s_top > pt['alpha']).astype(float)
    mask_p = (s_bot > pb['alpha']).astype(float)
    
    # Model prediction for y - b
    # Z_model = -b * (P + es*P') - beta*m
    
    total_data_minus_model = []
    
    for i, c in enumerate(cols):
        data_z = img[:half, c] - b_top[i]
        model_z = -b_top[i] * (p_self * mask + pt['echo_scale'] * p_other * mask_p) - pt['beta'] * mask
        
        # Only check sagged rows
        sagged = s_top > pt['alpha']
        if np.any(sagged):
            total_data_minus_model.extend(list(data_z[sagged] - model_z[sagged]))

    avg_diff = np.mean(total_data_minus_model)
    print(f"Mean Residual (Data - Model) in sagged rows: {avg_diff:.4f} counts")
    if avg_diff > 0:
        print("The model predicts MORE sag than observed (model is 'lower' on Z axis).")
    else:
        print("The model predicts LESS sag than observed.")
