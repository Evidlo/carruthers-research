
import numpy as np
import os
import sys
import json

# Add common and iplot to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath('../../../../research/iplot/'))

from common import load
import iplot as plt

def get_sag_val(S, knots, values):
    return np.interp(S, knots, values, left=values[0], right=values[-1])

if __name__ == "__main__":
    # 1. Load data
    img_path = 'images_20260115/oob_nfi_l0.pkl'
    img = np.asarray(load(img_path))
    S_full = np.sum(img, axis=1)
    half = 512
    
    # Parameters
    with open('gemini_flatten/params_multiplicative.json', 'r') as f:
        params = json.load(f)
    pt = params['top']
    pb = params['bot']
    
    # 2. Selection: 20 columns
    col_start = 100
    n_cols = 20
    cols = np.arange(col_start, col_start + n_cols)
    
    # 3. Reference bias (x is small)
    # Estimate from 200 rows as per AGENT.md
    b_top = np.median(img[150:362], axis=0)[cols]
    
    # 4. Prepare scatter data (Top half only)
    s_top = S_full[:half]
    s_bot = S_full[half:]
    
    X_scatter = [] # b_j
    Y_scatter = [] # s_i
    Z_scatter = [] # y - b
    
    for i, c in enumerate(cols):
        X_scatter.extend([b_top[i]] * half)
        Y_scatter.extend(list(s_top))
        Z_scatter.extend(list(img[:half, c] - b_top[i]))
        
    X_scatter = np.array(X_scatter)
    Y_scatter = np.array(Y_scatter)
    Z_scatter = np.array(Z_scatter)
    
    # 5. Prepare model surface
    b_grid = np.linspace(b_top.min() - 5, b_top.max() + 5, 10)
    B_mesh, S_mesh = np.meshgrid(b_grid, s_top)
    
    p_self = get_sag_val(s_top, pt['knots'], pt['values'])
    p_other = get_sag_val(s_bot, pb['knots'], pb['values'])
    
    # Calculate Z for the mesh: y - b ~ - (b*P + b*es*Po + beta)
    Z_mesh = np.zeros_like(B_mesh)
    for i in range(len(b_grid)):
        b_val = b_grid[i]
        Z_mesh[:, i] = -b_val * (p_self + pt['echo_scale'] * p_other) - pt['beta']

    plt.figure()
    
    # Plot scatter points
    plt.scatter3d(X_scatter, Y_scatter, Z_scatter, c=Y_scatter, cmap='Viridis', label='Data (y - b)')
    
    # Plot surface as a wireframe using scatter3d with lines
    for i in range(Z_mesh.shape[1]):
        plt.scatter3d(B_mesh[:, i], S_mesh[:, i], Z_mesh[:, i], mode='lines', line=dict(color='red', width=2), alpha=0.3)
    for j in range(0, Z_mesh.shape[0], 40): # sparse cross lines
        plt.scatter3d(B_mesh[j, :], S_mesh[j, :], Z_mesh[j, :], mode='lines', line=dict(color='red', width=2), alpha=0.3)

    plt.xlabel('True Charge (b_j) [counts]')
    plt.ylabel('Row Sum (S) [counts]')
    plt.zlabel('Sag Amount (y - b) [counts]')
    plt.title(f'Top Half Distortion (20 Cols) - {os.path.basename(img_path)}')
    
    output_path = '/www/gemini/distortion_3d_analytical_v2.html'
    plt.savefig(output_path)
    print(f"Updated 3D plot saved to {output_path}")
