import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

npz_path = '/home/evan/sync/research/carruthers/bias/opencode_cjrelate/cj.npz'
out_dir = '/www/opencode_cjrelate'

d = np.load(npz_path, allow_pickle=True)
data = d['arr_0'].item()
c_all = data['top']
dates = ['0316', '0317', '0318', '0319']
N = 1024
col = np.arange(N)

# Compute 0318 scale factors from overlap
c16 = c_all[0]
c17 = c_all[1]
c18 = c_all[2]
c19 = c_all[3]
overlap_mask = np.all(np.stack([np.isfinite(c16), np.isfinite(c17), np.isfinite(c18), np.isfinite(c19)]), axis=0)

scale_16 = np.median(c16[overlap_mask] / c18[overlap_mask])
scale_17 = np.median(c17[overlap_mask] / c18[overlap_mask])
scale_19 = np.median(c19[overlap_mask] / c18[overlap_mask])

# Best correction found: linear multiplicative from v2 + scaling 
def correct_linear_mul(c, scale, a):
    # c' = c * scale / (1 + a * d)
    d = np.abs(col - 512)
    return c * scale / (1.0 + a * d)

# Hand-optimized a to maximize R^2 with 0319
best_a = 0.002
cc16 = correct_linear_mul(c16, 1.0, 0.0)
cc17 = correct_linear_mul(c17, 1.0, 0.0)
cc18 = correct_linear_mul(c18, scale_19, best_a)
cc19 = correct_linear_mul(c19, 1.0, 0.0)

corr_all = {'0316': cc16, '0317': cc17, '0318': cc18, '0319': cc19}

# Plot: shared axis comparison of raw vs corrected
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

raw = {d: c_all[i] for i, d in enumerate(dates)}

for ax_idx, (c_dict, title) in enumerate([(raw, 'Raw'), (corr_all, f'Best correction: scale={scale_19:.3f}, a={best_a:.4f}')]):
    ax = axes[ax_idx]
    for i in range(4):
        for j in range(i+1, 4):
            ci, cj = c_dict[dates[i]], c_dict[dates[j]]
            mask = np.isfinite(ci) & np.isfinite(cj)
            x, y = cj[mask], ci[mask]
            if len(x) > 0:
                ax.scatter(x, y, s=5, alpha=0.3, label=f'{dates[i]} vs {dates[j]}')
    
    # Add 1:1 line
    all_vals = np.concatenate([v[np.isfinite(v)] for v in c_dict.values()])
    lo, hi = all_vals.min(), all_vals.max()
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_aspect('equal')
    ax.set_xlabel('c_j (date A)', fontsize=10)
    ax.set_ylabel('c_j (date B)', fontsize=10)
    ax.set_title(title)
    if ax_idx == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f'{out_dir}/40_raw_vs_corrected_overlay.png', dpi=150)
plt.close()

print('Done')
