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

print('Finite value ranges by date:')
for di, dname in enumerate(dates):
    c = c_all[di]
    finite = np.where(np.isfinite(c))[0]
    print(f'{dname}: cols {finite.min()}-{finite.max()}, count={len(finite)}')

print('\nOverlap across dates:')
for i in range(4):
    for j in range(i+1, 4):
        fi = np.isfinite(c_all[i])
        fj = np.isfinite(c_all[j])
        overlap = fi & fj
        cols = np.where(overlap)[0]
        print(f'{dates[i]} ∩ {dates[j]}: {len(cols)} cols, range {cols.min()}-{cols.max() if len(cols)>0 else "N/A"}')

print('\nAll 4 overlap:', np.sum(np.all(np.isfinite(np.stack(c_all)), axis=0)))

# Plot NaN masks
fig, axes = plt.subplots(4, 1, figsize=(14, 6), sharex=True)
for di, dname in enumerate(dates):
    ax = axes[di]
    mask = np.isfinite(c_all[di]).astype(float)
    ax.fill_between(col, 0, mask, alpha=0.5)
    ax.set_ylabel(dname, fontsize=10)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 1])

axes[-1].set_xlabel('Column index', fontsize=10)
plt.suptitle('Finite c_j by date (1=finite, 0=NaN)')
plt.tight_layout()
plt.savefig(f'{out_dir}/30_finite_masks.png', dpi=150)
plt.close()

# Now let's examine: what if 0318 also has Earth at center (512) instead of edge?
print('\n--- Testing if 0318 Earth center is actually at 512 ---')
dist_center = np.abs(col - 512)

distances_0318_center = dist_center
distances_0318_edge = np.abs(col - 1024)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot c_j vs distance for both assumptions
for idx, (dist_assumption, title) in enumerate([
    (distances_0318_edge, 'Assumption: Earth at right edge (d=|col-1024|)'),
    (distances_0318_center, 'Alternative: Earth at center (d=|col-512|)')
]):
    ax = axes[idx]
    for di, dname in enumerate(dates):
        c = c_all[di]
        if dname == '0318':
            dist = dist_assumption
        else:
            dist = dist_center
        mask = np.isfinite(c)
        ax.scatter(dist[mask], c[mask], s=5, alpha=0.4, label=dname)
    ax.set_xlabel('Distance from Earth center', fontsize=10)
    ax.set_ylabel('c_j', fontsize=10)
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig(f'{out_dir}/31_earth_position_hypothesis.png', dpi=150)
plt.close()

# Check pairwise correlations under both assumptions
def test_correlation(dist_fn_0318):
    r2s = {}
    for i in range(4):
        for j in range(i+1, 4):
            ci, cj = c_all[i], c_all[j]
            # Get distance for each under current assumption
            if dates[i] == '0318':
                di = dist_fn_0318
            else:
                di = dist_center
            if dates[j] == '0318':
                dj = dist_fn_0318
            else:
                dj = dist_center

            mask = np.isfinite(ci) & np.isfinite(cj)
            x, y = cj[mask], ci[mask]
            if len(x) < 3:
                continue
            ss_res = ((x - y)**2).sum()
            ss_tot = ((y - y.mean())**2).sum()
            r2s[f'{dates[i]}_{dates[j]}'] = 1 - ss_res / ss_tot
    return r2s

r2_edge = test_correlation(distances_0318_edge)
r2_center = test_correlation(distances_0318_center)

print('\nR^2 with 0318 Earth at edge (current):', r2_edge)
print('Avg:', sum(r2_edge.values())/len(r2_edge))
print('\nR^2 with 0318 Earth at center (alternative):', r2_center)
print('Avg:', sum(r2_center.values())/len(r2_center))

# Let's bin by distance and look at median profiles
print('\n--- Median profile by distance bin ---')
fig, ax = plt.subplots(figsize=(12, 6))
for di, dname in enumerate(dates):
    c = c_all[di]
    if dname == '0318':
        dist = distances_0318_edge
    else:
        dist = dist_center
    mask = np.isfinite(c)

    # Use fixed bins
    bins = np.arange(0, 1100, 50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    medians = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        in_bin = mask & (dist >= b0) & (dist < b1)
        if in_bin.sum() > 0:
            medians.append(np.median(c[in_bin]))
        else:
            medians.append(np.nan)

    ax.plot(bin_centers, medians, 'o-', label=dname, alpha=0.7)

ax.set_xlabel('Distance from Earth center (edge assumption for 0318)', fontsize=10)
ax.set_ylabel('Median c_j', fontsize=10)
ax.legend()
ax.set_title('Median c_j vs distance (50-pixel bins)')
plt.tight_layout()
plt.savefig(f'{out_dir}/32_median_profiles.png', dpi=150)
plt.close()


# --- Strategy: Can we find h(c, d) that makes 0316/0317/0319 and 0318 agree? ---
print('\n--- Testing if c_j differences correlate with distance signal ---')

# For overlapping columns between 0316 and 0318:
mask_16_18 = np.isfinite(c_all[0]) & np.isfinite(c_all[2])
dist_16 = dist_center[mask_16_18]
dist_18 = distances_0318_edge[mask_16_18]
c_16 = c_all[0][mask_16_18]
c_18 = c_all[2][mask_16_18]

print(f'0316-0318 overlap: {mask_16_18.sum()} columns')
print(f'Distance range 0316: {dist_16.min():.0f}-{dist_16.max():.0f}')
print(f'Distance range 0318: {dist_18.min():.0f}-{dist_18.max():.0f}')

# The key insight: if c_true is shared, and c_obs = c_true + contamination(distance)
# Then for same column j: c_16_j - c_18_j = contamination(dist_16_j) - contamination(dist_18_j)
# If contamination is some function f(d), then delta c should relate to delta d

# Let's compute delta c vs delta d
delta_c = c_16 - c_18
delta_d = dist_16 - dist_18  # Should be negative since 0318 distances are larger

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(dist_16, delta_c, s=10, alpha=0.5)
axes[0].set_xlabel('0316 distance (center ref)')
axes[0].set_ylabel('c_0316 - c_0318')
axes[0].set_title('Delta c vs 0316 distance')

axes[1].scatter(dist_18, delta_c, s=10, alpha=0.5)
axes[1].set_xlabel('0318 distance (edge ref)')
axes[1].set_ylabel('c_0316 - c_0318')
axes[1].set_title('Delta c vs 0318 distance')

plt.tight_layout()
plt.savefig(f'{out_dir}/33_delta_c_vs_dist.png', dpi=150)
plt.close()

# Correlation
corr_16 = np.corrcoef(dist_16, delta_c)[0, 1]
corr_18 = np.corrcoef(dist_18, delta_c)[0, 1]
print(f'Correlation(delta_c, dist_16): {corr_16:.4f}')
print(f'Correlation(delta_c, dist_18): {corr_18:.4f}')

# Try simple models for contamination:
# f(d) = a * d + b -> c_obs = c_true + f(d)
# Then c_true = c_obs - f(d)
# For 0316 and 0318: c_16 - a*d_16 - b = c_18 - a*d_18 - b => c_16 - c_18 = a*(d_16 - d_18)
# So delta_c = a * delta_d

# Fit a: delta_c = a * delta_d
A = delta_d.reshape(-1, 1)
a_est, _, _, _ = np.linalg.lstsq(A, delta_c, rcond=None)
a_est = a_est[0]
print(f'Estimated contamination slope a = {a_est:.4f} (delta_c = a * delta_d)')

# Apply correction and check
# c' = c - a*d for all dates
corrected = {}
for di, dname in enumerate(dates):
    c = c_all[di]
    if dname == '0318':
        dist = distances_0318_edge
    else:
        dist = dist_center
    corrected[dname] = c - a_est * dist

# Compute R^2
def print_r2(c_dict, label):
    r2s = {}
    for i in range(4):
        for j in range(i+1, 4):
            ci, cj = c_dict[dates[i]], c_dict[dates[j]]
            mask = np.isfinite(ci) & np.isfinite(cj)
            if mask.sum() < 3:
                continue
            x, y = cj[mask], ci[mask]
            ss_res = ((x - y)**2).sum()
            ss_tot = ((y - y.mean())**2).sum()
            r2s[f'{dates[i]}_{dates[j]}'] = 1 - ss_res / ss_tot
    if len(r2s) == 0:
        return -999
    avg = sum(r2s.values()) / len(r2s)
    print(f'--- {label} ---')
    for k, v in r2s.items():
        print(f'  {k}: R^2 = {v:.4f}')
    print(f'  avg R^2 = {avg:.4f}')
    return avg

print()
r2_corr = print_r2(corrected, f'Simple linear contamination (a={a_est:.4f})')

# Try quadratic contamination
# delta_c = a * delta_d + b * delta_d^2
A = np.vstack([delta_d, delta_d**2]).T
ab, _, _, _ = np.linalg.lstsq(A, delta_c, rcond=None)
a2, b2 = ab
print(f'Quadratic contamination: a={a2:.4f}, b={b2:.6f}')

corrected_q = {}
for di, dname in enumerate(dates):
    c = c_all[di]
    if dname == '0318':
        dist = distances_0318_edge
    else:
        dist = dist_center
    corrected_q[dname] = c - a2 * dist - b2 * dist**2

print()
r2_q = print_r2(corrected_q, f'Quadratic contamination (a={a2:.4f}, b={b2:.6f})')

# Now the big question: what if Earth is at center for ALL dates?
# Then distance is |col - 512| for all, and the only difference is contamination amplitude
print('\n--- Re-analyzing with Earth at center for ALL dates ---')
corrected_all_center = {}
for di, dname in enumerate(dates):
    c = c_all[di]
    # Fit contamination as function of distance for each date relative to 0319
    # (Use 0319 as reference)
    dist = dist_center
    if dname == '0319':
        corrected_all_center[dname] = c.copy()
    else:
        mask = np.isfinite(c) & np.isfinite(c_all[3])
        c_ref = c_all[3][mask]
        c_self = c[mask]
        d_mask = dist[mask]
        # Fit: c_self - c_ref = a * d + b
        delta = c_self - c_ref
        A = np.vstack([d_mask, np.ones(len(d_mask))]).T
        ab, _, _, _ = np.linalg.lstsq(A, delta, rcond=None)
        a_fit, b_fit = ab
        print(f'{dname}: contamination slope a={a_fit:.4f}, b={b_fit:.2f}')
        corrected_all_center[dname] = c - a_fit * dist - b_fit

print()
r2_all_center = print_r2(corrected_all_center, 'All dates at center assumption')

# Plot the corrected values for center assumption
fig, axes = plt.subplots(4, 4, figsize=(12, 12), squeeze=False)
for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        ci, cj = corrected_all_center[dates[i]], corrected_all_center[dates[j]]
        if i == j:
            vals = ci[np.isfinite(ci)]
            ax.hist(vals, bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
            ax.set_title(dates[i], fontsize=11)
            ax.tick_params(axis='both', labelsize=7)
        else:
            mask = np.isfinite(ci) & np.isfinite(cj)
            x, y = cj[mask], ci[mask]
            if len(x) > 0:
                ax.scatter(x, y, s=5, alpha=0.4, c='royalblue', edgecolors='none')
                lo = min(x.min(), y.min())
                hi = max(x.max(), y.max())
                pad = (hi - lo) * 0.05 + 1e-3
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_aspect('equal')
            ax.set_xlabel(dates[j], fontsize=9)
            ax.set_ylabel(dates[i], fontsize=9)
            ax.tick_params(axis='both', labelsize=7)
plt.suptitle("Corrected c' assuming Earth center for all dates", fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(f'{out_dir}/34_all_center_correction.png', dpi=150)
plt.close()

print('\nSummary of all approaches:')
print(f'  Raw edge assumption avg R^2: {sum(r2_edge.values())/len(r2_edge):.4f}')
print(f'  Raw center assumption avg R^2: {sum(r2_center.values())/len(r2_center):.4f}')
print(f'  Simple linear contamination R^2: {r2_corr:.4f}')
print(f'  Quadratic contamination R^2: {r2_q:.4f}')
print(f'  All center per-date fit R^2: {r2_all_center:.4f}')
