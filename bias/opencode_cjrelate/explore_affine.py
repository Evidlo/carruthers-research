import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

npz_path = '/home/evan/sync/research/carruthers/bias/opencode_cjrelate/cj.npz'
out_dir = '/www/opencode_cjrelate'

d = np.load(npz_path, allow_pickle=True)
c_all = d['arr_0'].item()['top']
dates = ['0316', '0317', '0318', '0319']

# Robust pairwise slope/intercept via Theil-Sen estimator
def theil_sen(x, y):
    slopes = []
    for ii in range(len(x)):
        for jj in range(ii+1, len(x)):
            dx = x[jj] - x[ii]
            if abs(dx) > 1e-6:
                slopes.append((y[jj] - y[ii]) / dx)
    med_slope = np.median(slopes)
    intercepts = y - med_slope * x
    med_int = np.median(intercepts)
    return med_slope, med_int

def make_plot(c_dict, title, fname, xlim=None):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), squeeze=False)
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ci, cj = c_dict[dates[i]], c_dict[dates[j]]
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
                    if xlim is None:
                        lo = min(x.min(), y.min())
                        hi = max(x.max(), y.max())
                        pad = (hi - lo) * 0.05 + 1e-3
                        ax.set_xlim(lo - pad, hi + pad)
                        ax.set_ylim(lo - pad, hi + pad)
                    else:
                        ax.set_xlim(*xlim)
                        ax.set_ylim(*xlim)
                    ax.set_aspect('equal')
                ax.set_xlabel(dates[j], fontsize=9)
                ax.set_ylabel(dates[i], fontsize=9)
                ax.tick_params(axis='both', labelsize=7)
    plt.suptitle(title, fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{fname}', dpi=150)
    plt.close()

def r2_dict(c_dict):
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
    return r2s

def report(c_dict, label):
    r2s = r2_dict(c_dict)
    avg = sum(r2s.values()) / len(r2s)
    print(f'--- {label} ---')
    for k, v in r2s.items():
        print(f'  {k}: R^2 = {v:.4f}')
    print(f'  avg R^2 = {avg:.4f}')
    return avg, r2s

# Raw
raw = {d: c_all[i] for i, d in enumerate(dates)}
make_plot(raw, 'Raw c_j', '01_raw.png')
r2_raw, _ = report(raw, 'raw')

# ============================================================================
# Model X: Simple per-date affine transform c = a_d * c' + b_d
# Fit to pairwise robust medians, anchored so 0319 c' is the reference
# ============================================================================

# Build robust pairwise slope matrix
slopes = np.zeros((4, 4))
intercepts = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        if i == j:
            slopes[i, j] = 1.0
            intercepts[i, j] = 0.0
            continue
        ci, cj = c_all[i], c_all[j]
        mask = np.isfinite(ci) & np.isfinite(cj)
        x, y = cj[mask], ci[mask]
        if len(x) < 3:
            slopes[i, j] = np.nan
            intercepts[i, j] = np.nan
        else:
            s, b = theil_sen(x, y)
            slopes[i, j] = s
            intercepts[i, j] = b

print('\nRobust pairwise slopes (row = y, col = x):')
for i in range(4):
    print(f'  {dates[i]}: {[f"{slopes[i,j]:.3f}" for j in range(4)]}')

print('\nRobust pairwise intercepts:')
for i in range(4):
    print(f'  {dates[i]}: {[f"{intercepts[i,j]:.1f}" for j in range(4)]}')

# Use 0319 (index 3) as reference. For each date d:
# c_obs_d = a_d * c' + b_d
# We can estimate a_d, b_d from the pairwise regression to 0319

# For each date, fit robust affine from 0319 using overlapping data
a_params = []
b_params = []
ref_idx = 3
for di in range(4):
    if di == ref_idx:
        a_params.append(1.0)
        b_params.append(0.0)
        continue
    ci, cj = c_all[di], c_all[ref_idx]
    mask = np.isfinite(ci) & np.isfinite(cj)
    # x = c_ref = 0319, y = c_date
    x, y = cj[mask], ci[mask]
    s, b = theil_sen(x, y)
    a_params.append(s)
    b_params.append(b)

a_params = np.array(a_params)
b_params = np.array(b_params)

print(f'\nAffine params vs 0319 (reference):')
for di, dname in enumerate(dates):
    print(f'  {dname}: a={a_params[di]:.4f}, b={b_params[di]:.1f}')

# Derive c' for each date: c' = (c_obs - b_d) / a_d
corrected_affine = {}
for di, dname in enumerate(dates):
    corrected_affine[dname] = (c_all[di] - b_params[di]) / a_params[di]

make_plot(corrected_affine, "Model X: affine-corrected c' (ref=0319)", '50_affine_corrected.png')
r2_affine, _ = report(corrected_affine, 'affine')

# ============================================================================
# Model Y: joint robust least-squares fit of shared c'
# ============================================================================
N = 1024
from scipy.optimize import minimize

def fit_shared_affine(all_c, dates):
    """Fit shared c'[col] and per-date (a_d, b_d) minimizing robust residuals"""
    all_finite = np.array([np.isfinite(c) for c in all_c])

    def objective(params):
        c_prime = params[:N]
        a = params[N:N+4]
        b = params[N+4:N+8]
        res = []
        for di in range(4):
            pred = a[di] * c_prime + b[di]
            mask = all_finite[di]
            r = pred[mask] - all_c[di][mask]
            # Robust L1-ish via sum of absolute
            res.append(np.abs(r))
        reg = 0.1 * np.sum(np.diff(c_prime, 2)**2)
        return np.sum(np.concatenate(res)) + reg

    p0 = np.zeros(N + 8)
    for j in range(N):
        vals = [all_c[di][j] for di in range(4) if all_finite[di, j]]
        p0[j] = np.median(vals) if vals else 0.0
    p0[N:N+4] = 1.0
    p0[N+4:N+8] = 0.0

    bounds = [(-10000, 10000)] * N + [(0.1, 10.0)]*4 + [(-5000, 5000)]*4
    result = minimize(objective, p0, method='L-BFGS-B', bounds=bounds)
    print(f'\n[Joint affine] success={result.success}, cost={result.fun:.2e}')

    c_prime = result.x[:N]
    a_fit = result.x[N:N+4]
    b_fit = result.x[N+4:N+8]

    # Derive c'
    c_dict = {}
    for di, dname in enumerate(dates):
        c_dict[dname] = (all_c[di] - b_fit[di]) / a_fit[di]
    return c_dict, a_fit, b_fit

c_dict_joint, a_joint, b_joint = fit_shared_affine(c_all, dates)
print(f'Joint fit params:')
for di, dname in enumerate(dates):
    print(f'  {dname}: a={a_joint[di]:.4f}, b={b_joint[di]:.1f}')

make_plot(c_dict_joint, "Model Y: jointly-fit affine c'", '51_joint_affine.png')
r2_joint, _ = report(c_dict_joint, 'joint_affine')

# ============================================================================
# Summary
# ============================================================================
print('\n=== Final Summary ===')
print(f'  Raw avg R^2:          {r2_raw:.4f}')
print(f'  Affine (ref=0319):    {r2_affine:.4f}')
print(f'  Joint affine:         {r2_joint:.4f}')
