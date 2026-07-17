import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

npz_path = '/home/evan/sync/research/carruthers/bias/opencode_cjrelate/cj.npz'
out_dir = '/www/opencode_cjrelate'

d = np.load(npz_path, allow_pickle=True)
data = d['arr_0'].item()

c_all = data['top']
dates = ['0316', '0317', '0318', '0319']
N = 1024

col = np.arange(N, dtype=float)
distances = {
    '0316': np.abs(col - 512),
    '0317': np.abs(col - 512),
    '0318': np.abs(col - 1024),
    '0319': np.abs(col - 512),
}

all_finite = np.array([np.isfinite(a) for a in c_all])


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


def print_r2(c_dict, label=''):
    r2s = {}
    for i in range(4):
        for j in range(i+1, 4):
            di, dj = c_dict[dates[i]], c_dict[dates[j]]
            mask = np.isfinite(di) & np.isfinite(dj)
            if mask.sum() < 3:
                continue
            x, y = dj[mask], di[mask]
            ss_res = ((x - y) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot
            r2s[f'{dates[i]}_{dates[j]}'] = r2
    avg = sum(r2s.values()) / len(r2s)
    print(f'--- {label} ---')
    for k, v in r2s.items():
        print(f'  {k}: R^2 = {v:.4f}')
    print(f'  avg R^2 = {avg:.4f}')
    return avg


# Raw
orig = {d: c_all[i] for i, d in enumerate(dates)}
orig_r2 = print_r2(orig, 'raw')


# ============================================================================
# Approach 1: Direct smoothness + variance minimization
# ============================================================================
# Rather than fitting forward model, directly minimize variance of c' across dates
# at each column, subject to smoothness constraints and the model form.
# For each column j, we have up to 4 observations c_j_d.
# We want c'_j shared across dates, and c_j_d = f_d(c'_j, d_j).
# We minimize sum_{d,j} (c_j_d - f_d(c'_j, d_j))^2 + smoothness(c')

def fit_by_variance_minimization(inverse_fn, n_dp, label, fname):
    """
    inverse_fn(c_obs, dp, di) -> c_prime_derived
    We'll optimize c_prime (1024) and per-date params (n_dp) jointly,
    but the objective is the variance of derived c' across dates plus smoothness.
    Actually easier: fit the forward model to minimize squared error,
    then derive c' per-date and look at the scatter.
    This is what we've been doing. Let me try a different variant:
    minimize the standard deviation of derived c' across dates at each column.
    """
    def objective(params):
        cp = params[:N]
        dp = params[N:]
        # For each date, derive c'
        derived = []
        for di in range(4):
            c_prime_d = inverse_fn(c_all[di], dp, di)
            derived.append(c_prime_d)
        derived = np.stack(derived)
        # Only consider columns where at least 2 dates are finite
        finite_counts = np.sum(np.isfinite(derived), axis=0)
        valid = finite_counts >= 2
        if not np.any(valid):
            return 1e10
        # Minimize variance of derived c' at each valid column
        variances = np.nanvar(derived[:, valid], axis=0, ddof=0)
        # Smoothness of shared c'
        smooth = np.sum(np.diff(cp, 2)**2)
        return np.nansum(variances) + 0.01 * smooth

    p0 = np.zeros(N + n_dp)
    p0[:N] = np.nanmedian(np.stack([np.where(np.isfinite(a), a, np.nan) for a in c_all]), axis=0)
    p0[:N] = np.nan_to_num(p0[:N], nan=0.0)

    bounds = [(-10000, 10000)] * N + [(-10, 10)] * n_dp

    result = minimize(objective, p0, method='L-BFGS-B', bounds=bounds)
    print(f"[{label}] success={result.success}, cost={result.fun:.2e}")

    cp_fit = result.x[:N]
    dp_fit = result.x[N:]

    c_dict = {}
    for di, dname in enumerate(dates):
        c_dict[dname] = inverse_fn(c_all[di], dp_fit, di)

    make_plot(c_dict, label, fname)
    return print_r2(c_dict, label), c_dict, result


# ============================================================================
# Approach 2: Use only far-from-Earth columns to estimate per-date scaling,
# then apply to all columns
# ============================================================================
# The idea: for columns far from Earth (say d > 700), contamination is small,
# so c_j ≈ true c'_j. Find per-date linear scaling that aligns far columns,
# then apply globally.

# Let's compute robust median scaling factors for each date relative to 0316
# in the far regions.

far_mask = distances['0316'] > 700  # At least 700 pixels from center
print(f'\nFar columns (d>700) finite counts:')
for di, dname in enumerate(dates):
    c = c_all[di]
    fin = np.isfinite(c) & far_mask
    print(f'  {dname}: {fin.sum()} finite values')

# For each pair, compute robust slope in far region
print('\nRobust slopes in far region (d>700):')
for i in range(4):
    for j in range(i+1, 4):
        ci, cj = c_all[i], c_all[j]
        mask = far_mask & np.isfinite(ci) & np.isfinite(cj)
        if mask.sum() < 10:
            continue
        # Use Theil-Sen or simple median ratio
        ratios = ci[mask] / cj[mask]
        med_ratio = np.median(ratios[np.isfinite(ratios)])
        print(f'  {dates[i]}/{dates[j]}: median ratio = {med_ratio:.4f}')


# ============================================================================
# Approach 3: Per-date smooth correction function
# For each date, model the deviation from a shared smooth baseline as a
# smooth function of distance. Then c' = c - correction(d).
# ============================================================================

def smooth_correction_approach():
    # Step 1: get a shared baseline using robust median across overlapping dates
    baseline = np.zeros(N)
    for j in range(N):
        vals = [c_all[di][j] for di in range(4) if all_finite[di, j]]
        if len(vals) > 0:
            baseline[j] = np.median(vals)
        else:
            baseline[j] = np.nan

    # Step 2: interpolate baseline to fill NaNs
    valid = np.isfinite(baseline)
    baseline_interp = np.copy(baseline)
    baseline_interp[~valid] = np.interp(
        np.where(~valid)[0],
        np.where(valid)[0],
        baseline[valid]
    )

    # Step 3: for each date, fit a smooth correction as function of distance
    # correction = spline of distance
    from scipy.interpolate import UnivariateSpline

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    c_dict = {}

    for di, dname in enumerate(dates):
        c = c_all[di]
        dist = distances[dname]
        mask = np.isfinite(c)
        residual = c[mask] - baseline_interp[mask]
        d_vals = dist[mask]

        # Sort by distance for spline
        order = np.argsort(d_vals)
        d_sorted = d_vals[order]
        r_sorted = residual[order]

        # Fit smoothing spline
        spline = UnivariateSpline(d_sorted, r_sorted, s=len(d_sorted)*100)

        # Evaluate correction everywhere
        correction = spline(dist)
        c_corrected = c - correction
        c_dict[dname] = c_corrected

        ax = axes[di // 2, di % 2]
        ax.scatter(d_vals, residual, s=5, alpha=0.3, label='raw residual')
        ax.plot(np.sort(d_vals), spline(np.sort(d_vals)), 'r-', lw=2, label='spline fit')
        ax.set_title(dname)
        ax.set_xlabel('Distance')
        ax.set_ylabel('c - baseline')
        ax.legend()

    plt.suptitle('Per-date correction vs distance')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/20_spline_corrections.png', dpi=150)
    plt.close()

    make_plot(c_dict, "Model S: spline-corrected c'", "21_modelS_spline.png")
    r2 = print_r2(c_dict, "Model S: spline correction")
    return r2, c_dict

r2_s, cd_s = smooth_correction_approach()


# ============================================================================
# Approach 4: Non-parametric shared c' with per-date distance nonlinearity
# Model: c_j_d = a_d * c'_j + b_d * g(d_j) where g is a shared smooth function
# We fit c'_j, a_d, b_d, and a smooth g simultaneously.
# ============================================================================

def nonparametric_shared_approach(n_knots=20):
    from scipy.interpolate import UnivariateSpline

    # g will be represented by values at knot positions
    # Knots: uniformly spaced in distance
    knot_distances = np.linspace(0, 1024, n_knots)

    def forward_fn(cp, params):
        a = params[:4]
        b = params[4:8]
        g_vals = params[8:]
        # Interpolate g at each column's distance
        predictions = []
        for di, dname in enumerate(dates):
            g_interp = np.interp(distances[dname], knot_distances, g_vals)
            pred = a[di] * cp + b[di] * g_interp
            predictions.append(pred)
        return predictions

    def objective(params):
        cp = params[:N]
        dp = params[N:]
        preds = forward_fn(cp, dp)
        res = []
        for di in range(4):
            mask = all_finite[di]
            res.append(preds[di][mask] - c_all[di][mask])
        smooth = np.sum(np.diff(cp, 2)**2)
        return np.sum(np.concatenate(res)**2) + 0.1 * smooth

    p0 = np.zeros(N + 8 + n_knots)
    p0[:N] = cp0
    p0[:4] = 1.0  # a_d starts at 1
    p0[4:8] = 0.0  # b_d starts at 0
    p0[8:] = 0.0  # g starts at 0

    bounds = [(-10000, 10000)] * N + [(0.5, 2.0)]*4 + [(-50000, 50000)]*4 + [(-5000, 5000)]*n_knots

    result = minimize(objective, p0, method='L-BFGS-B', bounds=bounds)
    print(f"[Model NP] success={result.success}, cost={result.fun:.2e}")

    cp_fit = result.x[:N]
    a_fit = result.x[:4]
    b_fit = result.x[4:8]
    g_fit = result.x[8:]

    # Derive c' per date: c' = (c - b*g) / a
    c_dict = {}
    for di, dname in enumerate(dates):
        g_interp = np.interp(distances[dname], knot_distances, g_fit)
        c_dict[dname] = (c_all[di] - b_fit[di] * g_interp) / a_fit[di]

    make_plot(c_dict, "Model NP: non-parametric shared c'", "22_modelNP_nonparam.png")
    return print_r2(c_dict, "Model NP: nonparametric"), c_dict

r2_np, cd_np = nonparametric_shared_approach()


# ============================================================================
# Approach 5: Simple robust direct alignment using only far-field overlap
# ============================================================================

def robust_scale_align():
    # Use columns where d > 700 and all dates have finite values
    far_all_finite = far_mask & np.all(all_finite, axis=0)
    print(f'\nFar region all-finite columns: {far_all_finite.sum()}')

    # Compute robust median c' in far region as baseline
    far_vals = [c_all[di][far_all_finite] for di in range(4)]
    far_baseline = np.median(np.stack(far_vals), axis=0)

    # For each date, find scale factor aligning far region to baseline
    scales = []
    for di in range(4):
        ratios = far_vals[di] / far_baseline
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        scale = np.median(ratios)
        scales.append(scale)
        print(f'  {dates[di]} scale factor: {scale:.4f}')

    scales = np.array(scales)

    # Also fit a linear distance correction per date using far region only
    # c_d / scale = c' + a_d * d + b_d in far region
    affine_params = []
    for di in range(4):
        c_scaled = c_all[di] / scales[di]
        mask = far_mask & np.isfinite(c_all[di])
        d_vals = distances[dates[di]][mask]
        c_vals = c_scaled[mask]
        # Fit c_vals = c_baseline + a * d + b
        # Use robust regression
        A = np.vstack([d_vals, np.ones(len(d_vals))]).T
        ab, _, _, _ = np.linalg.lstsq(A, c_vals, rcond=None)
        affine_params.append(ab)
        print(f'  {dates[di]} affine: a={ab[0]:.6f}, b={ab[1]:.2f}')

    # Apply correction globally
    c_dict = {}
    for di, dname in enumerate(dates):
        a, b = affine_params[di]
        c_dict[dname] = (c_all[di] / scales[di]) - a * distances[dname] - b

    make_plot(c_dict, "Model R: robust far-field alignment", "23_modelR_robust.png")
    return print_r2(c_dict, "Model R: robust alignment"), c_dict

r2_r, cd_r = robust_scale_align()


# ============================================================================
# Summary
# ============================================================================
print('\n=== Expanded model summary ===')
results = {
    'raw': orig_r2,
    'S_spline': r2_s,
    'NP_nonparam': r2_np,
    'R_robust': r2_r,
}
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v:.4f}')

best = max(results, key=results.get)
print(f'\nBest expanded model: {best} with R^2 = {results[best]:.4f}')
