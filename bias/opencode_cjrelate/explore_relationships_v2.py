import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy.polynomial.polynomial as poly

npz_path = '/home/evan/sync/research/carruthers/bias/opencode_cjrelate/cj.npz'
out_dir = '/www/opencode_cjrelate'

# Load data
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
make_plot(orig, 'Raw c_j (top half)', '01_raw.png')
orig_r2 = print_r2(orig, 'raw')


# --- Diagnostic: overlapping region ---
overlap_mask = np.all(all_finite, axis=0)
print(f'\nColumns with all 4 dates finite: {overlap_mask.sum()}')
if overlap_mask.sum() > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    for di, dname in enumerate(dates):
        c = c_all[di][overlap_mask]
        dist = distances[dname][overlap_mask]
        ax.scatter(dist, c, s=10, alpha=0.5, label=dname)
    ax.set_xlabel('Distance from Earth center (pixels)')
    ax.set_ylabel('c_j in overlap region')
    ax.legend()
    ax.set_title('c_j vs distance (all 4 dates overlap)')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/09_overlap_vs_dist.png', dpi=150)
    plt.close()


# --- Model fitting framework ---
# We fit: for each date d, observed c = f_d(c_prime_shared, distance, params_d)
# Params: c_prime_shared (1024) + per-date params
# After fitting, derive per-date c' = f_d^{-1}(observed, params_d)

def fit_and_eval(forward_fn, inverse_fn, params0, bounds, n_dp, label, fname, xlim=None):
    """
    forward_fn(c_prime, dp, di) -> predicted c_obs
    inverse_fn(c_obs, dp, di)  -> c_prime_derived
    params0: initial guess [c_prime(N), date_params(n_dp)]
    n_dp: number of per-date params
    """
    def residual(p):
        cp = p[:N]
        dp = p[N:]
        res = []
        for di in range(4):
            pred = forward_fn(cp, dp, di)
            mask = all_finite[di]
            res.append(pred[mask] - c_all[di][mask])
        # smoothness regularization
        reg = np.sqrt(0.1) * np.diff(cp, 2)
        return np.concatenate(res + [reg])

    result = minimize(lambda p: np.sum(residual(p)**2), params0, method='L-BFGS-B', bounds=bounds)
    print(f"[{label}] success={result.success}, cost={result.fun:.2e}")
    cp_fit = result.x[:N]
    dp_fit = result.x[N:]

    # Derive per-date c'
    c_dict = {}
    for di, dname in enumerate(dates):
        c_dict[dname] = inverse_fn(c_all[di], dp_fit, di)

    make_plot(c_dict, label, fname, xlim=xlim)
    return print_r2(c_dict, label), c_dict, result


# Init: median of finite values across dates
stacked = np.stack([np.where(np.isfinite(a), a, np.nan) for a in c_all])
cp0 = np.nanmedian(stacked, axis=0)
cp0 = np.nan_to_num(cp0, nan=0.0)

# ---- Model A: pure multiplicative linear (best so far) ----
# c = c' * (1 + a_d * d)
# c' = c / (1 + a_d * d)

def fwd_a(cp, dp, di):
    a = dp[:4]
    return cp * (1.0 + a[di] * distances[dates[di]])

def inv_a(c_obs, dp, di):
    a = dp[:4]
    return c_obs / (1.0 + a[di] * distances[dates[di]])

p0 = np.zeros(N + 4)
p0[:N] = cp0.copy()
p0[N:N+4] = 0.0
bounds = [(-10000, 10000)] * N + [(-0.01, 0.01)] * 4
r2_a, cd_a, res_a = fit_and_eval(fwd_a, inv_a, p0, bounds, 4,
    "Model A: c = c'*(1 + a*d)", "10_modelA_linear_mul.png")


# ---- Model B: additive exponential distance contamination ----
# c = c' + a_d * exp(-d / lam) + b_d
# c' = c - a_d * exp(-d/lam) - b_d
# With per-date amplitude a_d, offset b_d, shared lambda

def fwd_b(cp, dp, di):
    a = dp[:4]
    b = dp[4:8]
    lam = dp[8]
    return cp + a[di] * np.exp(-distances[dates[di]] / lam) + b[di]

def inv_b(c_obs, dp, di):
    a = dp[:4]
    b = dp[4:8]
    lam = dp[8]
    return c_obs - a[di] * np.exp(-distances[dates[di]] / lam) - b[di]

p0 = np.zeros(N + 4 + 4 + 1)
p0[:N] = cp0.copy()
p0[N:N+4] = 0.0
p0[N+4:N+8] = 0.0
p0[N+8] = 200.0
bounds = [(-10000, 10000)] * N + [(-50000, 50000)]*4 + [(-5000, 5000)]*4 + [(1.0, 10000.0)]
r2_b, cd_b, res_b = fit_and_eval(fwd_b, inv_b, p0, bounds, 4+4+1,
    "Model B: c = c' + a*exp(-d/λ) + b", "11_modelB_exp_add_offset.png")


# ---- Model C: polynomial multiplicative ----
# c = c' * (1 + a_d * d + b_d * d^2)
# c' = c / (1 + a_d * d + b_d * d^2)

def fwd_c(cp, dp, di):
    a = dp[:4]
    b = dp[4:8]
    d2 = distances[dates[di]]
    return cp * (1.0 + a[di]*d2 + b[di]*d2**2)

def inv_c(c_obs, dp, di):
    a = dp[:4]
    b = dp[4:8]
    d2 = distances[dates[di]]
    return c_obs / (1.0 + a[di]*d2 + b[di]*d2**2)

p0 = np.zeros(N + 4 + 4)
p0[:N] = cp0.copy()
p0[N:N+4] = 0.0
p0[N+4:N+8] = 0.0
bounds = [(-10000, 10000)] * N + [(-0.01, 0.01)]*4 + [(-1e-5, 1e-5)]*4
r2_c, cd_c, res_c = fit_and_eval(fwd_c, inv_c, p0, bounds, 4+4,
    "Model C: c = c'*(1 + a*d + b*d^2)", "12_modelC_quad_mul.png")


# ---- Model D: beam contamination centered at Earth ----
# c = c' + a_d * exp(-d^2 / (2 * lam^2))
# c' = c - a_d * exp(-d^2 / (2 * lam^2))

def fwd_d(cp, dp, di):
    a = dp[:4]
    lam = dp[4]
    d = distances[dates[di]]
    return cp + a[di] * np.exp(-(d**2) / (2.0 * lam**2))

def inv_d(c_obs, dp, di):
    a = dp[:4]
    lam = dp[4]
    d = distances[dates[di]]
    return c_obs - a[di] * np.exp(-(d**2) / (2.0 * lam**2))

p0 = np.zeros(N + 4 + 1)
p0[:N] = cp0.copy()
p0[N:N+4] = 0.0
p0[N+4] = 300.0
bounds = [(-10000, 10000)] * N + [(-50000, 50000)]*4 + [(1.0, 2000.0)]
r2_d, cd_d, res_d = fit_and_eval(fwd_d, inv_d, p0, bounds, 4+1,
    "Model D: c = c' + a*exp(-d^2/(2λ^2))", "13_modelD_gauss_beam.png")


# ---- Model E: non-parametric approach ----
# For each date, the mean c vs distance can be characterized as function f_d(d)
# Since c' is shared, f_d(d) should explain the date-specific offset
# Let's model c = c' * f_d(d) where f_d is a low-order polynomial (degree 3)

def fwd_e(cp, dp, di):
    # dp is 4 x 4 = 16 params (4 dates, 4 poly coeffs)
    coeffs = dp.reshape(4, 4)
    row_coeffs = coeffs[di]
    d = distances[dates[di]]
    f_d = poly.polyval(d, row_coeffs)
    return cp * f_d

def inv_e(c_obs, dp, di):
    coeffs = dp.reshape(4, 4)
    row_coeffs = coeffs[di]
    d = distances[dates[di]]
    f_d = poly.polyval(d, row_coeffs)
    return c_obs / f_d

p0 = np.zeros(N + 4*4)
p0[:N] = cp0.copy()
p0[N:N+4] = 1.0  # constant term = 1
bounds = [(-10000, 10000)] * N + [(-2, 2)]*16
r2_e, cd_e, res_e = fit_and_eval(fwd_e, inv_e, p0, bounds, 16,
    "Model E: c = c'*poly(d)", "14_modelE_poly_mul.png")


# ---- Model F: different Earth-center assumption for 0318 ----
# What if Earth center for 0318 is not at col 1024 but somewhere else?
# Let's treat 0318 center as a free parameter

def make_distances_f(center18):
    return {
        '0316': np.abs(col - 512),
        '0317': np.abs(col - 512),
        '0318': np.abs(col - center18),
        '0319': np.abs(col - 512),
    }

# Fit model A but with free center

def fit_model_f(initial_center=1024):
    def fwd(cp, dp, di, dists_f):
        a = dp[:4]
        return cp * (1.0 + a[di] * dists_f[dates[di]])

    def residual(params):
        cp = params[:N]
        a = params[N:N+4]
        center18 = params[N+4]
        dists_f = make_distances_f(center18)
        res = []
        for di in range(4):
            pred = fwd(cp, a, di, dists_f)
            mask = all_finite[di]
            res.append(pred[mask] - c_all[di][mask])
        reg = np.sqrt(0.1) * np.diff(cp, 2)
        return np.concatenate(res + [reg])

    p0 = np.zeros(N + 4 + 1)
    p0[:N] = cp0.copy()
    p0[N:N+4] = 0.0
    p0[N+4] = initial_center

    bounds = [(-10000, 10000)] * N + [(-0.01, 0.01)]*4 + [(512, 1024)]

    result = minimize(lambda p: np.sum(residual(p)**2), p0, method='L-BFGS-B', bounds=bounds)
    print(f"[Model F] success={result.success}, center18={result.x[N+4]:.1f}, cost={result.fun:.2e}")

    cp_fit = result.x[:N]
    a_fit = result.x[N:N+4]
    center18_fit = result.x[N+4]
    dists_f = make_distances_f(center18_fit)

    c_dict = {}
    for di, dname in enumerate(dates):
        c_dict[dname] = c_all[di] / (1.0 + a_fit[di] * dists_f[dname])

    make_plot(c_dict, f"Model F: c = c'*(1+a*d), 0318 center={center18_fit:.0f}", "15_modelF_free_center.png")
    return print_r2(c_dict, f"Model F (center={center18_fit:.0f})"), c_dict, result

r2_f, cd_f, res_f = fit_model_f()


# ---- Model G: additive + multiplicative contamination ----
# c = c' * (1 + a_d * d) + b_d * exp(-d/lam)

def fwd_g(cp, dp, di):
    a = dp[:4]
    b = dp[4:8]
    lam = dp[8]
    d = distances[dates[di]]
    return cp * (1.0 + a[di]*d) + b[di] * np.exp(-d/lam)

def inv_g(c_obs, dp, di):
    a = dp[:4]
    # Approximate inverse: c' = (c - b*exp(-d/lam)) / (1 + a*d)
    b = dp[4:8]
    lam = dp[8]
    d = distances[dates[di]]
    return (c_obs - b[di] * np.exp(-d/lam)) / (1.0 + a[di]*d)

p0 = np.zeros(N + 4 + 4 + 1)
p0[:N] = cp0.copy()
p0[N:N+4] = 0.0
p0[N+4:N+8] = 0.0
p0[N+8] = 200.0
bounds = [(-10000, 10000)] * N + [(-0.01, 0.01)]*4 + [(-50000, 50000)]*4 + [(1.0, 10000.0)]
r2_g, cd_g, res_g = fit_and_eval(fwd_g, inv_g, p0, bounds, 4+4+1,
    "Model G: c = c'*(1+a*d) + b*exp(-d/λ)", "16_modelG_mixed.png")


# ---- Model H: per-date linear transform + shared distance corr ----
# c = c' * s_d + a_d * d + b_d
# c' = (c - a_d * d - b_d) / s_d
# This allows each date to have its own overall scale

def fwd_h(cp, dp, di):
    s = dp[:4]
    a = dp[4:8]
    b = dp[8:12]
    d = distances[dates[di]]
    return cp * s[di] + a[di]*d + b[di]

def inv_h(c_obs, dp, di):
    s = dp[:4]
    a = dp[4:8]
    b = dp[8:12]
    d = distances[dates[di]]
    return (c_obs - a[di]*d - b[di]) / s[di]

p0 = np.zeros(N + 12)
p0[:N] = cp0.copy()
p0[:4] = 1.0
p0[4:12] = 0.0
bounds = [(-10000, 10000)] * N + [(0.5, 2.0)]*4 + [(-0.01, 0.01)]*4 + [(-5000, 5000)]*4
r2_h, cd_h, res_h = fit_and_eval(fwd_h, inv_h, p0, bounds, 12,
    "Model H: c = c'*s + a*d + b", "17_modelH_affine.png")


# ---- Summary ----
print('\n=== Summary of average R^2 ===')
results = {
    'raw': orig_r2,
    'A_lin_mul': r2_a,
    'B_exp_add': r2_b,
    'C_quad_mul': r2_c,
    'D_gauss': r2_d,
    'E_poly_mul': r2_e,
    'F_free_center': r2_f,
    'G_mixed': r2_g,
    'H_affine': r2_h,
}
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v:.4f}')

best = max(results, key=results.get)
print(f'\nBest model: {best} with R^2 = {results[best]:.4f}')
