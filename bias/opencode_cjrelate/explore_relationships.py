import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

npz_path = '/home/evan/sync/research/carruthers/bias/opencode_cjrelate/cj.npz'
out_dir = '/www/opencode_cjrelate'

# Load data
d = np.load(npz_path, allow_pickle=True)
data = d['arr_0'].item()

# top half c_j arrays, one per date
c_all = data['top']  # list of 4 arrays, each (1024,)
dates = ['0316', '0317', '0318', '0319']
N = 1024
col = np.arange(N, dtype=float)

distances = {
    '0316': np.abs(col - 512),
    '0317': np.abs(col - 512),
    '0318': np.abs(col - 1024),
    '0319': np.abs(col - 512),
}

# Precompute finite masks
finite_masks = [np.isfinite(c_all[i]) for i in range(4)]

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def make_plot(c_dict, title, fname, xlim=None):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), squeeze=False)
    date_list = dates
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ci = c_dict[date_list[i]]
            cj = c_dict[date_list[j]]
            if i == j:
                vals = ci[np.isfinite(ci)]
                ax.hist(vals, bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
                ax.set_title(date_list[i], fontsize=11)
                ax.tick_params(axis='both', labelsize=7)
            else:
                mask = np.isfinite(ci) & np.isfinite(cj)
                x = cj[mask]
                y = ci[mask]
                if len(x) > 0:
                    ax.scatter(x, y, s=5, alpha=0.4, c='royalblue', edgecolors='none')
                    if xlim is None:
                        lo = min(x.min(), y.min())
                        hi = max(x.max(), y.max())
                        pad = (hi - lo) * 0.05 + 1e-3
                        ax.set_xlim(lo - pad, hi + pad)
                        ax.set_ylim(lo - pad, hi + pad)
                    else:
                        lo, hi = xlim
                        ax.set_xlim(lo, hi)
                        ax.set_ylim(lo, hi)
                    ax.set_aspect('equal')
                ax.set_xlabel(date_list[j], fontsize=9)
                ax.set_ylabel(date_list[i], fontsize=9)
                ax.tick_params(axis='both', labelsize=7)
    plt.suptitle(title, fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{fname}', dpi=150)
    plt.close()


def compute_pairwise_r2(c_dict):
    r2s = {}
    for i in range(4):
        for j in range(i+1, 4):
            di = c_dict[dates[i]]
            dj = c_dict[dates[j]]
            mask = np.isfinite(di) & np.isfinite(dj)
            if mask.sum() < 3:
                continue
            x, y = dj[mask], di[mask]
            ss_res = ((x - y) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot
            r2s[f'{dates[i]}_{dates[j]}'] = r2
    return r2s


def print_r2(c_dict, label=''):
    r2s = compute_pairwise_r2(c_dict)
    avg = sum(r2s.values()) / len(r2s) if r2s else float('nan')
    print(f'--- {label} ---')
    for k, v in sorted(r2s.items()):
        print(f'  {k}: R^2 = {v:.4f}')
    print(f'  avg R^2 = {avg:.4f}')
    return avg


# ---------------------------------------------------------------------------
# Raw plots
# ---------------------------------------------------------------------------
orig = {d: c_all[i] for i, d in enumerate(dates)}
make_plot(orig, 'Raw c_j (top half)', '01_raw.png')
orig_r2 = print_r2(orig, 'raw')


def plot_raw_vs_distance():
    fig, ax = plt.subplots(figsize=(12, 6))
    for di, dname in enumerate(dates):
        c = c_all[di]
        dist = distances[dname]
        mask = finite_masks[di]
        ax.scatter(dist[mask], c[mask], s=5, alpha=0.4, label=dname)
    ax.set_xlabel('Distance from Earth center (pixels)')
    ax.set_ylabel('c_j')
    ax.legend()
    ax.set_title('Raw c_j vs distance from Earth')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/00_raw_vs_dist.png', dpi=150)
    plt.close()

plot_raw_vs_distance()


# ---------------------------------------------------------------------------
# Joint fitting + per-date inversion framework
# ---------------------------------------------------------------------------
# We jointly optimise a SHARED c' profile (1024 values) plus low-dim params.
# After fitting we *invert* the model per-date to get date-specific c' arrays.

c_med = np.nanmedian(np.stack([np.where(np.isfinite(a), a, np.nan) for a in c_all]), axis=0)
c_med = np.nan_to_num(c_med, nan=0.0)
# Use a heavily smoothed median as initial c' to stabilise the high-dim fit
c_prime_init = gaussian_filter1d(c_med, sigma=20, mode='nearest')


def optimize_generic(forward_fn, inverse_fn, params0, bounds, label, fname):
    """
    Optimise shared c' (first N values in params) + per-date params jointly,
    with a smoothness regularisation on c'.
    After fitting, derive per-date c' with the inverse model.
    """
    def residual(params):
        c_prime = params[:N]
        r = []
        for di in range(4):
            pred = forward_fn(c_prime, params, di)
            mask = finite_masks[di]
            r.append(pred[mask] - c_all[di][mask])
        # smoothness regularisation (weak but prevents wild c' oscillations)
        reg_weight = 1.0
        return np.concatenate(r + [np.sqrt(reg_weight) * np.diff(c_prime, 2)])

    result = minimize(lambda p: np.sum(residual(p)**2), params0,
                      method='L-BFGS-B', bounds=bounds)
    print(f"[{label}] success={result.success}, cost={result.fun:.2e}, nit={result.nit}")
    fitted = result.x

    # CRITICAL FIX: derive per-date c' by inverting, do NOT reuse shared c'
    c_derived = {}
    for di, dname in enumerate(dates):
        c_derived[dname] = inverse_fn(c_all[di], fitted, di)

    make_plot(c_derived, label, fname)
    r2 = print_r2(c_derived, label)
    return fitted, c_derived, r2


# ==========================================================================
# Model 1: Additive exponential
# c = c' + a_d * exp(-d/λ)  =>  c' = c - a_d * exp(-d/λ)
# ==========================================================================
def model1_forward(c_prime, params, di):
    a = params[N:N+4]
    lam = params[N+4]
    return c_prime + a[di] * np.exp(-distances[dates[di]] / lam)


def model1_inverse(c_obs, params, di):
    a = params[N:N+4]
    lam = params[N+4]
    return c_obs - a[di] * np.exp(-distances[dates[di]] / lam)


p0_m1 = np.zeros(N + 4 + 1)
p0_m1[:N] = c_prime_init.copy()
p0_m1[N:N+4] = 100.0
p0_m1[N+4] = 100.0
bounds_m1 = [(-50000, 50000)] * N + [(-50000, 50000)]*4 + [(1.0, 10000.0)]

fitted1, c_derived1, r2_1 = optimize_generic(
    model1_forward, model1_inverse, p0_m1, bounds_m1,
    "Model 1: c = c' + a*exp(-d/λ)", '02_model1_exp_add.png')


# ==========================================================================
# Model 2: Multiplicative exponential
# c = c' * (1 + a_d * exp(-d/λ))  =>  c' = c / (1 + a_d * exp(-d/λ))
# ==========================================================================
def model2_forward(c_prime, params, di):
    a = params[N:N+4]
    lam = params[N+4]
    return c_prime * (1.0 + a[di] * np.exp(-distances[dates[di]] / lam))


def model2_inverse(c_obs, params, di):
    a = params[N:N+4]
    lam = params[N+4]
    denom = 1.0 + a[di] * np.exp(-distances[dates[di]] / lam)
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12, denom)
    return c_obs / denom


p0_m2 = np.zeros(N + 4 + 1)
p0_m2[:N] = c_prime_init.copy()
p0_m2[N:N+4] = 0.1
p0_m2[N+4] = 100.0
bounds_m2 = [(-50000, 50000)] * N + [(-0.95, 5.0)]*4 + [(1.0, 10000.0)]

fitted2, c_derived2, r2_2 = optimize_generic(
    model2_forward, model2_inverse, p0_m2, bounds_m2,
    "Model 2: c = c'*(1 + a*exp(-d/λ))", '03_model2_exp_mul.png')


# ==========================================================================
# Model 3: Additive power-law
# c = c' + a_d / (d+ε)^p  =>  c' = c - a_d / (d+ε)^p
# ==========================================================================
epsilon = 1.0


def model3_forward(c_prime, params, di):
    a = params[N:N+4]
    p = params[N+4]
    d = distances[dates[di]] + epsilon
    return c_prime + a[di] / (d ** p)


def model3_inverse(c_obs, params, di):
    a = params[N:N+4]
    p = params[N+4]
    d = distances[dates[di]] + epsilon
    return c_obs - a[di] / (d ** p)


p0_m3 = np.zeros(N + 4 + 1)
p0_m3[:N] = c_prime_init.copy()
p0_m3[N:N+4] = 1000.0
p0_m3[N+4] = 1.0
bounds_m3 = [(-50000, 50000)] * N + [(-1e6, 1e6)]*4 + [(0.1, 5.0)]

fitted3, c_derived3, r2_3 = optimize_generic(
    model3_forward, model3_inverse, p0_m3, bounds_m3,
    "Model 3: c = c' + a/(d+ε)^p", '04_model3_power_add.png')


# ==========================================================================
# Model 4: Additive linear
# c = c' + a_d * d + b_d  =>  c' = c - a_d * d - b_d
# ==========================================================================
def model4a_forward(c_prime, params, di):
    a = params[N:N+4]
    b = params[N+4:N+8]
    d = distances[dates[di]]
    return c_prime + a[di] * d + b[di]


def model4a_inverse(c_obs, params, di):
    a = params[N:N+4]
    b = params[N+4:N+8]
    d = distances[dates[di]]
    return c_obs - a[di] * d - b[di]


p0_m4 = np.zeros(N + 4 + 4)
p0_m4[:N] = c_prime_init.copy()
p0_m4[N:N+4] = 0.0
p0_m4[N+4:N+8] = 0.0
bounds_m4 = [(-50000, 50000)] * N + [(-500, 500)]*4 + [(-5000, 5000)]*4

fitted4, c_derived4, r2_4 = optimize_generic(
    model4a_forward, model4a_inverse, p0_m4, bounds_m4,
    "Model 4: c = c' + a*d + b", '05_model4_linear_add.png')


# ==========================================================================
# Model 5: Multiplicative linear
# c = c' * (1 + a_d * d)  =>  c' = c / (1 + a_d * d)
# ==========================================================================
def model4b_forward(c_prime, params, di):
    a = params[N:N+4]
    d = distances[dates[di]]
    return c_prime * (1.0 + a[di] * d)


def model4b_inverse(c_obs, params, di):
    a = params[N:N+4]
    d = distances[dates[di]]
    denom = 1.0 + a[di] * d
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12, denom)
    return c_obs / denom


p0_m5 = np.zeros(N + 4)
p0_m5[:N] = c_prime_init.copy()
p0_m5[N:N+4] = 0.0
bounds_m5 = [(-50000, 50000)] * N + [(-0.01, 0.01)]*4

fitted5, c_derived5, r2_5 = optimize_generic(
    model4b_forward, model4b_inverse, p0_m5, bounds_m5,
    "Model 5: c = c'*(1 + a*d)", '06_model5_linear_mul.png')


# ==========================================================================
# Model 6: Gaussian bump
# c = c' + a_d * exp(-(d-shift_d)^2/(2σ^2))  =>  c' = c - a_d * exp(-(d-shift_d)^2/(2σ^2))
# ==========================================================================
def model5_forward(c_prime, params, di):
    a = params[N:N+4]
    shift = params[N+4:N+8]
    sigma = params[N+8]
    d = distances[dates[di]] - shift[di]
    return c_prime + a[di] * np.exp(-(d**2) / (2 * sigma**2))


def model5_inverse(c_obs, params, di):
    a = params[N:N+4]
    shift = params[N+4:N+8]
    sigma = params[N+8]
    d = distances[dates[di]] - shift[di]
    return c_obs - a[di] * np.exp(-(d**2) / (2 * sigma**2))


p0_m6 = np.zeros(N + 4 + 4 + 1)
p0_m6[:N] = c_prime_init.copy()
p0_m6[N:N+4] = 100.0
p0_m6[N+4:N+8] = 0.0
p0_m6[N+8] = 100.0
bounds_m6 = [(-50000, 50000)] * N + [(-50000, 50000)]*4 + [(-300, 300)]*4 + [(1.0, 5000.0)]

fitted6, c_derived6, r2_6 = optimize_generic(
    model5_forward, model5_inverse, p0_m6, bounds_m6,
    "Model 6: c = c' + a*exp(-(d-shift)^2/(2σ^2))", '07_model6_gauss.png')


# ==========================================================================
# Summary
# ==========================================================================
results = {
    'raw': orig_r2,
    'exp_add': r2_1,
    'exp_mul': r2_2,
    'power_add': r2_3,
    'linear_add': r2_4,
    'linear_mul': r2_5,
    'gauss_shift': r2_6,
}

print('\n=== Summary average R^2 (derived per-date c\') ===')
for k, v in results.items():
    print(f'  {k}: {v:.4f}')

best = max(results, key=results.get)
print(f'Best model by average R^2: {best}')

# Plot best model's derived c' vs distance
best_derived = {
    'raw': orig,
    'exp_add': c_derived1,
    'exp_mul': c_derived2,
    'power_add': c_derived3,
    'linear_add': c_derived4,
    'linear_mul': c_derived5,
    'gauss_shift': c_derived6,
}[best]

fig, ax = plt.subplots(figsize=(12, 6))
for dname in dates:
    c = best_derived[dname]
    dist = distances[dname]
    mask = np.isfinite(c)
    ax.scatter(dist[mask], c[mask], s=5, alpha=0.4, label=dname)
ax.set_xlabel('Distance from Earth center (pixels)')
ax.set_ylabel("c' (corrected)")
ax.legend()
ax.set_title(f'Corrected c_j vs distance ({best})')
plt.tight_layout()
plt.savefig(f'{out_dir}/08_corrected_vs_dist.png', dpi=150)
plt.close()
