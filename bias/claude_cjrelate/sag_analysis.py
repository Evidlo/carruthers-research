import sys
sys.path.insert(0, '../claude_flatten')
sys.path.insert(0, '..')

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import load, rob_bias

OUT_DIR = Path('/www/claude_cjrelate')
OUT_DIR.mkdir(parents=True, exist_ok=True)
FITS_DIR = Path('../claude_flatten/fits')
HOT_PIXELS = np.load(Path('../claude_flatten/hot_pixels.npy'))

# Nonflat ranges from the sanity_check run that produced the npz files
CONFIG = {
    '20260316': ('../images_20260316/oob_nfi_l0.pkl', 566),
    '20260317': ('../images_20260317/oob_nfi_l0.pkl', 566),
    '20260318': ('../images_20260318/star_nfi_l0.pkl', 1024),
    '20260319': ('../images_20260319/oob_nfi_l0.pkl', 566),
}

# Absolute sag row ranges (from FIT_ROWS and CJ_SAG_REL in sanity_check)
# top: FIT_ROWS[0]=150, CJ_SAG_REL = slice(212,362) → rows 362-511
# bot: FIT_ROWS[0]=512, CJ_SAG_REL = slice(0,150)   → rows 512-661
SAG_ROWS = {'top': (362, 512), 'bot': (512, 662)}
BIAS_ROW  = {'top': 0, 'bot': 512}

COLORS = {'20260316': 'C0', '20260317': 'C1', '20260318': 'C2', '20260319': 'C3'}
LABELS = {'20260316': '0316', '20260317': '0317', '20260318': '0318 (edge)', '20260319': '0319'}


def load_and_prep(path):
    img = load(path).astype(np.float64)
    img[HOT_PIXELS[:, 0], HOT_PIXELS[:, 1]] = np.nan
    for c in range(img.shape[1]):
        col = img[:, c]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)
    return img


col = np.arange(1024)
results = {}

for date, (img_path, earth_col) in CONFIG.items():
    short = date[-4:]
    print(f'Loading {date}...')
    npz = np.load(FITS_DIR / f'{date}_sanity_nocj.npz')
    actual = npz['actual'].astype(np.float64)
    fit    = npz['fit'].astype(np.float64)
    flat   = npz['sel_flat']

    sag_err = {}
    for half in ('top', 'bot'):
        r0, r1 = SAG_ROWS[half]
        resid = actual[r0:r1] - fit[r0:r1]   # NaN outside flat cols
        sag_err[half] = np.nanmedian(resid, axis=0)  # (1024,)

    img_np  = load_and_prep(img_path)
    bias_np = rob_bias(img_np, clip_out=150, clip_in=300)
    bj = {half: bias_np[BIAS_ROW[half]] for half in ('top', 'bot')}

    results[date] = {
        'sag_err': sag_err,
        'bj': bj,
        'dist': np.abs(col - earth_col),
        'flat': flat,
        'earth_col': earth_col,
    }


for half in ('top', 'bot'):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{half} half — sagged-row median error', fontsize=13)

    for date, r in results.items():
        flat = r['flat']
        err  = r['sag_err'][half][flat]
        bj   = r['bj'][half][flat]
        dist = r['dist'][flat]
        kw   = dict(s=4, alpha=0.5, color=COLORS[date], label=LABELS[date])

        axes[0].scatter(flat, err, **kw)
        axes[1].scatter(bj, err, **kw)
        axes[2].scatter(dist, err, **kw)

    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Sag error (counts)')
    axes[0].set_title('vs column')
    axes[0].legend(markerscale=3)

    axes[1].set_xlabel('b_j (counts)')
    axes[1].set_title('vs b_j')
    axes[1].set_ylabel('')

    axes[2].set_xlabel('|col − earth_col|')
    axes[2].set_title('vs distance from Earth')
    axes[2].set_ylabel('')

    plt.tight_layout()
    out = OUT_DIR / f'sag_error_{half}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved {out}')

# ---- Pairwise scatterplot matrix ----
DATES = list(CONFIG.keys())
SHORT = [d[-4:] for d in DATES]

for half in ('top', 'bot'):
    n = len(DATES)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    fig.suptitle(f'{half} half — pairwise sag error', fontsize=13)

    errs = [results[d]['sag_err'][half] for d in DATES]

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                vals = errs[i][np.isfinite(errs[i])]
                ax.hist(vals, bins=40, color=COLORS[DATES[i]], edgecolor='none')
                ax.set_title(SHORT[i], fontsize=10)
            else:
                mask = np.isfinite(errs[i]) & np.isfinite(errs[j])
                x, y = errs[j][mask], errs[i][mask]
                if len(x):
                    ax.scatter(x, y, s=3, alpha=0.4, color='royalblue', edgecolors='none')
                    lo = min(x.min(), y.min())
                    hi = max(x.max(), y.max())
                    pad = (hi - lo) * 0.05
                    ax.set_xlim(lo - pad, hi + pad)
                    ax.set_ylim(lo - pad, hi + pad)
                    ax.set_aspect('equal')
                ax.set_xlabel(SHORT[j], fontsize=8)
                ax.set_ylabel(SHORT[i], fontsize=8)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = OUT_DIR / f'sag_pairwise_{half}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved {out}')

# ---- Pairwise difference plots ----
ref = '20260316'
pairs = [('20260317', 'C1'), ('20260318', 'C2'), ('20260319', 'C3')]

for half in ('top', 'bot'):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{half} half — sag error difference vs 0316', fontsize=13)

    r0 = results[ref]
    for other, color in pairs:
        ro = results[other]
        # shared columns: non-NaN in both
        shared = np.where(
            np.isfinite(r0['sag_err'][half]) & np.isfinite(ro['sag_err'][half])
        )[0]
        diff = ro['sag_err'][half][shared] - r0['sag_err'][half][shared]
        bj   = r0['bj'][half][shared]
        dist = r0['dist'][shared]
        kw   = dict(s=4, alpha=0.5, color=color, label=f'{LABELS[other]} − 0316')

        axes[0].scatter(shared, diff, **kw)
        axes[1].scatter(bj, diff, **kw)
        axes[2].scatter(dist, diff, **kw)

    for ax in axes:
        ax.axhline(0, color='k', lw=0.5, ls='--')

    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Δ sag error (counts)')
    axes[0].set_title('vs column')
    axes[0].legend(markerscale=3)
    axes[1].set_xlabel('b_j 0316 (counts)')
    axes[1].set_title('vs b_j')
    axes[2].set_xlabel('|col − 0316 earth|')
    axes[2].set_title('vs distance from 0316 Earth')

    plt.tight_layout()
    out = OUT_DIR / f'sag_diff_{half}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved {out}')
