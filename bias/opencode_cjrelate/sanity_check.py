import sys
sys.path.insert(0, '../claude_flatten')
sys.path.insert(0, '..')

import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.image

from common import load, rob_bias
from fit import fit_model
from model_sharedpwl import Model as PWLModel
# from model_sharedpwlslope import Model as PWLModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

OUT_DIR = Path('/www/opencode_cjrelate')
OUT_DIR.mkdir(parents=True, exist_ok=True)
FITS_DIR = Path('../claude_flatten/fits')
FITS_DIR.mkdir(exist_ok=True)
HOT_PIXELS = np.load(Path('../claude_flatten/hot_pixels.npy'))

NCOLS = 1024
ECHO_TRIM = 150
FIT_ROWS = {'top': (ECHO_TRIM, 512), 'bot': (512, 1024 - ECHO_TRIM)}
HALF_ROW = {'top': 0, 'bot': 512}
CJ_SAG_REL = {'top': slice(212, 362), 'bot': slice(0, 150)}

IMAGE_LIST = [
    ('../images_20260113/oob_nfi_l0.pkl', (200, 800)),
    ('../images_20260316/oob_nfi_l0.pkl', (200, 800)),
    ('../images_20260317/oob_nfi_l0.pkl', (200, 800)),
    ('../images_20260318/star_nfi_l0.pkl', (800, 1024)),
    ('../images_20260319/oob_nfi_l0.pkl', (200, 800)),
]


def flat_cols(nonflat):
    lo, hi = nonflat
    return np.array(list(range(lo)) + list(range(hi, NCOLS)))


def load_and_prep(path):
    img = load(path).astype(np.float64)
    img[HOT_PIXELS[:, 0], HOT_PIXELS[:, 1]] = np.nan
    for c in range(img.shape[1]):
        col = img[:, c]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)
    return img


def prep_image(path):
    img_np = load_and_prep(path)
    bias_np = rob_bias(img_np, clip_out=150, clip_in=300)
    img_t = torch.from_numpy(img_np).to(device)
    bias_t = torch.from_numpy(bias_np).to(device)
    rs_t = img_t.sum(dim=1)
    return img_np, bias_np, img_t, bias_t, rs_t


def fit_half(img_t, bias_t, rs_t, flat_idx, half, c_full=None):
    r0, r1 = FIT_ROWS[half]
    s = rs_t[r0:r1].unsqueeze(1)
    y = img_t[r0:r1][:, flat_idx]
    b = bias_t[HALF_ROW[half], flat_idx]
    c = torch.from_numpy(c_full[flat_idx]).to(device) if c_full is not None else None
    m = PWLModel(b, s, c=c).to(device)
    fit_model(m, y, b, s, keep_ratio=0.98)
    return m, s, y, b


def extract_cj(img_t, bias_t, rs_t, flat_idx):
    """Pass-1 c_j extraction from sagged-row residuals. Returns dict keyed by half."""
    cj = {}
    for half in ('top', 'bot'):
        arr = np.full(NCOLS, np.nan)
        m, s, y, b = fit_half(img_t, bias_t, rs_t, flat_idx, half)
        with torch.no_grad():
            pred = m(b, s).cpu().numpy()
        sag_sl = CJ_SAG_REL[half]
        sag_resid = np.median((y.cpu().numpy() - pred)[sag_sl], axis=0)
        pwl_med = float(m.pwl(s[sag_sl]).mean())
        if abs(pwl_med) > 1e-10:
            arr[flat_idx] = sag_resid / pwl_med
        cj[half] = arr
    return cj


def build_residual(img_np, bias_np, img_t, bias_t, rs_t, flat_idx, cj=None):
    """Fit sharedpwl on flat_idx cols, return (resid, fit) full 1024×1024 arrays (NaN elsewhere)."""
    resid = np.full(img_np.shape, np.nan)
    fit = np.full(img_np.shape, np.nan, dtype=np.float32)
    for half in ('top', 'bot'):
        c_half = cj[half] if cj is not None else None
        m, s, y, b = fit_half(img_t, bias_t, rs_t, flat_idx, half, c_full=c_half)
        with torch.no_grad():
            pred = m(b, s).cpu().numpy()
        r0, r1 = FIT_ROWS[half]
        rows = np.arange(r0, r1)
        resid[np.ix_(rows, flat_idx)] = y.cpu().numpy() - pred
        fit[np.ix_(rows, flat_idx)] = pred.astype(np.float32)
    return resid, fit


def save_img(arr, path):
    # Normalize NaN to 0 for display, clip to [-10, 10]
    display = np.where(np.isnan(arr), 0.0, np.clip(arr, -10, 10))
    # Map [-10, 10] → [0, 1] for RdBu_r colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap('RdBu_r')
    rgba = cmap((display + 10) / 20.0)
    matplotlib.image.imsave(str(path), rgba)
    print(f'Saved {path}')


for path, nonflat in IMAGE_LIST:
    date = Path(path).parent.name.split('_')[-1]
    flat_idx = flat_cols(nonflat)
    print(f'\n[{date}] loading...')
    img_np, bias_np, img_t, bias_t, rs_t = prep_image(path)

    print(f'[{date}] fitting nocj...')
    resid_nocj, fit_nocj = build_residual(img_np, bias_np, img_t, bias_t, rs_t, flat_idx, cj=None)
    save_img(resid_nocj, OUT_DIR / f'{date}_nocj.png')
    np.savez(FITS_DIR / f'{date}_sanity_nocj.npz',
        actual=img_np.astype(np.float32), fit=fit_nocj,
        s=rs_t.cpu().numpy().astype(np.float32), sel_flat=flat_idx,
        fit_top_start=FIT_ROWS['top'][0], fit_top_stop=FIT_ROWS['top'][1],
        fit_bot_start=FIT_ROWS['bot'][0], fit_bot_stop=FIT_ROWS['bot'][1])

    print(f'[{date}] extracting c_j...')
    cj = extract_cj(img_t, bias_t, rs_t, flat_idx)

    print(f'[{date}] fitting flat...')
    resid_flat, fit_flat = build_residual(img_np, bias_np, img_t, bias_t, rs_t, flat_idx, cj=cj)
    save_img(resid_flat, OUT_DIR / f'{date}_flat.png')
    np.savez(FITS_DIR / f'{date}_sanity_flat.npz',
        actual=img_np.astype(np.float32), fit=fit_flat,
        s=rs_t.cpu().numpy().astype(np.float32), sel_flat=flat_idx,
        fit_top_start=FIT_ROWS['top'][0], fit_top_stop=FIT_ROWS['top'][1],
        fit_bot_start=FIT_ROWS['bot'][0], fit_bot_stop=FIT_ROWS['bot'][1])
