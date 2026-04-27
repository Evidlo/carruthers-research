#!/usr/bin/env python3
"""Global c_j fit from multiple OOB images.

For each image, fit sharedpwl with c=0, extract c_j from sagged-row residuals,
then average c_j across images. c_j is a detector property (not model-specific);
sharedpwl is just the tool used to derive it.

Usage: python global_fit.py [output_path]
  output_path defaults to params/nfi_glob_<date_of_first_image>.npz
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import torch
from pathlib import Path

from common import load, rob_bias
from fit import fit_model
from model_sharedpwl import Model

device = 'cuda'

ECHO_TRIM = 150
HOT_PIXELS = np.load(Path(__file__).parent / 'hot_pixels.npy')

# Images used for global c_j fit
IMAGE_LIST = [
    ('../images_20260316/oob_nfi_l0.pkl', np.array(list(range(333)) + list(range(666, 1024)))),
    ('../images_20260318/oob_nfi_l0.pkl', np.array(list(range(333)) + list(range(666, 1024)))),
    ('../images_20260319/oob_nfi_l0.pkl', np.array(list(range(333)) + list(range(666, 1024)))),
    # ('../images_20260113/oob_nfi_l0.pkl', np.array(list(range(400)) + list(range(750, 1024)))),
    # ('../images_20260115/oob_nfi_l0.pkl', np.array(list(range(400)) + list(range(750, 1024)))),
    # ('../images_20260117/oob_nfi_l0.pkl', np.array(list(range(400)) + list(range(750, 1024)))),
]

FIT_ROWS = {'top': (ECHO_TRIM, 512), 'bot': (512, 1024 - ECHO_TRIM)}
HALF_ROW = {'top': 0, 'bot': 512}

# Sagged rows relative to fit_start used for c_j residual extraction (most sagged region)
CJ_SAG_REL = {'top': slice(212, 362), 'bot': slice(0, 150)}


def image_tag(path):
    p = Path(path)
    date = p.parent.name.split('_')[-1]
    parts = p.stem.split('_')
    return f'{parts[1]}_{parts[0]}_{date}'


def load_and_prep(path):
    img_np = load(path).astype(np.float64)
    img_np[HOT_PIXELS[:, 0], HOT_PIXELS[:, 1]] = np.nan
    for c in range(img_np.shape[1]):
        col = img_np[:, c]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)
    return img_np


cj_estimates = {'top': [], 'bot': []}

for img_path, flat_idx in IMAGE_LIST:
    print(f'\n[{image_tag(img_path)}]: estimating c_j ...')
    img_np = load_and_prep(img_path)
    img = torch.from_numpy(img_np).to(device)
    bias = torch.from_numpy(rob_bias(img_np, clip_out=150, clip_in=150)).to(device)
    rs = img.sum(dim=1)

    for half in ['top', 'bot']:
        r0, r1 = FIT_ROWS[half]
        s = rs[r0:r1].unsqueeze(1)
        y = img[r0:r1][:, flat_idx]
        b = bias[HALF_ROW[half], flat_idx]

        m = Model(b, s, c=None).to(device)
        fit_model(m, y, b, s)

        with torch.no_grad():
            pred = m(b, s).cpu().numpy()
            sag_sl = CJ_SAG_REL[half]
            sag_resid = np.median((y.cpu().numpy() - pred)[sag_sl], axis=0)
            pwl_med = float(m.pwl(s[sag_sl]).mean())

        cj_full = np.zeros(1024)
        if abs(pwl_med) > 1e-10:
            cj_full[flat_idx] = sag_resid / pwl_med
        cj_estimates[half].append(cj_full)

cj_global = {half: np.mean(cj_estimates[half], axis=0) for half in ['top', 'bot']}


first_tag = image_tag(IMAGE_LIST[0][0])
date = first_tag.split('_')[-1]
out_path = sys.argv[1] if len(sys.argv) > 1 else f'params/nfi_glob_{date}.npz'
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

np.savez(out_path,
    cj_top=cj_global['top'],
    cj_bot=cj_global['bot'],
    images_used=np.array([str(p) for p, _ in IMAGE_LIST]),
)
print(f'\nSaved {out_path}')
