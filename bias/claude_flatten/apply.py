#!/usr/bin/env python3
"""Per-image fit: extract b_j, s, bias from a raw image and fit the model.

Usage: python apply.py <image_path> [--model MODEL]
Output: params/nfi_fit_<model>_<date>.npz
"""

import sys
import argparse
sys.path.insert(0, '..')

import numpy as np
import torch
from pathlib import Path

from common import load, rob_bias
from fit import fit_model
from registry import MODELS

ECHO_TRIM = 150
OOB_FLAT_IDX = np.array(list(range(400)) + list(range(750, 1024)))
HOT_PIXELS = np.load(Path(__file__).parent / 'hot_pixels.npy')
FIT_ROWS = {'top': (ECHO_TRIM, 512), 'bot': (512, 1024 - ECHO_TRIM)}
HALF_ROW = {'top': 0, 'bot': 512}

parser = argparse.ArgumentParser()
parser.add_argument('image_path')
parser.add_argument('--model', default='sharedpwl', choices=list(MODELS))
args = parser.parse_args()

ModelClass = MODELS[args.model]
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


image_path = args.image_path
flat_idx = OOB_FLAT_IDX

print(f'Loading {image_path} ...')
img_np = load_and_prep(image_path)
bias = rob_bias(img_np, clip_out=150, clip_in=150)

img_t = torch.from_numpy(img_np).to(device)
bias_t = torch.from_numpy(bias).to(device)
rs_t = img_t.sum(dim=1)

save_dict = dict(
    model_name=args.model,
    b_top=bias[HALF_ROW['top'], flat_idx],
    b_bot=bias[HALF_ROW['bot'], flat_idx],
    bias_top=bias[HALF_ROW['top'], :],
    bias_bot=bias[HALF_ROW['bot'], :],
    s_top=rs_t[FIT_ROWS['top'][0]:FIT_ROWS['top'][1]].cpu().numpy(),
    s_bot=rs_t[FIT_ROWS['bot'][0]:FIT_ROWS['bot'][1]].cpu().numpy(),
    flat_idx=flat_idx,
    fit_top_start=FIT_ROWS['top'][0],
    fit_top_stop=FIT_ROWS['top'][1],
    fit_bot_start=FIT_ROWS['bot'][0],
    fit_bot_stop=FIT_ROWS['bot'][1],
    image_path=str(Path(image_path).resolve()),
)

for half in ['top', 'bot']:
    print(f'\nFitting {half} ...')
    r0, r1 = FIT_ROWS[half]
    s = rs_t[r0:r1].unsqueeze(1)
    y = img_t[r0:r1][:, flat_idx]
    b = bias_t[HALF_ROW[half], flat_idx]

    m = ModelClass(b, s).to(device)
    fit_model(m, y, b, s)
    for k, v in m.to_params().items():
        save_dict[f'{k}_{half}'] = v

tag = image_tag(image_path)
date = tag.split('_')[-1]
out_path = f'params/nfi_fit_{args.model}_{date}.npz'
Path('params').mkdir(exist_ok=True)
np.savez(out_path, **save_dict)
print(f'\nSaved {out_path}')
