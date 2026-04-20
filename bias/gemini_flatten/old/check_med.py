
import numpy as np
from common import load
import json

files = [
    'images_20260111/oob_nfi_l0.pkl', 
    'images_20260113/oob_nfi_l0.pkl',
    'images_20260115/oob_nfi_l0.pkl', 
    'images_20260117/oob_nfi_l0.pkl', 
    'images_20260119/oob_nfi_l0.pkl',
    'images_20260305/oob_nfi_l0.pkl'
]

with open('params.json', 'r') as f:
    params = json.load(f)

all_x = []
alpha = params[2]
half = 512

for f in files:
    img = np.asarray(load(f))
    S = np.sum(img, axis=1)
    
    ut = np.where(S[:half] <= alpha)[0]
    ub = np.where(S[half:] <= alpha)[0]
    if len(ut) == 0: ut = np.arange(half)
    if len(ub) == 0: ub = np.arange(half)
    
    bt = np.median(img[:half][ut], axis=0)
    bb = np.median(img[half:][ub], axis=0)
    
    res = np.zeros_like(img)
    res[:half] = img[:half] - bt
    res[half:] = img[half:] - bb
    all_x.append(res)

med = np.median(all_x, axis=0)

print(f"Median image stats:")
print(f"  Mean: {np.mean(med):.4f}")
print(f"  Std:  {np.std(med):.4f}")
print(f"  Max:  {np.max(med):.4f}")
print(f"  Min:  {np.min(med):.4f}")

thresholds = [25, 50, 75, 100, 200, 500, 1000]
for t in thresholds:
    count = np.sum(med > t)
    print(f"  Count > {t:4}: {count:6} ({count/1024**2*100:.2f}%)")

# Check empty columns only
empty = np.concatenate([np.arange(0, 401), np.arange(750, 1024)])
med_empty = med[:, empty]
print(f"\nMedian image (empty cols) stats:")
print(f"  Mean: {np.mean(med_empty):.4f}")
print(f"  Std:  {np.std(med_empty):.4f}")
for t in thresholds:
    count = np.sum(med_empty > t)
    print(f"  Count > {t:4}: {count:6} ({count/med_empty.size*100:.2f}%)")
