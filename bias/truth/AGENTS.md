# CMOS Detector Distortion Model — AGENT.md

## Problem Statement

A CMOS detector with two halves exhibits directional readout crosstalk and distortion.
Pixel values are distorted by aggregate row statistics from both the same detector half (primary sag) and the
corresponding row in the opposite half (echo sag). Bias is also distorted in some way by the signal level.  The goal is to recover the true image `x`
from the observed distorted image `y`.

---

## Notation

| Symbol | Description                                                                    |
|--------|--------------------------------------------------------------------------------|
| i, j   | Row and column indices                                                         |
| xᵢⱼ    | True (undistorted) pixel value                                                 |
| yᵢⱼ    | Observed (distorted) pixel value                                               |
| bⱼ     | Per-column bias (known or estimated separately, per-image)                     |
| dⱼ     | Per-column dark stripes (known or estimated separately, `remove_dark_stripes`) |
| aᵢⱼ    | Flat field (L1A `dataset.flat`)                                                |
| sᵢ     | Row sum of the **same-side** detector half for row i                           |
| s'ᵢ    | Row sum of the **corresponding row** in the **opposite** detector half         |

---

## Forward Model

yᵢⱼ = f(xᵢⱼ, …) = xᵢⱼ + bⱼ + dⱼ + f₁(xᵢⱼ, bⱼ, sᵢ, s'ᵢ))

We suspect there may be multiple effects combined in the f₁ term affecting rows/cols separately.

**row-sag component**

suspect that a part of f₁ is P(sᵢ)·(xᵢⱼ - cⱼ) - σ'·(xᵢⱼ - cⱼ)·s'ᵢ·1(s'ᵢ > α')  (row sag)

Where P(sᵢ) is the primary sag shape (currently modeled as a piecewise-linear function of sᵢ, shared across columns). Stair-step terminology for P(s):
- **tread:** flat region at low sᵢ where P ≈ 0 (no sag)
- **riser:** sudden step where P jumps to its base value P₁ at s > α = bp[1]
- **below-riser slope:** linear descent continuing past the riser base as sᵢ increases further., and `cⱼ` is a per-column offset that has been confirmed to be a stable detector property (cross-image Pearson r ≈ 0.95–0.99).

Echo sag is still parametric (σ', α') and has not yet been fit in the current code.

**column structure component**

suspect that another part of f₁ is slopeⱼ•xᵢⱼ+offsetⱼ

## Image Properties

* **Detector Split:** 1024x1024 total; Top half (rows 0-511), Bottom half (rows 512-1023).  Separate bias/dark/etc
* **Earth Center:** Earth is usually centered horizontally and vertically (across the split), but this is image dependent.  Some images do not contain Earth.
* **Row Correspondence:** Physical readout correlation exists between row i and i+512.
* **Sag**: Bright stars and the earth usually induce row sag

## Goal

Obtain ground truth xᵢⱼ for parts of certain images to plot a scatter plot comparing

- yᵢⱼ - bⱼ - dⱼ - xᵢⱼ = f₁(xᵢⱼ, bⱼ, sᵢ, s'ᵢ) - error introduced by f₁
- xᵢⱼ - 
- sᵢ - row statistic
- j - column number

There are two approaches:

- use assumed rotational symmetry around earth of xᵢⱼ/aᵢⱼ for OOB image
  - expect minimal signal from earth - just PSF glow which is symmetric
- use assumed rotational symmetry around individual stars xᵢⱼ/aᵢⱼ

Both approaches follow the procedure:

1. find xᵢⱼ for (unsagged) pixels where f₁=0
2. divide out flatfield
3. apply symmetry to fill in xᵢⱼ/aᵢⱼ for (i,j) where f₁≠0 from (i,j) where f₁=0
4. multiply back in flatfield
5. plot f₁

The output should be saved in an npz file with two arrays 'truth' and 'corrupted', each a dict with keys as data name (e.g. "OOB_20260318") and values as np arrays. Use nans for masked pixels.

## Data Loading and Image Sets

```python
from glide.science_data_processing.L1A import L1A
from glide.common_components.camera import CameraNFI
from glide.validation.cam import load_lab_data
from bias_wavelet import remove_dark_stripes

# load mask and flat
cam = CameraNFI()
cam.spec = load_lab_data(cam.spec)
mask, flat = cam.spec.mask_fov, cam.spec.flat

# load and calibrate (L1A already has bias removed)
data = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-SCI_20251221_v1.0.nc')).data
img = data.images.where(xr.DataArray(mask, dims=['row', 'col']), 0) / data.n_frames
img = remove_dark_stripes(img, 'NFI')[0]
```

These dates contain off-nadir images:

``` python
dataset1 = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-STR_20251215_v1.0.nc')).data
dataset1 = dataset1.isel(time=[15, 16, 18, 24, 25])
dataset2 = L1A(xr.open_dataset('/data/L1A/CARRUTHERS_GCI-NFI_L1A-DRK_20251215_v1.0.nc')).data
dataset2 = dataset2.isel(time=[0])
```
