# CMOS Detector Distortion Model — AGENT.md

## Problem Statement

A CMOS detector with two halves exhibits directional readout crosstalk. Pixel values are
distorted by aggregate row statistics from both the same detector half (primary sag) and the
corresponding row in the opposite half (echo sag). The goal is to recover the true image `x`
from the observed distorted image `y`.

---

## Notation

| Symbol | Description                                                            |
|--------|------------------------------------------------------------------------|
| i, j   | Row and column indices                                                 |
| xᵢⱼ    | True (undistorted) pixel value                                         |
| yᵢⱼ    | Observed (distorted) pixel value                                       |
| bⱼ     | Per-column bias / dark-current offset (known or estimated separately)  |
| sᵢ     | Row sum of the **same-side** detector half for row i                   |
| s'ᵢ    | Row sum of the **corresponding row** in the **opposite** detector half |

---

## Forward Model

yᵢⱼ = f(xᵢⱼ, bⱼ, sᵢ, s'ᵢ)

yᵢⱼ = (xᵢⱼ + bⱼ) - P(sᵢ)·(xᵢⱼ + bⱼ - cⱼ) - σ'·(xᵢⱼ + bⱼ - cⱼ)·s'ᵢ·1(s'ᵢ > α')

Where P(sᵢ) is the primary sag shape (currently modeled as a piecewise-linear function of sᵢ, shared across columns), and `cⱼ` is a per-column offset that has been confirmed to be a stable detector property (cross-image Pearson r ≈ 0.95–0.99).

Echo sag is still parametric (σ', α') and has not yet been fit in the current code.

### Term Breakdown

1. **Signal + bias − c_j:** (xᵢⱼ + bⱼ − cⱼ) — baseline pixel with per-column offset.
2. **Primary sag** (same-side): − P(sᵢ)·(xᵢⱼ + bⱼ − cⱼ), pixel-proportional, shared shape P across columns.
3. **Echo sag** (opposite-side, active when s'ᵢ > α'): −σ'·(xᵢⱼ + bⱼ − cⱼ)·s'ᵢ, pixel-proportional, no floor term.

---

## Unknown Parameters

| Parameter | Role                                                              |
|-----------|-------------------------------------------------------------------|
| P(s)      | Primary sag shape vs sᵢ — shared across columns (piecewise linear)|
| cⱼ        | Per-column offset; treated as a global detector property          |
| σ'        | Echo sag gain                                                     |
| α'        | Echo sag activation threshold (on s'ᵢ)                            |

---

## Images

* **Detector Split:** 1024x1024 total; Top half (rows 0-511), Bottom half (rows 512-1023).
* **Safe Zone (Unsagged):** Rows 150-361 (Top) and 662-873 (Bottom), Columns 0-400.
* **Extreme Sagged:** Rows 437-511 (Top edge) and 512-587 (Bottom edge), typically where Earth signal is maximum.
* **Earth Center:** Earth is centered horizontally (cols ~400-750) and vertically (across the split).
* **Empty Columns:** Columns 0-400 and 750-1024 contain only background noise and sparse stars.
* **Row Correspondence:** Physical readout correlation exists between row i and i+512.

---

## Verification Strategy


Proposed verification approach:

Use `images_20260113/oob_nfi_l0.pkl`. This is a 1024x1024 image with earth centered in FOV (centered between the two halves).

The goal is to reverse the effect of `f` on this image as much as possible.  For validation we can analyze columns with no earth present (cols 0-400 and 750-1024).  This signal should be flat, barring additive noise and stars present.

Some useful facts:

- guaranteed nonsagged rows for this image (sᵢ, s'ᵢ not big enough due to no earth present on these rows): 150-362 and 662-874
  - rob_bias (robust bias) uses these nonsagged rows to estimate bias
  - making an assumption here that these pixels are yᵢⱼ = f(0 + bⱼ, [small constant sᵢ], [small constant s'ᵢ])

Other images similar to the above live at `images_*/oob_nfi_l0.pkl`. Only use these for analysis.

*DO NOT* edit any existing files in this directory.
*DO NOT* examine any files in this directory not mentioned here, as not to lead you in wrong directions.

Try to do autonomous validation before involving the user.

* **Grade Card Metrics:**
    * **Row Flatness (σ):** Target **0.22 counts**. Standard deviation of median row profile.
    * **Col Flatness (σ):** Target **0.26 counts**. Standard deviation of median column profile (striping).
    * **Half-Half Jump:** Target **< 0.5 counts**. Median difference across the row 511/512 split.
* **Heuristics:**
    * **Unsagged Benchmark:** Flattening must be verified first in the Safe Zone; failures here indicate bad bias (bⱼ) estimation.
    * **Multiplicative Scaling:** Sag must scale with total charge (x + bⱼ) to avoid introducing artificial striping near bright sources.
    * **Physical Consistency:** The model must use a global analytical shape P(s) for all columns to prevent overfitting to transient features (stars).

---

## Assumed Structure

- bⱼ is treated as known (pre-calibrated) or jointly estimated.
- sᵢ is the **row sum** (not mean or median).
- The activation inequalities 1(·) introduce piecewise behavior; the model is
  linear in parameters σ, β, σ' conditional on the active set being known.
- The model does **not** yet account for potential nonlinearity in s or s' as
  functions of pixel value; this is a candidate refinement if the linear model fails.


## Common Functions
`common.py` contains some useful functions for loading images and computing bias (heuristically).  Definitely use the loading function and optionally use bias (e.g. `rob_bias`)


## Validation plots

- 1D plots - 6 subplots rows, each row showing a column (1/2 detector) of actual values vs fitted values of the model (3 top, 3 bottom)
- 2D plot - just the flattened OOB image, (only 1), first row (bias subtracted, flattened), second row (same but absolute error)