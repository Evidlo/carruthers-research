# CMOS Detector Distortion Model — AGENT.md

## Problem Statement

A CMOS detector with two halves exhibits directional readout crosstalk. Pixel values are
distorted by aggregate row statistics from both the same detector half (primary sag) and the
corresponding row in the opposite half (echo sag). The goal is to recover the true image `x`
from the observed distorted image `y`.

---

## Notation

| Symbol | Description |
|---|---|
| $i, j$ | Row and column indices |
| $x_{ij}$ | True (undistorted) pixel value |
| $y_{ij}$ | Observed (distorted) pixel value |
| $b_j$ | Per-column bias / dark-current offset (known or estimated separately) |
| $s_i$ | Row sum of the **same-side** detector half for row $i$ |
| $s'_i$ | Row sum of the **corresponding row** in the **opposite** detector half |

---

## Forward Model

$$
y_{ij} = f(x_{ij},\, b_j,\, s_i,\, s'_i)
$$

$$
\boxed{
y_{ij} = (x_{ij} + b_j)
  - \bigl(\sigma(x_{ij} + b_j)\,s_i + \beta\bigr)\,\mathbf{1}(s_i > \alpha)
  - \sigma'(x_{ij} + b_j)\,s'_i\,\mathbf{1}(s'_i > \alpha')
}
$$

### Term Breakdown

1. **Signal + bias:** $(x_{ij} + b_j)$ — baseline pixel with column offset.

2. **Primary sag** (same-side, active when $s_i > \alpha$):
$$
-\bigl(\sigma(x_{ij} + b_j)\,s_i + \beta\bigr)\,\mathbf{1}(s_i > \alpha)
$$
   - $\sigma(x_{ij}+b_j)\,s_i$: pixel-proportional suppression scaling with row sum.
   - $\beta$: fixed additive floor of suppression (active whenever the threshold is exceeded).

3. **Echo sag** (opposite-side, active when $s'_i > \alpha'$):
$$
-\sigma'(x_{ij} + b_j)\,s'_i\,\mathbf{1}(s'_i > \alpha')
$$
   - Pixel-proportional only; **no floor term**.

---

## Unknown Parameters

All five unknowns are **scalar constants** (not position-dependent):

| Parameter | Role |
|---|---|
| $\sigma$ | Primary sag gain (pixel-proportional × row sum) |
| $\beta$ | Primary sag floor offset |
| $\alpha$ | Primary sag activation threshold (on $s_i$) |
| $\sigma'$ | Echo sag gain |
| $\alpha'$ | Echo sag activation threshold (on $s'_i$) |

---

## Verification Strategy

The model is **not yet confirmed**. Proposed verification approach:

Use `image_20260111/oob_nfi_l0.npy`. This is a 1024x1024 image with earth centered in FOV (centered between the two halves).

The goal is to reverse the effect of `f` on this image as much as possible.  For validation we can analyze columns with no earth present (cols 0-400 and 750-1024).  This signal should be flat, barring additive noise and stars present.

Some useful facts:

- guaranteed nonsagged rows for this image (s_i, s'_i not big enough due to no earth present on these rows): 150-362 and 662-874
  - rob_bias (robust bias) uses these nonsagged rows to estimate bias
  - making an assumption here that these pixels are y_ij = f(0 + b_j, [small constant s_i], [small constant s'_i])
  
- there might be an additional term c_j s.t. `y_ij = f(x_ij + b_j, s_i, s'_i) + c_j` 
- `c_j` might be constant across images

Other images similar to the above are `image_20260115/oob_nfi_l0.npy`, `image_20260117/oob_nfi_l0.npy`, `image_20260119/oob_nfi_l0.npy`.  Only use these for analysis.

*DO NOT* edit any existing files in this directory.

Try to do autonomous validation before involving the user.


---

## Assumed Structure

- $b_j$ is treated as known (pre-calibrated) or jointly estimated.
- $s_i$ is the **row sum** (not mean or median).
- The activation inequalities $\mathbf{1}(\cdot)$ introduce piecewise behavior; the model is
  linear in parameters $\sigma, \beta, \sigma'$ conditional on the active set being known.
- The model does **not** yet account for potential nonlinearity in $s$ or $s'$ as
  functions of pixel value; this is a candidate refinement if the linear model fails.


## Common Functions
`common.py` contains some useful functions for loading images and computing bias (heuristically).  Definitely use the loading function and optionally use bias (e.g. `rob_bias`)


