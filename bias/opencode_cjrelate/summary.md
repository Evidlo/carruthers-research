# Summary: c_j Cross-Date Relationship Analysis

**Goal:** Find a single model `c_j = h(c'_j, d_j)` mapping a shared true `c'` to observed `c_j` for all image dates.

**Key Findings:**

- **0316, 0317, 0319 correlate tightly** (pairwise R² ≈ 0.83–0.87) with slopes near 0.9 and small offsets (~65–85), suggesting they share a consistent underlying `c'`.
- **0318 is anomalous:** raw values are ~2.5× larger with a ~115 offset. Even after per-date affine rescaling to match 0319, 0318 residuals remain poor (R² ≈ 0.33).
- **Distance-based models failed:** Tested additive/multiplicative exponential, power-law, linear, quadratic, Gaussian beam, polynomial, spline, and mixed forms. None produced a shared `c'` with good 1:1 correlation across all four dates.
- **No overlap in far-field:** The only all-4 overlap is columns 0–199. Here 0318’s distance range (825–1024) is disjoint from the others (312–512), so distance contamination cannot be validated or corrected in shared coordinates.
- **0318 differs structurally:** Its column-to-column shape does not align with the other dates even after global scale/offset correction.

**Conclusion:** No tested single `h(c', d)` reconciles 0318 with the other dates. The 0318 anomaly appears to be a global calibration offset/gain rather than a simple distance-dependent contamination. For a unified `c'`, 0318 should likely be treated as an outlier or investigated for a preprocessing difference.
