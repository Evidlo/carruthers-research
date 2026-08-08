# work log

Highest level only, 1-3 lines per session. Newest at the bottom.

## 2026-07-25 — WFI-only 3D diagnostics
Built `sensitivity.py` (explicit design matrix, Gram/SVD null-space analysis) to
ask whether the 2-week arc supports 3D at all; answer is mostly no beyond A10.
Tried a phase-locked dipole basis (no-op), grid extent (confounded), and a
retrieval-free background measurement (floor is real but only beyond 24 Re).
Synthetic test then confirmed forward-model truncation as the cause of the A00
outer-knot blowup — gas beyond the 15 Re grid is 88% of the signal at TP 15 Re.

## 2026-07-26 — grid extent closed out
Found the three couplings to `size_r[1]` (truncation, `rmask`, knot span) and
that free outer knots, not grid extent, cost the temporal stability. Added
`clim=`/`tail=` to SphHarmSplineModel; real 50-date run kills the ~20 Re
reversal with no stability loss. Next: fold into `recon_1D.py`/`recon_3D.py`.

## 2026-07-27 — tail needs its own lr
Built `recon_1D_tail_page.py` (recon_1D's page doubled, WFI only, cols 1-2
current / 3-4 tail). `tail='power'` diverges at the inherited `lr=5e1`; 1e1
fixes it, arguments only.

### tail lr (open)
The outer knot anchors the r^-2.75 continuation to `size_r[1]`, so its gradient
is much larger and `lr=5e1` overshoots — loss goes *noisy*, not slow, and since
`gd` returns one best-loss snapshot for all dates the median date looks fine
(6.8%) while the worst is 140%. Sweep at 1500 iters / 37 dates
(`/tmp/claude_tail_lr.py`): 5e1 → 25% mean resid, **1e1 → 6.42%** (= current
config), 2e0 → 6.70%, 5e-1 → 23% — a window, not "smaller is better".
Per-retrieval lr is a smell; better fix is normalizing that knot's gradient in
`SphHarmSplineModel`. **Revisit `lr` when `tail=` lands in recon_1D/3D.**
Untested at max_l>0.

New evidence 07-28: at `lr=5e1` the `(3,25)+8` hybrid gave resid 7.72% but
**scatter@20Re 45%**, with date 0's knots still looking clean — the residual
barely moves while the ensemble diverges, so *resid cannot detect this failure*.
`recon_1D_tail.py` now pins `LR = 1e1` for all configs.

## 2026-07-28 — hybrid grid: 8 outer shells replace the wide grid
Decoupled radial *extent* from radial *resolution*: `DefaultGrid(outer_r=,
outer_bins=)` appends coarse log shells past `size_r[1]`, and `DiffLoss` now
divides by Δlog r so a spacing discontinuity isn't read as a density gradient.
`(3,22)x200 + 8 to 100` reproduces `(3,100)x200` at 200 in-range shells.

### grid extent (settled, pending commit)
- **8 outer shells suffice.** before `(3,25)x200` → knots 28.9, 7.4, **10.0**
  (reversal), resid 6.61%, scatter@20Re 8.81%; hybrid `(3,22)x200 +8 to 100
  clim=(3,21)` → 38.9, 13.7, 5.8 monotonic, resid 6.60%, scatter 7.96%. Matches
  the 07-26 `(3,100)` numbers (33.6, 13.0, 5.7). `recon_1D_tail.py`,
  /www/storm/1D_tail_compare.html.
- **Boundary goes at TP_MAX (22), not clim[1] (21).** Knots are set by `clim` and
  don't move with the boundary, but the mask still admits LOS tangenting to 22
  and discretization error concentrates at tangency, so the coarse region must
  start beyond it. That is also why only 8 shells are needed: nothing tangents
  out there.
- **Coarse bins need geometric centers.** `SphericalGrid`'s manual-`r_b` path uses
  arithmetic centers; on wide outer bins that biases them outward into density
  orders of magnitude too low. `DefaultGrid` sets them itself.
- **DiffLoss weight drops by 1/Δlog r²** (8.9e3 on `(3,25)x200`): 5e4 → 5.6.
  Rescaled at every call site. Verified behavior-preserving — `synth_truncation.py`
  at 3.24 reproduces the 07-25 coefficients to ~0.1 (`full` 62.0 → 141.9).
  On a smooth r^-2.75 profile the *unnormalized* loss spikes 36x at the junction.
- Fixed a pre-existing tomosphero bug: dynamic grid + `spacing=` + `datetime64`
  dates crashed in `SphericalGrid.__init__`; `recon_1D_tail.py` could not run.
- Not yet folded into `recon_1D.py`/`recon_3D.py`. Untested at max_l>0.


## 2026-07-28 — fake_simulate repaired, hypersweep blocked
Fixed 3 bugs leaving `fake_simulate` 1.2e4 too quiet; now matches `simulate` to
3.4%. Staged uncommitted in glide-sdc + tomosphero.

### TODO
- **Validate 200 shells over 3-100 Re** (tuned for 3-25, ~94/e-fold → ~57); affects `tail='power'` and `hypersweep`.
- **`recon_fake.py`**: `recon.py` with `fake_simulate`, check carderr vs `/www/sph/two_week_100x50lin.html`.
- **`hypersweep.py`**: mockup approved, `evaluate_real` written but unrun; drop the inherited 300-shell biggrid.
