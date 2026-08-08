# tasks

Handoff for the hybrid-grid change. Written 2026-07-28. Delete once committed and
folded into the production recons.

## State: everything below is STAGED BUT NOT COMMITTED

Four repos, staged with `git add`. Nothing is committed — Evan commits.

| repo | files |
|---|---|
| `tomosphero` | `tomosphero/geometry.py` |
| `glide-sdc` (branch `evan`) | `glide/science/model_sph.py`, `glide/science/recon/loss_sph.py` |
| `carruthers` | `recon/recon_1D_tail.py`, `recon/WORK.md`, `recon/synth_truncation.py`, `recon/hypersweep.py`, `recon/recon_1D.py`, `recon/recon_3D.py`, `recon/recon_3D_reduced.py` |
| `carruthers/prelim` (nested repo, stages separately) | `AGENTS.md` (new) |

Check with `git status --short` in each; `prelim` will NOT show up in the
`carruthers` status — it is its own repo, not a submodule.

## What changed

1. **`DefaultGrid(outer_r=, outer_bins=)`** — appends coarse log shells past
   `size_r[1]` so radial *extent* (complete the LOS integral) is decoupled from
   radial *resolution* (bins inside the fitted range). `outer_bins=0` reproduces
   the old grid bit-for-bit, so every existing call site is unaffected.
   Sets **geometric** bin centers itself; `SphericalGrid`'s manual-`r_b` path uses
   arithmetic centers, which on wide outer bins bias them outward into density
   orders of magnitude too low.
2. **`DiffLoss` divides by Δlog r** — otherwise a spacing discontinuity reads as a
   density gradient. On a smooth r^-2.75 profile the unnormalized loss spikes
   **36x** at the junction; normalized it is flat (0.98).
   **Weights drop by 1/Δlog r²** — 8.9e3 on `(3,25)x200`, so 5e4 → 5.6. Already
   rescaled at every call site, including commented-out ones.
3. **tomosphero bugfix** (pre-existing, unrelated) — dynamic grid + `spacing=` +
   `datetime64` dates crashed in `SphericalGrid.__init__`. `recon_1D_tail.py`
   could not run at all before this.
4. **`recon_1D_tail.py`** — `hybrid` config added, `after` dropped from the page,
   `LR = 1e1` pinned for all configs, per-config `bins`.

## Results (converged, lr=1e1, 50-date March, WFI+NFI geometry, TP ≤ 22)

| config | resid | scatter@20Re | outer A₀₀ knots |
|---|---|---|---|
| before `(3,25)x200` | 6.61% | 8.81% | 28.9 → 7.4 → **10.0** reversal |
| hybrid `(3,22)x200 +8 to 100` | 6.60% | 7.96% | 38.9 → 13.7 → 5.8 monotonic |

Matches the 07-26 `(3,100)x200` result (33.6 → 13.0 → 5.7) with 200 in-range
shells instead of ~130. Page: `/www/storm/1D_tail_compare.html`.

`synth_truncation.py` at the rescaled weight 3.24 reproduces the 07-25
coefficients to ~0.1 (`full` 62.0 → 141.9, `trunc` monotonic), confirming the
DiffLoss change is behavior-preserving.

## Design points worth not re-deriving

- **Boundary goes at TP_MAX (22), not clim[1] (21).** Knots are set by `clim` and
  do not move with the grid boundary. The mask still admits LOS tangenting to 22,
  and discretization error concentrates at tangency, so the coarse region must
  start beyond the last tangent point.
- **That is also why 8 shells suffice** — nothing tangents in the coarse region,
  so it is only ever crossed obliquely. Earlier estimates suggested ~8 gets under
  ~1% column error to TP 25 Re and ~20 gets under 0.2% at WFI's full 31 Re reach.
- **100 Re is only adequate because of the TP ≤ 22 mask.** If the fit ever widens
  toward WFI's 31 Re reach, `outer_r` must grow regardless of shell count.
- **A single wide outer shell does not work** (30–500 Re scored −7.5% vs −3.9%
  for 30–100): the log bin center drags out to where density is ~40x too low.

## Open tasks

1. **Commit the staged work** (Evan). Four separate commits, one per repo; the
   tomosphero bugfix is independent and could go first.
2. **Fold `outer_r`/`outer_bins` + `tail=`/`clim=` into `recon_1D.py` and
   `recon_3D.py`.** They still use plain wide grids (`(3,25)` and `(3,60)`) and
   have their DiffLoss lines commented out. Rescaled weights are already written
   into those comments (11.2 for `(3,60)x200`, 11.2 / 2.25 for `(3,25)x200`).
3. **Re-check `lr` when this lands in recon_1D/3D.** See WORK.md "tail lr (open)".
   `lr=5e1` produced resid 7.72% with **scatter@20Re 45%** and a clean-looking
   date 0 — the residual cannot detect this failure, so check the scatter too.
   The real fix is normalizing the outer knot's gradient in `SphHarmSplineModel`.
4. **Untested at `max_l > 0`.** All hybrid-grid validation so far is `max_l=0`.
5. **Sweep the outer shell count** (8 was chosen as the easiest number to explain,
   not measured). Cheap: vary `outer_bins` in `recon_1D_tail.py`'s hybrid config.
6. **Thesis update** — see `prelim/AGENTS.md`. `discretization_table` and the
   summary sentence at `main.typ:2199-2230` still describe a single uniform log
   grid and report total bins rather than bins/decade inside the fitted range.
   The tangency argument that justifies the coarse region is not made in the text.
7. **`/www/tmp/knot_sensitivity.png` is the only evidence for the 21 Re knot
   cutoff** and it is a bare PNG with no write-up (generator:
   `/tmp/claude_grid/dydn4.py`, both dated 07-26, still present but in `/tmp`).
   Worth regenerating somewhere durable before `/tmp` is cleared.

## Reproducing

```bash
cd ~/sync/research/carruthers/recon
PYTORCH_ALLOC_CONF=expandable_segments:True python3 recon_1D_tail.py before
PYTORCH_ALLOC_CONF=expandable_segments:True python3 recon_1D_tail.py hybrid
python3 -c "import recon_1D_tail as m; m.compare()"    # -> /www/storm/1D_tail_compare.html
PYTORCH_ALLOC_CONF=expandable_segments:True python3 synth_truncation.py
```

Cache is `/tmp/claude_tail/*.npz`; `compare()` reads labels from `CFGS`, not the
cache, so label edits need no re-run. 208 shells fit in 11.7 GB without touching
`NDATES`; 225 did not.
