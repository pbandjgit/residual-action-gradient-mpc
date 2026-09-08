# D's `lambda_jac` grid extension — follow-up protocol (2026-09-03)

## 0. Provenance and scope

This is a narrow follow-up to the frozen go/no-go ablation protocol
(`docs/JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_2026_08_31.md`, v7) and its
real execution (`results/ablation_execution_2026_08_31/`, Freeze Manifest
`docs/JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_FREEZE_MANIFEST_2026_08_31.json`).
It answers R1.4/R2.M3's "general Jacobian/Sobolev matching" fair-baseline
request, which the Decision Record (2026-09-01) flagged as unsettled
because condition D's `lambda_jac` selection landed on its tuning grid's
upper boundary (`LAM_JAC_GRID=[0.001,0.005,0.01,0.05,0.1,0.5]`) on all 10
seeds (directly re-verified from the real `ablation_training_manifest_
seed*_2026_08_31.json` files before writing this document: 10/10 seeds
select `lambda_jac=0.5`, validation MSE still monotonically decreasing
into the boundary, and closed-loop success at `lambda_jac=0.5` is still
climbing relative to `0.1` for most seeds — e.g. seed 2 jumps from 0.2 to
1.0 between `0.1` and `0.5`).

**None of the original frozen artifacts are modified.** This protocol
only adds new files under a new namespace and a new Freeze Manifest that
references (not replaces) the original one.

## 1. Contract (user-specified, 2026-09-03)

- Extend the existing grid with exactly `{1.0, 2.0, 5.0}` — one extension,
  not an open-ended search.
- Only the 3 new points are trained: `10 seeds x 3 lambda = 30` new
  models. The original 60 models (10 seeds x 6 lambda) are reused as-is
  from the frozen M2 files, never retrained.
- Final `lambda_jac` selection is performed over the **combined** 9-point
  grid (6 old + 3 new) using **validation MSE only**, with the same
  deterministic tie-break (smallest `lambda_jac` on a tie) already
  implemented in `select_lambda_jac` (`src/cstr/ablation_lib.py:590-614`).
  Test MSE, closed-loop success, and oracle gradient metrics play **no
  role** in selection — they are computed only for the already-selected
  candidate (plus, for Section 5.3's existing full-grid transparency
  convention, for each of the 3 new candidates individually).
- If the newly selected `lambda_jac` is again the top of the (now
  extended) grid, i.e. `5.0`, the outcome is recorded as
  `BOUNDARY_PERSISTS` and the grid is **not** extended again. Any other
  selected value is recorded as `INTERIOR_OPTIMUM_FOUND`.
- Executed as a separate follow-up namespace/manifest chain
  (`results/ablation_lamjac_extension_2026_09_03/`,
  `docs/JPC_LAMJAC_GRID_EXTENSION_FREEZE_MANIFEST_2026_09_03.json`), never
  overwriting `ablation_execution_2026_08_31/`.

## 2. Input reuse and provenance verification

The new driver reconstructs `pool`/`clean`/`scale` using the exact same
recipe as `ablation_driver_2026_08_31.run_real_protocol` (`alib.
load_pool()`, `alib.build_clean_query_table(pool, sim, pr.precompute_phi)`,
`alib.shared_scale_constants(pool, clean)`), then verifies this
reconstruction against the **original** frozen M1 (Query Manifest) via
`verify_query_manifest`, and verifies the original Freeze Manifest is
still unmodified via `verify_freeze_manifest`. Both are read-only checks
against the 2026-08-31 artifacts. For each seed, the original frozen M2 is
also read-and-verified (`verify_training_manifest`) to extract its 6-point
`lambda_jac_grid_trace` and its `init_state_hashes["D"]`, which the new
run's own `init_state_hash` for the same seed must match exactly (model
init depends only on `torch.manual_seed(seed)`, not on `lambda_jac`, so
this is a hard equality check, not a tolerance).

## 3. New training and evaluation

For each of the 10 pre-registered seeds and each `lambda_jac` in
`{1.0, 2.0, 5.0}`: `alib.train_base_condition("D", seed, pool, clean,
scale, lam_jac=lam, checkpoint_dir=<new dir>, checkpoint_prefix=
f"D_lamjac{lam}_ext")` — the same function, same module-level
hyperparameters (`EPOCHS=50`, `WARMUP=30`, `BATCH=256`,
`CHECKPOINT_EVERY=10`) as the original run, unmodified.

Post-selection evaluation (for the 3 new candidates, using the same
machinery `run_seed` used for D's original full grid):
`alib.freeze_any(model)` + `_closed_loop_success_rate`-equivalent call
into the real `budget_sweep_solver.closed_loop` at the primary
`horizon=3, budget=20, steps=120` endpoint over `bs.ICS`, and
`alib.diagnose_fixed_idx(model, pool, sim, oracle_idx)` with
`oracle_idx = alib.fixed_oracle_idx(pool["test_idx"])` (same fixed
800-point/meta-seed-90210 set used everywhere else in this protocol
family).

## 4. Reporting

The new Results Manifest reports, per seed: the merged 9-point trace, the
newly selected `lambda_jac`, and (for full-grid transparency) each new
candidate's val/test MSE, closed-loop success, and oracle metrics.
Aggregated: mean/CI (bootstrap, `BOOTSTRAP_SEED=31415`, `N_BOOTSTRAP=
10000`, reusing `alib.bootstrap_ci`/`exact_permutation_pvalue` verbatim)
for D(extended)-vs-E on closed-loop success, and the `BOUNDARY_PERSISTS`
/ `INTERIOR_OPTIMUM_FOUND` outcome tag. No claim beyond this scope: this
protocol does not re-litigate `target_specificity`'s original definition
or any other outcome from the 2026-08-31 protocol; it only supplies the
grid-extension evidence that was missing.
