# N-family / N-scale / FD / L-grad sensitivity — design note (2026-09-03)

## 0. Scope and status

Answers R2.M3's "FD-step/noise/gradient-weight sensitivity" request,
R2.m8's "justify Cauchy noise choice/scale; add noiseless/Gaussian/
correlated-noise cases," and the strategy document's explicit requirement
(`docs/JPC_MAJOR_REVISION_RESPONSE_STRATEGY_2026_08_31.md:151-152`,
`:138-143`) to sweep FD step size, label-noise scale, `lambda_g`, and to
resolve the real `eps=1e-3` (main experiments) vs. `eps=0.02`
(`reviewer_supplement_cstr.py`) discrepancy directly.

**Non-gating, but not result-blind.** Does not recompute
`target_specificity`, `MSE_ATTRIBUTION`, or the primary 10-seed E-vs-B
contrast. An unfavorable result here must narrow the manuscript's
robustness-scope language, not be omitted. Condition D is out of scope.

Axis names below are **N-family, N-scale, FD, L-grad** (not "Axis
A/B/C/D" — "Axis D" collided with the manuscript's condition D).

**Design note, not yet frozen or implemented.** Scientific design and
experimental axes are settled as of v3; v4 and v5 are operational/
statistical hardening only — no new experimental axis is added in either.

## v3 -> v4 corrections

User review of v3 found 6 P0 + 6 P1 issues:

1. **P0 — fixed-oracle-index provenance was missing.** Recomputing
   `alib.fixed_oracle_idx(pool["test_idx"])` fresh is not sufficient
   proof it is the SAME 800-point set the primary protocol used, even
   though it is deterministic. **Fixed**: the Setup Manifest (Section 6)
   now pins the ORIGINAL Results Manifest's own
   `oracle_fixed_idx_path`/`oracle_fixed_idx_sha256`
   (`scripts/ablation_driver_2026_08_31.py:670-682`, confirmed to exist
   in the real M3 payload) and loads the index array from THAT file,
   never recomputing it independently.
2. **P0 — no compatibility gate for the continuous-metric evaluator
   itself.** Freeze/hash verification of the solver-audit artifacts
   proves the FILES are unmodified, not that THIS protocol's own wiring
   into `single_cstr_rollout` (`src/cstr/solver_audit_lib.py:335`,
   confirmed to exist) is correct. **Fixed**: before evaluating any new
   checkpoint, re-run `single_cstr_rollout` on at least one already-frozen
   checkpoint (B and E, seed 0) and require its per-IC `final_V`/`max_V`/
   `settling_time`/`success` to reproduce Battery 1's frozen values
   exactly — a real evaluator-wiring smoke check, not just a manifest
   hash check. Added to Section 5.
3. **P0 — incomplete failure/statistics rule.** Recording
   `TRAINING_DIVERGED` is necessary but not sufficient: v3 never said
   what happens to a CONTRAST when one of its paired seeds diverged.
   **Fixed** (Section 4): if any seed contributing to a
   level-vs-reference contrast has `TRAINING_DIVERGED` on either side,
   that entire contrast is reported as `DESCRIPTIVE_CONTRAST_INCOMPLETE`
   and **no CI/bootstrap is computed for it** — silently excluding the
   failed seed and computing a CI on the remaining ones is explicitly
   **prohibited** (that would understate variance and could manufacture
   a tighter interval than the evidence supports).
4. **P0 — no crash-resume contract for training.** A bare no-overwrite
   writer would collide if a crash leaves a partial 10-epoch checkpoint
   before its Training Record is written. **Fixed** (Section 6): reuses
   the EXACT SAME "fresh, never-before-used attempt subdirectory per
   retry" pattern the primary ablation driver already established
   (`ablation_driver_2026_08_31.run_full_protocol`'s `attempt_n` logic) —
   not a new invention, the same mechanism applied to this protocol's
   per-model training loop.
5. **P0 — compatibility-artifact ordering and pinning were unspecified.**
   **Fixed** (Section 6): the literal required order is implementation ->
   synthetic smoke -> the 1 discarded real compatibility reconstruction
   (produces its own result JSON) -> Freeze Manifest (which pins that
   JSON's path AND hash, not a bare "PASS" label) -> Setup Manifest ->
   the 80-model execution. Freezing is blocked if the compatibility
   result JSON does not report all four exact-equality checks (Section
   v2->v3 item 4) as passed.
6. **P0 — compatibility cost arithmetic was imprecise.** Verifying the
   parameterized module's `clean`-builder path (not just its training
   path) at the literal baseline `FD_EPS=1e-3` also requires one real
   `clean` reconstruction (`sim.step()` calls) to compare byte-for-byte
   against the frozen original `clean`. **Corrected accounting**: **80
   evidence training runs + 1 discarded training reconstruction**, **3
   evidence `clean` rebuilds + 1 discarded compatibility `clean`
   reconstruction** (the latter also discarded after verification, never
   used as data).

P1:

7. **"supplement-matching" overclaimed.** The real
   `reviewer_supplement_cstr.py` differs from the main experiments in
   more than just `eps`. Renamed the `0.02` FD level's label to
   "`eps=0.02` (matches the supplement FD step only)" throughout.
8. **`disp` recompute rule fixed.** For N-family/N-scale, `disp` (the
   MAD-based robust scale inside `build_noise_realized_table`) IS
   recomputed per realized-noise table, per the existing recipe — this is
   documented as an intentional, prescribed derived quantity that
   legitimately changes with the injected noise, not an oversight. For
   FD/L-grad, `disp` is NOT recomputed — the frozen baseline's `disp`
   value is reused unchanged (neither axis touches the base label noise).
9. **Noise-diagnostic definitions pinned exactly.** MAD and RMS: computed
   per OUTPUT DIMENSION, over `train_idx` rows, on the realized noise
   array. Clipping fraction: fraction of the full noise-scalar array
   (both output dimensions, all train rows) with `|n_raw| > 5` before
   clipping. Correlation: Pearson correlation between the two
   output-dimension noise columns. `noiseless`'s correlation field is
   JSON `null` (undefined for an all-zero array), never `0.0`.
   **Superseded by v4->v5 item 3**: "the realized noise array" above is
   ambiguous between pre-clip and post-clip; v5 resolves this by naming
   both `n_raw`/`n_applied` explicitly and assigning MAD/RMS/correlation
   to `n_applied`, clipping fraction to `n_raw`. This entry is kept as a
   historical record of the v3->v4 wording, not a currently active
   definition.
10. **`init_state_hash`/shuffle-seed gate added.** Every new model's
    `init_state_hash` must equal the frozen baseline's
    `init_state_hashes` entry for the SAME seed (proving none of
    N-family/N-scale/FD/L-grad perturbs model initialization, as
    expected since `torch.manual_seed(seed)` precedes all of them), and
    `shuffle_seed(k)=k+5000` is asserted literally unchanged (no new
    shuffle scheme introduced by this protocol).
11. **FD `0.02` common-support placeholder convention made explicit.**
    The 232 pairs dropped by the common-support mask are handled by the
    SAME existing placeholder mechanism `build_clean_query_table` already
    uses for its own natively-invalid pairs
    (`ablation_lib.py`: `np.where(valid_mask[...], ..., 0.0)`) — masked-out
    pairs get `valid_mask=False` and `true_ug`/`true_df_du` set to `0.0`,
    not a new convention.
12. **Per-axis reference must be stated in the Results Manifest.**
    N-family/N-scale/L-grad contrasts are against the frozen 2026-08-31
    baseline; FD contrasts are against the new common-support `eps=1e-3`
    reference (never the frozen baseline) — recorded as an explicit
    `reference` field per axis, not left implicit.

Also fixed: a leftover mid-edit artifact ("seed-0 D... (E, not D)") in
the discarded-reconstruction description, now reads cleanly as
"the frozen 2026-08-31 seed-0 E record."

## v4 -> v5 corrections (this revision)

User review of v4 found 4 P0 + 6 completeness-precision issues, all
confirmed directly before being incorporated:

1. **P0 — evaluator gate's "exact" criterion contradicted the upstream
   tolerance contract.** `src/cstr/solver_audit_io.py:47-76` (confirmed
   by direct read) fixes `ISCLOSE_RTOL=1e-5`/`ISCLOSE_ATOL=1e-8` for
   continuous floats and reserves exact (`==`) comparison for booleans/
   integers. **Fixed**: the Section 5 evaluator-compatibility gate now
   compares `final_V`/`max_V` via `numpy.isclose(rtol=1e-5,
   atol=1e-8)` (reusing `solver_audit_io.values_match`/`ISCLOSE_RTOL`/
   `ISCLOSE_ATOL` verbatim, not reinvented) and `settling_time`/`success`
   via exact equality — comparing everything exact risked a false
   `EVALUATOR_WIRING_FAILURE` from ordinary floating-point noise.
2. **P0 — evaluator configuration source was unstated.** **Fixed**:
   `horizon=3`, `budget=20`, `steps=120`, `lr=0.1`, `rho_u=0.01`,
   `success_v=2.0` are read as literal fields from the frozen
   `results/solver_audit_execution_2026_09_01/solver_audit_setup_
   manifest_2026_09_01.json` (confirmed present: `budget=20, horizon=3,
   lr=0.1, rho_u=0.01, steps=120, success_v=2.0`), not re-typed as fresh
   literals in the new driver. Normalization (`norm`) and input bounds
   (`un_lo`/`un_hi`) are not stored as literal arrays in that manifest —
   confirmed by direct read of its keys — so they are reconstructed via
   `make_norm(data_tuple)` from the SAME pool the solver-audit Setup
   Manifest already cross-verifies (`pool_content_sha256_matches_m1`),
   exactly as the primary and lambda_jac-extension drivers already do,
   rather than hand-typed. **Gate target expanded**: BOTH B and E's
   seed-0 checkpoints, across **all 5 ICs each** (10 rollouts total, not
   a single spot-check).
3. **P0 — noise-diagnostic definitions used an incoherent quantity.**
   "Post-injection, pre-clip" does not name a single well-defined array
   (injection and clipping are two separate steps: `n_raw = noise_scale *
   draw`, `n_applied = clip(n_raw, -5, 5)`, and `Yn[train_idx] +=
   n_applied` is the actual injection). **Fixed**: both `n_raw` and
   `n_applied` are stored; MAD/RMS/correlation are computed on
   `n_applied` (what the model actually trains against), and clipping
   fraction remains `|n_raw| > 5` (the pre-clip array, since that is what
   determines whether clipping activated).
4. **P0 — noise companion artifact was promised but never actually
   pinned in the manifest bullets.** Section 6's Setup Manifest/Training
   Record bullets listed `clean` rebuilds, `ug_scale`, `valid_mask`, and
   `oracle_fixed_idx`, but never the noise arrays themselves. **Fixed**:
   a per-`(level, seed)` companion NPZ (`n_raw`, `n_applied`, `Yn`,
   `disp`) is hash-locked for every N-family/N-scale level, and B/E's
   Training Records for the same `(level, seed)` reference the identical
   companion-NPZ hash — the only way the common-random-number claim
   (Section 2) is independently checkable rather than merely asserted.

Completeness precision (non-P0, folded in per the user's framing "not
scope expansion, closing verifiability"):

5. **Failure taxonomy expanded and scoped per-endpoint.** Beyond
   `TRAINING_DIVERGED` (invalidates ALL 8 endpoints for that seed, since
   no model exists): `EVALUATION_FAILED`, a non-finite endpoint value, or
   an incomplete rollout (`n_steps_completed < 120`) each invalidate only
   the endpoints that depend on the closed-loop rollout (`success`,
   `final_V`, `max_V`, `settling_time`) — they do NOT invalidate the
   oracle/test-MSE endpoints (`align_cos`, wrong-sign fraction, relative
   gradient error, test MSE), which never touch the closed-loop
   evaluator. Any of these on either side of a paired contrast makes that
   specific endpoint's contrast (not necessarily all 8) `DESCRIPTIVE_
   CONTRAST_INCOMPLETE`.
6. **Training Records store raw `per_ic` summaries**, not just the 8
   aggregated seed-level endpoints — the 5 per-IC `final_V`/`max_V`/
   `settling_time`/`success` values, **plus `solver_completed`,
   `n_steps_completed`, and `failure_reason`** (mirroring Battery 1's own
   `per_ic[i]["summary"]` schema in full, not a subset — added so the
   incomplete-rollout judgment itself (`n_steps_completed<120`) can be
   independently re-verified from the stored raw record, not merely
   trusted from a derived flag), are persisted so the seed-level
   aggregate can be independently re-derived.
7. **Oracle record expanded**: `excluded_fraction` and the actual
   `n_used` are recorded alongside `align_cos`/wrong-sign
   fraction/relative-gradient-error. `n_used` is stored as the literal
   INTEGER count of non-excluded (boolean `~excluded`) oracle points
   actually used in the median/fraction computation, never derived as
   the floating-point product `800 * (1 - excluded_fraction)` (which
   would round-trip through a lossy division/multiplication instead of
   being the count that was directly summed). Both are
   explicitly computed over the non-excluded sample only (denominator =
   `n_used`, matching `alib.diagnose_fixed_idx`'s own existing
   exclusion convention, not a new one).
8. **Setup Manifest records the execution environment**, mirroring the
   solver-audit Setup Manifest's own schema exactly (`environment`:
   `cpu_architecture`/`numpy_version`/`os_platform`/`processor`/
   `python_version`/`torch_version`, plus `torch_num_threads`/
   `compute_device`) — and the discarded compatibility reconstruction's
   own recorded environment must match the real 80-model execution's
   environment field-for-field before the latter is allowed to proceed.
9. **Freeze Manifest covers the new code's transitive local-import
   closure**, not a hand-curated file list (which risks silently missing
   a real dependency) — computed programmatically from the new driver's
   and module's imports, restricted to repo-internal (`src/cstr`,
   `scripts`) modules. **Canonical entry point restricted to one
   function**, `run_real_sensitivity_protocol()`, which hardcodes
   `seeds=[0,1,2,3,4]` and asserts the Results Manifest accounts for
   exactly 80 evidence-model slots before returning.
10. **`attempt_id` recorded per Training Record.** When a crash-resume
    (Section 6, `attempt_n` logic) causes a model to be retried, the
    Training Record for the ADOPTED attempt records its own `attempt_id`
    and the full checkpoint hash for that specific attempt, so an
    orphaned earlier attempt's checkpoint (left on disk, per the
    no-delete convention) can never be confused with the adopted one.

## 1. What already exists (reuse, not rebuild)

Base noise mechanism (`alib.build_noise_realized_table`), the separate
B/C augmented-target sensitivity arm (untouched by this protocol),
`FD_EPS=1e-3` (`ablation_lib.py:36`), `LAM_G=0.05` (`ablation_lib.py:39`),
`alib.fixed_oracle_idx`/`alib.diagnose_fixed_idx`, `single_cstr_rollout`
(`src/cstr/solver_audit_lib.py:335`).

Loss family is fixed throughout: every model on every axis/level trains
with the same Cauchy NLL loss (`models_onestep.cauchy_nll`). Only the
label-corruption process (N-family, N-scale), the gradient-supervision
construction (FD), or the gradient-loss weight (L-grad) varies — never
the loss function itself.

## 2. Proposed axes

**Baseline** = frozen primary recipe: Cauchy(scale 0.2, clip 5.0),
`FD_EPS=1e-3` (native mask), `LAM_G=0.05`.

### N-family (R2.m8), conditions B and E, 5 seeds -- 30 models

| level | construction | new / reused |
|---|---|---|
| noiseless | `n = 0` | new |
| gaussian | `sigma=0.2/Phi^-1(0.75)=0.29652044...` (MAD-matched to Cauchy baseline: `scipy.stats.norm.ppf(0.75)=0.6744897501960817`), seed `k+7000`, no clip | new |
| cauchy (baseline) | existing mechanism, unchanged, seed `k+1000` | **reused** |
| correlated-gaussian | SAME seed `k+7000` as `gaussian` (common random numbers), same `sigma`, Cholesky-transformed with cross-dimension `rho=0.7`; a single synthetic stress case for positive correlation between the two normalized output dimensions, not a physical-noise claim | new |

`disp` recomputed per realized-noise table (Section v3->v4 item 8).
3 new families x 2 conditions x 5 seeds = **30**.

### N-scale (strategy doc Tier 2 item 6), conditions B and E, 5 seeds -- 20 models

Common-random-number design: the SAME `k+1000`-seeded `standard_cauchy`
draw is reused across all three scales -- only the multiplying scale (and
the resulting `+/-5` clip) differs.

| level | value | new / reused |
|---|---|---|
| lower | `0.1` | new |
| baseline | `0.2` | **reused** |
| higher | `0.4` | new |

`disp` recomputed per realized-noise table. 2 new scales x 2 conditions x
5 seeds = **20**.

### FD common-support (R2.M3 + the real `eps=0.02` discrepancy), condition E only, 5 seeds -- 20 models, ALL new

| level | value | new / reused |
|---|---|---|
| finer | `1e-4` | new |
| common-support reference | `1e-3` | new (differs from frozen E: common-support mask, not native mask) |
| coarser | `1e-2` | new |
| supplement FD step | `0.02` (matches the supplement FD step only, not its full recipe) | new |

All four trained on the common-support mask (== the `0.02` native mask,
per the nesting result verified against the real pool: `mask[1e-4] ⊇
mask[1e-3] ⊇ mask[1e-2] ⊇ mask[0.02]`, `n_pairs_valid/21000` =
20997/20985/20880/20768 respectively). Masked-out pairs use the existing
`0.0`-placeholder convention (Section v3->v4 item 11). `ug_scale` fixed
at the value computed once from the `1e-3` common-support reference,
reused unchanged across all four; `disp` is NOT recomputed for this axis
(reuses the frozen baseline's `disp`, Section v3->v4 item 8). 4 eps
values x 5 seeds = **20**, all new — reference is the new common-support
`1e-3` model, never the frozen baseline (Section v3->v4 item 12).

### L-grad (R2.M3), condition E only, 5 seeds -- 10 models

| level | value | new / reused |
|---|---|---|
| lower | `0.025` | new |
| baseline | `0.05` | **reused** |
| higher | `0.1` | new |

`disp` NOT recomputed (reuses frozen baseline). 2 new values x 1
condition x 5 seeds = **10**.

**Total: 80 evidence models** (30 + 20 + 20 + 10) **+ 1 discarded
training reconstruction + 1 discarded compatibility `clean`
reconstruction.** New `clean` rebuilds: **3 evidence** (`1e-4`, `1e-2`,
`0.02`) **+ 1 discarded compatibility** (`1e-3`, to verify the
parameterized `clean`-builder path itself, Section v3->v4 item 6).

## 3. Seed set

5 seeds (`0,1,2,3,4`). Reused baseline points for N-family/N-scale/L-grad
are read from the frozen `ablation_execution_2026_08_31` M2 files (never
retrained); reused continuous-metric baseline points are read from the
frozen `solver_audit_execution_2026_09_01` Battery-1 records (Section 5).

## 4. Reporting

**Aggregation hierarchy (fixed)**: seed-level summary first, THEN
aggregate the 5 seed-level summaries (mean, `ddof=1` sd, paired bootstrap
CI on `variant - reference` seed-level differences — fixed sign
convention). Never pool raw IC-level or oracle-point-level values
directly across seeds.

**Failure rule (Section v3->v4 item 3, scoping fixed in v4->v5 item 5)**:
`TRAINING_DIVERGED` on either side of a contrast invalidates ALL 8
endpoints for that seed. `EVALUATION_FAILED` / a non-finite endpoint /
an incomplete rollout (`n_steps_completed<120`) invalidates only the
closed-loop-derived endpoints (`success`, `final_V`, `max_V`,
`settling_time`) for that seed, leaving the oracle/test-MSE endpoints
unaffected. Whichever endpoints are invalidated on either side of a
contrast make THAT endpoint's contrast `DESCRIPTIVE_CONTRAST_INCOMPLETE`
(no CI/bootstrap computed for it). Excluding the failed seed and
computing a CI on the remaining seeds is prohibited for any endpoint.

**Eight scalar endpoints** per axis/level: closed-loop success,
`align_cos` median, wrong-sign fraction, relative gradient error median,
test MSE, final `V`, max `V`, failure-penalized settling time. Wrong-sign
fraction and the two medians are computed over the oracle's non-excluded
sample only (denominator = `n_used`, the literal integer count of
non-excluded points actually summed over, never the floating-point
product `800 * (1 - excluded_fraction)` — matching `alib.
diagnose_fixed_idx`'s existing exclusion convention); `excluded_fraction`
and `n_used` are recorded alongside them (v4->v5 item 7).

**Raw per-IC storage (v4->v5 item 6)**: each Training Record also stores
the 5 per-IC `final_V`/`max_V`/`settling_time`/`success`/
`solver_completed`/`n_steps_completed`/`failure_reason` values (Battery
1's own `per_ic[i]["summary"]` schema, in full), not just the seed-level
aggregate, so `final_V`/`max_V`/`settling_time` AND the
incomplete-rollout judgment itself can be independently re-derived
rather than trusted from a stored aggregate/flag alone.

**Noise diagnostics** (N-family/N-scale only): both `n_raw` (pre-clip)
and `n_applied` (post-clip, what is actually injected into `Yn`) are
stored. Per-output-dimension MAD and RMS and cross-output Pearson
correlation are computed on `n_applied`; clipping fraction is computed on
`n_raw` (`|n_raw|>5` before clipping). `noiseless`'s correlation field is
JSON `null`. Plus `disp` (v3->v4 item 8's recompute rule).

**Per-axis reference, stated explicitly in the Results Manifest**:
N-family/N-scale/L-grad -> frozen 2026-08-31 baseline. FD -> the new
common-support `eps=1e-3` reference.

Bootstrap: `alib.bootstrap_ci` reused verbatim -- `BOOTSTRAP_SEED=31415`,
`N_BOOTSTRAP=10000`, a fresh `default_rng(31415)` per contrast.

**Explicit non-claims**: no re-opening of `target_specificity` or
`MSE_ATTRIBUTION`; no full cross-product of axes; 5-seed descriptive
results are not held to the primary 10-seed battery's statistical bar;
an unfavorable result narrows manuscript claims rather than being
omitted.

## 5. Reused-baseline provenance and evaluator compatibility

- N-family/N-scale/L-grad baseline points: frozen `results/
  ablation_execution_2026_08_31/ablation_training_manifest_seed{s}_
  2026_08_31.json`, verified via `verify_training_manifest` against the
  unmodified original Freeze/M1.
- **Fixed oracle index (new)**: loaded from the ORIGINAL Results
  Manifest's own companion file, path and hash taken from that
  manifest's `oracle_fixed_idx_path`/`oracle_fixed_idx_sha256` fields
  (`scripts/ablation_driver_2026_08_31.py:670-682`) -- never
  independently recomputed via `alib.fixed_oracle_idx`, even though that
  would be deterministic, because determinism alone does not prove it is
  the SAME array the original protocol used.
- Continuous closed-loop metrics (final `V`, max `V`, settling time):
  `results/solver_audit_execution_2026_09_01/solver_audit_battery1_
  {condition}_seed{s}_2026_09_01.json`, `per_ic[i]["summary"]`, verified
  via its own `canonical_payload_sha256` and a live re-verification that
  `predecessor_setup_manifest_sha256` still chains to
  `docs/JPC_SOLVER_AUDIT_PROTOCOL_FREEZE_MANIFEST_2026_09_01.json`.
- **Continuous-metric evaluator compatibility gate (v4->v5 items 1-2)**:
  before evaluating any new checkpoint, `single_cstr_rollout` is run on
  BOTH B's and E's already-frozen seed-0 checkpoints, across ALL 5 ICs
  each (10 rollouts). Evaluator configuration (`horizon=3`, `budget=20`,
  `steps=120`, `lr=0.1`, `rho_u=0.01`, `success_v=2.0`) is read as
  literal fields from the frozen `solver_audit_setup_manifest_2026_09_
  01.json` (confirmed present); `norm`/`un_lo`/`un_hi` are reconstructed
  via `make_norm(data_tuple)` from the same pool that manifest's own
  `pool_content_sha256_matches_m1` field already cross-verifies, never
  hand-typed. Comparison against Battery 1's frozen per-IC values uses
  the SAME tolerance contract Battery 1 itself uses
  (`solver_audit_io.ISCLOSE_RTOL=1e-5`/`ISCLOSE_ATOL=1e-8`, reused
  verbatim via `values_match`): `final_V`/`max_V` via `isclose`,
  `settling_time`/`success` via exact equality — comparing floats exact
  would risk a false `EVALUATOR_WIRING_FAILURE` from ordinary
  floating-point noise; comparing counts/booleans via `isclose` would
  under-detect a real wiring bug. This is a real execution-wiring check
  (wrong horizon/budget/lr/rho_u passes a file-hash check but fails this
  one), not merely a manifest-hash verification.

## 6. Manifest chain, ordering, and completion contract

**Required order**: implementation -> synthetic smoke -> the 1 discarded
real compatibility reconstruction (training path: baseline settings,
seed 0, exact-equality check against the frozen seed-0 E record on
`val_mse`/`test_mse`/`init_state_hash`/final-checkpoint-state-dict-hash;
`clean`-builder path: rebuild `eps=1e-3`'s `clean` table with the new
parameterized builder and diff it byte-for-byte against the frozen
original `clean` companion NPZ) -> Freeze Manifest -> Setup Manifest ->
the 80-model execution.

- **Freeze Manifest**: covers this protocol doc, the smoke results file,
  the compatibility reconstruction result JSON, and the new code's
  **transitive local-import closure** (computed programmatically from
  the new driver/module, restricted to repo-internal `src/cstr`/
  `scripts` modules — not a hand-curated file list, v4->v5 item 9);
  records `predecessor_original_ablation_freeze_sha256`,
  `predecessor_solver_audit_freeze_sha256`, and the **compatibility
  reconstruction result JSON's path and hash** (not a bare "PASS"
  label). Freezing is blocked unless that JSON reports all of: training
  `val_mse`/`test_mse`/`init_state_hash`/checkpoint-state-dict-hash exact
  match, `clean`-rebuild byte-for-byte match, AND its recorded execution
  `environment` matches what the real 80-model execution will use
  (v4->v5 item 8), as passed. **Canonical entry point restricted to
  `run_real_sensitivity_protocol()`**, hardcoding `seeds=[0,1,2,3,4]` and
  asserting the Results Manifest accounts for exactly 80 evidence-model
  slots before returning (v4->v5 item 9).
- **Setup Manifest**: hash-locks the 3 evidence `clean` rebuilds
  (companion NPZ each, `pool_content_sha256` cross-check), the fixed
  `ug_scale` for the FD axis, the common-support `valid_mask` array
  (companion NPZ), the pinned `oracle_fixed_idx` path+hash (Section 5)
  copied in verbatim from the original Results Manifest, **a per-`(level,
  seed)` noise companion NPZ for every N-family/N-scale level** (`n_raw`,
  `n_applied`, `Yn`, `disp`; v4->v5 item 4), and the **execution
  environment** (`environment`: `cpu_architecture`/`numpy_version`/
  `os_platform`/`processor`/`python_version`/`torch_version`, plus
  `torch_num_threads`/`compute_device` — mirroring the solver-audit Setup
  Manifest's own schema exactly, v4->v5 item 8).
- **Per-model Training Records**: one per (axis, level, condition, seed)
  new model -- `val_mse`, `test_mse`, `init_state_hash` (gated against
  the frozen baseline's value for the same seed, Section v3->v4 item
  10), checkpoint path+hash, `attempt_id` and that attempt's full
  checkpoint hash (v4->v5 item 10), the 8 scalar endpoints, the 5 raw
  per-IC summaries (v4->v5 item 6), `excluded_fraction`/`n_used` (v4->v5
  item 7), and noise diagnostics (`n_raw`/`n_applied`-derived, v4->v5
  item 3) where applicable. N-family/N-scale Training Records for B and E
  at the same `(level, seed)` reference the identical noise-companion-NPZ
  hash, making the common-random-number claim independently checkable.
  `TRAINING_DIVERGED`/`EVALUATION_FAILED` recorded explicitly, never
  silently dropped; Results Manifest completeness accounts for all 80
  slots.
- **Crash-resume**: reuses the primary ablation driver's own
  "fresh, never-before-used attempt subdirectory per retry" pattern
  (`run_full_protocol`'s `attempt_n` logic) verbatim -- a crash before a
  Training Record is written never collides with a retry's checkpoint
  writes; the partial attempt's checkpoints are left on disk untouched,
  and the adopted attempt's `attempt_id` is recorded so an orphaned
  attempt's checkpoint can never be confused with the adopted one.
- **Results Manifest**: per-axis aggregated tables (Section 4, including
  the `reference` field and per-endpoint `DESCRIPTIVE_CONTRAST_INCOMPLETE`
  handling), bootstrap resample-index arrays (companion NPZ), and a
  `completeness` block (`80/80` or the actual count with per-slot
  outcomes).
- **Resume-or-verify / no-overwrite**: unchanged from the primary
  ablation and lambda_jac-extension drivers.
