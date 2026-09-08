# D's `lambda_jac` grid extension — Decision Record (2026-09-03)

## 0. Provenance

- Protocol: [`docs/JPC_LAMJAC_GRID_EXTENSION_PROTOCOL_2026_09_03.md`](JPC_LAMJAC_GRID_EXTENSION_PROTOCOL_2026_09_03.md)
- Freeze Manifest: [`docs/JPC_LAMJAC_GRID_EXTENSION_FREEZE_MANIFEST_2026_09_03.json`](JPC_LAMJAC_GRID_EXTENSION_FREEZE_MANIFEST_2026_09_03.json),
  self-hash `3c7aa5f8fb2972b3cd04b87e4163a4968261a523e39daaf80494c65df759b003`
  (independently `shasum -a 256`-verified against all 3 covered files;
  its own `predecessor_original_freeze_sha256` field verified live against
  the still-unmodified 2026-08-31 Freeze Manifest).
- Smoke: `results/ablation_lamjac_extension_smoke_results_2026_09_03.json`,
  11/11 checks passed, including one real end-to-end 1-seed x 1-lambda run
  against the real frozen pool/simulator (not a fabricated fake).
- Results Manifest: [`results/ablation_lamjac_extension_2026_09_03/ablation_lamjac_extension_results_2026_09_03.json`](../results/ablation_lamjac_extension_2026_09_03/ablation_lamjac_extension_results_2026_09_03.json),
  self-hash `66e22ff8b1ac884991572ccbf7ac0e91454f819c7e7abef280255e7a24ebff9c`
  (re-derived from raw payload, independently re-verified, not trusted
  from its own recorded field).
- The original `results/ablation_execution_2026_08_31/` artifacts and
  Freeze Manifest were **not modified**; this run only read and
  hash-verified them.

## 1. What was executed

Extended `alib.LAM_JAC_GRID` (originally `[0.001,0.005,0.01,0.05,0.1,0.5]`)
with `{1.0, 2.0, 5.0}`. Trained exactly 30 new D models (10 seeds x 3
lambda), reusing the frozen `pool`/`clean`/`scale` (verified byte-identical
to the original 60-model run via `verify_query_manifest`) and the frozen
per-seed RNG/init-hash setup (`init_state_hash_consistent_with_original`
confirmed `True` for all 10 seeds). Final `lambda_jac` selected over the
combined 9-point grid using **validation MSE only** (never test MSE,
closed-loop success, or oracle metrics) — confirmed by direct code
reading and by the `selection_ignores_closed_loop_and_test_mse_fields`
smoke check.

## 2. Boundary outcome

7/10 seeds (0,1,2,3,4,6,8) again selected the new top of the grid,
`lambda_jac=5.0` -> `BOUNDARY_PERSISTS`. 3/10 seeds (5,7,9) selected
`lambda_jac=1.0` -> `INTERIOR_OPTIMUM_FOUND`. Overall outcome: `MIXED`.
Per the pre-registered contract, **the grid is not extended again** —
this result is reported as-is, including its own limitation (below).

## 3. Closed-loop success: a ceiling, not equivalence

All 10 seeds' finally-selected D(extended) model reached closed-loop
success `1.0` (5/5 ICs), exactly matching E's `1.0` on all 10 seeds
(re-verified against the frozen 2026-08-31 `closed_loop_success["E"]`
values). The resulting contrast is degenerate: `point_estimate=0.0`,
95% CI `[0.0, 0.0]`, exact permutation `p=1.0` (`d_extended_vs_e_contrast`/
`d_extended_vs_e_permutation` in the Results Manifest, both independently
recomputed from the raw per-seed values above, not merely read off the
manifest's own field).

**This must not be reported as "D and E are equivalent."** The primary
closed-loop endpoint (5 fixed ICs, budget=20, horizon=3) has a ceiling of
`1.0`; both conditions saturating it removes the endpoint's power to
discriminate between them at all — the same ceiling-effect confound
already documented for R3.3's B-FNN condition. A CI of exactly `[0,0]`
here is an artifact of that saturation, not evidence of no underlying
difference.

Additionally: because 7/10 seeds again selected the new boundary
(`5.0`), **D's true optimum — and therefore its true best-achievable
closed-loop success and gradient fidelity — remains unknown and
right-censored.** This grid extension answers "does D catch up to E's
closed-loop success once fairly tuned past the original artificial
ceiling" (yes, on this saturating endpoint) but does **not** answer "what
is D's best possible performance" (still open, and not being pursued
further per the pre-registered stopping rule).

## 4. Where D and E still differ: oracle gradient fidelity (descriptive)

Computed on the same fixed 800-point oracle set (`meta_seed=90210`) used
everywhere else in this protocol family, comparing E against each seed's
*selected* D(extended) candidate:

| metric | mean(D−E) | direction | seeds where E wins |
|---|---|---|---|
| `align_cos` median | −0.0326 | E higher | 10/10 |
| wrong-sign fraction (`frac_neg`) | +0.0168 | E lower | 10/10 |
| relative gradient error, median | +0.327 | E lower | 10/10 |
| test MSE | +7.10e-5 | mixed | 6/10 (D worse) |

E's `align_cos` median ranges `0.9727`–`0.9989` across seeds (corrected
from an earlier, less precise verbal range) and is above D(extended)'s on
every single seed. Wrong-sign fraction and relative gradient error show
the same uniform pattern. Test MSE is the one metric that does **not**
uniformly favor E — the two are close and the sign is mixed across seeds,
unlike the much larger, one-sided MSE gap E showed against B.

## 5. Revised claim

**Withdrawn**: "residual action-gradient target E is superior to the
general Jacobian/Sobolev target D on closed-loop performance." Not
supported once D is fairly (validation-only) retuned past its original
artificially narrow grid — both reach the same saturated closed-loop
success on this benchmark's primary endpoint.

**Retained, reframed as descriptive**: E shows consistently better oracle
action-gradient fidelity (alignment, wrong-sign fraction, relative error)
than a validation-selected D, on all 10 seeds — reported as an observed
association, not as evidence that this fidelity difference is the unique
cause of any closed-loop performance gap (the closed-loop endpoint here
cannot show such a gap at all, being saturated for both).

**Explicit prohibitions for the manuscript/response letter**:
- Do not cite the `CI=[0,0]`/`p=1.0` contrast as "D and E are
  equivalent" — it is a ceiling artifact of this specific 5-IC endpoint,
  not a general equivalence result.
- Do not claim D's optimal tuning or best-achievable gradient fidelity
  has been established — 7/10 seeds are still boundary-censored.
- Do not use the Oracle-20/Certified-NLP result
  (`docs/JPC_ORACLE_NLP_COMPARISON_DECISION_RECORD_2026_09_03.md`,
  `gap_E≈0.18` vs `gap_B=43.6`) as evidence for E-vs-D superiority — that
  comparison never included a D controller; it bears only on E-vs-B.
- Do not describe E's gradient-fidelity advantage over D as a
  "target-specific" or "uniquely causal" result — report it descriptively.

**Suggested response-letter language** (addresses R1.4/R2.M3's fair
Jacobian-matching-baseline request directly, including the unfavorable
part):

> Extending the Jacobian-loss grid revealed that the original comparison
> had truncated the baseline. After validation-only retuning, the generic
> Jacobian baseline reached the same saturated closed-loop success as the
> residual-gradient model. We therefore removed the claim of closed-loop
> target specificity. The residual-gradient model nevertheless retained
> consistently better oracle action-gradient fidelity across all ten
> seeds, which we report descriptively rather than as evidence of unique
> causal superiority.

## 6. Next step

Bundle 2 (FD-step / `lambda_grad` / noise-family sensitivity, R2.M3 +
R2.m8) is next. The Review Response Matrix's R1.4/R2.M3 rows and the
"Summary of real gaps" list need updating to reflect this record before
that.
