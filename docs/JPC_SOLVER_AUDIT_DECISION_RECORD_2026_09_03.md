# JPC Solver Audit (R1.2/R2.M1/R2.m4/R2.M6/R3.8) — Decision Record

Status: FIXED, based on the real, frozen-protocol execution. This record
is descriptive of a completed real result; it is not itself part of the
solver-audit Freeze Manifest's hash-locked file set and may be extended
(never retroactively altered) as follow-up work is completed. Mirrors the
provenance/structure convention of `docs/JPC_ABLATION_DECISION_RECORD_2026_09_01.md`.

## 0. Provenance

- Frozen protocol: `docs/JPC_SOLVER_AUDIT_PROTOCOL_2026_09_01.md` (v11),
  hash-locked by `docs/JPC_SOLVER_AUDIT_PROTOCOL_FREEZE_MANIFEST_2026_09_01.json`
  (`canonical_payload_sha256` = `43a2f92def68d4b432121bd2e558fa5f6684cb007c82cd616eb59462900ceac0`,
  11 files, smoke 128/128). Built after 6 review rounds (5 of which found
  and fixed real P0s; round 6 = `FREEZE_READY`, no new findings).
  Independently `shasum -a 256`-verified against all 11 covered files.
- Zero edits to any of the 12 files frozen by the underlying go/no-go
  ablation's own Freeze Manifest (`docs/JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_FREEZE_MANIFEST_2026_08_31.json`,
  self-hash `19bd6cab...3242a`) — re-verified via `shasum -a 256` after
  every review round and again after real execution.
- Real execution: `run_real_solver_audit()` and
  `run_real_two_cstr_minimum_audit()` (`scripts/solver_audit_driver_2026_09_01.py`),
  both entry points verify the solver-audit Freeze Manifest before running
  and record its hash as `predecessor_solver_audit_freeze_sha256` in their
  respective Setup Manifests. Both Setup Manifests independently confirmed
  to reference the exact freeze hash above.
- Artifacts (`results/solver_audit_execution_2026_09_01/`): single-CSTR
  Setup Manifest, Battery1 (80 records, 8 conditions x 10 seeds) + 1
  Summary Manifest, Battery2 (20 records, B/E x 10 seeds) + 1 Summary
  Manifest, Battery3 (20 records, B/E x 10 seeds) + 1 Summary Manifest;
  two-CSTR Setup Manifest, 20 checkpoints (2 configs x 10 seeds), 1
  Replica Manifest, Battery5 (20 records) + 1 Summary Manifest. 190
  artifacts total, 0 stray/partial files.
- Independent re-verification (this record's author, not the process that
  produced the results): every one of the above manifests re-checked via
  `read_and_verify_json_artifact` (self-hash recomputed from stored
  payload, not read off a label) — all passed. All gate `passed`/
  `mismatches` fields re-aggregated directly from the raw per-record JSON
  (not inferred from the fact that `require_gate_pass()` did not raise):
  Gate1 (Battery2 per-IC rollout match) 100/100, Gate2 (seed-level success
  rate match, Battery1 all 8 conditions + Battery2) 100/100, Gate3
  (Battery3 per-IC GGN metric match) 100/100, Gate4 (Replica Manifest
  reconstruction-vs-legacy, 20 config/seed pairs, re-checked again inside
  Battery5 phase 4) 20/20 + 20/20, Battery5 reload-equivalence (per-IC)
  100/100. **340/340 gate checks passed, 0 failures, 0 non-empty
  `mismatches`.** All condition/seed completion rates = 1.0 (no
  survivorship downgrades triggered). The primary success-rate contrast
  (E vs B) was independently recomputed from raw `per_ic` records via
  seed-first aggregation and matched the Summary Manifest's stored
  contrast exactly (point estimate 0.34, CI [0.14, 0.56]), as did every
  other headline figure in Section 1 below.

## 1. Headline result

The actual-solver audit answers a different question than the go/no-go
ablation: not "does E beat B under an equal query budget" (already
`SUPPORTED`, Decision Record 2026-09-01 Sec. 1) but "when E's advantage is
realized through the ACTUAL projected-Adam/GGN solver at finite budget,
is it visible beyond the binary success endpoint, and is it attributable
to improved local gradient alignment as Proposition 1's mechanism
predicts?"

```
closed_loop_continuous_effect:   SUPPORTED (Battery 1, 6/6 pre-registered
                                  continuous metrics significant, all in
                                  the direction favoring E)
gradient_alignment_bridge:       NOT SUPPORTED (Battery 2, actual-solver
                                  cosine/counterfactual-decrease/
                                  directional-derivative contrasts all
                                  cross zero)
state_constraint_minimum_audit:  COMPLETE (input saturation + box
                                  feasibility + temperature-cap reported
                                  for every rollout, both benchmarks)
two_cstr_extension:              DESCRIPTIVE ONLY (not equal-query; large
                                  directional improvement, constraint
                                  violation NOT resolved)
```

This is the same asymmetry pattern the ablation's own MSE-attribution
finding warned about (`mse_attribution = UNRESOLVED`): a real, robust
closed-loop effect whose first-order mechanistic story does not survive
contact with the actual constrained, finite-budget solver.

## 2. Battery 1 — closed-loop continuous metrics (single-CSTR, E vs B, 10 seeds, seed-first aggregation)

All six pre-registered continuous metrics move in the direction favoring
E, all six are statistically significant at the exact permutation test's
resolution, and all six were independently re-derived from the raw
per-IC records (not merely read off the Summary Manifest):

| metric | B | E | diff (E-B) | 95% CI | p |
|---|---|---|---|---|---|
| success rate | 0.66 | 1.00 | +0.34 | [0.14, 0.56] | 0.0156 |
| final V | 43.64 | 0.18 | -43.46 | [-93.57, -10.73] | 0.00195 |
| settling time (failure-penalized, full 5-IC denominator) | 63.0 | 9.8 | -53.2 | [-76.08, -31.12] | 0.00195 |
| control TV | 17.02 | 5.33 | -11.69 | [-14.66, -9.15] | 0.00195 |
| Lyapunov-increase count | 60.46 | 48.54 | -11.92 | [-21.42, -3.68] | 0.0352 |
| max temperature deviation (K) | 14.91 | 6.88 | -8.03 | [-13.88, -3.40] | 0.00586 |

Failed ICs are penalized (`settling_time = 121`, one beyond the 120-step
horizon), not excluded from the denominator, satisfying R2.m4's request
that settling time not be reported "only over successful ICs." A full
settling-time ECDF (`solver_audit_battery1_summary_2026_09_01.json` ->
`settling_time_ecdf_path`, hash-verified) is available for a
time-to-target distribution figure if needed, also per R2.m4.

Solver-core wall-clock (median, per step, logging overhead excluded):
E-B point estimate = 1.16e-6 s, CI [-1.54e-4, 1.72e-4], p = 0.996 — no
meaningful difference. E's continuous-metric improvement is not bought
with a slower per-step solve.

State-constraint minimum audit (R2.M6/R3.8): `input_box_satisfied`
(genuine `lo-tol <= u <= hi+tol`, distinct from "never saturated")
contrast E-B = 0, p = 1.0 — **zero box violations in either condition,
all 800 (8 conditions x 10 seeds x 5 IC x 2 actuators... rows recorded)
rollouts feasible.** `temperature_cap_satisfied_posthoc` and per-actuator
`input_saturation_frequency` are both recorded per-IC for every condition
(not just B/E), directly filling the "input saturation: absent" /
"feasibility: absent" gap identified in the review response matrix's
prior pass over `two_cstr_constraint.json`.

## 3. Battery 2 — actual-solver gradient-alignment bridge (single-CSTR, E vs B, 10 seeds)

This is the audit R1.2 and R2.M1 ask for by name: does the theorized
mechanism (Proposition 1's local descent-direction improvement) show up
as improved alignment when the ACTUAL projected-Adam solver runs at its
real finite budget, not just in the oracle gradient diagnostics already
reported in the ablation Decision Record (Sec. 2, cosine 0.937 -> 0.993)?

| metric | E-B point estimate | 95% CI | p | reading |
|---|---|---|---|---|
| cosine(true, learned) at the actual solver's iterate | -0.083 | [-0.244, 0.066] | 0.357 | not significant; if anything trends the WRONG way |
| counterfactual true-objective decrease (post-clamp) | -0.0125 | [-0.098, 0.084] | 0.813 | not significant |
| directional derivative, true objective, raw step | +0.00916 | [-0.023, 0.045] | 0.631 | not significant |
| directional derivative, true objective, projected step | -0.00364 | [-0.021, 0.011] | 0.672 | not significant |
| learned-objective decrease (`j_learned_decrease`) | -0.0322 | [-0.077, 0.011] | 0.229 | not significant |
| raw gradient inner product | -5.46 | [-13.46, 0.75] | 0.191 | not significant |

None of the six actual-solver gradient/objective-alignment contrasts
reject the null. This is not merely "underpowered": `g_hat` and `g_true`
(the raw residual magnitudes entering the cosine, reported separately)
DO differ significantly (E-B = -4.66, p = 0.021 for `g_hat`; -2.55,
p = 0.00195 for `g_true`) — E's residuals are smaller in magnitude at the
solver's actual iterates, consistent with E's better closed-loop
trajectory. But the DIRECTIONAL alignment metrics built from the same
iterates do not separate E from B. `projection_active_rate` (E-B =
-0.028, CI [-0.054, -0.001] excludes zero at the 95% bootstrap level, but
the exact permutation p = 0.084 does not clear 0.05 -- reported as-is,
not resolved into a single verdict) hints E's solver spends marginally
less time against the input bound, which is at most a weak, unresolved
secondary observation, not a claim.

**Conclusion for Proposition 1's bridge to the actual solver**: the
oracle-diagnostic gradient improvement reported in the go/no-go ablation
(cosine 0.937 -> 0.993, an idealized one-shot comparison against the true
gradient at fixed sample points) does **not** carry through, at
statistically detectable strength, to the six actual-solver
alignment/objective-decrease diagnostics measured at the real projected-
Adam iterate under the real finite budget. The robust closed-loop
improvement documented in Section 2 is real and independently
reproduced, but this audit does not license attributing it causally to
improved local gradient alignment at the solver's actual working point.

## 4. Battery 3 — GGN diagnostics (single-CSTR, E vs B, 10 seeds, descriptive only, not pre-registered for significance testing)

| metric | B | E |
|---|---|---|
| accepted step fraction | 0.551 | 0.447 |
| mean accepted step size (alpha) | 0.592 | 0.774 |
| mean true residual decrease (accepted steps) | 1.809 | 1.811 |
| true-residual-increased fraction (accepted-but-wrong-direction) | 0.505 | 0.380 |
| singular-solve fraction | 0.0 | 0.0 |

No bootstrap/permutation test was pre-registered for these (same
descriptive-only status as the ablation's oracle diagnostics, Decision
Record 2026-09-01 Sec. 8 item 1). E accepts fewer GGN steps overall but
has a lower true-residual-increased-despite-acceptance rate and a larger
mean accepted step -- directionally consistent with E's better closed-loop
outcome, but this is GGN's own internal accept/reject bookkeeping, not
the same quantity as Battery 2's cosine/objective-decrease bridge, and is
reported here as supplementary color, not as a second confirmation of
the mechanism claim.

## 5. Battery 5 — two-CSTR minimum audit (descriptive, NOT equal-query, auxiliary only)

Explicitly out of scope for the equal-query causal claim (this comparison
reuses the two-CSTR value-only vs. value+grad checkpoints from an earlier
study, not a budget-matched pair) -- included only as a second-benchmark
directional check, consistent with how it was scoped in the frozen
protocol.

| metric | value_only | value+grad |
|---|---|---|
| success rate (mean of 10 seeds) | 0.20 | 0.96 |
| temperature-cap satisfied (posthoc, mean) | 0.26 | 0.58 |
| worst-case max temperature deviation, reactor 1 (K, true max over all 50 rollouts, not seed-averaged) | 361.9 | 125.0 |
| worst-case max temperature deviation, reactor 3 (K) | 288.2 | 112.2 |

Large directional improvement on both success and temperature-cap
satisfaction. **Constraint violation is not resolved**: even value+grad's
worst-case deviation (125.0 K over a 70 K cap) is far outside bounds.
This must be reported as "large directional improvement, constraint
satisfaction NOT achieved" -- not as evidence the state-constraint problem
(R2.M6/R3.8) is solved.

## 6. Defensible manuscript-level conclusion (current evidence ceiling)

> Under an equal offline identification-query budget, residual
> action-gradient consistency training produces a robust, multi-metric
> improvement in the ACTUAL closed-loop behavior of a finite-budget
> learned Lyapunov MPC (success rate, terminal Lyapunov value, settling
> time, control smoothness, and peak constraint violation all improve
> significantly, at no solver-timing cost, with zero input-box
> violations). However, a dedicated actual-solver audit -- measuring
> gradient cosine alignment and true-objective decrease at the solver's
> real projected-Adam iterates, not at idealized oracle sample points --
> does not detect a statistically significant improvement in local
> gradient alignment. The causal mechanism proposed in Proposition 1
> (improved local descent-direction quality) is therefore not established
> as the explanation for the observed closed-loop gain; the paper's
> claim must separate the (well-supported) empirical closed-loop result
> from the (not established) causal mechanism.

This does not weaken Proposition 1 further than the 2026-09-01 Decision
Record's narrowing already did (Prop 1 remains "local descent-direction
preservation," a conditional/local statement) -- but it does mean
Proposition 1 can no longer be presented even informally as "the reason"
the closed-loop numbers improved. The paper's defensible structure is now
two separate, honestly-labeled claims: a theorem (local, conditional) and
an empirical closed-loop result (robust, multi-metric) -- with an
explicit, reported negative finding that the actual-solver bridge between
them is not supported by data.

## 7. Mapping to review response matrix rows

- **R1.2** ("explain how the proposed gradient consistency condition
  applies to the actual solvers and objectives") -- the audit this
  comment asks for has been run. The honest answer is: it applies at the
  level of the resulting residual magnitude (`g_hat`/`g_true` both
  significantly smaller under E) and the resulting closed-loop
  trajectory (Section 2), but NOT at the level of directional alignment
  or objective-decrease at the solver's real iterate (Section 3). Status
  update: `PARTIALLY RESOLVED` -> `RESOLVED` in the sense that the
  requested analysis now exists and is reported; the finding itself is
  mixed, not favorable, and must be written up as such.
- **R2.M1** (present Prop 1 as local + empirical, contingent on real
  projection-activity/accepted-step evidence) -- that evidence now
  exists (Battery 2/3) and is reported. Status update: `PARTIALLY
  RESOLVED` -> `RESOLVED`, with the caveat text above.
- **R2.m4** (failure-penalized settling time / ECDF, not success-only
  ICs) -- resolved by Battery 1's `settling_time` (121-step penalty for
  failed ICs) and the persisted ECDF artifact. Status update: `NOT YET
  ADDRESSED` -> `RESOLVED`.
- **R2.M6 / R3.8 minimum bar** (input saturation + explicit feasibility
  statement) -- resolved by `input_box_satisfied`, `input_never_
  saturated`, `input_saturation_frequency`, and `temperature_cap_
  satisfied_posthoc`, recorded per-IC for every condition in both
  benchmarks. Status update: `PARTIALLY RESOLVED` -> `RESOLVED` for the
  minimum bar (the active-constraint/CLBF EXTENSION remains
  `OPTIONAL/DEFERRABLE`, unchanged).

## 8. Next steps

1. **This document** — fixed as of 2026-09-03, no modification to any of
   the 11 solver-audit frozen files or the 12 ablation frozen files.
2. Update `docs/JPC_REVIEW_RESPONSE_MATRIX_2026_09_01.md` in place (its
   own stated convention, being a living control document) to reflect
   Section 7 above, and re-rank the "Summary of real gaps" list — item 1
   (actual-solver audit) is now done; **R2.M4's oracle-gradient /
   converged-NLP comparison becomes the top remaining item**, since it is
   the only way to determine whether the Section 3 negative bridge result
   reflects a genuine finite-optimizer-budget bottleneck or a more
   general surrogate-model mismatch.
3. Design (not yet implement) the oracle-gradient / converged-NLP
   comparison protocol per R2.M4, reusing the existing B/E checkpoints
   (no retraining), scoped to single-CSTR only for this round.
4. Only after 2-3: `lambda_jac` grid extension (D), FD/noise/weight
   sensitivity sweep, and the R3.4 auxiliary-controller freshness check.

Items 3-4 are new work, not yet started; each needs its own explicit
go-ahead before implementation, per this project's standing convention.

## 9. Precision addendum (2026-09-03, post-review)

Four wording/count corrections identified on a code-verified review pass,
appended rather than edited in place per this document's own stated
policy (Sections 0-8 remain exactly as originally written).

1. **"800 ... rollouts" (Section 2) is wrong; the real count is 400.**
   8 conditions x 10 seeds x 5 ICs = 400 rollouts. 800 only results from
   also counting each rollout's 2 actuators as a separate unit (400 x 2
   actuator-channel checks) -- that is a channel count, not a rollout
   count, and must not be conflated with it.
2. **"6/6 pre-registered continuous metrics" (Section 1, 2) overstates
   the word "continuous."** The six endpoints are `success` (binary),
   `final_V`, `settling_time`, `tv`, `lyapunov_increase_count` (a count,
   not continuous), and `max_temperature_deviation_K`. The precise
   description is "six prespecified closed-loop endpoints," not "six
   continuous metrics." Additionally: no multiple-comparison correction
   was applied across these six exact-permutation tests -- each p-value
   in Section 2's table is an unadjusted per-endpoint exact p < 0.05,
   which must be stated explicitly if the six-endpoint agreement is used
   as evidence in the manuscript (the same caveat the go-no-go ablation's
   own Decision Record applies to its `secondary_family` tests).
3. **"This is not merely 'underpowered'" (Section 3) is an unsupported
   inference and must be deleted.** Significance on five OTHER endpoints
   (success, final_V, settling_time, tv, lyapunov_increase_count,
   max_temperature_deviation_K) does not establish that the Battery 2
   alignment/objective-decrease tests specifically had adequate power --
   these are different quantities computed from different (iteration-
   level vs. rollout-level) data with different variances. The correct,
   power-neutral statement is simply: the six actual-solver alignment/
   objective-decrease contrasts (Section 3) do not reject the null: no
   claim about why (power vs. a genuine null effect) is licensed by this
   data alone.
4. **"at no solver-timing cost" (Section 6) interprets a non-significant
   difference as equivalence, which a single non-significant contrast
   cannot establish.** Correct phrasing: "with no detectable timing
   increase" (point estimate 1.16e-6 s, CI [-1.54e-4, 1.72e-4], p=0.996)
   -- consistent with, but not proof of, no true difference.

None of these four corrections change any headline number, gate result,
or the Section 6/7 conclusions -- they are wording/count precision fixes
only, caught on independent re-review of the exact phrasing against the
underlying data and record counts.

## 10. Precision addendum, round 2 (2026-09-03, post second review)

Two further corrections, again appended rather than edited in place.

1. **Section 9 item 3's "five OTHER endpoints" is a counting error.**
   The sentence itself lists six: `success`, `final_V`, `settling_time`,
   `tv`, `lyapunov_increase_count`, `max_temperature_deviation_K`. Read
   "six OTHER endpoints" wherever Section 9 item 3 says "five."
2. **Section 7's combined "R2.M6 / R3.8 minimum bar" bullet incorrectly
   extends `OPTIONAL/DEFERRABLE` framing to R3.8.** Per
   `docs/JPC_REVIEW_RESPONSE_MATRIX_2026_09_01.md` (which has this
   right): `OPTIONAL/DEFERRABLE` applies only to R2.M6's "preferably"
   active-temperature-constraint/CLBF extension. R2.M6 itself does have
   a legitimate minimum-bar structure that the new data satisfies
   (RESOLVED for the minimum bar). **R3.8 has no such minimum-bar
   structure** -- it is a flat "critical limitation" statement -- and
   must stay `PARTIALLY RESOLVED` regardless of the minimum-bar data now
   existing, since the underlying constraint violation (two-CSTR
   worst-case 125 K over a 70 K cap) is not resolved. Section 7's bullet
   should be read as applying its RESOLVED-for-the-minimum-bar /
   OPTIONAL-extension structure to R2.M6 only; R3.8's status is
   `PARTIALLY RESOLVED` as stated in the Review Response Matrix, not
   modified by this Decision Record.
