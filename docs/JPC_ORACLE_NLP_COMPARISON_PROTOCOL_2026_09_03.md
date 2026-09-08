# Oracle-Gradient / Converged-NLP Comparison — Protocol Draft v12

Status: **IMPLEMENTED. A preliminary real execution (2026-09-03,
217s wall-clock) was run under a v11 freeze that turned out to have a
doc-vs-code semantic mismatch (Sec. 23) -- that execution and its Freeze
Manifest are PRESERVED as a historical/diagnostic record
(`docs/JPC_ORACLE_NLP_COMPARISON_PROTOCOL_FREEZE_MANIFEST_2026_09_03_PRELIMINARY.json`,
`results/oracle_nlp_execution_2026_09_03_preliminary/`), not treated as
confirmatory.** This v12 revision corrects the mismatch and closes two
further implementation gaps found during the same review (Sec. 23);
a corrective re-freeze and full re-execution (including the previously-
deferred timing pass) follow before any Decision Record is written.
Ten design-review rounds preceded implementation (v1-v10, Sections
17-21); v10 declared design review complete, v11 finalized Section 14
into the Sec. 14.1 schema table and marked all thresholds "CONFIRMED,"
which is precisely why the v11-vs-code drift found in Sec. 23 matters:
a "confirmed, must not relax post-hoc" value was already wrong the day
it was written down.

## 0. Motivation and the question this protocol must answer

The solver-audit real execution (`docs/JPC_SOLVER_AUDIT_DECISION_RECORD_2026_09_03.md`
Sec. 3) found a robust closed-loop improvement (E over B) with a
NON-significant actual-solver gradient-alignment bridge. This protocol
does not cleanly separate the "finite optimizer-budget" explanation from
the "surrogate-mismatch" explanation for E's advantage over B -- it
**triangulates/decomposes** the B/E-vs-truth gap into two additive
components (`optimizer_gap`, `gap_B`/`gap_E`, Sec. 12) that bear on, but
do not conclusively isolate, either explanation. Oracle-20 (Sec. 7)
replaces the entire learned model (dynamics, objective, gradient) with
the true model at once -- it cannot isolate the gradient contribution
alone. **No output of this protocol, under any outcome, may be cited as
evidence for or against a gradient-specific mechanism.**

## 1. Scope

Single-CSTR only. No retraining -- B/E checkpoints already produced and
gate-verified by the solver audit (`results/solver_audit_execution_2026_09_01/`)
are read-only inputs. Explicitly deferred: two-CSTR extension, active-
state-constraint study, `lambda_jac` grid extension, FD/noise/
`lambda_grad` sensitivity sweep, any modification to B/E checkpoints.

## 2. Four conditions

| condition | dynamics/objective | gradient source | solver | budget | reference type |
|---|---|---|---|---|---|
| B (reused) | learned value-only surrogate | autograd, learned model | projected Adam | 20 | 10-seed distribution |
| E (reused) | learned value+grad surrogate | autograd, learned model | projected Adam | 20 | 10-seed distribution |
| Oracle-20 (new) | true dynamics, true objective (`J_true`, Sec. 6) | exact AD of `J_true` | projected Adam, identical loop to B/E (Sec. 7) | 20 | fixed, 5 deterministic IC trajectories |
| Certified-NLP (new) | true `J_true` | exact AD, IPOPT-internal | IPOPT to certified local stationarity (Sec. 8-9) | not budget-limited; 3 fixed deterministic multi-starts | fixed, 5 deterministic IC reference values |

B and E are not re-run for their Battery 1 continuous metrics; only a
separate timing pass (Sec. 12) re-executes a subset of them.

## 3. Shared setup and literal constants (all confirmed by direct code read)

- `P = [[1060.0, 22.0], [22.0, 0.52]]`, `INPUT_LO = [-3.5, -5.0e5]`,
  `INPUT_HI = [3.5, 5.0e5]`, `RHO_U = 0.01`, `SUCCESS_V = 2.0`,
  `HORIZON = 3`, `STEPS = 120`, `BUDGET = 20`, `LR = 0.1`
  (`scripts/solver_audit_driver_2026_09_01.py:61`).
- Input normalization: `un_lo = (INPUT_LO - u_mean) / u_std`,
  `un_hi = (INPUT_HI - u_mean) / u_std`, `u_mean, u_std` = the pool's
  input mean/std (`_shared_norm()`/`make_norm()`,
  `solver_audit_driver_2026_09_01.py:433-447`).
- Simulator: `LCNNPaperCSTRSimulator`, `shifted=True`
  (`scripts/budget_sweep_solver.py:90`), `dt_hr=1e-3` h = **3.6 s**
  real-world sampling period, `integration_substeps=100`
  (`lcnn_paper_simulator.py:54-55`).
- Objective: `J(U_n) = sum_{h=0}^{H-1} [ y_{h+1}^T P y_{h+1} + RHO_U *
  ||u_{n,h}||^2 ]`, confirmed against `_adam_mpc_step`
  (`src/cstr/solver_audit_lib.py:274-325`).
- 5 initial conditions (`bs.ICS`).
- Warm-start scheme is **deterministic** (confirmed,
  `single_cstr_rollout`, `solver_audit_lib.py:342,387`): `un =
  torch.zeros(horizon, 2)` at rollout start, `un = torch.cat([un_next[1:],
  un_next[-1:]])` between control steps. No randomness anywhere in the
  existing B/E algorithm.

## 4. Build/freeze/execution ordering

1. Implementation + synthetic smoke suite.
2. **Preflight phase** -- produces exactly two new immutable artifacts
   (Sec. 5): `oracle_nlp_compat_sample_2026_09_XX.npz` (fixed sample,
   including the measured `un_lo`/`un_hi`/quartile-start values, Sec.
   8) and `oracle_nlp_preflight_gate_results_2026_09_XX.json` (all six
   gate results, including the upstream identities Gate 4 depends on,
   Sec. 5.3).
3. **This protocol's Freeze Manifest** -- directly includes both
   preflight artifacts' file hashes in its own covered-file list (like
   the solver audit's Freeze Manifest pins source files directly), plus
   all new code files and this design document. Independently
   `shasum -a 256`-verified. No further code or preflight-artifact
   changes after this point without re-freezing.
4. **Runtime Setup Manifest** -- built at the start of real execution;
   references this protocol's Freeze Manifest hash, the solver-audit's
   Freeze Manifest hash, B/E Battery 1 record hashes, and RE-VERIFIES
   the same upstream identities Gate 4 used (checkpoint hash, pool/
   normalization hash, simulator config) still match (evidence-aware
   resume pattern, Sec. 5.3 item 4).
5. **Runtime Compatibility Gate Manifest** -- does NOT recompute the
   preflight gates. Re-hashes the two frozen preflight artifacts and
   asserts their hashes match the Freeze Manifest's record -- a pure
   hash-chain integrity check.
6. Rollout execution (Oracle-20, Certified-NLP) and the timing pass
   (Sec. 12), subject to the Sec. 6 runtime floor-inactivity invariant.
7. Summary Manifest (Sec. 14).
8. Decision Record (append-only, post-execution).

## 5. Preflight compatibility gates (six gates, run once before freeze)

### 5.1 Fixed compatibility sample

5 ICs (`bs.ICS`) x 3 fixed control vectors (zero, `un_lo`, `un_hi`) = 15
one-step test points; 5 ICs each paired with the all-zero `U_n` plan = 5
multi-step test points. This exact index list, the resulting array, and
its content hash are written into `oracle_nlp_compat_sample_2026_09_XX.npz`
BEFORE any gate runs -- fixed at design time, never selected post-hoc.

### 5.2 Error-metric definition

For two arrays/scalars `a, b` (vector or scalar): **`err(a,b) = max_i (
|a_i - b_i| / max(1, |a_i|, |b_i|) )`** -- the outer `max_i` applies to
the WHOLE per-component ratio, not just the numerator (parenthesization
made explicit per P1 review feedback: the division is computed
component-wise FIRST, then the maximum is taken over components). The
`max(1, ...)` denominator floor gives an absolute-error regime when
`|a|,|b| < 1` and a relative-error regime otherwise, avoiding both
division blowup and spurious pass/fail from scale mismatch.

### 5.2a Tolerance-rule scope (v9 fix -- two rules had drifted into
overlapping, inconsistently-named use)

**`err()` (Sec. 5.2) is used ONLY for gates 1-3** -- comparisons between
TWO DIFFERENT COMPUTATIONAL PATHS that could differ in scale for
legitimate reasons (symbolic CasADi vs. numeric Python; AD vs. FD),
where a relative-error regime is the right tool.

**`values_match(new, old, exact)` (`solver_audit_io.py:62-77`,
`ISCLOSE_RTOL=1e-5`, `ISCLOSE_ATOL=1e-8` when `exact=False`), applied
via the `flatten_leaves()` wrapper (Sec. 5.3 gate 4) to any list/nested-
dict-valued field, is used for EVERYTHING ELSE**: Gate 4's summary AND
per-step comparison, Gate 6's reproducibility check, and BOTH endpoint-
equivalence gates (Oracle-20 Sec. 7, Certified-NLP Sec. 9) -- these all
compare the SAME implementation run twice (a fresh run vs. a frozen one,
or a clean pass vs. an instrumented one), which is exactly the
regression/reproducibility scenario `values_match()`'s existing,
already-frozen tolerance convention was designed for. Any prior mention
of `err<=1e-8`/`err<=1e-6` in this document outside gates 1-3 is
superseded by this rule -- e.g. Sec. 9's endpoint-equivalence gate and
Gate 4's per-step reconstruction both use `values_match()`, not `err()`.

### 5.3 The six gates

1. **Symbolic-vs-numeric one-step**: CasADi one-RK4-step function vs.
   `LCNNPaperCSTRSimulator.step()`, `err<=1e-8` on all 15 fixed points.
2. **Symbolic-vs-numeric `J_true`**: `J_true(x0,U_n)` vs.
   `counterfactual_true_objective()` (`solver_audit_lib.py:135`),
   `err<=1e-8` on all 5 fixed multi-step points.
3. **AD-vs-FD gradient**: CasADi AD gradient of `J_true` vs. central FD
   (fixed `eps=1e-5`), `err<=1e-4` on all 5 points, using the SAME
   `err()` formula elementwise on the 6-dim gradient. Both AD and FD
   gradients must be entirely finite before `err()` is computed -- a
   non-finite value on either side is an automatic gate failure.
4. **Oracle-20 harness reproduction gate**: Oracle-20's harness,
   TEMPORARILY fed autograd through condition B's seed-0 checkpoint
   instead of the true AD gradient, must reproduce `single_cstr_rollout`'s
   own frozen B-seed-0 rollout at BOTH the rollout-summary level AND the
   per-step level (v8 fix -- see the "per-step strengthening" note
   below), using a **flattening comparator built on the existing
   `values_match()` primitive, not a new tolerance algorithm** (v8 fix --
   `values_match(..., exact=False)` only applies `np.isclose` when BOTH
   inputs are `numbers.Real` scalars; for a Python list (e.g.
   `input_tv_per_actuator_physical`) it silently falls through to exact
   `list == list` comparison, and for a numpy array it would raise
   `ValueError` on `bool(array)` -- confirmed by direct re-read of
   `solver_audit_io.py:62-77`. Neither behavior is what a per-actuator or
   per-temperature-index "isclose" comparison needs):
   - **`flatten_leaves(obj, prefix="")`**: a small new recursive helper
     (built ON TOP of, not replacing, `values_match()`) that walks a
     dict/list-valued structure and yields `(dotted_key, scalar_value)`
     pairs at every leaf -- e.g. `input_tv_per_actuator_physical` (a
     2-list) yields `input_tv_per_actuator_physical.0`,
     `input_tv_per_actuator_physical.1`; `state_constraint.per_
     temperature_index` (a `{str(idx): {...}}` dict of dicts) yields
     `state_constraint.per_temperature_index.1.max_temperature_deviation_K`
     etc. Each `(new_leaf, old_leaf)` pair is then compared via the
     EXISTING `values_match(new_leaf, old_leaf, exact=<per-key
     designation>)` -- the only new code is the flattening walk, not a
     new comparison rule.
   - **EXACT fields** (leaf-level, `exact=True`): `success`,
     `solver_completed`, `failure_reason`, `failed_at_step`,
     `n_steps_completed`, `lyapunov_increase_count`; within
     `state_constraint`: `input_box_satisfied`, `input_never_saturated`,
     `temperature_cap_satisfied_posthoc`, `denominator_note`.
   - **NUMERIC-ISCLOSE fields** (leaf-level, `exact=False`, via
     `flatten_leaves` for the list/dict-valued ones): `final_V`,
     `max_V`, `settling_time`, `settling_time_hours`, `tv`,
     `input_tv_per_actuator_physical.{0,1}`,
     `input_tv_per_actuator_normalized.{0,1}`,
     `state_constraint.input_saturation_frequency.{actuator key}`,
     `state_constraint.per_temperature_index.{index key}.{cap_absolute_K,
     cap_deviation_K, max_temperature_absolute_K,
     max_temperature_deviation_K, temperature_violation_frequency,
     temperature_violation_integral}`. Wall-clock fields are EXCLUDED
     entirely (this is a correctness gate, not a timing gate, Sec. 12).
   - Implementation shape: `flatten_leaves()` produces the leaf pairs;
     `gate_compare_dict()`-style bookkeeping (`solver_audit_io.py:80-105`,
     its `MISSING_FIELD`-not-`KeyError` behavior reused conceptually)
     collects mismatches over the flattened leaf set, calling
     `values_match()` per leaf with the exact/isclose designation fixed
     above -- `gate_compare_dict()` itself is not modified, only fed
     pre-flattened scalar leaves instead of the raw nested structure it
     was not designed to handle.
   - **Per-step strengthening (v9 fix -- corrected; the v8 version could
     not check the last of 120 steps)**: comparing only rollout-level
     summary fields cannot detect a harness bug where intermediate
     actions/states differ but happen to land on the same final value by
     coincidence. **v8's approach of using "the next step's `x_before`"
     as that step's realized `x_after` cannot cover the 120th (last)
     step, since there is no 121st record -- and the existing Battery 2
     `step_records` do not store `x_after` at all (confirmed by direct
     re-read: only `x_before`, `u0`, `warm_start_un`, `final_plan_un`,
     `V_before`, `V_after_realized`, `realized_decrease` are stored,
     `solver_audit_lib.py`'s `single_cstr_rollout` step-record `dict`).**
     Fixed approach: for EVERY one of the 120 stored `(x_before, u0)`
     pairs (including the 120th), independently RECONSTRUCT
     `x_after_ref = sim.step(x_before, u0)` using the fixed, deterministic
     `LCNNPaperCSTRSimulator` -- this covers all 120 transitions
     uniformly, with no dependency on a "next record" that may not
     exist, and no reliance on anything the original run did not already
     store. Compare `u0`, `final_plan_un` (isclose, elementwise via
     `flatten_leaves`), the reconstructed `x_after_ref` against Oracle-
     20's own realized next state (isclose elementwise), and
     `V_after_realized` (isclose) -- against the frozen B-seed-0 Battery
     2 per-step NPZ arrays. This is a strictly stronger check than the
     rollout-summary-only comparison and directly targets the failure
     mode Finding 3 (round 6) originally described, now without the
     last-step gap.
   **Upstream identities this gate depends on** (v9 fix -- extended;
   the checkpoint/pool/simulator identities alone do not cover the
   Battery 2 evidence this gate actually reads per-step): checkpoint
   file path + SHA-256, pool/normalization file path + SHA-256,
   simulator config (`dt_hr`, `integration_substeps`, `shifted`), **AND,
   newly added: the specific Battery 2 JSON record's path + self-hash
   (`solver_audit_battery2_B_seed0_2026_09_01.json`), its companion
   per-step NPZ's path + SHA-256, and a cross-check that the NPZ hash
   RECORDED INSIDE the JSON record matches the NPZ file's actual current
   hash (defense against silent NPZ drift undetected by the JSON's own
   self-hash), AND the upstream solver-audit Setup/Freeze Manifest hash
   these Battery 2 records themselves descend from** -- all recorded
   explicitly in `oracle_nlp_preflight_gate_results_2026_09_XX.json`, so
   the Freeze Manifest (step 3) pins them and the Runtime Setup Manifest
   (step 4) can re-verify none have silently changed.
5. **Certified-NLP harness unit-regression gate**: candidate
   initialization (Sec. 8), generic deduplication (Sec. 8), certification
   (Sec. 9), and winner-selection logic, exercised against a FIXED
   SYNTHETIC test problem `J_test(U) = ||U||^2` over the same box
   (hand-computable optimum at the box point nearest zero) -- confirms
   the harness's control logic independent of the true CSTR dynamics.
   **This gate is not just a single "success path" smoke check (v7 fix,
   P1) -- it must separately exercise, each as its own synthetic smoke
   case**: (a) the dedup path (construct a synthetic box where two of
   the three starts are made to coincide by construction, confirm only
   one solve runs and `starts_deduplicated` is logged correctly); (b) a
   forced non-whitelisted `return_status` (e.g. mock/force
   `Maximum_Iterations_Exceeded` on one candidate) confirming it is
   correctly rejected regardless of whether it would numerically pass
   the KKT/feasibility checks; (c) an uncertified-candidate fallback
   case (force the lowest-`J_true` candidate to fail certification,
   confirm the harness correctly falls back to selecting the next-best
   CERTIFIED candidate rather than the globally-lowest-objective one).
6. **Certified-NLP reproducibility gate (v10 fix -- tolerance rule
   aligned with Sec. 5.2a, which this gate had been left out of)**:
   solving the SAME fixed synthetic problem (gate 5) twice must
   reproduce: `return_status` EXACTLY (`values_match(..., exact=True)`)
   for every candidate; `selected_start_id` EXACTLY; and `J_test`
   (the synthetic problem's objective at the winner) via
   `values_match(..., exact=False)` (`ISCLOSE_RTOL=1e-5`,
   `ISCLOSE_ATOL=1e-8`, Sec. 5.2a) -- NOT the mixed `err()` formula
   (which is reserved for gates 1-3's cross-implementation comparisons),
   and not a bare `1e-12` numeral, which was a leftover from before
   Sec. 5.2a's scope rule existed.

All six gates are evaluated ONCE, before freeze, against the fixed
Sec. 5.1 sample and the fixed synthetic problem -- never re-evaluated
per real rollout, and never dependent on any Oracle-20/Certified-NLP
real execution having happened.

## 6. `J_true` and the runtime floor-inactivity invariant

**`J_true`**: explicit inline unroll of 300 RK4 integration steps (100
substeps/control-step x `HORIZON=3`), each with 4 RHS/stage evaluations
(`k1-k4` per `_rk4_single`) = **1,200 RHS evaluations total**, matching
`step()` exactly, including `temp = ca.fmax(T, 1.0)` (the true RHS,
`rhs_absolute`, `lcnn_paper_simulator.py:117`, is NOT fully smooth --
this floor must be represented, not omitted).

**Floor-inactivity invariant**: a fail-closed RUNTIME check (not a
preflight gate -- it cannot be evaluated before real trajectories
exist). For any `J_true` evaluation, define `minimum_stage_temperature`
= the minimum of the 1,200 RHS-evaluation input temperatures, and
`argmin_stage` = which of the 1,200 evaluations achieved it (NOT the
full 1,200-length array -- storing only the min and its location is
sufficient to determine `floor_active = (minimum_stage_temperature <=
1.0 + margin)`, `margin` OPEN, proposed `1.0` K).

**Consequence, by condition (v6 fix -- was unspecified in v5)**:
- **Oracle-20**: if ANY of the 20 Adam iterations' gradient-evaluation
  points in a given control step is `floor_active`, the ENTIRE control
  step's optimizer trajectory is contaminated (the kink affects the
  gradient used for that step's update), which propagates forward
  through the deterministic warm-start chain to every subsequent step.
  **The WHOLE IC's rollout is therefore marked reference-INVALID**
  (Sec. 10) if `floor_active` occurs at any step of that IC's rollout --
  not just the affected step.
- **Certified-NLP**: a candidate whose FINAL returned plan is
  `floor_active` is **disqualified from winner selection** (folded into
  the certification criteria, Sec. 9) -- if another certified,
  non-floor-active candidate exists among that step's 2-3 (post-dedup)
  solves, it is selected instead. If ALL certified candidates for a step
  are `floor_active`, that step has no valid winner and is treated as an
  uncertified step under the Sec. 10 failure contract.
- Every `floor_active` occurrence (by condition, IC, step, candidate) is
  logged as an explicit count in the Summary Manifest -- never silently
  absorbed, whether or not it changes the final reference-validity
  outcome.
- A failure of the invariant CHECK ITSELF to execute (e.g. an exception
  computing `minimum_stage_temperature`) is an
  `IMPLEMENTATION_OR_VALIDATION_BLOCKED` condition (Sec. 13), distinct
  from the invariant correctly detecting and flagging a floor-active
  point (a normal, reportable occurrence).

## 7. Oracle-20 controller

Structurally identical to `_adam_mpc_step` (`solver_audit_lib.py:274-325`):
1. CasADi float64 AD gradient of `J_true`, cast to `torch.float32`,
   manually assigned to `un.grad` (replacing `J.backward()`); everything
   downstream (`opt.step()`, clamp) is unchanged code.
2. A FRESH `torch.optim.Adam([un], lr=LR)` every control step (confirmed
   `_adam_mpc_step` creates a new optimizer per call -- no cross-step
   momentum).
3. `opt.step()` BEFORE the clamp (`un.clamp_(un_lo, un_hi)` under
   `torch.no_grad()`), exact order preserved.
4. Preflight Gate 4 (Sec. 5.3) validates this harness's correctness
   before its true-gradient output is trusted.

Deterministic warm start (Sec. 3) means exactly 5 fixed trajectories, no
seed axis.

**Clean-vs-instrumented timing split** (v6 fix -- floor-monitoring
computation itself could contaminate a naive single-pass timing
measurement): a **clean pass** (`clean_online_solve_wall_clock_seconds`,
no floor-monitoring bookkeeping beyond the minimum needed to determine
`floor_active` post-hoc from already-computed intermediate values, no
extra logging) is used for the deadline comparison (Sec. 12) AND doubles
as the reference rollout for gap computation (Sec. 12a) -- it is
executed exactly ONCE, not twice (see Sec. 12's execution-ownership
clarification). An **instrumented pass** (full per-iteration logging:
`un_before`, `un_projected`, `j_true` (v7 fix: renamed from `j_learned`,
which was a copy-paste artifact from B/E's own learned-model field name
-- Oracle-20 has no "learned" quantity, only `J_true`), floor-check
diagnostics) is a SEPARATE, additional execution used ONLY for
diagnostic reporting -- never for the reference/gap values or the
timing numbers.

**Endpoint-equivalence gate (v9 fix -- tolerance unified, field list
completed)**: comparing only rollout-level `success`/`final_V`/
`floor_active` cannot catch a bug where intermediate actions differ but
happen to coincide at the final value. The gate instead requires, for
every one of the 5 ICs' 120 control steps: the applied action `u0`, the
FULL final plan `final_plan_un` (v9 fix -- was listed in the v8
corrections summary but omitted from this section's actual field list),
the realized next state `x_after`, `V_after`, and `minimum_stage_
temperature` (Sec. 6) to match between the clean and instrumented
passes -- via `flatten_leaves`+`values_match(..., exact=False)`
(`ISCLOSE_RTOL=1e-5`, `ISCLOSE_ATOL=1e-8`, Sec. 5.2a's unified tolerance
rule), NOT the mixed `err()` formula (Sec. 5.2, reserved for gates 1-3's
cross-implementation comparisons only) -- not merely the rollout's
final aggregate values.

## 8. Certified-NLP controller — deterministic starts, corrected assertion scope

U-only single shooting: decision variable `U_n in R^{HORIZON x 2}`,
`J_true(x0, U_n)` minimized directly, subject only to
`un_lo <= U_n <= un_hi`.

**Three fixed, deterministic starting VECTORS (2-dim, one component per
actuator), each converted to a full `(HORIZON, 2)` starting PLAN by
tiling identically across all 3 horizon blocks** (`plan[h,:] =
start_vector` for every `h in {0,1,2}` -- this tiled `plan`, not the bare
2-vector, is what is actually passed to `opti.set_initial(U_n, plan)`,
Sec. 9):
1. **Zero**: `start_vector = [0, 0]`.
2. **Warm-shifted**: the previous control step's accepted FULL PLAN
   (already `(HORIZON,2)`-shaped), shift-repeated (Sec. 3's rule) --
   this one is NOT re-tiled from a 2-vector, since it already carries a
   genuine per-horizon-block plan from the prior solve. At a rollout's
   FIRST control step, no previous solution exists, so this equals
   start 1's tiled zero plan -- an EXPECTED, INTENTIONAL degeneracy at
   that specific step, not an error.
3. **Upper-quartile corner**: `start_vector = un_lo + 0.75 * (un_hi -
   un_lo)`, tiled identically across all 3 blocks.
   **Measured for this frozen pool** (direct numeric computation against
   the real `xm`/`xs` normalization): `un_lo = [-1.7333, -1.7350]`,
   `un_hi = [1.7278, 1.7440]`, quartile start `= [0.8625, 0.8743]`. This
   is an EMPIRICAL property of the current pool, not a general
   mathematical guarantee -- recorded in `oracle_nlp_compat_sample_2026_09_XX.npz`
   (Sec. 5.1), pinned by the Freeze Manifest.

**Startup assertion (corrected scope, v7 P1 addition)**: checked ONCE,
at protocol startup, BEFORE any rollout: `||zero - quartile||_inf >
tol_dedup_input`, AND `quartile` is strictly within `(un_lo, un_hi)`
component-wise, **AND `zero` (the all-zeros vector) is also strictly
within `(un_lo, un_hi)` component-wise** (v7 fix -- v6 only checked
quartile's box-interior containment; zero must be checked too, since a
future re-derived normalization could in principle shift the box such
that zero itself falls outside it, which would silently make start 1
an infeasible initial guess). **This assertion covers ONLY the two
STEP-INVARIANT fixed vectors (zero, quartile)** -- it does NOT check
"all three starts are pairwise distant," since start 2 (warm-shifted)
is EXPECTED to coincide with start 1 at the first control step by
design. Any collision INVOLVING warm-shifted, at any step, is handled
entirely by the per-step generic deduplication rule below, not by this
startup assertion.

**Generic deduplication (every control step, every pair, v6 unchanged
from v5)**: for every pair of the 3 candidate vectors, if
`||start_i - start_j||_inf < tol_dedup_input` (`tol_dedup_input=1e-9`,
dimensionless normalized-input units), only ONE IPOPT solve is run for
that pair, logged as `starts_deduplicated: [i,j]` for that step.

Winner selection: lowest `J_true` among CERTIFIED, non-floor-active
candidates (Sec. 6, Sec. 9); tie-break (`J_true` equal to
`tol_tie_objective=1e-9`, cost/objective units, a DIFFERENT constant
from `tol_dedup_input`): lowest-numbered start wins (zero > warm-shifted
> upper-quartile).

## 9. Convergence certification and IPOPT execution contract

**v6 fix -- restored an explicit status whitelist, ANDed with the
self-computed criteria (v5 had dropped `return_status` from
certification entirely, which would have made a solver-flagged
`Maximum_Iterations_Exceeded` or `Infeasible_Problem_Detected` run
eligible if it happened to pass the numeric checks -- not defensible to
a reviewer)**:

A candidate is **CERTIFIED** (and eligible to WIN) iff ALL of:
1. **`return_status == Solve_Succeeded`** (v9 fix -- narrowed from the
   v8 whitelist `{Solve_Succeeded, Solved_To_Acceptable_Level}`; see the
   rationale below). Any other status (`Solved_To_Acceptable_Level`
   included, plus `Maximum_Iterations_Exceeded`,
   `Infeasible_Problem_Detected`, `Restoration_Failed`, etc.) is an
   automatic non-certification for WINNER-SELECTION purposes,
   REGARDLESS of what the self-computed checks below would say.
2. **Primal feasibility**: finite, within `[un_lo-tol_primal,
   un_hi+tol_primal]`, `tol_primal=1e-6` (v12 fix -- see Sec. 23 item 1;
   distinct from IPOPT's own `ipopt_tol` option, Sec. 9's execution
   contract, since the two are conceptually different quantities).
3. **Projected-gradient stationarity**: `r(U*) = ||U* - Proj_box(U* -
   alpha*grad_J(U*))||_inf <= tol_kkt` (`alpha=1.0`, `tol_kkt=1e-6`).
4. **Finite trajectory**: no NaN/Inf anywhere.
5. **Not `floor_active`** (Sec. 6) -- a candidate whose final plan
   touches the temperature floor is disqualified from certification
   regardless of 1-4.

**`Solved_To_Acceptable_Level` candidates are DESCRIPTIVE-ONLY, never
winner-eligible (v9 fix -- the safer of two options, adopted per
review)**: v8 had made `Solved_To_Acceptable_Level` winner-eligible
contingent on pinning `acceptable_iter`/`acceptable_tol`, but IPOPT's
"acceptable" convergence is actually governed by a WIDER set of
options -- acceptable dual infeasibility, constraint violation,
complementarity, and objective-change tolerances -- not just those two,
and leaving any of them at version-dependent implicit defaults would let
an IPOPT version change silently alter what counts as an acceptable
solution. Rather than enumerate and pin the full `acceptable_*` option
set now, the SAFER default is adopted: **only `Solve_Succeeded`
candidates are certified/winner-eligible.** `Solved_To_Acceptable_Level`
candidates still pass through primal feasibility/KKT/finite/floor checks
2-5 and have their `return_status` and `would_be_certified_if_status_
relaxed` boolean (plus their own `J_true`) RECORDED, per candidate, for
transparency -- but they cannot be selected as the winner and do not
affect `optimizer_gap`/`gap_B`/`gap_E`. **v10 fix (P1)**: since the full
`acceptable_*` option set is not pinned, the RAW COUNT/frequency of
`would_be_certified_if_status_relaxed==True` across a run is NOT itself
a stable, reproducible statistic (it could shift with an IPOPT version
change along with the unpinned options) and must not be reported or
interpreted as a descriptive finding (e.g. "X% of candidates would have
been acceptable") -- only the raw per-candidate record is kept, for
debugging/transparency, not for analysis. If a full `acceptable_*`
option enumeration is done later and this restriction is relaxed, that
is a separate, explicit design change, not a default assumed here.
`acceptable_iter` (IPOPT default `15`) and
`acceptable_tol=1e-6` are still explicitly pinned and recorded into the
Runtime Setup Manifest's IPOPT-
options block (Sec. 14), so a future IPOPT version change cannot
silently alter what "acceptable" means for this protocol's results
without it being visible in provenance.

**IPOPT execution contract**: `max_iter=500`, `ipopt_tol=1e-8` (v7:
renamed from `tol` to avoid clashing with `tol_primal` above -- this is
IPOPT's own internal convergence tolerance option, a different
quantity from the post-hoc primal-feasibility check even though both
are numerically `1e-8`), `acceptable_tol=1e-6` (explicitly looser than
and distinct from `ipopt_tol`), Hessian = exact (only `HORIZON*2=6`
decision variables), `linear_solver=mumps`, `mu_strategy=adaptive`. **No
`warm_start_init_point` option** -- this would imply passing dual
(multiplier) warm-start data, which this design does not construct for
any start (including "warm-shifted," which only carries a primal
initial guess). Every start uses a plain primal-only initial guess
(`opti.set_initial(U_n, plan)`, where `plan` is the EXPLICIT
`(HORIZON, 2)` tiled array, Sec. 8 -- never the bare 2-vector
`start_vector` itself, which is not shape-compatible with the decision
variable `U_n`), matching `ipopt_lmpc_baseline.py`'s own existing
convention. Per-step logging: `iter_count`, IPOPT's reported line-search
trial counts, `return_status`, all candidates' `J_true` values (not just
the winner's), `certified` boolean per candidate, `floor_active`
boolean per candidate, `starts_deduplicated`.

**CasADi/IPOPT object construction and reuse contract (v7 -- missing in
v6)**: `J_true`, its AD gradient function, and the parameterized `Opti`
NLP (with `x0` as an `opti.parameter()` and `U_n` as the decision
variable) are each built **exactly ONCE, at the start of real
execution** -- mirroring `ipopt_lmpc_baseline.py`'s own `make_solver()`
pattern (built once outside its closed-loop step loop, reused every
step via `opti.set_value(x0p, x)`/`opti.set_initial(...)`). Rebuilding
the `Opti`/solver object per candidate or per control step would itself
be a large, uncontrolled timing confound and must not happen. Between
candidates/steps, only `opti.set_value(x0p, x_current)` and
`opti.set_initial(U_n, plan)` change; the underlying `Opti` object,
its symbolic graph, and the configured IPOPT options persist unchanged
for the entire rollout (and across ICs, re-parameterized only by
`x0p`). **After a failed/uncertified solve**: the SAME `Opti` object is
reused for the next candidate's attempt (CasADi's `Opti` holds no
solve-to-solve state beyond what `set_value`/`set_initial` explicitly
set) -- no reset or reconstruction is needed or performed between
candidates, only fresh `set_value`/`set_initial` calls for the next
candidate's starting point.

**Clean-vs-instrumented split applies to Certified-NLP too**: a **clean
pass** runs all post-dedup candidates with only the logging strictly
required to certify and select a winner (no `iter_count`/line-search
capture) -- this clean pass DOUBLES as the reference rollout used for
gap computation (Sec. 12a), executed exactly ONCE (Sec. 12's execution-
ownership clarification), not re-run separately for timing. An
**instrumented pass** is a SEPARATE, additional execution with full
per-step logging (the "Per-step logging" list above), used only for
diagnostic reporting -- never for reference/gap values or timing.

**Endpoint-equivalence gate (v9 fix -- tolerance unified to Sec. 5.2a)**:
"IDENTICAL `J_true` values" is not a well-posed contract for two
independently-run IPOPT solves (floating-point non-determinism across
runs, even single-threaded, is possible at the level of the solver's
internal iteration path) -- replaced with an explicit, per-field rule,
checked for every (IC, step): `certified` booleans and `selected_
start_id` must be EXACTLY equal (discrete fields, `values_match(...,
exact=True)`); `J_true` at the selected winner, the winner's final plan
`U_n*`, the realized next state `x_after`, and `minimum_stage_
temperature` must match via `values_match(..., exact=False)`
(`ISCLOSE_RTOL=1e-5`, `ISCLOSE_ATOL=1e-8`, Sec. 5.2a) through
`flatten_leaves()` -- NOT the mixed `err()` formula, which is reserved
for gates 1-3's cross-implementation comparisons. If any of these
diverge, the extra instrumentation itself is perturbing the solve
(e.g. a different code path triggered by logging), which must be
resolved before either pass's numbers are trusted. `clean_online_solve_wall_
clock_seconds` (Sec. 12) is measured ONLY in the clean pass; the
instrumented pass's own wall-clock is reported separately and never
used for the deadline comparison.

## 10. Failure / termination / reference-validity contract

First uncertified control step (Sec. 9) stops the rollout immediately --
no fallback action computed or applied. Record fields match Battery 1's
own failure contract exactly (`single_cstr_rollout`,
`solver_audit_lib.py:404-421`): `success=False`, `settling_time=
steps+1=121`, `failed_at_step=<index>`, `n_steps_completed=<count
strictly before the failure>`. The identical contract applies to
Oracle-20 if its Adam solve ever produces a non-finite iterate, OR if
`floor_active` occurs anywhere in that IC's rollout (Sec. 6).

**Reference validity vs. controller success, explicitly separated, AND
defined per condition (v8 fix -- Sec. 9's "certified" concept is an
IPOPT/Certified-NLP notion and does not apply to Oracle-20, which has no
solver-status/KKT-residual object at all; v7's single shared definition
was not applicable to Oracle-20 as written)**:
- **Oracle-20 reference validity** (per IC): all 120 control steps
  completed, EVERY one of the 20 Adam iterations' iterates at every step
  is finite (no NaN/Inf anywhere in the optimization trajectory), and
  the rollout is non-floor-active throughout (Sec. 6).
- **Certified-NLP reference validity** (per IC): all 120 control steps
  completed, and at EVERY step the SELECTED WINNER candidate passed
  Sec. 9's full certification (status whitelist + feasibility + KKT +
  finite + non-floor-active).

Both are purely procedural/numerical criteria, **completely independent
of whether `success` (`final_V<=2.0`) is True or False** for those
completed rollouts. A reference-valid but poorly-performing rollout
(`success=False`) is a VALID reference data point for either condition.

**Aggregate consequence**: if any IC fails reference validity for
Oracle-20 or Certified-NLP, that condition's outcome is
`REFERENCE_NOT_ESTABLISHED` (Sec. 13) -- no partial-IC averaging.

## 11. (reserved -- merged into Sec. 7-10 above in v6; no content, number retained for cross-reference stability with prior review rounds)

## 12. Timing pass — unit accounting, execution ownership, clean/instrumented split

**Unit count (v8 -- reduced-subset proposal WITHDRAWN, full 10-seed
re-run ADOPTED)**: 22 controllers (10 B-seeds + 10 E-seeds + Oracle-20 +
Certified-NLP) x 5 ICs = **110 rollouts, all executed**. Prior drafts
proposed re-running only 3 seeds each of B/E to save compute; this is
withdrawn on sign-off: Battery 1's own measured `solver_core_wall_clock_
median_seconds` is on the order of ~7 ms/control-step, so the 70
rollouts a 3-seed subset would have skipped (7 seeds x 2 conditions x 5
ICs) cost approximately `70 x 120 x 7ms ~= 59s` -- roughly one minute,
small relative to Oracle-20/Certified-NLP's own cost (Sec. 6's 1,200
RHS-evaluation `J_true` calls, and Certified-NLP's multi-candidate IPOPT
solves) and not worth the added complexity of a documented, sign-off-
gated exception to "re-run everything." All 10 B/E seeds are therefore
re-executed for the timing pass, using exactly the same raw-per-IC
reproduction gate as before but now against the FULL 100 raw Battery-1
records (`solver_audit_battery1_{B,E}_seed{0-9}_2026_09_01.json`'s
`per_ic` entries), not a 30-record subset.

**Execution ownership (v8 fix -- Finding 4, resolves an ambiguity
between Sec. 14's rollout records and this section's timing record that
could have been read as re-running Oracle-20/Certified-NLP's clean
rollout twice)**: Oracle-20 and Certified-NLP each have exactly TWO
executions total, not three: (1) the **clean pass**, run once per IC,
which SIMULTANEOUSLY serves as (a) the reference rollout whose `final_V`
etc. feed Sec. 12a's gap computation, and (b) the source of
`clean_online_solve_wall_clock_seconds` for this timing pass -- there is
no separate "reference rollout" execution distinct from the clean
timing pass; they are the same execution, reported into two consuming
manifests (Oracle-20/Certified-NLP rollout records AND the timing-pass
record, both referencing the SAME underlying run via a shared record
hash, Sec. 14). (2) The **instrumented pass**, a genuinely separate,
additional execution, run purely for diagnostic logging (`iter_count`,
line-search trials, full per-iteration records) and consumed ONLY by
the endpoint-equivalence gates (Sec. 7, 9) -- never by gap computation
or timing. **B and E, by contrast, ARE re-executed specifically and
solely for this timing pass** (their non-timing endpoints already exist,
frozen, from the original Battery 1 execution, and are not recomputed
here) -- B/E have no "instrumented pass" of their own in this protocol.

**Execution, exact iteration order (v10 fix -- added deterministic
rotation; v9's fixed controller order left B-seed0 always first and
Certified-NLP always last, a systematic cache/thermal-position bias
across the whole campaign)**: fix the 22-controller list ONCE
(`controllers = [B-seed0, ..., B-seed9, E-seed0, ..., E-seed9,
Oracle-20, Certified-NLP]`, index `0..21`), then for each `(step, ic)`
pair (`step` the outer loop 0-119, `ic` the inner loop over `bs.ICS`'s
index 0-4), compute a deterministic rotation offset `r = (step * 5 +
ic) % 22` and execute the 22 controllers in the ROTATED order
`[controllers[(r+k) % 22] for k in range(22)]` for that `(step, ic)` --
every controller still takes its turn at every `(step, ic)` (round-robin
coverage is unchanged), but WHICH controller goes first/last cycles
deterministically across the campaign instead of being fixed, so no
single controller systematically benefits from or is penalized by a
warm cache/thermal state the others didn't also
just experience. Single-threaded
(`configure_deterministic_single_thread()`), same `TIMING_WARMUP_STEPS`
convention as Battery 1 applied per-controller (the first N steps of
EACH controller's own timing series are excluded, not a single global
warmup window across the whole nested loop).

**Timing fields, disambiguated (v7 -- split what v6 collapsed into one
`graph_build_wall_clock_seconds` field, per P1 review feedback)**:
- `symbolic_build_wall_clock_seconds`: one-time cost of constructing the
  `J_true` CasADi symbolic graph and its AD gradient function (Sec. 6),
  measured once before any solving happens.
- `solver_construction_wall_clock_seconds`: one-time cost of building
  the parameterized `Opti` object and configuring IPOPT options (Sec.
  9's object-reuse contract) -- a DIFFERENT one-time cost from symbolic
  construction, kept as its own field since the two could scale
  differently (e.g. if IPOPT's setup cost depends on options not on
  graph size).
- `first_call_warmup_wall_clock_seconds` (v8 fix -- precise definition,
  was under-specified): a **throwaway dry-run solve/optimization at a
  FIXED, dedicated `(IC, plan)` pair -- the first of `bs.ICS` paired
  with the all-zero `U_n` plan (the same pair already used by Sec. 5.1's
  compatibility sample, reused here rather than inventing a new fixed
  point) -- run ONCE before any real IC's rollout begins, and DISCARDED
  (not counted as, and not reused as, that IC's actual first control
  step)**. After this dry run, the controller/solver object's `set_
  value`/`set_initial` state is reset to the REAL first IC's actual
  `x0` and warm-start plan (Sec. 3/8) before real rollouts begin -- the
  dry run's own result never leaks into any real IC's trajectory or
  warm-start chain. This isolates JIT/compilation-cache effects from
  contaminating either the graph-build cost above OR the first REAL
  IC's own per-step timing.
  All three of the above are one-time costs, reported separately, and
  NEVER included in any per-step timing figure, for both Oracle-20 and
  Certified-NLP.
- `clean_online_solve_wall_clock_seconds` (per control step): for
  Oracle-20, the clean pass's 20-iteration Adam solve time (Sec. 7);
  for Certified-NLP, the SUM of ALL post-dedup candidate solves +
  certification-check compute + selection overhead for that step
  (Sec. 8-9) -- since the winner is not known until all candidates are
  solved and certified, this sum, not just the winner's solve time, is
  the real online cost. THIS is the figure compared against the 3.6 s
  deadline.
- `TIMING_WARMUP_STEPS` excludes the first N control steps from both the
  median/p95 summary and the deadline-exceedance fraction.
- Deadline-exceedance fraction denominator = the number of TIMED
  (post-warmup) control steps for that condition/controller.

**B/E reproduction gate (v8 -- full 10-seed count, per the withdrawn-
reduction sign-off above)**: before this pass's new timing numbers are
trusted, the fresh same-session run's `success, final_V, max_V,
settling_time, tv, lyapunov_increase_count, input_tv_per_actuator_physical,
input_tv_per_actuator_normalized`, and the full `state_constraint` block
must match the corresponding RAW per-IC values in the frozen Battery 1
records for **all 10 seeds x {B,E} x 5 ICs = 100 raw records**
(`solver_audit_battery1_{B,E}_seed{0-9}_2026_09_01.json`'s `per_ic`
entries -- not the Summary Manifest's aggregates), using the
`flatten_leaves`+`values_match()` comparator defined in Sec. 5.3 Gate
4 (`src/cstr/solver_audit_io.py:62-105`).

## 12a. Gap definitions, aggregation order, decomposition identity

**Primary endpoint**: `final_V` (lower is better, cost-like).

**Aggregation, explicit order**:
- `optimizer_gap = mean_IC( Oracle20_final_V[IC] - Certified_NLP_final_V[IC] )`
  -- per-IC difference first, then averaged over the 5 ICs.
- `oracle_mean = mean_IC( Oracle20_final_V[IC] )` (fixed, shared across
  all seeds, computed once).
- `gap_B(seed) = mean_IC( B_final_V[seed, IC] ) - oracle_mean` (average
  B's 5 ICs for that seed FIRST, then subtract the fixed Oracle mean --
  mirrors Battery 1's seed-first convention). `gap_E(seed)` analogous.

**Mandatory decomposition-identity check** (implementation-level sanity
assertion, blocks reporting on failure -- not a mathematical footnote):
for each seed and each of `{B, E}`:
```
(gap_X(seed) + optimizer_gap) == mean_IC(X_final_V[seed,IC]) - mean_IC(Certified_NLP_final_V[IC])
```
to within `1e-9`. Failure indicates an implementation bug (e.g. an
IC-indexing mismatch across the B/Oracle/NLP arrays) and blocks
reporting entirely -- this is not a substantive finding to interpret.

**Secondary descriptive reporting**: `success`, `settling_time`, `tv`,
`lyapunov_increase_count`, `max_temperature_deviation_K` (Battery 1's
own endpoint set) are ALSO computed for Oracle-20 and Certified-NLP and
reported descriptively alongside the primary `final_V` analysis.

## 13. Quantified outcome procedure — descriptive-primary, margin-based labels demoted to sensitivity-only (v8 sign-off)

**v8 sign-off decision**: `margin_V=0.1` has no defensible domain
justification (it was an arbitrary 5%-of-`SUCCESS_V` placeholder). Per
explicit user sign-off, the PRIMARY report for each of `optimizer_gap`,
`gap_B`, `gap_E` is now the raw quantities themselves -- point estimate,
per-IC dispersion (min/max range across the 5 ICs, descriptive), 95% CI
(`gap_B`/`gap_E` only, via `bootstrap_ci()`, Sec. 15), and Holm-adjusted
two-sided exact p-value (`gap_B`/`gap_E` only) -- NOT a categorical
margin-based label. This is more defensible to a reviewer than a
categorical verdict resting on an unjustified threshold.

**Primary report (no `margin_V` involved)**:
- `optimizer_gap`: point estimate, per-IC range (`max_IC - min_IC` of
  the per-IC `Oracle20-NLP` differences feeding the mean).
- `gap_B`, `gap_E`: point estimate (seed-mean), 95% CI (percentile
  bootstrap, `BOOTSTRAP_SEED=31415`, `N_BOOTSTRAP=10_000`), Holm-adjusted
  exact two-sided p-value (`exact_permutation_pvalue()`, Sec. 15), sign
  (`gap>0` = learned/Oracle-20 worse than the reference it is compared
  to; `gap<0` = better).

**Secondary, OPTIONAL descriptive sensitivity classification** (retained
only as a supplementary sensitivity note, never the primary reported
result, and never gating any conclusion): IF a `margin_V` is later
given an actual domain justification (not fabricated here), the same
3-category (`optimizer_gap`) / 4-category (`gap_B`/`gap_E`) partition
from prior drafts may be computed AS A SENSITIVITY CHECK ONLY, clearly
labeled as such and reported alongside, never in place of, the raw
numbers above.

**Top-level precedence** (checked in this exact order):
1. **`IMPLEMENTATION_OR_VALIDATION_BLOCKED`** if ANY of: a preflight
   gate (Sec. 5) failed; the freeze-time verification (Sec. 4) failed;
   a provenance/hash/resume-chain failure occurred at any manifest stage
   (Sec. 14); OR the runtime floor-inactivity invariant ITSELF failed to
   execute (an exception computing `minimum_stage_temperature`, as
   opposed to correctly detecting and flagging a floor-active point,
   which is a normal, reportable, non-blocking occurrence). STOP -- no
   further outcome is computed.
2. **`REFERENCE_NOT_ESTABLISHED`** if Oracle-20 or Certified-NLP fails
   reference validity (Sec. 10, condition-specific definitions) for any
   IC. STOP -- no gap is computed.
3. Otherwise: report `optimizer_gap`, `gap_B`, `gap_E`'s point
   estimates/dispersion/CI/Holm-adjusted p (the primary report above),
   the decomposition-identity check's pass/fail (Sec. 12a), and the
   secondary descriptive endpoints (Sec. 12a). **No forced single
   dominance verdict or categorical margin-based label is the primary
   output** -- three independently reported quantities; synthesis is
   left to descriptive prose in the eventual Decision Record. **No
   report, under any combination, may be described as confirming or
   restoring a gradient-specific mechanism** (Sec. 0).

## 14. Provenance / execution manifest chain

Per-stage detail (v8 fix -- Freeze-Blocking precision item: each stage
now states its exact keyset, expected count, predecessor callback, and
companion-artifact hash requirement, matching the solver audit's own
manifest-design depth; this remains a schema SKETCH, not final code,
and still needs its own dedicated provenance-focused review pass, Sec.
16 item 6 -- v11 fix: corrected from a stale "item 8" reference, Section
16 currently has 6 items):

1. **Preflight artifacts**: `oracle_nlp_compat_sample_2026_09_XX.npz`
   (fixed sample array + measured `un_lo`/`un_hi`/quartile values, Sec.
   5.1, 8; expected count: exactly 1 file), `oracle_nlp_preflight_gate_
   results_2026_09_XX.json` (exact keyset: one entry per of the 6 gates
   in Sec. 5.3, each with `passed: bool`, `measured_error` or
   `measured_values`, and Gate 4's FULL upstream-identity fields --
   **v10 fix (P0): the Battery 2 evidence Gate 4 actually reads was
   pinned in Gate 4's own prose (Sec. 5.3) but never propagated into
   this keyset** -- `checkpoint_path`, `checkpoint_sha256`, `pool_path`,
   `pool_sha256`, `simulator_config`, AND `battery2_json_path`,
   `battery2_json_self_hash`, `battery2_npz_path`,
   `battery2_npz_sha256`, `battery2_npz_hash_cross_check_passed: bool`
   (comparing the NPZ hash recorded INSIDE the JSON against the NPZ
   file's actual current hash), `upstream_solver_audit_setup_manifest_
   sha256`, AND `upstream_solver_audit_freeze_manifest_sha256` (v11 fix
   -- Gate 4's prose promises both the upstream Setup AND Freeze
   Manifest hashes, but only the Setup field had made it into this
   keyset; added the Freeze field directly rather than relying on
   transitive verification through the Setup Manifest alone, the safer
   of the two options); expected count: exactly 1 file, exactly 6 gate
   entries).
2. **This protocol's Freeze Manifest** -- directly pins artifact 1's
   two file hashes (no predecessor callback needed; this IS the root of
   this protocol's own provenance chain).
3. **Runtime Setup Manifest** -- predecessor callback: re-hash and
   compare artifact 2 (this protocol's Freeze Manifest) and the solver-
   audit's own Freeze Manifest; re-verify Gate 4's pinned upstream
   identities -- **v10 fix (P0): checkpoint/pool hashes AND the
   Battery 2 JSON/NPZ hashes and their cross-check (item 1's full
   keyset above)** -- still match current files.
   **v9 fix (P1, exact path specified)**: this Manifest must also
   SEMANTICALLY verify (not merely hash-match) all 20 referenced B/E
   Battery 1 records, checking EXACTLY: `record["payload"]["per_ic"][i]
   ["summary"]["solver_completed"] == True` for every `i` in `range(5)`;
   `len(record["payload"]["per_ic"]) == 5`; `record["payload"]["per_ic"][i]
   ["summary"]["final_V"]` is finite for every `i`; and
   `[record["payload"]["per_ic"][i]["initial_condition_index"] for i in
   range(5)] == [0, 1, 2, 3, 4]` (exact list equality, not just "same
   set") -- for every one of the 20 B/E records. A hash match alone
   proves byte-identity to a previously-verified state, not that the
   statistical INPUT this protocol is about to consume is complete and
   well-formed; this is cheap defense-in-depth given the 340/340 gates
   already passed for these records upstream, not a
   re-litigation of the solver audit's own results.
4. **Runtime Compatibility Gate Manifest** -- predecessor callback:
   re-hash the two preflight artifacts (stage 1), assert they match what
   the Freeze Manifest (stage 2) recorded. No recomputation of gate
   logic.
5. **Oracle-20 rollout records** (exact count: 5, one per IC; each
   containing the clean-pass record used as reference AND, separately, a
   companion instrumented-pass record whose hash is cross-referenced,
   not merged in -- Sec. 12's ownership rule) and **Certified-NLP
   rollout records** (exact count: 5, one per IC; same clean/
   instrumented split, each step's 2-3 candidate solves per Sec. 8-9).
   Predecessor callback: re-verify against stage 3's Setup Manifest hash.
6. **Timing-pass record** -- exact count: 1; contains references (by
   hash) to the SAME clean-pass rollout records from stage 5 (not a
   duplicate execution, Sec. 12) plus the newly-executed B/E timing
   rollouts (20 controllers x 5 ICs = 100 rollouts, the full 10-seed
   count per Sec. 12's sign-off) and their own companion NPZ of per-step
   wall-clock arrays, content-hashed. Predecessor callback: re-verify
   against stage 3 and stage 5's hashes.
7. **Summary Manifest**: `optimizer_gap`, `gap_B`, `gap_E` (point
   estimate/dispersion/CI/Holm-p, Sec. 13's primary report),
   decomposition-identity result, secondary endpoints, `resample_idx`
   arrays (Sec. 15, companion-NPZ-hashed), `floor_active` occurrence
   counts, and the raw per-candidate `return_status`/`would_be_
   certified_if_status_relaxed` records (v10 fix: renamed from the
   retired `ACCEPTABLE_NOT_PRIMARY` field name, kept as raw records only,
   not an interpreted frequency statistic, Sec. 9). Predecessor
   callback: re-verify against stages 5-6's hashes.
8. **Decision Record** (append-only, post-execution, no predecessor
   callback beyond citing stage 7's hash in its Provenance section, per
   the two existing Decision Records' convention).

Evidence-aware resume (re-deriving each stage's predecessor callback,
not trusting a resumed file's internal self-consistency alone), atomic
no-overwrite writes, and a single named canonical real-entry-point
function (e.g. `run_real_oracle_nlp_comparison()`) apply throughout.

### 14.1 Exact artifact schema table (v11 -- replaces the prose sketch above with a concrete, implementation-ready reference)

| # | Filename pattern | `kind` value | Top-level keys (name: type/shape) | Self-hash field | Expected count | Predecessor field(s) |
|---|---|---|---|---|---|---|
| 1a | `oracle_nlp_compat_sample_2026_09_XX.npz` | n/a (raw NPZ, not a JSON manifest) | `one_step_x0: float64[15,2]`, `one_step_u: float64[15,2]`, `multi_step_x0: float64[5,2]`, `multi_step_U: float64[5,3,2]`, `un_lo: float64[2]`, `un_hi: float64[2]`, `quartile_start: float64[2]` | file SHA-256 (computed externally, not self-embedded -- NPZ has no JSON `self_hash` convention) | 1 | none (root artifact) |
| 1b | `oracle_nlp_preflight_gate_results_2026_09_XX.json` | `"oracle_nlp_preflight_gate_results"` | `gates: dict[str,dict]` with exactly 6 keys (`"gate1".."gate6"`), each `{passed: bool, measured_error: float or null, measured_values: dict or null}`; `checkpoint_path: str`, `checkpoint_sha256: str`, `pool_path: str`, `pool_sha256: str`, `simulator_config: dict`, `battery2_json_path: str`, `battery2_json_self_hash: str`, `battery2_npz_path: str`, `battery2_npz_sha256: str`, `battery2_npz_hash_cross_check_passed: bool`, `upstream_solver_audit_setup_manifest_sha256: str`, `upstream_solver_audit_freeze_manifest_sha256: str` | `self_hash` (top-level, matching this codebase's existing JSON-manifest convention) | 1 | references artifact 1a's file hash inline (`compat_sample_sha256` field) |
| 2 | `docs/JPC_ORACLE_NLP_COMPARISON_PROTOCOL_FREEZE_MANIFEST_2026_09_XX.json` | `"oracle_nlp_freeze_manifest"` | `canonical_payload_sha256: str`, `file_sha256: dict[str,str]` (covers all new code files + this design doc + artifacts 1a/1b), `smoke_n_pass: int`, `smoke_n_total: int`, `generated_at_utc: str` | `canonical_payload_sha256` (top-level, matching the solver-audit's own Freeze Manifest shape exactly) | 1 | none (root of this protocol's own chain) |
| 3 | `oracle_nlp_setup_manifest_2026_09_XX.json` | `"oracle_nlp_setup_manifest"` | `predecessor_freeze_sha256: str` (artifact 2), `predecessor_solver_audit_freeze_sha256: str`, `battery1_record_hashes: dict[str,str]` (20 entries, keys `"{B,E}_seed{0-9}"`), `battery2_semantic_check_passed: bool`, `environment: dict`, `ipopt_options: dict` (all of Sec. 9's pinned values), `casadi_version: str`, `ipopt_version: str` | `self_hash` | 1 | artifact 2, plus the 20 individual Battery 1 record hashes |
| 4 | `oracle_nlp_compat_gate_manifest_2026_09_XX.json` | `"oracle_nlp_compat_gate_manifest"` | `predecessor_setup_manifest_sha256: str`, `rehashed_compat_sample_sha256: str`, `rehashed_preflight_results_sha256: str`, `hash_chain_verified: bool` | `self_hash` | 1 | artifact 3 |
| 5a | `oracle_nlp_oracle20_ic{0-4}_2026_09_XX.json` | `"oracle_nlp_oracle20_rollout"` | `initial_condition_index: int`, `clean: dict` (`summary`, `per_step: list[120]` of `{u0: float64[2], final_plan_un: float64[3,2], x_after: float64[2], V_after: float, minimum_stage_temperature: float, floor_active: bool}`), `instrumented_record_ref: {path: str, sha256: str}` | `self_hash` | 5 | artifact 3 |
| 5b | `oracle_nlp_oracle20_instrumented_ic{0-4}_2026_09_XX.json` | `"oracle_nlp_oracle20_instrumented"` | same per-step shape as 5a plus `iterations: list[20]` per step (`un_before`, `un_projected`, `j_true`, `j_true` after) | `self_hash` | 5 | artifact 5a (cross-referenced, not merged) |
| 5c | `oracle_nlp_certnlp_ic{0-4}_2026_09_XX.json` | `"oracle_nlp_certnlp_rollout"` | `initial_condition_index: int`, `clean: dict` (`summary`, `per_step: list[120]` of `{candidates: list[2-3] of {start_id, return_status, J_true, certified: bool, floor_active: bool, would_be_certified_if_status_relaxed: bool}, selected_start_id: int, starts_deduplicated: list or null}`), `instrumented_record_ref: {path: str, sha256: str}` | `self_hash` | 5 | artifact 3 |
| 5d | `oracle_nlp_certnlp_instrumented_ic{0-4}_2026_09_XX.json` | `"oracle_nlp_certnlp_instrumented"` | same per-step/candidate shape as 5c plus `iter_count`, `line_search_trials` per candidate | `self_hash` | 5 | artifact 5c |
| 6 | `oracle_nlp_timing_2026_09_XX.json` + companion `oracle_nlp_timing_arrays_2026_09_XX.npz` | `"oracle_nlp_timing_manifest"` | JSON: `predecessor_setup_manifest_sha256`, `reference_rollout_hashes: dict[str,str]` (references artifacts 5a/5c, NOT duplicated), `be_timing_rollout_hashes: dict[str,str]` (100 entries), `companion_npz_sha256: str`, `deadline_exceedance_fractions: dict`, `be_reproduction_gate_result: dict`. NPZ: `symbolic_build_wall_clock_seconds: float`, `solver_construction_wall_clock_seconds: float`, `first_call_warmup_wall_clock_seconds: float`, `clean_online_solve_wall_clock_seconds: float[...]` per controller | `self_hash` (JSON) + file SHA-256 (NPZ) | 1 JSON + 1 NPZ | artifact 3, artifacts 5a/5c (by reference, not re-execution) |
| 7 | `oracle_nlp_summary_manifest_2026_09_XX.json` | `"oracle_nlp_summary_manifest"` | `optimizer_gap: {point_estimate, per_ic_range}`, `gap_B`/`gap_E: {point_estimate, ci_lower, ci_upper, p_holm, resample_idx_sha256}`, `decomposition_identity_passed: bool`, `secondary_endpoints: dict`, `floor_active_counts: dict`, `acceptable_level_records: list` (raw, uninterpreted) | `self_hash` | 1 | artifacts 5a/5c/6 hashes |
| 8 | `docs/JPC_ORACLE_NLP_COMPARISON_DECISION_RECORD_2026_09_XX.md` | n/a (markdown, append-only) | prose, cites artifact 7's hash in a "Provenance" section | none (markdown has no self-hash convention in this project) | 1 | artifact 7 (cited, not hash-chained) |

This still needs its own dedicated provenance-focused review pass (the
class of defect the solver audit's own round 2 found, e.g. a missing
required key in a fresh-build manifest) before implementation --
explicitly unresolved (Sec. 16 item 6).

## 15. Statistical machinery — reuse, not reinvent

`gap_B`/`gap_E`'s one-sample contrasts reuse `src/cstr/ablation_lib.py`'s
existing, already-frozen machinery exactly: `bootstrap_ci()`
(`BOOTSTRAP_SEED=31415`, `N_BOOTSTRAP=10_000`, fresh
`np.random.default_rng(seed)` per call, 95% two-sided percentile-method
CI, `resample_idx` array returned and persisted -- `ablation_lib.py:250-268`)
and `exact_permutation_pvalue()` (exhaustive two-sided sign-flip test
over all `2^10=1024` sign patterns for a one-sample test against 0).
Holm correction across the `{gap_B, gap_E}` pair follows the same
procedure as the go/no-go ablation's own `secondary_family` Holm
adjustment (Decision Record 2026-09-01 Sec. 2). `resample_idx` arrays
are persisted and hashed into the Summary Manifest.

## 16. Numeric thresholds — CONFIRMED (v11), plus remaining actions

**All numeric thresholds below are CONFIRMED as of this review round,
not merely "proposed" -- per explicit sign-off, they are reasonable for
their stated purpose and are now PRE-REGISTERED.** Per this project's
standing discipline (frozen-protocol values are not to be loosened
post-hoc based on how real results turn out), none of the values below
may be relaxed after real execution begins without an explicit,
separately-justified amendment recorded the same way a frozen ablation
protocol's own post-hoc corrections are recorded (append-only, never
silently edited in place).

**Resolved (no longer open)**: timing-pass subset size (full 10-seed
re-run adopted, Sec. 12); `margin_V` categorical classification
(descriptive primary report adopted, Sec. 13); endpoint-equivalence/
Gate-4-per-step tolerance ambiguity (unified onto `values_match()`'s
`ISCLOSE_RTOL`/`ISCLOSE_ATOL`, Sec. 5.2a); Certified-NLP winner-
eligibility policy for `Solved_To_Acceptable_Level` (restricted to
descriptive-only, Sec. 9); Section 14 finalized into an exact-schema
table (Sec. 14.1).

**Confirmed thresholds** (Sec. 5.3): `1e-8` (gates 1-2), `1e-4` (gate
3), `ISCLOSE_RTOL=1e-5`/`ISCLOSE_ATOL=1e-8` (gates 4 and 6, per Sec.
5.2a). **Confirmed** (Sec. 6): floor-inactivity margin `1.0` K.
**Confirmed** (Sec. 8): `tol_dedup_input=1e-9`, `tol_tie_objective=1e-9`
(distinct constants, distinct units). **Confirmed** (Sec. 9):
`alpha=1.0`, `tol_kkt=1e-6`, `tol_primal=1e-6` (v12 fix -- corrected
from `1e-8`, see Sec. 23 item 1; the v11 text was wrong the day it was
frozen, not "relaxed later"), `ipopt_tol=1e-8`, `acceptable_tol=1e-6`,
`acceptable_iter=15` (IPOPT default, explicitly pinned).

**Remaining actions (not numeric sign-offs -- execution steps)**:
1. Wall-clock budget probe (1 IC, 5 control steps, both Oracle-20 and
   Certified-NLP) to confirm the full 110-rollout timing campaign's cost
   is tractable before committing to it (Sec. 12) -- an empirical check,
   not a design decision.
2. Section 14.1's exact-schema table needs its own dedicated,
   implementation-focused review pass (the class of defect the solver
   audit's own round 2 found, e.g. a missing required key in a
   fresh-build manifest) before coding begins.

Per the review, closing item 2 (a code-level review of Sec. 14.1, to
happen once implementation exists) and running item 1 (the timing
probe) are what remain before `구현 → smoke → 동결 → timing probe →
전체 실행` can proceed -- no further design-level review of this
document is expected.
6. Section 14's provenance schema sketch needs its own dedicated,
   implementation-focused review pass (exact manifest field types,
   companion-NPZ formats, canonical function signatures) before coding
   begins.

## 17. v5 -> v6 corrections (2026-09-03, code-verified review round 5)

1. **Restored standalone completeness** (the v5 review's primary
   finding): v6 inlines every section in full rather than deferring to
   "unchanged from v4" -- including the exact `margin_V`-based
   inequalities for outcome classification (Sec. 13) and the full
   timing-interleave/unit-accounting rule (Sec. 12), neither of which
   existed in v5's own body text.
2. Fixed the startup assertion's scope (Sec. 8): it now checks ONLY the
   two step-invariant fixed constants (zero vs. quartile distance and
   box-interior containment), evaluated ONCE at startup -- v5's blanket
   "all three starts pairwise distant" assertion would have failed
   immediately and incorrectly at every rollout's first control step,
   where warm-shifted is DESIGNED to equal zero. Warm-shifted collisions
   at any step are handled entirely by the existing per-step generic
   dedup rule, not by this assertion.
3. Restored an explicit `return_status` whitelist
   (`{Solve_Succeeded, Solved_To_Acceptable_Level}`) as a NECESSARY
   condition for certification, ANDed with the self-computed feasibility/
   KKT/finite/non-floor-active checks (Sec. 9) -- v5 had dropped
   `return_status` from certification entirely, which would have made a
   solver-flagged `Maximum_Iterations_Exceeded` or `Infeasible_Problem_
   Detected` run eligible if it happened to pass the numeric checks, not
   defensible to a reviewer. `Solved_To_Acceptable_Level` candidates
   remain eligible to win if they also pass 2-5, tagged
   `ACCEPTABLE_NOT_PRIMARY` for transparency, with counts/winner-status
   reported separately.
4. Specified floor-active consequences by condition (Sec. 6): for
   Oracle-20, ANY floor-active gradient-evaluation point contaminates
   the WHOLE IC's rollout (reference-invalid), since the deterministic
   warm-start chain propagates the effect forward; for Certified-NLP, a
   floor-active candidate is disqualified from winner selection (a
   non-floor-active certified alternative is preferred if one exists;
   if none exists, the step is uncertified). Replaced "store all 1,200
   stage temperatures" with a compact `minimum_stage_temperature` +
   `argmin_stage` contract, sufficient to derive `floor_active`.
5. Folded Gate 4's upstream-identity dependencies (B seed-0 checkpoint
   path+hash, pool/normalization identity, simulator config) into the
   preflight-gate-results artifact explicitly, so the Freeze Manifest
   pins them and the Runtime Setup Manifest re-verifies them (evidence-
   aware resume). Moved the measured `un_lo`/`un_hi`/quartile-start
   values (Sec. 8) into the PRE-FREEZE `oracle_nlp_compat_sample`
   artifact, not the post-freeze Runtime Compatibility Gate Manifest
   (which per Sec. 4 is a pure hash-chain check and must not be the
   first place a substantive measured value appears).
6. P1 fixes: made the mixed-error formula's parenthesization explicit
   (`max_i (|a_i-b_i|/max(1,|a_i|,|b_i|))`, the ratio computed
   component-wise before the max); replaced "floating-point tolerance"/
   "identical" (Gates 4, 6) with concrete numeric tolerances (`1e-6`,
   `1e-12` respectively) and an exact-match requirement on
   `selected_start_id`; kept `graph_build_wall_clock_seconds` as an
   explicitly separate field from `clean_online_solve_wall_clock_seconds`
   for both new conditions; extended the clean-vs-instrumented timing
   split (previously Certified-NLP only) to Oracle-20 as well, since
   floor-monitoring bookkeeping could contaminate its timing too (Sec.
   7).

Section 14's manifest schema and the 8 items in Section 16 remain open --
freezing is still not procedurally possible until those are closed, per
the v5 review's own final observation.

## 18. v6 -> v7 corrections (2026-09-03, code-verified review round 6)

1. **Restored the Certified-NLP clean-vs-instrumented endpoint-
   equivalence gate**, which existed for Oracle-20 (Sec. 7) but had been
   dropped from Certified-NLP's own section in v6 despite Sec. 12's
   timing-field text implying it existed -- now explicit in Sec. 9: both
   passes must produce identical `certified`, `J_true`, `floor_active`,
   and `selected_start_id` per (IC, step).
2. **Fixed the timing-gate record count**: the reduced timing subset is
   3 seeds x 2 conditions x 5 ICs = **30** raw records, not the 100 that
   would correspond to re-running all 10 B/E seeds -- v6 conflated the
   full-campaign count with the reduced-subset count (Sec. 12).
3. **Added the missing CasADi/IPOPT object construction/reuse
   contract** (Sec. 9): `J_true`, its AD gradient, and the parameterized
   `Opti` object are built exactly ONCE at execution start (mirroring
   `ipopt_lmpc_baseline.py`'s own `make_solver()` pattern), reused via
   `set_value`/`set_initial` across all candidates/steps/ICs, including
   after a failed/uncertified solve (no reset needed) -- rebuilding
   per-candidate would itself be an uncontrolled timing confound.
4. **Replaced Gate 4's single numeric tolerance with a split exact/
   isclose comparator** (Sec. 5.3), reusing the existing
   `values_match()`/`gate_compare_dict()` (`solver_audit_io.py:62-105`)
   rather than inventing a new one, since `state_constraint` contains
   booleans/strings/`None`/nested dicts that cannot all take one numeric
   `err<=` check. Also completed the exact field keyset (added `max_V`
   and the full nested `state_constraint` sub-keys, which v6's
   parenthetical list had incompletely enumerated).
5. P1 fixes: renamed the overloaded `tol` into `tol_primal` (Sec. 9
   item 2) and `ipopt_tol` (Sec. 9's IPOPT options) as two distinct
   constants; renamed Oracle-20's instrumented field `j_learned` to
   `j_true` (a copy-paste artifact from B/E's learned-model naming,
   Sec. 7); specified that IPOPT's initial guess is the explicit tiled
   `(HORIZON,2)` `plan`, never the bare 2-dim `start_vector` (Sec. 8-9);
   added a zero-plan box-interior check to the startup assertion
   alongside the existing quartile check (Sec. 8); split
   `graph_build_wall_clock_seconds` into `symbolic_build_wall_clock_
   seconds`, `solver_construction_wall_clock_seconds`, and
   `first_call_warmup_wall_clock_seconds` (Sec. 12); expanded Gate 5's
   synthetic-problem smoke coverage to explicitly exercise the dedup
   path, a forced non-whitelisted `return_status` rejection, and an
   uncertified-candidate fallback, not just the single success path
   (Sec. 5.3 gate 5).

Section 14's manifest schema and Section 16's numeric sign-off items
remain open -- freezing is still not procedurally possible until those
are closed.

## 19. v7 -> v8 corrections (2026-09-03, code-verified review round 7)

1. **Fixed Gate 4's/endpoint-equivalence's comparator reuse claim**:
   direct re-read of `values_match()` (`solver_audit_io.py:62-77`)
   confirmed it only applies `np.isclose` when BOTH inputs are
   `numbers.Real` scalars -- a Python list falls through to exact
   equality, and a numpy array would raise `ValueError` on `bool(array)`.
   Added a small `flatten_leaves()` recursive helper that walks nested
   dict/list structures down to scalar leaves, each then compared via
   the EXISTING `values_match()` -- not a new tolerance algorithm, a
   flattening wrapper around the one that already exists (Sec. 5.3 Gate
   4, Sec. 7/9's endpoint-equivalence gates).
2. **Split reference validity by condition** (Sec. 10): Oracle-20 (no
   solver-status/KKT concept) now has its own criterion (120 steps
   completed, all Adam iterates finite, floor-inactive throughout),
   separate from Certified-NLP's (120 steps completed, every step's
   selected winner passed Sec. 9 certification, floor-inactive) -- v7's
   single shared "certified" wording was not applicable to Oracle-20 as
   written.
3. **Strengthened both endpoint-equivalence gates (Oracle-20 Sec. 7,
   Certified-NLP Sec. 9) and Gate 4 (Sec. 5.3) to compare per-step
   trajectory data** (`u0`, final/selected plan, realized next state,
   `V_after`, `minimum_stage_temperature`), not just rollout-level
   `success`/`final_V`/`floor_active` aggregates -- closes a real gap
   where a harness bug producing different intermediate actions/states
   could still pass a final-value-only check by coincidence. Gate 4
   additionally now cross-checks against the existing Battery 2 per-step
   NPZ fields (`x_before`, `u0`, `final_plan_un`, `V_after_realized`)
   already produced by the solver audit, rather than only the rollout
   summary. Replaced "IDENTICAL `J_true` values" (not well-posed across
   independent IPOPT runs) with an explicit exact-vs-isclose split
   (`certified`/`selected_start_id` exact; `J_true`/plan/state/floor-
   diagnostic isclose at `err<=1e-8`).
4. **Clarified rollout-execution ownership** (Sec. 12): Oracle-20 and
   Certified-NLP each have exactly two executions -- a clean pass that
   DOUBLES as both the reference rollout (feeding Sec. 12a's gaps) and
   the timing measurement (executed once, not twice), and a separate
   instrumented pass used only for diagnostics. B/E are re-executed
   solely for timing (their non-timing endpoints are not recomputed).
5. **Adopted two explicit sign-offs** per the review's recommendation:
   (a) withdrew the reduced 3-seed timing subset in favor of re-running
   all 10 B/E seeds for timing (the ~1-minute additional cost is small
   relative to Oracle-20/Certified-NLP's own cost, and removes a
   documented-exception complexity for little savings) -- the B/E
   reproduction gate now checks the full 100 raw records again, not 30;
   (b) demoted the `margin_V`-based categorical classification (no
   defensible domain justification existed for `0.1`) to an optional,
   clearly-labeled sensitivity check, with the PRIMARY report now being
   the raw point estimate, per-IC dispersion, CI, and Holm-adjusted p
   for each of `optimizer_gap`/`gap_B`/`gap_E` (Sec. 13).
6. P1/precision fixes: pinned `acceptable_iter` (and reiterated
   `acceptable_tol`) as explicit IPOPT options recorded in the Runtime
   Setup Manifest, not left at implicit defaults (Sec. 9); precisely
   defined `first_call_warmup_wall_clock_seconds` as a throwaway dry run
   at a fixed `(IC, zero-plan)` pair, discarded and never leaking into
   any real IC's trajectory or warm-start state (Sec. 12); added a
   semantic (not just hash-based) verification of all 20 referenced B/E
   Battery 1 records to the Runtime Setup Manifest
   (`solver_completed==true`, exactly 5 ICs, finite `final_V`, consistent
   IC ordering) as cheap defense-in-depth (Sec. 14); filled in Section
   14's per-stage exact keysets, expected counts, predecessor callbacks,
   and companion-artifact hash requirements, which had previously been
   left as category names only.

Per the review, the comparison design itself is not expected to change
further; Section 14's provenance schema and Section 16's remaining
numeric sign-offs are what stand between this document and
implementation readiness.

## 20. v8 -> v9 corrections (2026-09-03, code-verified review round 8)

1. **Fixed Gate 4's per-step comparison's last-step blind spot**: v8's
   design used "the next step's `x_before`" as the current step's
   realized `x_after`, which has no analogue for the 120th (last) step,
   and the existing Battery 2 `step_records` do not store `x_after` at
   all (confirmed: only `x_before`, `u0`, `warm_start_un`,
   `final_plan_un`, `V_before`, `V_after_realized`, `realized_decrease`).
   Replaced with independently reconstructing `x_after_ref =
   sim.step(x_before, u0)` for all 120 stored `(x_before, u0)` pairs
   directly, covering every transition including the last, with no
   dependency on a possibly-absent "next record" (Sec. 5.3 gate 4).
2. **Pinned the Battery 2 evidence Gate 4 actually reads**: added the
   specific Battery 2 JSON record's path+self-hash, its companion
   per-step NPZ's path+SHA-256, a cross-check that the NPZ hash recorded
   INSIDE the JSON matches the NPZ file's current hash, and the upstream
   solver-audit Setup/Freeze Manifest hash these records descend from --
   previously only checkpoint/pool/simulator identities were pinned,
   leaving the actual per-step comparison evidence unprotected against
   silent drift (Sec. 5.3 gate 4).
3. **Unified two drifting tolerance rules into one explicit scope rule**
   (new Sec. 5.2a): `err()` (Sec. 5.2) is reserved for gates 1-3's
   cross-implementation comparisons (symbolic-vs-numeric, AD-vs-FD);
   `values_match()`'s existing `ISCLOSE_RTOL=1e-5`/`ISCLOSE_ATOL=1e-8`
   (via `flatten_leaves()`) is used for everything that compares the
   SAME implementation run twice -- Gate 4, Gate 6, and both endpoint-
   equivalence gates (Oracle-20 Sec. 7, Certified-NLP Sec. 9), all of
   which had inconsistently mixed `err<=1e-8`/`err<=1e-6` language with
   `values_match()` references in earlier drafts.
4. **Restricted Certified-NLP winner eligibility to `Solve_Succeeded`
   only** (Sec. 9), the safer of two options: `Solved_To_Acceptable_
   Level`'s trigger conditions depend on a WIDER set of IPOPT options
   (acceptable dual infeasibility, constraint violation, complementarity,
   objective-change tolerances) than the `acceptable_iter`/
   `acceptable_tol` v8 had pinned, and leaving the rest at version-
   dependent defaults risked a silent, undetected change in what
   "acceptable" means across IPOPT versions. `Solved_To_Acceptable_Level`
   candidates are now recorded descriptively
   (`would_be_certified_if_status_relaxed`) but never selected as winner
   and never affect the reported gaps.
5. P1 fixes: added `final_plan_un` to Oracle-20's endpoint-equivalence
   gate's actual field list (it was claimed in a prior corrections
   summary but missing from the live contract, Sec. 7); specified the
   exact `(step, IC, controller)` nested iteration order for the 22-
   controller timing round-robin, with per-controller (not global)
   warmup exclusion (Sec. 12); specified the Runtime Setup Manifest's
   B/E semantic-verification check as exact dict-path assertions
   (`per_ic[i].summary.solver_completed`, `len(per_ic)==5`, finite
   `final_V`, exact `initial_condition_index` list equality to
   `[0,1,2,3,4]`) rather than a prose description (Sec. 14).

Section 14's manifest schema (now closer to, but still short of, an
implementation-ready table per the review's own framing) and Section
16's remaining numeric sign-offs are what stand between this document
and implementation.

## 21. v9 -> v10 corrections (2026-09-03, code-verified review round 9)

1. **Propagated Battery 2 provenance into Section 14's actual keyset**
   (P0): Gate 4's own prose (Sec. 5.3) had already been extended with
   the Battery 2 JSON/NPZ path+hash fields and the upstream solver-audit
   manifest hash, but Section 14's preflight-artifact exact keyset and
   the Runtime Setup Manifest's re-verification callback still only
   listed `checkpoint`/`pool`/`simulator_config` -- the two sections had
   drifted apart. Section 14 item 1's keyset and item 3's callback now
   both explicitly include `battery2_json_path`,
   `battery2_json_self_hash`, `battery2_npz_path`, `battery2_npz_sha256`,
   `battery2_npz_hash_cross_check_passed`, and
   `upstream_solver_audit_setup_manifest_sha256`.
2. **Fixed Gate 6's stale tolerance** (P0): Gate 6 still used
   `err<=1e-12` after Section 5.2a introduced the `err()`-vs-
   `values_match()` scope rule (`err()` for gates 1-3 only). Gate 6 now
   uses `values_match(..., exact=True)` for `return_status`/
   `selected_start_id` and `values_match(..., exact=False)`
   (`ISCLOSE_RTOL`/`ISCLOSE_ATOL`) for `J_test`, consistent with Gate 4
   and both endpoint-equivalence gates. Removed the now-stale `1e-12`
   entry from Section 16's open-items list.
3. **Added deterministic controller-order rotation to the timing pass**
   (P1): the previous fixed 22-controller order (B-seed0..9, E-seed0..9,
   Oracle-20, Certified-NLP) put B-seed0 first and Certified-NLP last at
   EVERY `(step, IC)`, a systematic cache/thermal-position bias across
   the whole campaign. Fixed via a deterministic rotation offset
   `r = (step*5 + ic) % 22` applied to the fixed controller list at each
   `(step, IC)` -- full round-robin coverage is unchanged, but the
   starting position cycles deterministically instead of being fixed
   (Sec. 12).
4. **Corrected the `Solved_To_Acceptable_Level` reporting contract**
   (P1): renamed the retired `ACCEPTABLE_NOT_PRIMARY` field to the
   already-introduced `would_be_certified_if_status_relaxed` naming
   throughout (including a stale mention in Section 14's Summary
   Manifest item), and removed the claim that its FREQUENCY across a run
   is a meaningful descriptive statistic -- since the full `acceptable_*`
   option set is not pinned, that frequency could itself shift with an
   unpinned IPOPT option's default, so only the raw per-candidate record
   is kept (for transparency/debugging), not an aggregated, interpreted
   count (Sec. 9).

Per the review, closing these items plus finalizing Section 14 into an
exact-type/shape/filename/self-hash table is what remains before design
review can end and implementation (`구현 → smoke → 동결 → timing probe
→ 전체 실행`) begins.

## 22. v10 -> v11 corrections (2026-09-03, code-verified review round 10 — design review declared complete)

**No new design-level P0s found this round.** Two minor documentation-
consistency issues, both fixed:

1. **Stale cross-reference**: Section 14 referenced "Sec. 16 item 8"
   twice, but Section 16 has only 6 items (the count shrank across
   earlier rounds as items were resolved) -- corrected both references
   to "Sec. 16 item 6."
2. **Incompletely-propagated upstream hash field**: Gate 4's prose (Sec.
   5.3) promises both the upstream solver-audit Setup Manifest hash AND
   Freeze Manifest hash, but Section 14's preflight-artifact keyset had
   only gained the Setup Manifest field in the v10 fix -- added
   `upstream_solver_audit_freeze_manifest_sha256` directly to the
   keyset (the safer of two options, per the review, rather than relying
   on transitive verification through the Setup Manifest alone).

**Also completed per the review's explicit next-step request**:
- Section 14 rewritten as a concrete Section 14.1 exact-schema table
  (filename pattern, `kind` value, top-level keys with types/shapes,
  self-hash field, expected count, predecessor field(s) for all 8
  manifest stages plus their companion NPZ artifacts) -- replacing the
  prior prose sketch.
- Section 16 restructured: all previously-"proposed" numeric thresholds
  (preflight-gate tolerances, floor-inactivity margin, dedup/tie
  tolerances, KKT/primal/IPOPT tolerances) are now stated as CONFIRMED/
  pre-registered, with an explicit rule that none may be relaxed
  post-hoc once real execution begins (mirroring this project's
  standing frozen-protocol discipline). Only two genuinely remaining
  actions are left: an empirical wall-clock probe (not a design
  decision) and a future code-level review of Section 14.1 once
  implementation exists.

Per the review: **design-level review of this protocol ends here.** The
next steps are implementation, synthetic/real smoke, freeze, the timing
probe, and then real execution -- not further rounds of design
correction.

## 23. v11 -> v12 corrections (2026-09-03, post-implementation review of the preliminary real execution)

Implementation and a preliminary real execution (217s wall-clock, all
gates real-verified, `optimizer_gap`/`gap_B`/`gap_E`/decomposition-
identity all computed) happened under a v11 freeze. A review of that
execution found three issues -- none invalidate the underlying
mathematics, but the FROZEN PACKAGE (doc text + code) was internally
inconsistent, and two manifest-chain contracts were not fully
implemented. **Hash integrity of the v11 freeze was intact throughout
(every covered file matched its recorded hash) -- the problem is
semantic drift between two hash-verified files, not a provenance
failure**, which is exactly why it required a content review to catch,
not another hash check.

1. **`tol_primal` doc/code mismatch**: v11's Sec. 9 and Sec. 16 both
   stated `tol_primal=1e-8` as a "CONFIRMED, pre-registered" value, but
   the actual frozen code (`src/cstr/oracle_nlp_io.py`) already had
   `TOL_PRIMAL=1e-6`, changed during implementation (Sec. 9's own
   history: a real IPOPT solve at `ipopt.tol=1e-8` left a residual bound
   violation of ~1.7e-8 at an active constraint, which a `1e-8` check
   would spuriously reject) -- but the design document was never updated
   to match, so the "confirmed" value written down was already false the
   day v11 was frozen. This is corrected here to `tol_primal=1e-6` in
   both locations (Sec. 9, Sec. 16). The code comment describing this as
   "adjusted DOWN" was also imprecise and is corrected: `1e-6` is a
   LARGER (looser) tolerance than `1e-8`, i.e. the constraint was
   RELAXED/LOOSENED, not "adjusted down" -- a wording error, now fixed
   in `oracle_nlp_io.py`'s comment as well.
2. **Summary Manifest missing timing-pass linkage**: Sec. 14.1 item 7
   requires `resample_idx` arrays to be companion-NPZ-hashed and the
   Summary Manifest to reference the timing-pass record's predecessor
   hash (item 6). The preliminary execution's `build_summary_manifest()`
   did neither -- it computed `gap_B`/`gap_E`'s `resample_idx` in memory
   and recorded only a `dict_arrays_sha256` digest, never publishing the
   raw resample arrays as their own artifact, and never built or
   referenced a timing-pass record at all (the 22-controller interleaved
   timing pass, Sec. 12, was deferred as out of scope for the
   preliminary run). Both gaps must be closed before a confirmatory run.
3. **Certified-NLP solver object rebuilt per IC**: Sec. 9's object-
   construction contract requires the `Opti` object to be built EXACTLY
   ONCE per pass (clean, instrumented) and reused across all 5 ICs via
   `set_value(x0p, ...)`. The preliminary implementation's
   `run_real_oracle_nlp_comparison()` called `build_certified_nlp_solver()`
   freshly INSIDE the per-IC loop (once for clean, once for
   instrumented, per IC -- 10 total `Opti` objects across 5 ICs instead
   of 2), violating the "build once, reuse via `set_value`" contract
   this section itself specifies. Does not invalidate the preliminary
   `gap`/`decomposition-identity` result (each freshly-built `Opti`
   still solves the correct problem), but must be fixed before the
   timing pass (Sec. 12) is implemented, since Sec. 9's whole rationale
   for building once is avoiding an uncontrolled timing confound from
   repeated `Opti` construction.

**Disposition**: per explicit instruction, the preliminary execution and
its v11 Freeze Manifest are preserved, unmodified, as a historical/
diagnostic record -- NOT deleted, NOT presented as confirmatory. A
corrective revision (this v12) fixes item 1 in the design text and
directs items 2-3 to be fixed in code, followed by: full smoke re-run,
a NEW corrective Freeze Manifest (distinct filename from the preliminary
one), and a full re-execution (in a new results namespace) that
includes the previously-deferred timing pass. Only that confirmatory
execution's results are to be used in the eventual Decision Record.

## 24. Timing-pass scope reduction, disclosed (found while implementing item 4 of Sec. 23)

Implementing the full 22-controller interleaved timing pass (Sec. 12)
surfaced a genuine conflict with Sec. 12's OWN execution-ownership rule:
Oracle-20/Certified-NLP's clean pass is required to execute EXACTLY
ONCE, simultaneously serving as both the reference rollout (feeding
Sec. 12a's gaps, already completed in a separate call before any timing
pass exists) and the source of `clean_online_solve_wall_clock_seconds`.
Literally interleaving Oracle-20/Certified-NLP INTO the 22-controller
timing loop, as Sec. 12's execution-order text describes, would require
their clean pass to run a SECOND time inside that loop -- directly
violating the "exactly two executions total" rule this same section
states, which is the more load-bearing constraint (repeatedly emphasized
across multiple review rounds, e.g. v8 Finding 4).

**Resolution adopted (a disclosed scope reduction, not a silent
deviation)**: the interleaved timing pass, as implemented, rotates only
the 20 B/E controllers among THEMSELVES (their own internal fairness is
still achieved -- no single B/E seed is systematically first/last).
Oracle-20/Certified-NLP's timing figures are taken from their
ALREADY-COMPLETED clean reference pass's own `rollout_solver_core_wall_
clock_seconds` (Sec. 7/9), NOT re-measured inside the B/E interleaved
loop. This means Oracle-20/Certified-NLP's timing is NOT literally
interleaved with B/E's fresh re-execution at the wall-clock level --
a genuine, disclosed reduction from Sec. 12's full 22-way design, kept
because avoiding double-execution is the stricter, more explicitly-
reviewed constraint. Any manuscript or Decision Record text citing the
timing comparison must state this explicitly (B/E measured via a
20-controller interleaved rotation in one session; Oracle-20/
Certified-NLP measured via their own single clean-pass execution,
not concurrently interleaved with B/E's) rather than claiming full
22-way interleaving occurred.
