# Single-CSTR Query-Matched Solver Audit + Two-Plant State-Constraint Minimum Audit — Protocol

Status: DRAFT, NOT FROZEN, v11 (2026-09-02). Follows the same discipline
established for `JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_2026_08_31.md`:
design review -> implement -> synthetic smoke -> freeze -> real execution,
in that order. Implementation is IN PROGRESS (`src/cstr/solver_audit_lib.py`,
`src/cstr/solver_audit_io.py`, `scripts/solver_audit_driver_2026_09_01.py`,
`scripts/solver_audit_synthetic_smoke_2026_09_01.py`) and has been through
five implementation-review rounds (real code-level defects found and fixed
every time -- full round-by-round record kept in this project's session
memory, not duplicated here, since this document's job is to pin DESIGN
decisions, not track implementation-bug history). v8 pinned two decisions
the design left implicit that implementation had to resolve concretely
(Section 2 item 3, Section 5) -- see "v7 -> v8 corrections". v9 corrected
one textual inaccuracy Section 4a's own text asserted (where gate results
actually live) -- see "v8 -> v9 corrections". v10 pinned the `input_box_
satisfied` field name (Section 1) and corrected Section 5's Battery-2
denominator text to be metric-specific -- see "v9 -> v10 corrections". v11
corrects v10's OWN `input_box_satisfied` definition, which was itself
wrong (conflated box-feasibility with "never saturated") -- see "v10 ->
v11 corrections" at the end. Not yet frozen; no real execution has
happened.

**Revision history**: v1 found 6 P0 issues, fixed in v2 (per-iteration/
realized-decrease confusion, missing Proposition-1 residual bridge,
conflated `feasible` boolean, missing coverage battery for E/F
separation, false "checkpoints only" precondition, missing statistical
contract). v2 found 6 more P0 issues, fixed in v3 (coordinate-system
ambiguity, single-CSTR-vs-two-CSTR scope conflict resolved by the user
as single-CSTR-primary + a small non-causal two-CSTR minimum audit, GGN
predicted/true-decrease denominator mismatch, timing contamination,
ambiguous shifted-vs-absolute temperature coordinates, no boundary-
handling contract for the true-gradient FD). v3 found 4 more P0 issues
on a still-deeper implementation check, fixed here in v4 -- most
importantly, **v3's core assumption that two-CSTR checkpoints already
exist was FALSE**: `scripts/two_cstr_lgrad.py` trains fresh every run and
never calls `torch.save()` (confirmed by direct read); Section 8 is
rewritten around this fact. Also fixed: Battery 5's seed count now
matches the manuscript's actual 10-seed two-CSTR convention (not
`two_cstr_constraint.py`'s separate 3-seed probe); the statistical
contrast family (Section 5) now includes F-E (ceiling separation, the
whole point of Battery 1's 8-condition coverage) and C-B/E-FNN-vs-B-FNN
alongside the primary E-B, not E-B alone; and a legacy-result
compatibility gate is now required before any instrumented run is
trusted, for both the reimplemented single-CSTR Adam/GGN batteries and
the reconstructed two-CSTR checkpoints. v4 found 4 more P0 contractual
contradictions on a still-deeper review, fixed here in v5: the
compatibility gate required per-IC data M2/M3 don't contain (split into
Gates 1/2), GGN had no legacy result on the query-matched checkpoints at
all (Gate 3, an internal old-vs-new-code check instead), the statistical
contrast family requested gradient-bridge data no battery produces for
F/C/FNN (scoped per data source), and the reconstructed two-CSTR model's
identity/failure-mode were unstated (`metric_compatible_replica` naming,
`TWO_CSTR_RECONSTRUCTION_INCOMPATIBLE` failure state, Gate 4). See
"v3 -> v4 corrections" and "v4 -> v5 corrections" at the end for full
detail. v5 found 3 more P0 contractual gaps (not structural issues --
the reviewer's own characterization): two literal values left as
placeholders (the completion-rate downgrade threshold, Gate 4's
tolerance), no defined float-vs-exact tolerance split for Gates 1/3, and
two field-name mismatches between this document's prose and the actual
source dict keys (`Jx` -> the real key `state_jac_med`; `violations`
needing an explicit mapping to this document's `lyapunov_increase_
count`). All fixed here in v6 -- see "v5 -> v6 corrections" at the end.
v6's review found **no remaining P0s** and judged the design ready for
implementation, modulo two non-blocking wording fixes ("Three gates"
miscounted (there are four); the tolerance contract's justification
applied "same checkpoint" reasoning to Gate 4, which is actually a
same-recipe REPLICA check, not a same-checkpoint one) -- both fixed here
in v7; see "v6 -> v7 corrections" at the end. Per the user's explicit
guidance, once v6's remaining items are addressed, this protocol's scope
should not be widened further.

## 0. Scope and precondition

**Renamed and re-scoped in v3**: this is explicitly a **single-CSTR**
protocol. The real go/no-go ablation's checkpoints (`POOL_PATH =
data/processed/lcnn_paper_cstr_onestep_20k.npz`) and
`scripts/budget_sweep_solver.py` are a 2-state/2-input single CSTR only
-- confirmed by direct read (`P` is 2x2, `INPUT_LO`/`INPUT_HI` are
length-2). Two-CSTR GGN (`scripts/two_cstr_ggn.py`) and state-constraint
(`scripts/two_cstr_constraint.py`) scripts exist SEPARATELY but have NO
query-matched/equal-query design (confirmed: zero matches for "query",
"augment", or "2m" in either file) -- there is no query-matched two-CSTR
B/E checkpoint pair, and producing one would mean designing, freezing,
and running an entire second go/no-go protocol for a different plant, a
much larger undertaking than this document. Given the 2026-09-20
deadline (~19 days remaining as of this writing) and that the equal-
query fairness rebuttal is already established on single-CSTR (Decision
Record Sec. 1), that structural risk is not worth taking now (Section
10 has the full rationale and deferred-scope conditions).

This protocol therefore covers, for **single-CSTR only**: the actual-
solver audit (R1.2/R2.M1), the ceiling-free continuous metrics (R2.m4),
and the state-constraint minimum audit's single-CSTR half (R2.M6/R3.8).
Section 8 separately covers a small, explicitly non-causal two-CSTR
minimum audit for the SAME reviewer comments' two-CSTR half, since both
benchmarks are discussed in the current manuscript and R2.M6/R3.8 do not
name a specific plant. Neither this protocol nor Section 8 covers
R2.M4's oracle-gradient-controller/converged-NLP item, the `lambda_jac`
grid extension, the FD/noise/weight sensitivity sweep, the MPC-necessity
freshness check, the optional active-constraint case study, or a
two-CSTR equal-query ablation -- those remain separate, later (or
deferred) items.

**Precondition**: real execution needs the real simulator, normalization/
scaling data, the GGN step mechanics, and a concrete software/hardware
environment -- not checkpoints alone (Section 4). None of the 12
Freeze-Manifest-covered files may be edited to implement this protocol;
new instrumentation lives in new files, and this protocol creates its
own manifest chain rather than resuming or extending the frozen
ablation's M1/M2/M3.

## 1. Naming corrections

1. `budget_sweep_solver.closed_loop()`'s `violations` field counts steps
   where the LYAPUNOV FUNCTION INCREASED, not a state-constraint
   violation. Any manuscript table or new evaluator output reporting
   this existing quantity MUST label it `lyapunov_increase_count`.
2. A single `feasible` boolean conflates numerical solver completion,
   input-box feasibility, and retrospective temperature-cap
   satisfaction. Three separate fields; the temperature one is
   `temperature_cap_satisfied_posthoc` -- never `state_feasible`, since
   this method does not enforce the constraint. **Input-box semantics,
   corrected in v11 (v10's definition was WRONG)**: `input_box_satisfied`
   is a genuine box-FEASIBILITY check -- `True` iff every applied action
   satisfies `lo - tol <= u <= hi + tol` and is finite. A value sitting
   EXACTLY at the boundary is still feasible (the constraint is `<=`, not
   `<`); v10 incorrectly defined this field as "never saturated," which
   would have wrongly reported `False` for a legitimately-feasible
   boundary value. The genuinely useful "was the solver's solution ever
   pinned at the bound" diagnostic v10 actually computed is now its own,
   separately-named field, `input_never_saturated` (`True` iff every
   per-actuator saturation frequency, Section 3.6, is exactly 0) --
   complementing, not duplicating, the existing per-actuator saturation-
   frequency rates, and never sharing the "satisfied" framing that would
   incorrectly imply saturation is itself a constraint violation. In this
   protocol's own system, every applied action is always clamped into
   the box before use (Adam/GGN both clip), so `input_box_satisfied` is
   trivially `True` in practice -- but it is computed as a REAL,
   independently verified check here, not assumed.

## 2. Five batteries with distinct coverage

1. **Continuous endpoint battery** (single-CSTR) -- ALL 8 conditions
   (A-F, B-FNN, E-FNN) x 10 seeds x 5 initial conditions. Keeps the FULL
   `closed_loop()` return dict (`final_V`, `max_V`, `success`,
   `lyapunov_increase_count`, `tv`) instead of discarding everything but
   the boolean. Separates E from F's ceiling-saturated success.
2. **Detailed Adam audit** (single-CSTR) -- B and E only, x 10 seeds x
   5 initial conditions. Full per-iteration logging (Section 3.4) plus
   the one-step residual/gradient bridge (Section 3.5). Since this
   battery necessarily REIMPLEMENTS (not merely calls) the Adam inner
   loop to insert per-iteration logging, it is subject to the legacy-
   result compatibility gate (Section 4a) before its output is trusted.
3. **GGN descriptive audit** (single-CSTR) -- B and E only, 10 seeds x
   5 ICs, reusing `scripts/ggn_mpc_probe.py`'s diagnostic family via a
   new driver pointed at the real ablation's trained B/E checkpoints,
   WITH the denominator-mismatch fix (Section 3.4 item 4, Section 6
   item 2). Descriptive/non-gating. Also subject to the compatibility
   gate (Section 4a) -- this battery reimplements the GGN inner loop.
   **Objective choice, pinned in v8**: `ggn_mpc_probe.py` supports two
   residual objectives (`"V"`, the rolled-out Lyapunov value; `"residual"`,
   the exact one-step residual against the Sontag auxiliary law) -- this
   battery uses `objective="V"` throughout, matching `ggn_mpc_probe.
   main()`'s own default/CLI-argument default. This was left unpinned in
   v1-v7 and resolved ad hoc during implementation; pinned here as a
   fixed literal, not left to be chosen per run.
4. **State-constraint summary** (single-CSTR) -- piggybacks on battery 1
   (same rollouts, no separate re-execution), so it covers all 8
   conditions at no extra rollout cost. Calls the FROZEN
   `budget_sweep_solver.closed_loop()` DIRECTLY (no reimplementation,
   just retains fields the wrapper functions currently discard), so no
   compatibility gate is strictly required here, though a cheap sanity
   check against M3's recorded per-seed success rates is still worth
   including.
5. **Two-CSTR minimum audit** (Section 8, REWRITTEN in v4 -- see "v3 ->
   v4 corrections" item 1) -- small relative to a full two-CSTR
   equal-query redesign, but NOT free: `scripts/two_cstr_lgrad.py`
   trains fresh in memory and never persists a checkpoint (confirmed by
   direct read -- no `torch.save()` call anywhere in the file), so this
   battery requires deterministically RECONSTRUCTING the `value_only`/
   `value+grad` two-CSTR models (10 seeds, matching the manuscript's
   actual two-CSTR convention, not the separate 3-seed
   `two_cstr_constraint.py` probe), verifying them against a legacy-
   result compatibility gate, saving them as NEW checkpoints, and only
   then running the instrumented state-constraint rollouts. Explicitly
   non-causal (no query-matched design exists for two-CSTR).

Coverage note: batteries 2/3 are B/E-only -- this protocol does not
produce a projection-activity/gradient-fidelity bridge for D, C, or the
FNN conditions unless explicitly extended.

Coverage vs. duplication note (new in v4): B and E are members of both
Battery 1 (all 8 conditions, coarse) and Battery 2 (B/E only, detailed).
To avoid running the same (condition=B or E, seed, IC) rollout twice
under two independently-executed code paths -- which wastes compute and
risks the two paths silently disagreeing -- **Battery 2's detailed run
for B/E is the SOLE source of B/E's rollout-summary and state-constraint
records**; Battery 1 only independently executes A, C, D, F, B-FNN, and
E-FNN, and its Summary record merges in B/E's rollout-summary-level
fields extracted from Battery 2's already-collected data rather than
re-running them.

## 3. Required logging schema

### 3.1 Identifiers (every record)

`plant` (`single_cstr` | `two_cstr`, only battery 5 uses `two_cstr`),
`condition`, `seed`, `initial_condition_index`, `solver` (`adam` |
`ggn`), `horizon`, `budget`.

### 3.2 Rollout summary (one record per condition x seed x IC x solver)

`final_V`, `max_V`, `success`, `settling_time` (Section 6 item 1),
`input_tv_per_actuator_physical` AND `input_tv_per_actuator_normalized`
(both units stored separately, corrected in v3 -- P1), `rollout_
solver_core_wall_clock_seconds`, `rollout_audit_counterfactual_wall_
clock_seconds`, `rollout_instrumented_total_wall_clock_seconds`
(three-way timing split, new in v3, Section 6 item 3 -- **only
`solver_core_wall_clock` should ever be cited as "control-update time"
in the response letter**), `lyapunov_increase_count`.

**Timing methodology** -- CPU wall-clock numbers alone do not satisfy
R2.M4's request without a fixed measurement protocol:
- Use `time.perf_counter()` (monotonic, high-resolution), never
  `time.time()`.
- **Warm-up count pinned as a literal in v5 (P1)**: the first **2**
  control steps of every rollout are executed and timed but EXCLUDED
  from the reported summary statistics, to avoid first-call overhead
  (JIT/cache warm-up, lazy imports) contaminating the reported numbers.
  Recorded in the Setup Manifest.
- **Thread count pinned as a literal in v5 (P1)**: `torch.set_num_
  threads(1)` -- single-threaded, for maximal reproducibility and
  cross-run comparability (avoids core-contention variance that a
  multi-threaded default would introduce). Recorded in the Setup
  Manifest alongside the rest of the software/hardware environment
  (Section 4).
- No concurrent CPU load during timed runs -- explicitly no other
  instrumented battery, smoke test, or unrelated process runs
  simultaneously on the same machine while timing data is being
  collected (mirrors the go/no-go ablation's own Paper-M-concurrency
  precondition, Section 0 of that protocol).
- Report MEDIAN and P95 (not just mean) of per-step solve wall-clock
  across all timed (non-warm-up) steps, since OS-scheduling-induced
  outliers can skew a mean without reflecting typical control-update
  latency.
- The simulator's actual physical sampling period is `dt_hr = 1e-3`
  hours `= 3.6` seconds per control step (confirmed by direct read of
  `lcnn_paper_simulator.py`, computed and stated explicitly here so a
  reviewer can directly compare `solver_core_wall_clock`'s median/P95
  against the real-time budget a deployed controller would have).

### 3.3 Per-control-step (one record per step within a rollout)

Current (real) state, warm-start input sequence, final planned input
sequence, applied input `u0`, predicted next state/`V` (from the FINAL
plan's model rollout), REALIZED next state/`V` (from `SIM.step`), the
REALIZED decrease (`V_before - V_after` -- the only genuine "realized"
quantity anywhere in this protocol), a projection-activity summary for
the step, and per-step solve wall-clock (split per Section 3.2).

### 3.4 Per-iteration (battery 2, B/E, all iterations)

Corrected in v3 (P0, coordinate system + scope):

1. **Coordinate system, fixed in v3**: Adam optimizes the NORMALIZED
   action `u_n` (confirmed: `budget_sweep_solver.py`'s inner loop treats
   `un` itself as the optimized/clamped tensor). ALL gradients, inner
   products, and directional derivatives in this section and Section 3.5
   MUST be computed in normalized-`u_n` space, matching both Adam's
   actual optimization variable and `diagnose_fixed_idx()`'s existing
   convention (`learned_u_grad(un_np)` already operates on `un`).
   Physical-space gradients must never be dotted against normalized-
   space steps -- that inner product has no meaningful units or scale.
2. **Evaluation point, fixed in v3**: the true/learned residual and
   their gradients are evaluated at `u_n` BEFORE the iteration's update
   (`un_before`, the start-of-iteration point), matching standard
   descent-direction analysis (directional derivative = gradient at the
   start point, dotted with the step taken from that point).
3. **Scope, fixed in v3**: Proposition 1 is a ONE-STEP statement. The
   residual/gradient bridge (Section 3.5) uses ONLY the plan's FIRST
   horizon block, `un[0]` (`Delta u_0`), not the full multi-step plan --
   the one-step residual `g(x, u)` is evaluated at the current state `x`
   with `u = un[0]` specifically.
4. Learned objective value `J_learned` (the actual multi-step quantity
   Adam optimizes, unchanged from v2) and its iteration-to-iteration
   change.
5. Counterfactual true-objective value -- the CANDIDATE plan `un` at
   this iteration, rolled out from the CURRENT real state through the
   TRUE plant (`SIM`, not the learned surrogate) for `horizon` steps,
   using the EXACT SAME functional form as `J_learned` (Section 6 item 7
   pins the formula). Never applied to anything; a counterfactual only.
6. **GGN's version of item 5, fixed in v3 (P0, denominator mismatch)**:
   `ggn_mpc_probe.py`'s existing code averages `pred_dec_applied`
   (logged once per ACCEPTED inner iteration, across the whole rollout)
   against `mean_true_dec` (logged once per CONTROL STEP) -- different
   denominators, confirmed by direct read of `ggn_closed_loop()` lines
   89-142. Fixed: for every ACCEPTED GGN update, additionally compute
   the counterfactual true-objective before/after THAT SAME accepted
   candidate plan (same construction as item 5), so predicted and
   counterfactual-true decrease are compared at the SAME (per-accepted-
   update) granularity. The genuinely REALIZED plant decrease
   (`Vbefore - Vafter` from an actual `SIM.step`) remains a separate,
   once-per-control-step statistic (Section 3.3) and must never be
   averaged together with the per-accepted-update counterfactual
   comparison.
7. Raw (pre-projection) step and projected (post-projection) step at
   this iteration, plus count of clamped coordinates (both in
   normalized-`u_n` space, per item 1).

### 3.5 One-step residual/gradient bridge to Proposition 1

At each logged point (Section 3.4's scope: `un_before`, first block
only), in NORMALIZED-`u_n` space throughout (item 1 above):

- Learned one-step residual `g_hat(x, u_n[0])` and true one-step
  residual `g(x, u_n[0])` (same functional form as
  `diagnose_fixed_idx()`, evaluated at the actually-visited state/action
  rather than the fixed 800-point oracle set).
- Learned `nabla_{u_n} g_hat` and true `nabla_{u_n} g` (matching
  `diagnose_fixed_idx()`'s `learned_u_grad`/central-FD `true_grad`
  construction, in normalized space).
- Their cosine similarity AND raw inner product (both; per the Response
  Matrix's R1.3 correction).
- The directional derivative of the TRUE residual gradient along the RAW
  (pre-projection) step and along the PROJECTED (post-projection) step,
  i.e. `true_grad . raw_step` and `true_grad . projected_step`, both
  evaluated at `un_before` and both in normalized space -- this is the
  direct empirical bridge from Proposition 1 (a statement about the
  learned negative-gradient direction relative to the true residual) to
  the actual solver's realized step.
- Multi-step learned/counterfactual-true objective change (Section 3.4
  items 4-5), reported alongside for comparison.

**Boundary-handling contract for the true-gradient central-FD (new in
v3, P0)**: `diagnose_fixed_idx()`'s existing central-FD (`eps=1e-3` in
normalized space) has NO boundary check at all -- confirmed by direct
read, it never verifies `un +/- eps` stays within `[un_lo, un_hi]` before
calling `sim.step`. This is tolerable for the pre-registered fixed
800-point oracle set (unlikely to sample exactly at the box edge), but
NOT tolerable here: this audit specifically targets projection-active
states, which by construction sit AT or NEAR the box edge, so a naive
central-FD would routinely evaluate the true plant at inadmissible
inputs exactly where this audit needs it most. Contract, pre-registered
here: (a) reuse `eps=1e-3` (normalized) for continuity with the existing
oracle diagnostics; (b) if `un_before[j] + eps` or `un_before[j] - eps`
would exceed `[un_lo[j], un_hi[j]]`, switch to a ONE-SIDED finite
difference toward the INTERIOR for that coordinate (not simple
exclusion) -- excluding boundary-active points would bias the sample
away from exactly the region under study, unlike the existing oracle
set's exclusion rule, which is safe there only because boundary points
are a small, non-targeted minority of that fixed set.

**One-sided FD formula, pinned in v4 (P1)** -- v3 specified only the
DIRECTION (toward the interior) without an order, and central FD is
second-order accurate (`O(eps^2)`) while a naive one-sided FD is only
first-order (`O(eps)`); comparing a first-order boundary estimate
against second-order interior estimates would confound "gradient error
because of proximity to the constraint" with "gradient error because of
a cruder finite-difference scheme," undermining exactly the boundary-
vs-interior comparison this audit exists to make. The one-sided
replacement MUST be the 3-point, SECOND-ORDER one-sided formula (forward
example, toward increasing `un_before[j]`): `(-3 f(un_before) + 4
f(un_before + eps e_j) - f(un_before + 2 eps e_j)) / (2 eps)` (the
backward form is the mirror image, used when the interior direction is
decreasing). This requires `2 eps` of interior clearance on the chosen
side, not just `eps`; if even that is unavailable (an extremely tight
box, or an unusually large `eps`), the point is excluded and the
exclusion reason (`fd_clearance_unavailable`) is logged explicitly --
never silently substituted with the lower-order formula, which would
reintroduce the same order-mismatch problem this fix exists to remove.

**Boundary/exclusion reporting, new in v5 (P1)**: since excluded points
are disproportionately likely to be exactly the projection-active states
this audit targets, silently averaging exclusions away would remove
visibility into the region of primary interest. Report, per (condition,
seed): the fraction of logged points using the one-sided (vs. central)
formula, the fraction excluded via `fd_clearance_unavailable`, and the
fraction of logged control steps where projection was active at all
(Section 3.3's projection-activity summary) -- so a reader can see
whether, e.g., condition D's exclusion/one-sided rate is much higher
than E's, which would itself be informative about D's proximity to the
input constraints.

### 3.6 State-constraint statistics

- `max_temperature_deviation_K` and `max_temperature_absolute_K` (both
  stored -- **fixed in v3 (P0)**: the state is shifted/deviation
  coordinates by default (`x = [CA - CAs, T - Ts]`, confirmed by direct
  read of `lcnn_paper_simulator.py`), and `T_CAP = 70.0` in
  `two_cstr_constraint.py` is a DEVIATION cap, not an absolute-Kelvin
  cap -- computed via each simulator's own `to_absolute_state()`/
  `xs_abs` (not a hardcoded constant), so this generalizes correctly to
  both plants without assuming a single steady-state temperature).
  Single-CSTR reports state index 1 only; two-CSTR (battery 5) reports
  indices 1 and 3 separately (reactor 1 and reactor 2).
- `cap_deviation_K` (`T_CAP` as currently used, `70.0`) and
  `cap_absolute_K` (`T_CAP + Ts`, per-reactor if the two reactors' steady
  states differ, via the same `to_absolute_state()`/`xs_abs` accessor).
- `temperature_violation_frequency` = fraction of steps with
  `T_deviation > cap_deviation`.
- `temperature_violation_integral` = `dt_hr * sum(max(0, T_deviation -
  cap_deviation))` over the rollout, `dt_hr` = the simulator's actual
  physical control-step interval (default `1e-3` hours, confirmed by
  direct read of `lcnn_paper_simulator.py`) -- distinct from the
  training-time soft penalty's squared functional form.
- `input_saturation_frequency` per actuator -- **tolerance fixed in v3
  (P1, removes v2's internal self-contradiction)**: "at bound" means
  within `1e-6 * (INPUT_HI[i] - INPUT_LO[i])` of `INPUT_LO[i]` or
  `INPUT_HI[i]` -- a tolerance PROPORTIONAL to each actuator's own range
  (confirmed the two actuators differ by five orders of magnitude,
  `[-3.5, 3.5]` vs. `[-5e5, 5e5]`), NOT a shared absolute constant. v2
  stated both an absolute `1e-6` form and this relative form in the same
  paragraph; only the relative form is correct and used. Denominator is
  total control steps in the rollout, reported per actuator.
- `temperature_cap_satisfied_posthoc` (Section 1 item 2) -- boolean,
  `max_temperature_deviation_K <= cap_deviation_K` for the entire
  rollout.

### 3.7 Failure-handling contract

Unchanged from v2: `solver_completed` boolean (narrowed exception types,
matching the frozen ablation's discipline); on failure, `settling_time`
is the fixed `steps+1` sentinel (never imputed), an explicit failure
record is written, and every OTHER continuous metric is aggregated only
over completed rollouts with an explicit `completion_rate` reported
alongside -- never silently excluded from `success`/`settling_time`'s
own denominator, which always includes every attempted rollout.

## 4. New artifact/manifest chain

- **Setup Manifest**: checkpoints (paths + hashes, re-verified against
  the real ablation's M2 `checkpoint_sha256`), coverage per battery, the
  logging schema version, simulator code identity (file hash of
  `src/cstr/lcnn_paper_simulator.py` for batteries 1-4, ADDITIONALLY
  `src/cstr/two_cstr_series.py` for battery 5), normalization/scaling
  provenance (source ablation's M1 self-hash and `pool_content_sha256`),
  source M2/M3 self-hashes, the GGN mechanics file hash
  (`scripts/ggn_mpc_probe.py`), and the software/hardware environment --
  **fixed in v3 (P1)**: record the ACTUAL torch device tensors were
  computed on (e.g. `next(model.parameters()).device`), not MPS/CUDA
  "availability" -- confirmed by direct grep that NONE of
  `budget_sweep_solver.py`, `ggn_mpc_probe.py`, or `ablation_lib.py`
  ever call `.to(...)`/pass `device=`/reference `cuda`/`mps`, so all
  computation is on CPU regardless of `torch.backends.mps.is_available()`
  being `True` on the development machine; the setup manifest should
  assert and record this directly rather than report availability as a
  proxy for actual placement.
- Battery 5 (two-CSTR) uses its OWN, smaller setup record (different
  checkpoints, different simulator file hash), kept separate from
  batteries 1-4's single-CSTR setup record rather than merged into one.
- Per-(condition, seed, solver, battery) **Rollout records**: battery
  2's per-iteration data (Section 3.4/3.5, all iterations, B/E only) as
  compressed NPZ arrays; batteries 1/3/4/5's coarser records may use
  JSON.
- A **Summary record** per battery: aggregated statistics per the
  statistical contract (Section 5) -- descriptive/diagnostic, not a
  pre-registered go/no-go gate.

## 4a. Compatibility gates (REWRITTEN in v5, P0 -- three distinct gates, not one)

v4 proposed a single "legacy-result compatibility gate" comparing new
evaluators against M2/M3. **This was wrong for two independent reasons,
corrected here:**

- M2/M3 do not contain per-IC data at all -- `_closed_loop_success_
  rate()` (`scripts/ablation_driver_2026_08_31.py`) returns only
  `float(np.mean(successes))` across the 5 ICs; there is no per-IC
  `final_V` anywhere in the real ablation's manifests to compare a new
  per-IC-instrumented evaluator against.
- `ggn_mpc_probe.py`'s existing logged results are NOT from the real
  ablation's query-matched B/E checkpoints at all -- confirmed by direct
  read of its `main()`: it calls `L.train(ARCH, seed, data, ...)`,
  training its OWN models fresh with a different (EAAI-era, non-query-
  matched) recipe, entirely independent of the go/no-go ablation's
  frozen checkpoints. There is no "legacy result on the same model" for
  GGN to compare against.

Four gates, each targeting what actually CAN be verified:

**Tolerance contract, pinned as literals in v6 (P0) -- fixed BEFORE
implementation, never chosen by looking at results (that would be
circular)**: for every gate below, boolean and integer fields (`success`,
`lyapunov_increase_count`/`violations`, `accepted_frac`'s underlying
counts) are compared EXACT (`==`); continuous float fields use
`numpy.isclose(a, b, rtol=1e-5, atol=1e-8)` (numpy's own default, chosen
deliberately rather than inventing a bespoke value). **Precision fixed
in v7**: this default is adequate for Gates 1-3, which are genuine
SAME-CHECKPOINT equivalence checks -- both sides evaluate the identical,
already-frozen checkpoint, differing only in whether logging is enabled,
never two independently-trained models. Gate 4 is different in kind: it
compares a NEWLY-TRAINED `metric_compatible_replica` against a
previously-logged scalar from a run whose original checkpoint was never
saved (Section 8) -- a deterministic SAME-RECIPE replica-compatibility
check, not a same-checkpoint one. The same `numpy.isclose(rtol=1e-5,
atol=1e-8)` value is still used for Gate 4 (Section 4a's Gate 4
description), since same-seed/same-code training is expected to
reproduce very closely -- but this is a distinct justification from
Gates 1-3's, not "the same checkpoint" reasoning applied a fourth time.

**Gate 1 -- controller-equivalence (Battery 2, Adam)**: on the SAME
checkpoint, seed, and IC, run BOTH the frozen `budget_sweep_solver.
closed_loop()` AND the new instrumented evaluator (with logging
disabled for this check), and compare their full per-IC output dicts
directly -- a genuine per-IC, apples-to-apples implementation check that
does not depend on M2/M3 containing per-IC data at all. **Field mapping,
new in v6 (P0)**: `closed_loop()`'s own dict keys are `final_V`, `max_V`,
`success`, `violations`, `tv` (confirmed by direct read of
`budget_sweep_solver.py` line 86) -- the new evaluator's schema uses the
renamed `lyapunov_increase_count` for `violations` (Section 1). Gate 1's
comparison logic must explicitly map `violations <-> lyapunov_
increase_count`, not compare dict keys by exact string match, or this
gate would spuriously fail on a naming difference that is not a real
implementation bug.

**Gate 2 -- legacy-summary gate (Batteries 1/2, single-CSTR)**:
separately, aggregate the new evaluator's per-IC successes into a
seed-level mean using the SAME aggregation `_closed_loop_success_rate()`
uses, and compare that seed-level value against M2's recorded
`closed_loop_success[condition]` for that seed (float comparison per the
tolerance contract above). This catches an aggregation-level bug (e.g. a
wrong IC subset or averaging error) that Gate 1's per-IC comparison
alone would not.

**Gate 3 -- implementation-equivalence gate (Battery 3, GGN)**: since no
external legacy result exists for GGN on the real ablation's B/E
checkpoints, load those SAME checkpoints into BOTH the ORIGINAL,
unmodified `ggn_mpc_probe.py` mechanics (`ggn_closed_loop()`, called
as-is, not the new instrumented wrapper) and the NEW instrumented
version, on the same seed/IC/budget, and verify the two produce matching
core outputs (`success` exact, `final_V`/`accepted_frac` via the float
tolerance above) -- i.e. the check is INTERNAL (old code vs. new code on
the same new inputs), not against any pre-existing JSON. No field-name
mapping issue here since both sides are variants of the same
`ggn_mpc_probe.py` mechanics using the same key names.

**Gate 4 -- two-CSTR legacy-scalar gate (Battery 5), tolerance and field
names pinned in v6 (P0)**: only the aggregate scalars already logged in
`results/interim/logs/two_cstr_lgrad.json` are available for comparison
(no per-IC data exists there either). **Field names, corrected from an
earlier draft that used the console-print variable name `Jx` instead of
the actual stored JSON key**: the reconstructed replica's metrics must
match `two_cstr_lgrad.json`'s existing per-(config, seed) dict, whose
real keys (confirmed by direct read of `two_cstr_lgrad.py` line 313) are
`test_mse`, `align`, `frac_neg`, `b3`, `b20`, `state_jac_med` (NOT `Jx`).
Comparison rule: `b3`, `b20`, and `frac_neg` are already exact means of
boolean/exact quantities over a fixed, deterministic IC set and RNG seed
-- compared EXACT. `test_mse`, `align`, and `state_jac_med` are
floating-point training/evaluation outputs -- compared via the SAME
`numpy.isclose(rtol=1e-5, atol=1e-8)` tolerance as Gates 1-3 (chosen
identically rather than a separate bespoke value, since this is likewise
a same-recipe reproduction, not an independent replication expected to
land in a wider band). **On failure for any (config, seed) pair, the
protocol halts with the explicit named state
`TWO_CSTR_RECONSTRUCTION_INCOMPATIBLE` -- the tolerance is never
loosened to force a pass** (Section 8 restates this).

For every gate: only after it passes for every covered (condition, seed[,
IC]) does that battery's fully-instrumented run proceed, and only its
output is written into the manifest chain. A gate failure blocks that
battery entirely and must be root-caused (an implementation bug, not an
excuse to relax tolerance) before retrying. **Gate record location,
corrected in v9**: each gate's pass/fail record does NOT live in the
Setup Manifest -- Gates 1/2's per-IC results live in the relevant
Battery 1/2 record, Gate 3's in the Battery 3 record, and Gate 4's in
the two-CSTR Replica Manifest (Section 8's phase-3 freeze artifact); the
Setup Manifest instead pins the UPSTREAM evidence chain (freeze/M1/M2/M3/
checkpoints/pool/environment) every battery depends on. Not a formality
either way -- every gate result is still permanently recorded, just not
all in the same file.

## 5. Statistical contract

Shared machinery for batteries 1-4 (single-CSTR): aggregate the 5 ICs
within each seed FIRST, then treat the resulting 10 per-seed values as
paired samples across conditions, using the frozen ablation's own
bootstrap-CI/exact-permutation machinery -- **literal seed `31415`**
(matching `alib.BOOTSTRAP_SEED`, fixed in v3), fresh generator per
contrast, `2^10` permutation enumeration. No Holm correction is required
(not a pre-registered gating family the way the main ablation's is), but
every reported contrast carries its CI, and the family below is fixed
before execution, not selected post hoc.

**Contrast family, corrected in v5 (P0) -- scoped per battery's actual
data coverage, not one family applied uniformly to everything:**

**Aggregation convention, pinned in v8, denominator precision fixed in
v10**: every quantity below is aggregated "5 ICs within each seed FIRST"
per the paragraph above -- for Battery 2's per-ITERATION quantities
specifically (logged at every Adam iteration of every control step of
every IC), this collapses in TWO stages, both fixed here rather than
left to be chosen ad hoc: (1) within one IC's rollout, take the mean
over logged iterations to get one IC-level value; (2) average that
IC-level value across the IC's 5 initial conditions to get one
seed-level value, which is what Section 5's bootstrap/permutation
machinery then treats as a single paired sample. This was left
unspecified in v1-v7 (the design pinned the seeds-level aggregation but
not how per-iteration data within a single IC collapses to a single
per-IC number) and resolved ad hoc during implementation; pinned here.

**The stage-1 (within-IC) denominator is METRIC-SPECIFIC, not uniform,
corrected in v10** -- v8's text asked for one blanket rule ("NON-EXCLUDED
logged iterations" for every quantity), which does not actually fit
every metric this battery reports:
- The Section 3.5 gradient-BRIDGE quantities (`g_hat`, `g_true`, cosine,
  raw inner product, directional derivatives) use ONLY non-excluded
  logged iterations (Section 3.5's `excluded` flag, set when either
  gradient's norm is too small for a direction to be meaningful) --
  exactly as v8 specified.
- The objective-decrease metrics (`j_learned_decrease`,
  `counterfactual_objective_decrease`, Section 6 item 7's before/after
  pair) use EVERY logged iteration with a finite before/after pair --
  `excluded` is a gradient-alignment concept and has no bearing on
  whether an objective value was computed.
- `projection_active_rate` uses every logged iteration (it is a
  first-block clamp indicator, not a gradient quantity).
- `fd_one_sided_rate`/`fd_excluded_rate` use every logged (iteration,
  coordinate) pair across BOTH normalized-action coordinates -- these
  describe the finite-difference boundary-handling contract itself
  (Section 3.5's own boundary/exclusion reporting requirement), not the
  gradient-alignment `excluded` flag, and conflating the two would hide
  exactly the boundary-proximity information this reporting exists to
  surface.

- **Continuous-endpoint metrics** (Battery 1/4's `final_V`, `max_V`,
  `success`, `settling_time`, TV, `lyapunov_increase_count`, state-
  constraint stats -- available for all 8 conditions): **E-B** (primary),
  **F-E** (ceiling/loss-interaction separation -- the reason Battery 1
  covers all 8 conditions), **C-B** (symmetric-residual comparison,
  matching the ablation's `C_vs_B`), **E-FNN-B-FNN** (architecture
  interaction, matching the ablation's `EFNN_vs_BFNN`).
- **Proposition/gradient-bridge quantities** (Section 3.5's `g`,
  `nabla_{u_n} g`, cosine, inner product, directional derivatives --
  Battery 2 covers ONLY B and E, by design): **E-B only**. F, C, and the
  FNN conditions have no gradient-bridge data to contrast at all -- v4's
  wording ("every Section 3.5 gradient-bridge quantity" gets the full
  4-contrast family) asked for data this design cannot produce for any
  condition but B/E.
- **GGN quantities** (Battery 3, B/E only, explicitly descriptive/non-
  gating): report B and E's own summary statistics (with their own CIs
  if useful for context) but **no formal E-B contrast/CI test** --
  consistent with Battery 3 never having been scoped as a causal
  comparison in the first place (Section 2).

D is intentionally left OUT of every family above and treated as
descriptive-only context (consistent with `lambda_jac` still being at
its tuning grid's boundary, per the Decision Record) until the grid
extension (Response Matrix gap #4) is resolved.

**Survivorship-bias rule for continuous-endpoint contrasts, threshold
pinned as a literal in v6 (P0)**: continuous metrics are computed only
over COMPLETED rollouts (Section 3.7); if the two conditions in a
contrast have `abs(completion_rate_A - completion_rate_B) > 0.10`
(a fixed literal, not an "e.g." placeholder -- recorded in the Setup
Manifest), the completed-only sample is no longer a like-for-like
comparison (the
harder cases that failed to complete are excluded asymmetrically), so
that specific contrast is DOWNGRADED to descriptive-only (point values
reported, no CI-based causal claim) rather than treated as a fully valid
paired contrast.

For battery 5 (two-CSTR minimum audit): **not a causal contrast** --
explicitly no equal-query control exists for two-CSTR (Section 0), so no
CI-based causal claim is made. Report `value_only`/`value+grad`
descriptively (mean/min/max across the 10 seeds reconstructed in Section
8, matching the manuscript's actual two-CSTR convention -- corrected in
v4, not `two_cstr_constraint.py`'s separate, smaller 3-seed probe),
explicitly labeled as NOT a rigorously
powered comparison.

## 6. Resolved design decisions

1. **Settling time** -- fixed in v3 (P1, `t=0` and physical-time
   representation): the first step index (`t=0` = the initial state,
   BEFORE any control action, counts as a valid settling point if
   `V(x_0) <= success_V` already) such that `V` drops below
   `success_V` and stays below it for the rest of the rollout. If never
   achieved, the sentinel is `steps + 1` -- reported alongside its
   physical-time equivalent `settling_time_hours = settling_time_steps *
   dt_hr` for completed rollouts, with the `steps+1` sentinel explicitly
   flagged as a STEP-COUNT sentinel, never itself converted to a
   physical-time value that could be misread as a real settling
   duration. Report the full ECDF across ICs/seeds alongside the point/
   CI summary.
2. **GGN denominator mismatch** -- fixed per-accepted-update
   counterfactual (Section 3.4 item 6); real plant decrease stays a
   separate once-per-control-step statistic.
3. **Timing contamination** -- fixed via the three-way split (Section
   3.2): `solver_core_wall_clock` (the actual Adam/GGN optimization
   only) vs. `audit_counterfactual_wall_clock` (the extra instrumentation
   cost from Section 3.4 item 5/6's true-plant rollouts) vs.
   `instrumented_total_wall_clock` (sum). Only `solver_core_wall_clock`
   should ever be cited as "control-update time" for R2.M4.
4. **Per-iteration logging (battery 2, B/E)**: log ALL iterations, no
   subsampling, compressed NPZ arrays. If a smoke-test cost check shows
   the counterfactual-true-objective computation (Section 3.4 item 5) is
   prohibitive, the reduction must be decided and the protocol
   re-frozen BEFORE the real run.
5. **Continuous endpoint battery coverage**: all 8 conditions (battery
   1).
6. **GGN battery coverage**: B/E, 10 seeds x 5 ICs first (battery 3);
   reduce only via an explicit pre-execution freeze update.
7. **Counterfactual-objective formula, pinned in v3 (P1)**: must be
   EXACTLY the learned objective's functional form -- same physical-plan
   conversion (`un -> u0` via `xs[2:]`/`xm[2:]`), same true-plant
   rollout structure (`horizon` steps), same `V`-term (`y @ Pt @ y`), and
   the same normalized-input penalty `rho_u * (un[h]**2).sum()` -- the
   only difference from the learned objective is substituting `SIM.step`
   for the learned surrogate `seq(...)`. Any deviation from this exact
   form (e.g. omitting the `rho_u` penalty) makes the learned-vs-
   counterfactual comparison apples-to-oranges.
8. **State-constraint audit**: piggybacked on the continuous endpoint
   battery (single-CSTR, all 8 conditions); battery 5 handles two-CSTR
   separately (Section 8).
9. **Temperature-violation integral units**: `dt_hr * sum(max(0,
   T_deviation - cap_deviation))`, `dt_hr = 1e-3` hours by default
   (confirmed by direct read of `lcnn_paper_simulator.py`).
10. **Input-saturation tolerance/denominator**: proportional to each
    actuator's own range (Section 3.6), NOT an absolute constant;
    denominator is total control steps, reported per actuator.
11. **Failure handling**: `settling_time` uses the `steps+1` sentinel;
    every other continuous metric is reported only over completed
    rollouts, alongside an explicit `completion_rate`.
12. **Actual compute backend**: record the tensor device actually used
    (confirmed CPU throughout the relevant files, per Section 4), not
    MPS/CUDA "availability."

## 7. Solver-instrumentation applicability note (for direct use in the response letter)

Projected Adam has no line-search or accept/reject step -- "line search"
and "accepted-step fraction" are **not applicable** to the Adam battery
and must be stated as such explicitly in the response letter.
Backtracking and acceptance/rejection are reported ONLY for the GGN
battery (battery 3), which genuinely has those mechanics.

## 8. Two-CSTR minimum audit (battery 5, small and non-causal, REWRITTEN in v5)

**Purpose**: close R2.M6/R3.8's minimum disclosure bar (temperature
maxima, input saturation, feasibility) for the two-CSTR benchmark, since
neither reviewer comment is single-plant-specific and the current
manuscript discusses both. This is explicitly NOT an equal-query causal
experiment and must never be presented as one.

**No checkpoints exist**: direct read of `scripts/two_cstr_lgrad.py`
confirms `main()` calls `train(seed, ...)` fresh for every (config,
seed) pair, evaluates the returned in-memory `model` for `test_mse`/
`align`/`frac_neg`/`b3`/`b20`/`state_jac_med`, and writes ONLY those scalar metrics
to `results/interim/logs/two_cstr_lgrad.json` -- there is no
`torch.save()` call anywhere in the file. The only checkpoints anywhere
in the repository are the real go/no-go ablation's single-CSTR ones.
This battery is therefore not free; it requires four sequential phases:

1. **Deterministic reconstruction, precisely named in v5 (P0)**: retrain
   `value_only` and `value+grad` (the two conditions this audit needs;
   `grad_only` is not required for the R2.M6/R3.8 disclosure purpose)
   using EXACTLY `two_cstr_lgrad.py`'s existing recipe -- same `configs`
   entries, same `SEEDS = list(range(10))` (10 seeds, matching the
   manuscript's actual two-CSTR convention, not `two_cstr_
   constraint.py`'s separate 3-seed probe), same `torch.manual_seed(seed)`
   call, same `n_data=12000`/`epochs=70` -- with ONE code addition:
   `torch.save()` the trained model's `state_dict()` after training,
   hash it, and record the hash. **Since no original checkpoint from
   whatever produced `two_cstr_lgrad.json`'s existing numbers was ever
   saved, this new model is NOT "the original model recovered" -- it is
   a `metric_compatible_replica`, a freshly-trained model whose only
   claim is reproducing the ALREADY-LOGGED scalar metrics under the same
   code and seed. This term must be used in any manifest/manuscript text
   describing it, never "the original two-CSTR model" or similar
   language implying continuity with a specific prior artifact that in
   fact never existed as a file.** This is genuinely new execution (10
   seeds x 2 configs = 20 trainings), not free, but far smaller than a
   query-matched redesign.
2. **Gate 4 (Section 4a)**: the freshly reconstructed replicas'
   `test_mse`/`align`/`frac_neg`/`b3`/`b20`/`state_jac_med` (recomputed with the
   same evaluation code `two_cstr_lgrad.py`'s `main()` already uses)
   must match `results/interim/logs/two_cstr_lgrad.json`'s existing
   per-(config, seed) values -- `b3`/`b20`/`frac_neg` EXACT, `test_mse`/
   `align`/`state_jac_med` via `numpy.isclose(rtol=1e-5, atol=1e-8)`, per
   Section 4a's pinned tolerance contract (not a value deferred to
   implementation time).
   **On failure for any (config, seed) pair, the protocol halts with the
   explicit named state `TWO_CSTR_RECONSTRUCTION_INCOMPATIBLE` -- the
   tolerance is never loosened to force a pass; a failure means either a
   reconstruction bug or that the replica genuinely cannot stand in for
   the original numbers, and either way the minimum audit cannot proceed
   on that (config, seed) until root-caused.**
3. **Freeze the new checkpoints**: once Gate 4 passes for all 20
   (config, seed) pairs, the replica checkpoints and their hashes become
   the fixed input to step 4 -- recorded in this battery's own Setup
   Manifest (Section 4), analogous to how the main ablation's frozen
   checkpoints anchor batteries 1-4.
4. **Instrumented state-constraint audit**: run BOTH `value_only` and
   `value+grad` (at `rho_c=0.0` for a clean apples-to-apples baseline;
   the existing `rho_c` sweep remains available as separate descriptive
   context, not the primary comparison here) through an instrumented
   rollout logging the Section 3.6 state-constraint fields (per-reactor
   absolute/deviation temperature, saturation, `temperature_cap_
   satisfied_posthoc`) plus `solver_completed`, across all 10
   reconstructed seeds.

This is more work than "reuse existing checkpoints" (v3's incorrect
assumption), but still substantially smaller than a full two-CSTR
equal-query redesign (Section 10) -- no NEW loss functions, query
contracts, or FNN/D-style conditions are needed, only reproducing two
ALREADY-DEFINED training configs with a checkpoint saved this time.

## 9. Manuscript claim scoping (new in v3)

- The causal equal-query conclusion (E vs. B, +0.34, p=0.0156) is
  established ONLY on single-CSTR; it must not be described as
  independently confirmed, replicated, or established on two-CSTR.
- Two-CSTR results (existing GGN/state-constraint work, plus this
  protocol's battery 5) should be reframed as SUPPORTIVE or TRANSFER
  evidence -- consistent with, but not independently causally
  establishing, the single-CSTR mechanism -- since they were never run
  under an equal-query control.
- Any manuscript language implying the mechanism was "independently
  confirmed on two plants" or similar must be located (a text-search
  pass against `manuscript_jpc/paper_h_jpc.tex` for phrasing along these
  lines is a prerequisite writing task, not yet performed) and revised
  to state the single-CSTR/two-CSTR evidentiary asymmetry explicitly.

## 10. Deferred: two-CSTR equal-query expansion

Explicitly OUT OF SCOPE for the current revision cycle. Rationale
(user-provided, recorded for the record): the equal-query rebuttal's
core burden is already discharged by the single-CSTR 10-seed result;
R2 explicitly permits narrowing generality language INSTEAD of adding a
third structural benchmark (R2.M5); two-CSTR shares the single-CSTR's
reaction kinetics, so a large new two-CSTR ablation would not
meaningfully answer the "structurally different benchmark" critique
either; and more directly necessary work remains (oracle/converged-NLP,
timing, sensitivity sweeps, and the manuscript/response text itself).
Estimated cost if ever undertaken: design/freeze 1-2 days, query-matched
data/training/checkpoint implementation + smoke 2-3 days, real training +
GGN/Adam evaluation several hours to 1 day, verification/statistics/
manuscript integration 1-2 days -- a minimum of 4-7 working days against
a ~19-day total budget, more if results are ambiguous or defects are
found (as every prior round of this project's protocols has found real
defects on review). Revisit ONLY after the single-CSTR audit (this
document), the oracle/converged-NLP item, and the core response-letter
writing are complete and schedule allows; if undertaken even then, scope
strictly to a B/E-only, 10-seed targeted transfer study, not a full
8-condition repeat of the single-CSTR design.

## v3 -> v4 corrections

A review of v3 found 4 P0 issues, all independently re-verified against
actual code before being incorporated, plus 4 P1 precision items:

1. **v3's core assumption that two-CSTR checkpoints already exist was
   false.** Direct re-read of `scripts/two_cstr_lgrad.py` confirms
   `main()` trains fresh every run and never calls `torch.save()` -- only
   scalar metrics are persisted to `two_cstr_lgrad.json`, never the
   trained model. Fixed: Section 8 rewritten around a 4-phase process
   (deterministic reconstruction with checkpoint-saving added,
   legacy-result compatibility gate, freeze the new checkpoints, THEN
   the instrumented audit) -- smaller than a full equal-query redesign,
   but not free.
2. **Battery 5's seed count (3) didn't match the manuscript's actual
   two-CSTR convention (10, `two_cstr_lgrad.py`'s `SEEDS = list(range
   (10))`, confirmed by direct read)** -- the 3-seed count belonged to
   the separate, smaller `two_cstr_constraint.py` probe, not the main
   comparison this audit needs to support. Fixed: Section 2/5/8 all
   corrected to 10 seeds.
3. **The statistical contract (Section 5) fixed the reported family to
   E-vs-B alone, but Battery 1's entire justification for covering all 8
   conditions (separating E from F's ceiling-saturated success) requires
   an E-vs-F contrast, which was missing** -- without it the audit could
   not actually answer the question its own coverage was designed to
   answer. Fixed: contrast family expanded to E-B (primary), F-E
   (ceiling separation), C-B (symmetric residual), E-FNN-vs-B-FNN
   (architecture interaction); D stays descriptive-only pending the
   `lambda_jac` grid extension.
4. **No gate verified that a battery's REIMPLEMENTED solver loop
   actually reproduces the original, already-logged behavior before its
   instrumented output was trusted** -- a reimplementation bug in
   Battery 2/3/5's instrumented loops could silently characterize a
   DIFFERENT controller than the one that actually produced the
   manuscript's headline numbers. Fixed: Section 4a's legacy-result
   compatibility gate, required before any reimplemented battery's
   logging-enabled run is trusted.

P1: the one-sided FD's order was unspecified, risking an order mismatch
between boundary (first-order) and interior (second-order central)
gradient-error estimates -- fixed to a pinned 3-point second-order
one-sided formula (Section 3.5). Timing lacked a measurement protocol
(`perf_counter`, warm-up exclusion, fixed thread count, no concurrent
load, median/P95 reporting) -- fixed (Section 3.2). The physical sampling
period was not stated in seconds for reviewer-facing comparison -- fixed
(`dt_hr = 1e-3` h `= 3.6` s, Section 3.2). Battery 1 and Battery 2 could
have redundantly re-run the same B/E rollouts under two independently-
executed code paths -- fixed via an explicit dedup contract (Section 2)
making Battery 2's detailed run for B/E the sole source, with Battery 1
merging those results rather than re-executing them.

## v4 -> v5 corrections

A review of v4 found 4 P0 contractual contradictions, all independently
re-verified against actual code before being incorporated:

1. **The single "legacy-result compatibility gate" (v4 Section 4a)
   required per-IC data that does not exist in M2/M3** -- confirmed
   `_closed_loop_success_rate()` returns only a seed-level mean across
   the 5 ICs, never persisting per-IC `final_V`. Fixed: Section 4a now
   defines Gate 1 (controller-equivalence -- direct per-IC comparison of
   the frozen `closed_loop()` against the new evaluator on the SAME
   checkpoint/seed/IC, independent of M2/M3) and Gate 2 (legacy-summary
   -- the new evaluator's seed-level aggregate compared against M2's
   actual recorded seed-level value) as two separate checks.
2. **GGN has no legacy result on the query-matched B/E checkpoints at
   all** -- confirmed `ggn_mpc_probe.py`'s `main()` trains its own,
   different (EAAI-era, non-query-matched) models via `L.train(...)`,
   entirely independent of the real ablation's checkpoints. Fixed: Gate
   3 (implementation-equivalence) compares the ORIGINAL, unmodified
   `ggn_closed_loop()` against the NEW instrumented version on the SAME
   (real ablation) checkpoints -- an internal old-code-vs-new-code check,
   not a comparison against any external prior JSON.
3. **The statistical contract (v4 Section 5) required the full 4-
   contrast family (E-B, F-E, C-B, E-FNN-B-FNN) for "every Section 3.5
   gradient-bridge quantity," but Battery 2 (which produces Section 3.5's
   data) only ever covers B and E** -- asking for gradient-bridge
   contrasts involving F, C, or the FNN conditions requests data no
   battery in this design produces. Fixed: contrast family now scoped
   per data source -- continuous-endpoint metrics get all 4 contrasts,
   Proposition/gradient-bridge quantities get E-B only, GGN quantities
   get descriptive reporting with no formal contrast at all.
4. **The reconstructed two-CSTR model's identity was unstated, and
   there was no explicit named failure state if the legacy-scalar check
   failed.** Fixed: Section 8 now requires the term
   `metric_compatible_replica` (never "the original model") for the
   reconstructed checkpoints, and Gate 4's failure mode is the explicit
   `TWO_CSTR_RECONSTRUCTION_INCOMPATIBLE` state -- tolerance is never
   loosened to force a pass.

P1: a survivorship-bias rule was added (Section 5) downgrading any
continuous-endpoint contrast to descriptive-only when the two conditions'
`completion_rate` differ materially, since completed-only continuous
metrics are not a like-for-like comparison under asymmetric non-
completion. Warm-up step count and thread count, left as placeholders in
v4, are now pinned literals (2 warm-up steps, `torch.set_num_threads(1)`,
Section 3.2). One-sided-FD usage/exclusion rates and projection-activity
fraction are now required to be reported per (condition, seed) rather
than only in aggregate, since exclusions disproportionately affect
exactly the projection-active region this audit targets (Section 3.5).

Per the user's guidance accompanying this review: once these four P0s
and the P1 items are fixed, this protocol's scope (single-CSTR query-
matched causal audit + two-CSTR reconstructed-replica minimum audit)
should not be widened further -- it is judged strategically sufficient
for the reviewers' requirements without a full two-CSTR equal-query
study, which would risk both the focus and the schedule of this major
revision (Section 10 already defers that expansion).

## v5 -> v6 corrections

The reviewer's own characterization of this round: the core structure
(Sections 2/4a/5/8's gate/battery/contrast design) is settled; the
remaining 3 P0s are contractual precision gaps, not new structural
problems. All independently re-verified against actual code:

1. **Two literal values were left as unpinned placeholders**: the
   completion-rate survivorship-bias threshold (Section 5, was "e.g.
   differing by more than 1 seed's worth... exact threshold to be
   pinned at implementation time") and Gate 4's tolerance (Section 4a/8,
   was "a PRE-REGISTERED tolerance (value to be pinned at implementation
   time)"). Both fixed to literals now: `abs(completion_rate_A -
   completion_rate_B) > 0.10` for the survivorship rule; `numpy.isclose
   (rtol=1e-5, atol=1e-8)` for Gate 4's float fields. Deciding a
   tolerance only after seeing implementation results would be circular
   -- a threshold chosen to make a specific run pass is not a threshold
   at all.
2. **Gates 1 and 3 had no defined split between exact and
   floating-point comparison.** Fixed (Section 4a): booleans/integers
   compared EXACT (`==`), continuous floats via the same `numpy.isclose
   (rtol=1e-5, atol=1e-8)` used for Gate 4 -- justified identically
   across all four gates since every comparison is the SAME checkpoint
   evaluated by code differing only in instrumentation, never two
   independently-trained models that would warrant a wider "acceptable
   model variation" band.
3. **Two field-name mismatches between this document's prose and the
   actual source code's dict keys, confirmed by direct read:**
   - `two_cstr_lgrad.py` line 313 stores `state_jac_med` as the actual
     JSON key -- this document had written `Jx` (the local Python
     variable name used only for console printing) in three places
     (Sections 5, 8, and Gate 4's description). All three corrected to
     `state_jac_med`. Left uncorrected, an exact-key comparison in Gate
     4 would have failed on every run, not because of a real
     reconstruction defect but because the document itself named the
     wrong key.
   - `budget_sweep_solver.py` line 86 returns `violations`, not this
     document's renamed `lyapunov_increase_count` (Section 1's naming
     correction is a MANUSCRIPT/schema-level rename, not something that
     exists in the source dict). Gate 1's comparison logic must
     explicitly map `violations <-> lyapunov_increase_count`, now stated
     directly in Gate 1's description, rather than relying on an
     implicit exact-key match between the two evaluators' dicts.

## v6 -> v7 corrections

v6's review found no remaining P0s and judged the design ready for
implementation; two non-blocking wording fixes, both applied here:

1. Section 4a said "Three gates, each targeting what actually CAN be
   verified" immediately before listing four (Gates 1-4). Fixed to
   "Four gates."
2. The tolerance contract's justification ("both sides of every gate
   comparison are literally the same checkpoint evaluated by code
   differing only in whether logging is enabled") is accurate for Gates
   1-3 but not Gate 4, which compares a newly-trained
   `metric_compatible_replica` against a previously-logged scalar from a
   run whose original checkpoint was never saved -- a deterministic
   same-recipe replica-compatibility check, not a same-checkpoint one.
   Fixed: the justification is now split, with Gate 4 given its own
   (still using the same numeric tolerance value, but for a distinct
   reason).

Per the user's confirmation, this protocol is now ready to proceed to
implementation -> synthetic smoke -> full re-review -> freeze -> real
execution, in that order.

## v7 -> v8 corrections

Implementation began after v7. Two implementation-review rounds followed
(round 1: 7 P0s + 1 P1, all real, in the production orchestration code --
missing resume-or-verify, a B/E dedup-contract violation, an incomplete
Adam/GGN logging schema, a genuine GGN timing-contamination bug, missing
two-CSTR phase ordering, incomplete Setup Manifest provenance, and the
single-thread pin never being called; round 2: 6 more P0s + 1 P1 --
resume-or-verify not actually re-checking upstream evidence, the two-CSTR
"freeze" step never producing a real manifest, Section 5's aggregation
falling far short of the pre-registered metric/contrast scope, an
inverted failure-denominator rule, still-missing raw per-step/per-
iteration data, and residual timing contamination inside the "detailed"
Adam/GGN logging passes). Full per-round technical detail is kept in this
project's session memory (`project_paper_h_status.md`), not duplicated
here -- those are implementation defects against an already-clear
specification, not gaps in this document's own text. Two decisions,
however, WERE genuinely left unspecified by v1-v7's text and had to be
resolved ad hoc while implementing; both are now pinned as explicit
literals rather than left implicit:

1. **GGN objective choice** (Section 2 item 3): `objective="V"`, matching
   `ggn_mpc_probe.py`'s own default.
2. **Battery 2's per-iteration -> per-IC -> per-seed aggregation
   convention** (Section 5): mean over non-excluded iterations within an
   IC, then mean over the IC's 5 initial conditions.

Status remains DRAFT, NOT FROZEN -- the implementation-review cycle is
still open as of v8; freeze happens only once a review round finds no
remaining P0s, per this protocol's own Section 0 discipline.

## v8 -> v9 corrections

A third implementation-review round found 5 more P0s (all in the
production orchestration code, not in this document's own text --
missing signed before/after decrease for the learned/counterfactual
objective-change metrics; Battery 1/3/5 Summary aggregation still short
of the pre-registered metric set per battery; Summary resume not
re-verifying its own companion NPZ files; the survivorship-rule gap
computed as a max of PER-SEED completion-rate differences instead of the
protocol's own OVERALL per-condition rate; and NaN CI/p-values silently
computed instead of an explicit incomplete-evidence status when a paired
seed's difference is non-finite) + 2 P1s. Full technical detail in this
project's session memory, per this document's established convention of
not duplicating implementation-bug history here. The ONE genuine textual
inaccuracy in this document's own text, now fixed: Section 4a claimed
"each gate's pass/fail record lives in the Setup Manifest," which was
never actually true of the implementation as designed (Gates 1-3 live in
their respective battery records, Gate 4 in the two-CSTR Replica
Manifest) -- corrected in place in Section 4a rather than left standing.

Status remains DRAFT, NOT FROZEN -- freeze happens only once a review
round finds no remaining P0s.

## v9 -> v10 corrections

A fourth implementation-review round found 3 more P0s (all in the
production orchestration code, not in this document's own text --
`input_box_satisfied` was never actually implemented despite Section 1
naming the concept; Battery 5's two-CSTR "maximum temperature" summary
averaged 5 ICs per seed before taking a max, diluting a single IC's
genuine worst excursion, and never reported the absolute-K maximum at
all; the Adam battery's `rollout_instrumented_total_wall_clock_seconds`
was built from the CLEAN pass's core timing instead of the LOGGED pass's
own total, silently undercounting the real instrumented cost by exactly
the new `j_learned_after` evaluation this round's v9 predecessor added)
+ 2 P1s, all fixed. Full technical detail in this project's session
memory. Two genuine textual gaps in this document's own text, now fixed:

1. **Section 1**: the input-box field this section required was never
   given a concrete name (only the temperature field was pinned) --
   `input_box_satisfied` is now pinned explicitly, with its actual
   semantics (never-saturated-for-the-whole-rollout, not a box-violation
   check, since the box is never actually violated by construction).
2. **Section 5's Battery-2 aggregation paragraph** (added in v8) asked
   for one blanket "non-excluded logged iterations" denominator for
   EVERY per-iteration quantity, which does not fit the objective-
   decrease/projection-rate/FD-rate metrics (the gradient-alignment
   `excluded` flag has no bearing on any of those) -- corrected to state
   the denominator per metric family.

Status remains DRAFT, NOT FROZEN -- freeze happens only once a review
round finds no remaining P0s.

## v10 -> v11 corrections

A fifth implementation-review round found 2 more P0s + 1 P1, all fixed.
Full technical detail in session memory. One genuine error in THIS
document's own v10 text, now fixed:

1. **Section 1's `input_box_satisfied` definition (added in v10) was
   itself wrong.** v10 defined it as "never saturated," conflating box-
   FEASIBILITY (satisfying `lo <= u <= hi`, which a boundary value still
   does) with never being PINNED at the boundary (a solver-behavior
   diagnostic, not a feasibility statement). Corrected: `input_box_
   satisfied` is now a genuine feasibility check; the diagnostic v10
   actually computed is renamed `input_never_saturated` and kept as a
   separate field, never sharing the "satisfied" framing.

The other 2 fixes (a solver-audit Freeze Manifest hash never being
recorded into either Setup Manifest despite both entry points verifying
it, breaking post-hoc traceability of which frozen protocol version
produced a given result; the two-CSTR Setup Manifest missing the full
software/hardware environment the single-CSTR one already records) were
implementation gaps against an already-clear specification, not gaps in
this document's own text.

Status remains DRAFT, NOT FROZEN -- freeze happens only once a review
round finds no remaining P0s.

