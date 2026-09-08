# JPC Oracle-Gradient / Converged-NLP Comparison (R2.M4) — Decision Record

Status: FIXED, based on the real, frozen-protocol CORRECTIVE execution.
This record is descriptive of a completed real result; it is not itself
part of either Freeze Manifest's hash-locked file set and may be
extended (never retroactively altered) as follow-up work is completed.
Mirrors the provenance/structure convention of
`docs/JPC_ABLATION_DECISION_RECORD_2026_09_01.md` and
`docs/JPC_SOLVER_AUDIT_DECISION_RECORD_2026_09_03.md`.

## 0. Provenance

- Design protocol: `docs/JPC_ORACLE_NLP_COMPARISON_PROTOCOL_2026_09_03.md`,
  v12 (10 design-review rounds, v1-v10; implementation-review rounds
  v11-v12, Sections 22-24).
- **CORRECTIVE Freeze Manifest** (the only one whose results are
  confirmatory): `docs/JPC_ORACLE_NLP_COMPARISON_PROTOCOL_FREEZE_MANIFEST_2026_09_03_CORRECTIVE.json`,
  `canonical_payload_sha256` = `fa56e4d31e472eca2093461b23599b19c1c509df480bfe720154af2ef55d7e70`,
  8 files, smoke 56/56. Independently `shasum -a 256`-verified against
  all 8 covered files.
- Real execution: `run_real_oracle_nlp_comparison()`
  (`scripts/oracle_nlp_driver_2026_09_03.py`), namespace
  `results/oracle_nlp_execution_2026_09_03_corrective/`, 273.35s
  wall-clock.
- Summary Manifest: `oracle_nlp_summary_manifest_2026_09_03.json`,
  self-hash `bb3ef491b8d264f335047ff8a2707baecc1dd5accfe2d6faae751a306adfa377`
  (independently re-verified via `read_and_verify_json_artifact`, not
  trusted from a printed label).
- Timing Manifest: `oracle_nlp_timing_2026_09_03.json`, self-hash
  `01b17aa6a5f6622530c6e44c317b49117727358cd86b759fa40e8c5006534486`
  (independently re-verified).
- Zero edits to any of the 12 go/no-go-ablation frozen files or the 11
  solver-audit frozen files -- re-verified via direct `file_sha256`
  recomputation against both of those Freeze Manifests immediately
  before and after the corrective execution.

## 1. Preliminary execution — excluded from confirmatory evidence, preserved as historical record only

A first real execution (217.00s wall-clock, all six preflight gates
real-passed, `optimizer_gap`/`gap_B`/`gap_E`/decomposition-identity all
computed) ran under a v11 freeze
(`docs/JPC_ORACLE_NLP_COMPARISON_PROTOCOL_FREEZE_MANIFEST_2026_09_03_PRELIMINARY.json`).
A post-execution review found the v11 freeze package to be **internally
inconsistent** -- hash integrity was intact throughout (every covered
file matched its recorded hash both then and re-confirmed now), but the
frozen DESIGN TEXT and frozen CODE disagreed on a numeric contract value
(`tol_primal`, Section 2 below), and two further manifest-chain
contracts (Sections 24, item 3 below) were not correctly implemented.
**Hash verification alone cannot detect this class of defect** -- it
requires a content/semantics review, which is why this was caught only
after the fact, not by any automated gate.

Per explicit instruction, the preliminary execution and its Freeze
Manifest are **preserved, unmodified, under renamed paths**
(`results/oracle_nlp_execution_2026_09_03_preliminary/`,
`docs/JPC_ORACLE_NLP_COMPARISON_PROTOCOL_FREEZE_MANIFEST_2026_09_03_PRELIMINARY.json`)
as a historical/diagnostic record. **It is NOT confirmatory evidence and
must not be cited as such in any manuscript or response-letter text.**
Only the corrective execution (Sections 3 onward) is confirmatory.

## 2. `tol_primal` corrective revision — cause, evidence, timing

v11's design text (Sec. 9 and Sec. 16) stated `tol_primal=1e-8` as a
"CONFIRMED, pre-registered" value. The actual implementation already
used `tol_primal=1e-6` (`src/cstr/oracle_nlp_io.py`) -- changed DURING
IMPLEMENTATION, **before the preliminary execution was ever run**, for
a concrete, observed reason: a real IPOPT solve at `ipopt.tol=1e-8`
left a residual bound violation of `~1.7e-8` at an active box
constraint (directly observed on condition B seed 0's near-boundary
solution), which a `tol_primal=1e-8` feasibility check would have
rejected as spurious infeasibility on an otherwise correct
`Solve_Succeeded` solution. `1e-6` gives ~2 orders of magnitude margin
over the observed `1.7e-8` slack.

**This was a genuine, evidence-based RELAXATION (loosening) of the
tolerance, decided before any confirmatory result existed** -- not a
post-hoc loosening in response to how results looked, which would be a
form of data-dependent tuning this project's own discipline explicitly
prohibits. The design document was simply never updated to match the
code at the time -- an oversight, corrected in v12 (Sec. 23 item 1 of
the design doc) by updating the design text's `tol_primal` value from
`1e-8` to `1e-6` in both locations, and correcting an accompanying code
comment that had described the change as "adjusted DOWN" (backwards:
`1e-6` is a LARGER/looser tolerance than `1e-8`, i.e. a relaxation, not
a reduction).

## 3. Reproducibility — preliminary vs. corrective results are numerically identical

Despite three real contract fixes between the preliminary and corrective
runs (Sec. 23 items 1-3 of the design doc: `tol_primal` doc/code
reconciliation, per-IC NLP-solver-rebuild fix, timing-pass/companion-NPZ
completion), **the corrective run's `optimizer_gap`, `gap_B`, `gap_E`
point estimates, confidence intervals, and Holm-adjusted p-values are
identical to full float precision** to the preliminary run's:

| quantity | preliminary | corrective |
|---|---|---|
| `optimizer_gap` point estimate | `0.0005955525750385866` | `0.0005955525750385866` |
| `optimizer_gap` per-IC range | `0.0009878810833807475` | `0.0009878810833807475` |
| `gap_B` point estimate | `43.64168070155425` | `43.64168070155425` |
| `gap_B` 95% CI | `[10.886289423923623, 93.75914934357192]` | identical |
| `gap_B` Holm-adjusted p | `0.00390625` | `0.00390625` |
| `gap_E` point estimate | `0.180310460621094` | `0.180310460621094` |
| `gap_E` 95% CI | `[0.09337934877049477, 0.2857382391957021]` | identical |
| `gap_E` Holm-adjusted p | `0.00390625` | `0.00390625` |
| decomposition-identity check | `True` | `True` |

This confirms the three fixed defects were **provenance/contract
violations, not result-corrupting bugs** -- none of them altered any
certification outcome, gradient computation, or aggregation in this
particular real dataset. The corrective re-execution was still
necessary: a result computed under an internally-inconsistent frozen
package cannot be trusted as confirmatory merely because it happens to
reproduce under the fix, and the missing timing-pass linkage (Sec. 24)
did not exist in the preliminary run at all.

## 4. Primary statistical result (corrective execution, confirmatory)

Primary endpoint: `final_V` (lower is better, cost-like), Sec. 12a.

- **`optimizer_gap = mean_IC(Oracle20_final_V - Certified_NLP_final_V)`
  = `0.000596`** (per-IC range `0.000988`) -- effectively zero relative
  to the scale of `gap_B`. Oracle-20 (finite 20-step projected-Adam
  budget, EXACT true gradient) reaches almost exactly the same
  terminal cost as Certified-NLP (IPOPT-certified local-stationary
  reference) on all 5 ICs.
- **`gap_B = mean_IC(B_final_V) - Oracle20_mean_final_V` (seed-first,
  10 seeds) = `43.64`, 95% CI `[10.89, 93.76]`, Holm-adjusted exact
  p = `0.0039`** -- large, statistically significant. Condition B
  (value-only learned surrogate) is far from the Oracle-20 reference at
  the SAME finite budget.
- **`gap_E = mean_IC(E_final_V) - Oracle20_mean_final_V` = `0.180`,
  95% CI `[0.093, 0.286]`, Holm-adjusted exact p = `0.0039`** --
  statistically significant but roughly 240x smaller in magnitude than
  `gap_B`. Condition E (value+grad learned surrogate) is much closer to
  the Oracle-20 reference than B is, though not exactly at it.
- **Decomposition-identity check: PASSED** (`gap_X + optimizer_gap ==
  mean_IC(X) - mean_IC(Certified_NLP)` to `<1e-9` for both X in
  {B, E}) -- the mandatory implementation-level sanity assertion that
  would have blocked reporting on any indexing/aggregation bug found no
  such bug.
- Secondary descriptive endpoints (all 5 ICs, both conditions):
  Oracle-20 and Certified-NLP both achieve `success=True` on every IC;
  Certified-NLP shows zero Lyapunov-increase events on any IC (consistent
  with it being a certified local-stationary reference), while Oracle-20
  shows 47-53 Lyapunov increases per IC (consistent with normal
  finite-budget exploratory behavior of a first-order method); control
  TV is modestly higher for Oracle-20 (4.4-5.2) than Certified-NLP
  (1.3-2.0) on every IC.

## 5. B/E reproduction gate — 100/100 passed

The fresh, same-session B/E timing re-execution (Sec. 6 below) was
required to reproduce the frozen Battery 1 raw per-IC summary
(`success`, `final_V`, `max_V`, `tv`, `input_tv_per_actuator_physical`,
`input_tv_per_actuator_normalized`, exact/isclose split per field) for
**all 10 seeds x {B, E} x 5 ICs = 100 raw records**. Result:
**`be_reproduction_gate_passed = True`**, zero mismatches. This confirms
the corrective `LearnedModelTimingController` (a fresh wrapper around
`solver_audit_lib._adam_mpc_step`, that frozen function called as-is,
never copied) reproduces the exact frozen closed-loop behavior for
every one of the 20 B/E controllers.

## 6. Timing — deadline exceedance and measurement provenance

- **Oracle-20**: `deadline_exceedance_fraction = 0.0` across all 5 ICs.
  Per-IC clean-pass median step time `12.5-13.2 ms` (p95 `13.1-13.7 ms`)
  -- ~275-290x margin under the literal `dt_hr=1e-3` h = 3.6 s
  sampling-period deadline.
- **Certified-NLP**: `deadline_exceedance_fraction = 0.0` across all 5
  ICs. Per-IC clean-pass median step time `141-155 ms` (p95
  `184-212 ms`) -- ~17-25x margin under the 3.6 s deadline.
- **Measurement provenance, explicit** (Sec. 24 of the design doc):
  Oracle-20's and Certified-NLP's timing figures above come from their
  OWN already-completed clean reference pass (Sec. 7/9 of the design
  doc) -- NOT from a re-execution inside the B/E interleaved loop. The
  20 B/E controllers (10 seeds x {B, E}) were re-executed together in a
  SEPARATE interleaved pass (`be_interleaved_wall_clock_seconds =
  61.20`), with a deterministic `(step*5+ic) % 20` rotation ensuring no
  single B/E seed is systematically first/last across the whole
  campaign. **Oracle-20/Certified-NLP's timing is therefore NOT
  literally interleaved with B/E's fresh re-execution at the wall-clock
  level** -- this is a disclosed scope reduction from the design
  protocol's originally-specified full 22-way interleaving (Sec. 24),
  adopted because interleaving Oracle-20/Certified-NLP into the B/E
  timing loop would have required a THIRD execution of their clean
  pass, violating the stricter, more-reviewed "exactly two executions
  total" rule (Sec. 7/9). Any citation of these timing figures must
  state this measurement-provenance split explicitly, not claim a
  single unified 22-way interleaved measurement occurred.

## 7. Execution ownership — clean-pass reuse, no double-execution

Oracle-20 and Certified-NLP each executed exactly TWO times per IC, per
the design protocol's own contract: (1) a CLEAN pass, which
simultaneously serves as (a) the reference rollout whose `final_V` feeds
Section 4's gaps, and (b) the source of the Section 6 timing figures --
the SAME execution, not two separate runs; (2) an INSTRUMENTED pass
(full per-iteration/per-candidate logging), used ONLY for the
endpoint-equivalence gates (comparing clean vs. instrumented trajectories
step-by-step: `u0`, final plan, realized next state, `V_after`,
`minimum_stage_temperature` -- both gates passed with zero mismatches
for all 5 ICs of both conditions). B and E have no instrumented pass of
their own in this protocol; they are re-executed once each (within the
20-controller interleaved loop) solely for the Section 6 timing
measurement, since their non-timing endpoints already exist, frozen,
from the original Battery 1 execution (re-confirmed exactly via
Section 5's 100/100 reproduction gate, not recomputed for the gap
analysis).

## 8. Freeze integrity — three independent file sets, all re-verified

Immediately before and after the corrective real execution:

- Go/no-go ablation Freeze Manifest (12 files,
  `19bd6cab...3242a`): **intact**, every file's `file_sha256`
  recomputation matched.
- Solver-audit Freeze Manifest (11 files, `43a2f92d...ceac0`): **intact**.
- This protocol's CORRECTIVE Freeze Manifest (8 files,
  `fa56e4d3...`): **intact**, and independently cross-checked via
  `shasum -a 256` (a distinct tool from the Python `hashlib`-based
  self-verification) against all 8 covered files, all matching.

The PRELIMINARY Freeze Manifest (v11, `docs/JPC_ORACLE_NLP_COMPARISON_PROTOCOL_FREEZE_MANIFEST_2026_09_03_PRELIMINARY.json`)
now shows drift against 5 of its 8 covered paths (the shared design
document and three shared code files, intentionally revised for the v12
corrective fixes) -- this is EXPECTED and does not indicate corruption:
the two artifacts unique to the preliminary results directory
(`oracle_nlp_compat_sample_2026_09_03.npz`,
`oracle_nlp_preflight_gate_results_2026_09_03.json`, re-located under
`results/oracle_nlp_execution_2026_09_03_preliminary/`) were directly
re-hashed at their new path and still match their originally-recorded
hashes exactly -- confirming the preliminary results were preserved
without modification, only the shared LIVE source files were (correctly)
revised.

## 9. Defensible manuscript-level conclusion (current evidence ceiling)

> Under an equal offline identification-query budget (per the go/no-go
> ablation) and an equal finite optimizer budget (`BUDGET=20`,
> `HORIZON=3`, per this protocol), a controller using the EXACT true
> dynamics/objective gradient (Oracle-20) reaches almost the same
> terminal cost as a converged, IPOPT-certified local-stationary
> reference (Certified-NLP) on the tested single-CSTR benchmark's 5
> initial conditions -- i.e., the finite 20-step projected-Adam budget
> itself is NOT the dominant bottleneck when the gradient is exact.
> Under the SAME finite budget, the value-only learned surrogate
> (condition B) is far from that Oracle-20 reference (mean gap 43.6,
> Holm-adjusted p=0.004), while the value+gradient-consistency surrogate
> (condition E) is roughly 240x closer to it, though still measurably
> different (mean gap 0.18, Holm-adjusted p=0.004).

**Interpretation limits, explicit and binding** (per the design
protocol's own Section 0 constraint, unaffected by any of the
corrective fixes above): Oracle-20 replaces the ENTIRE learned model --
dynamics, objective, AND gradient -- with the true model simultaneously.
**This result cannot be read as evidence for or against a
gradient-specific mechanism** (i.e., it does NOT show that Proposition
1's local gradient-alignment property is what causes E's advantage over
B). What it DOES support, on this fixed single-CSTR benchmark, this
horizon (`HORIZON=3`), and these 5 initial conditions specifically: the
gap between the learned surrogates (B, E) and a fully-informed reference
is attributable primarily to residual SURROGATE-MODEL mismatch, not to
the finite optimizer budget itself -- and E's surrogate is
substantially, but not completely, closer to that fully-informed
reference than B's is. Generalization beyond this specific benchmark/
horizon/IC set is not established by this data.

## 10. Next steps

1. **This document** -- fixed as of 2026-09-03, no modification to any
   of the 12 ablation / 11 solver-audit / 8 corrective-freeze files.
2. Update `docs/JPC_REVIEW_RESPONSE_MATRIX_2026_09_01.md` in place (its
   own "living document" convention) to reflect R2.M4's status change
   from "NOT YET ADDRESSED" to a resolved-with-caveats state matching
   Section 9's conclusion and interpretation limits above.
3. Manuscript and response-letter text revisions incorporating Section 9
   as written (the headline result AND its binding interpretation
   limits together, not the headline alone).

Items 2-3 are new work, not yet started; each still needs its own
explicit go-ahead before implementation, per this project's standing
convention.
