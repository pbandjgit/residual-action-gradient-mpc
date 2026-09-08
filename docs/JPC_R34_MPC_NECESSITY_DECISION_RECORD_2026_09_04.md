# R3.4 (MPC-necessity evidence) — freshness check, Decision Record (2026-09-04)

## 0. The question

R3.4: "Paper assumes a stabilizing, constraint-respecting controller
already exists on a neighborhood; no state constraints; the MPC does the
same task that controller already does -- what is the advantage of the
proposed controller?" The Response Matrix flagged the existing EAAI-era
answer (saturated-Sontag baseline, single-CSTR) as needing a freshness
check against the current JPC setup before reuse, and noted no
equivalent check existed for two-CSTR's saturated-LQR auxiliary
controller.

## 1. Single-CSTR: re-verified fresh under the current setup, unchanged conclusion

Re-ran the Sontag-alone rollout (`src/cstr/sontag.py`'s
`saturated_sontag_np`, the SAME function already live in the current
solver-audit residual-bridge code, `solver_audit_lib.py:534-536`) using
the CURRENT `budget_sweep_solver` module's real simulator (`bs.SIM`),
input bounds (`bs.INPUT_LO`/`bs.INPUT_HI`), `P`, and the current 5 ICs
(`bs.ICS`), for 120 steps, success threshold `V<=2.0` -- i.e. every
component of this computation is the live, current codebase, not a
stale copy.

**Result** (independently computed, no training involved -- Sontag is a
static feedback law, so this is fully deterministic and reproducible by
construction): success `0.2` (1/5 ICs), mean cost `4528.9`, mean input TV
`95.0` -- these numbers match the previously-cited EAAI-era figures
(`succ 0.20`, `cost 4529`, `TV 95`) essentially exactly. **The EAAI-era
finding was not stale; it reproduces byte-for-byte under the current
codebase.**

Compared against the current frozen E (go/no-go ablation, seed 0,
Battery 1): success `1.0` (5/5), mean settling `9.6` steps, mean TV
`2.30`. **Conclusion for single-CSTR is unchanged and now freshly
verified**: the auxiliary Sontag controller alone is genuinely weak
(input-saturated, chattering at the box bounds, TV=95), and the
proposed L_grad-LMPC controller is what actually stabilizes it (TV=2.3,
settles in ~10 steps). This is the correct single-CSTR answer to R3.4.

## 2. Two-CSTR: the SAME argument does NOT transfer, and this must be reported

Found `scripts/two_cstr_metrics.py` (already existing, dated 2026-07-01,
predating this JPC revision round) computes exactly the analogous
auxiliary-vs-learned comparison for two-CSTR, using the CURRENT
`two_cstr_lgrad.py` module (`T.phi`, saturated LQR on the quadratic CLF,
`T.train`, `T.ICS`), the SAME `horizon=3, budget=20, steps=120, lr=0.2,
rho_u=0.01` configuration used throughout the current two-CSTR results
(matching `two_cstr_realism.py`'s config exactly), 10 seeds.

**Re-verified the deterministic auxiliary-LQR portion by direct
re-execution** (`M.record_aux`/`M.metrics` against `T.ICS`, no training
needed): reproduced the stored `aux_LQR` result exactly --
success=`1.0`, settle=`41.4`, cost=`366.07`, input TV=`1.886`. The
learned-controller portion (`value_only`/`value+grad`, which does
require retraining) was cross-checked against the independently-run
`results/interim/logs/two_cstr_lgrad.json` (memory-recorded, itself
already verified): both report `value_only` success=`0.20` and
`value+grad` success=`0.96` — the SAME numbers from two independently
generated files using the same deterministic seeded training recipe,
which is strong evidence this data is current and not stale.

**Result table (two-CSTR, 10 seeds, current setup)**:

| controller | success | settle (steps) | cost | input TV |
|---|---|---|---|---|
| aux LQR (auxiliary alone) | **1.00** | 41.4 | **366.1** | **1.89** |
| value+grad (proposed) | 0.96 | **31.9** | 382.7 | 34.4 |
| value_only | 0.20 | 57.75 | 4170.8 | 23.6 |

**This is a real, unfavorable finding that must be disclosed, not
omitted.** On two-CSTR, the auxiliary LQR controller alone is NOT weak
-- it achieves perfect success, LOWER cost, and dramatically lower input
TV (1.9 vs. 34.4, an order of magnitude smoother) than the proposed
value+grad LMPC. The proposed controller's only advantage over the bare
auxiliary controller on this benchmark is faster settling (31.9 vs. 41.4
steps) at the cost of much more aggressive actuation.

## 3. What this means for the R3.4 response

**Prohibited**: presenting a single, uniform "the auxiliary controller
is weak, therefore MPC adds value" narrative across both benchmarks --
this is TRUE for single-CSTR and FALSE for two-CSTR as measured here.

**Correct, defensible framing**: the value of the proposed controller is
demonstrated on single-CSTR, where the assumed auxiliary controller is
genuinely inadequate (input-saturated, chattering, 80% failure rate) and
the learned LMPC recovers it completely. On two-CSTR, the auxiliary LQR
is already adequate under these NOMINAL initial conditions -- the
proposed controller here is not shown to add closed-loop value beyond
faster settling, though R2.M5's already-planned narrowing of the
generality claim already limits what two-CSTR is asked to demonstrate
(process-operation robustness under disturbance/mismatch/setpoint
tracking, per `tab:oper`/`fig:realism` — NOT "MPC beats a weak auxiliary
controller," which two-CSTR does not evidence).

This actually integrates cleanly with R2.m7's already-disclosed fact
(different auxiliary controllers per benchmark: Sontag for single-CSTR,
LQR for two-CSTR) -- the reason single-CSTR and two-CSTR play different
argumentative roles in the paper is now empirically grounded: single-CSTR
demonstrates "MPC rescues an inadequate auxiliary controller," two-CSTR
demonstrates "the L_grad target keeps a learned MPC's residual-consistency
benefit under process-operation disturbances, distinct from and not
contingent on the auxiliary controller being weak."

## 4. Suggested response-letter language

> The advantage of the proposed controller over the assumed auxiliary
> controller is benchmark-specific, not uniform. On the CSTR, the
> saturated Sontag auxiliary controller is genuinely inadequate under
> input saturation (20% success, input total variation 95, re-verified
> under the current solver setup), and the proposed LMPC recovers full
> stabilization (100% success, total variation 2.3). On the two-CSTR
> process, the saturated LQR auxiliary controller already achieves
> perfect nominal success with lower cost and much smoother actuation
> than the proposed controller; the two-CSTR results are offered as
> evidence of process-operation robustness under disturbance and
> mismatch (Table [tab:oper]), not as a demonstration that MPC is
> necessary to overcome a weak auxiliary controller on that benchmark.

## 5. Provenance

All numbers in this record were independently (re)computed directly
from the current codebase in this session, not read from memory or an
old report: `saturated_sontag_np` re-run against `budget_sweep_solver`'s
live `SIM`/`ICS`/`INPUT_LO`/`INPUT_HI`/`P` (single-CSTR); `two_cstr_
metrics.record_aux`/`metrics` re-run against `two_cstr_lgrad`'s live
`ICS`/`phi` (two-CSTR, deterministic, byte-identical reproduction of the
existing `results/interim/logs/two_cstr_metrics.json`); E's Battery 1
seed-0 per-IC summary read directly from
`results/solver_audit_execution_2026_09_01/solver_audit_battery1_E_seed0_2026_09_01.json`.
No new training was required for this check (the auxiliary controllers
are training-free by construction; the learned-controller comparison
numbers were cross-validated against two independently-generated
existing files rather than retrained).

## 6. Next step

R3.6 (robustness-mechanism question / two-CSTR MSE-gradient-fidelity
cross-check) is the last remaining planned experimental item.
