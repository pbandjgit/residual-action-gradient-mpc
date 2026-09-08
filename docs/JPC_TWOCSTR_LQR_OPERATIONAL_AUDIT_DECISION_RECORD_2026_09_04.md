# Two-CSTR LQR operational audit — Decision Record (2026-09-04)

## 0. Purpose

Closes the remaining ambiguity in `docs/JPC_R34_MPC_NECESSITY_DECISION_
RECORD_2026_09_04.md` (which only checked the auxiliary LQR at the
NOMINAL two-CSTR operating point) and reinforces `docs/JPC_R36_
ROBUSTNESS_MECHANISM_DECISION_RECORD_2026_09_04.md`: does the auxiliary
LQR controller remain adequate under the SAME disturbance/mismatch/
setpoint conditions the manuscript's robustness table (`tab:oper`) uses,
or does it degrade there (which would make two-CSTR a genuine
MPC-necessity case after all, just not at the nominal point)?

## 1. Pre-registered contract (fixed before running)

- `K_LQR` = the nominal gain (`two_cstr_lgrad.K_LQR`), never retuned per
  scenario.
- Same 5 ICs (`two_cstr_lgrad.ICS`), 120 steps, input box (`BOX_LO`/
  `BOX_HI`) as every other two-CSTR result in this codebase.
- Feed-temp `+5K` / `k0 +5%` / `dH +5%`: reused `two_cstr_realism.py`'s
  EXISTING plant-perturbation objects verbatim, not re-derived.
- Setpoint tracking: fair baseline `u = clip(u_sp - K_LQR@(x-x_sp), lo, hi)`,
  reusing `two_cstr_tracking.py`'s `X_SP`/`U_SP`.
- Metrics: success, final `V`, failure-penalized settling time (`steps+1`
  sentinel on failure, never imputed from a partial trajectory),
  accumulated `V` (cost), input TV (normalized), saturation frequency.
- LQR results are a deterministic 5-IC set; the learned-MPC comparison
  numbers are a 10-seed distribution -- reported separately, never
  conflated.
- Interpretation rule, fixed in advance: LQR degrades under
  disturbance while the learned controller does not -> real
  distribution-shift advantage for the learned MPC. LQR stays dominant
  -> two-CSTR is not an MPC-necessity case, only a surrogate-training
  comparison benchmark. Mixed -> report per-scenario.

No training was involved (LQR and the plant perturbations are both
training-free); this is a deterministic real-dynamics computation.

## 2. Result: LQR dominates in every scenario tested

`scripts/two_cstr_lqr_operational_audit_2026_09_04.py`,
`results/interim/logs/two_cstr_lqr_operational_audit_2026_09_04.json`
(hash `286314538fbd237c6ed7284969355005e45eb8b0771e0b91bdde1e7769c895e3`).
The nominal-regulation row was cross-verified against the
already-existing `two_cstr_metrics.json`'s `aux_LQR` entry and matches
to full precision (settle `41.4`, cost `366.07`, TV `1.886`), confirming
the new script's correctness before trusting its new (disturbance/
mismatch/tracking) rows.

| scenario | LQR success | LQR cost | LQR TV | LQR settle | value_only success | value+grad success |
|---|---|---|---|---|---|---|
| nominal | **1.00** | 366.1 | 1.89 | 41.4 | 0.20 | 0.96 |
| feed-temp +5K | **1.00** | 370.7 | 1.89 | 41.6 | 0.18 | 0.94 |
| k0 +5% | **1.00** | 383.5 | 1.91 | 42.0 | 0.22 | 0.94 |
| dH +5% | **1.00** | 412.1 | 1.93 | 43.4 | 0.24 | 0.94 |
| setpoint tracking | **1.00** | 603.5 | 1.22 | 43.8 | 0.26 | 0.94 |

(`value_only`/`value+grad` success columns independently re-pulled
directly from `two_cstr_realism.json`/`two_cstr_tracking.json` in this
session, not recalled.)

**The auxiliary LQR controller does not degrade under any of the tested
+5% disturbance/mismatch conditions or under setpoint tracking** --
success stays exactly `1.00`, cost and TV increase only mildly (cost
366->412, TV 1.89->1.93 across regulation scenarios), remaining far
below the proposed value+grad controller's cost and TV in every single
scenario (e.g. TV `1.89` vs. `34.4` at nominal -- an order of magnitude
smoother).

Per the pre-registered interpretation rule, this is the **"LQR stays
dominant"** branch, not the "mixed" or "LQR degrades" branch.

## 3. Consequence: two-CSTR's role is redefined, cleanly and completely

**This is a real, unfavorable, but fully decisive finding.** Two-CSTR is
not shown to be an "MPC advantage" benchmark at ANY tested operating
condition -- nominal or perturbed. The auxiliary LQR alone is uniformly
superior in cost and actuation smoothness while matching the proposed
controller's success rate (and beating it, since value+grad's success is
`0.94`-`0.96`, not `1.00`).

**Two-CSTR's sole defensible role in this paper is as a benchmark
comparing SURROGATE TRAINING OBJECTIVES within the learned-MPC
framework** -- `value_only` fails catastrophically (`0.18`-`0.26`
success) under every condition, and gradient-consistency training
(`value+grad`) rescues it (`0.94`-`0.96`). This is exactly the R3.6
"co-occurred, not uniquely attributed" framing's natural companion: the
mechanism explaining WHY `value+grad` beats `value_only` is scoped to
single-CSTR (Section 2 of the R3.6 record); the two-CSTR result itself
is real and robust, but it is evidence about surrogate training, not
about MPC's necessity over a reasonable auxiliary controller.

**No further experiments are needed on this question.** The result is
unambiguous (LQR wins on every metric, every scenario), which is exactly
the kind of clean, decisive (if unfavorable) outcome the pre-registered
interpretation rule was designed to produce -- it forecloses further
ambiguity rather than inviting more probing.

## 4. Manuscript-language corrections required

1. **Do not claim or imply, anywhere in the two-CSTR discussion, that
   the proposed controller demonstrates an advantage over the auxiliary
   LQR controller.** The only defensible two-CSTR claim is: "value+grad
   substantially outperforms value_only across nominal operation,
   disturbance, parameter mismatch, and setpoint tracking" -- never "the
   learned MPC is needed because the auxiliary controller is
   inadequate," which is single-CSTR-only language (correctly scoped
   there in `docs/JPC_R34_MPC_NECESSITY_DECISION_RECORD_2026_09_04.md`).
2. **`\pm5\%` (manuscript line 841) is wrong and must be corrected to
   `+5\%`.** Confirmed by direct code read: `two_cstr_realism.py`'s
   `scenarios` dict only constructs `k0*1.05`/`dH*1.05` (a `+5%`
   perturbation); no `-5%` condition exists anywhere in the codebase.
   This is a wording fix, not a new experiment -- do not run a `-5%`
   condition to make `\pm5\%` literally true unless there is a
   independent reason to want that additional data point.
3. Table `tab:oper`/Fig. `fig:realism` currently present only
   `value_only` vs. `value+grad` (correct, since that IS the valid
   two-CSTR comparison) -- no change needed to the table itself, only to
   surrounding prose that might frame it as an MPC-vs-auxiliary-controller
   result.

## 5. Suggested response-letter language

> An additional operational audit evaluated the auxiliary LQR controller
> alone, using its fixed nominal gain, under the same disturbance,
> parameter-mismatch, and setpoint-tracking conditions used for the
> two-CSTR robustness table. The auxiliary controller did not degrade
> under any tested condition, remaining at 100% success with lower cost
> and substantially smoother actuation than the proposed controller in
> every case. We therefore do not claim an MPC-versus-auxiliary-controller
> advantage on the two-CSTR benchmark under any tested condition; its
> role in this paper is restricted to comparing surrogate training
> objectives within the learned-MPC framework, where the gradient-
> consistency term's benefit over value-only training is large and
> consistent (success improving from 18-26% to 94-96% across all tested
> conditions).

## 6. Status

This closes both R3.4 (two-CSTR portion) and R3.6 with a single,
decisive result. **All planned experimental work for this JPC revision
is now complete**, including this final audit. Next: fix the `\pm5\%`
manuscript wording (Section 4 item 2), re-extract R1/R2/R3 review quotes
from source, then manuscript/response-letter drafting.
