# R2.m4 correction: failure-penalized settling time for the manuscript's actual Table 4 — Decision Record (2026-09-04)

## 0. What went wrong and how it was caught

Re-extracting R1/R2/R3's exact wording from the newly-repersisted source
text (`docs/source_reviews/JPC_review_JPROCONT-D-26-00624_2026-08-31.txt`,
SHA-256 `ab71d5909349f28ba8b98896f586e89a05bfcde5b4832b79a0f56d12323a472a`;
every quoted line number in the Response Matrix independently
re-verified against this file and found to match exactly) led to
directly reading the manuscript's literal Table 4 to confirm which
table R2.m4 refers to.

Counting `\begin{table}`/`\label{tab:...}` occurrences in document order
confirms: **Table 4 = `tab:metrics`** (two-CSTR closed-loop control
metrics), **Table 5 = `tab:oper`** (two-CSTR process-operation
robustness, already correctly identified in the R3.6/R2.M6 Decision
Records). `tab:metrics`'s own caption reads: "settling is the mean
number of closed-loop steps to reach the target **over the initial
conditions that reach it**" -- and `scripts/two_cstr_metrics.py:32`'s
`metrics()` confirms this in code: failed ICs get `settle=NaN`, and
`agg()`'s `np.nanmean` silently excludes them.

**This is exactly R2.m4's complaint, on the exact table it names, and it
was NOT fixed.** The Response Matrix's prior "RESOLVED" mark for R2.m4
cited `docs/JPC_SOLVER_AUDIT_DECISION_RECORD_2026_09_03.md`'s Battery 1
`settling_time` -- which is single-CSTR data, a different benchmark and
a different table than the two-CSTR `tab:metrics` R2.m4 literally
points at. This was a real mapping error, caught only by directly
reading the manuscript table the review comment names, which the
original resolution never did.

## 1. Correction: re-aggregated from EXISTING data, no new rollout or training

`results/interim/logs/two_cstr_metrics.json` (already real, already
frozen, self-hash `7632d164f15e2521370d5cc14339b46681bb6cbb338257b861d02f97998f5d46`)
stores, per seed, `success` (fraction of the 5 ICs reaching the target)
and `settle` (the CONDITIONAL mean, over successful ICs only -- the
exact quantity R2.m4 objects to). This is sufficient to reconstruct the
failure-penalized value with no information loss:

```
n_success = round(success * 5)
t_penalized = (n_success * conditional_settle + (5 - n_success) * 121) / 5
```

`121 = steps(120) + 1`, the same fixed-sentinel convention already used
by `single_cstr_rollout` and this session's `two_cstr_lqr_operational_
audit_2026_09_04.py` -- never imputed from a partial trajectory.
Aggregation is seed-first (per-seed penalized value computed, then
mean/`ddof=0`-sd taken across the 10 seeds), matching the original
table's own aggregation convention exactly.

`scripts/two_cstr_failure_penalized_settling_2026_09_04.py` implements
this and writes `results/interim/logs/two_cstr_failure_penalized_
settling_2026_09_04.json` (self-hash
`9504dcbafc427fb7dfa69a1d15dfc2b4894a1762618757dfffbfbf78468e2ebc`,
records `predecessor_source_sha256` for the exact `two_cstr_metrics.json`
version it was derived from).

**Result** (independently verified, matches exactly):

| controller | conditional settle (old, Table 4) | failure-penalized settle (corrected) |
|---|---|---|
| value_only | 57.8 ± 12.0 | **109.48 ± 12.93** |
| value+grad | 31.9 ± 3.5 | **35.54 ± 7.44** |
| auxiliary LQR | 41.4 (deterministic) | 41.4 (unchanged -- no failures to penalize, success=1.00 on all 5 ICs) |

## 2. What this changes about the conclusion (net: clarifying, not damaging)

- `value_only`'s reported settling time was flattered by the conditional
  metric exactly as R2.m4 warned: `57.8` (conditional) understates how
  bad it actually is; `109.5` (failure-penalized, close to the `121`
  sentinel itself, since only 20% of ICs ever succeed) is the honest
  number.
- `value+grad` degrades only mildly under penalization (`31.9` ->
  `35.5`, since it succeeds on 94-96% of ICs, so few sentinel values
  enter the mean) and **remains faster than the auxiliary LQR** (`35.5`
  vs. `41.4`).
- The auxiliary LQR is unaffected (deterministic, always succeeds).

**Combined with the LQR operational audit's own finding** (`docs/JPC_
TWOCSTR_LQR_OPERATIONAL_AUDIT_DECISION_RECORD_2026_09_04.md`): LQR alone
remains superior on success, cost, and input TV; the proposed
controller's one genuine remaining edge on two-CSTR is faster
failure-penalized recovery, at the cost of much more aggressive
actuation (TV `34.4` vs. LQR's `1.9`). **Correct framing**: "value+grad
trades faster recovery for more aggressive actuation, relative to the
auxiliary LQR alone" -- not a blanket robustness or necessity claim.

## 3. Required manuscript changes

1. Table 4 (`tab:metrics`)'s `settling` column: replace `57.8±12.0` /
   `31.9±3.5` with the failure-penalized `109.48±12.93` / `35.54±7.44`.
   `auxiliary LQR`'s `41.4` is unchanged.
2. Caption: state the `121`-step (`steps+1`) sentinel convention and
   that aggregation is seed-first (per-seed penalized value, then
   mean/s.d. across seeds) -- not IC-first.
3. The old conditional-only values MAY be retained in a supplement table
   if useful for context, explicitly labeled "conditional on success,"
   never as the primary reported number.
4. Response Matrix's R2.m4 row: replace the single-CSTR Battery-1
   citation with this two-CSTR reaggregation as the actual resolution
   evidence.

## 4. Status

This is a correction of a mismapped review response, not a new
experiment -- required before any response-letter claim about R2.m4
being resolved. All planned experimental work remains complete; this is
a writing-phase data-accuracy fix uncovered by the review-quote
re-verification step itself.
