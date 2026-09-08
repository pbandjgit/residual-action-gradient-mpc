# R3.6 (robustness-mechanism question) — Decision Record (2026-09-04)

## 0. The question

R3.6: "Robustness study (Table 5): why would a derivative-matching
controller be more robust to model uncertainty? No analysis of why this
should help." The Response Matrix noted a candidate mechanism already
existed in the go/no-go ablation data (single-CSTR: E has WORSE one-step
test MSE than B but BETTER closed-loop success and gradient fidelity --
`docs/JPC_LAMJAC_GRID_EXTENSION_DECISION_RECORD_2026_09_03.md` and the
earlier ablation work) but flagged that this had never been checked
against the two-CSTR data the actual robustness table (`tab:oper`) uses,
and that connecting the two requires care since they come from different
training runs.

## 1. Single-CSTR mechanism candidate (re-confirmed, unchanged from prior verification)

Direct recomputation from `results/ablation_execution_2026_08_31/`
(10-seed go/no-go ablation): E's mean test MSE `0.000708` is roughly
2.6x WORSE than B's `0.000269` (9/10 seeds), while E's closed-loop
success is near-ceiling and its oracle gradient fidelity (`align_cos`
median, wrong-sign fraction) is uniformly better than B's. This
dissociation -- worse one-step prediction accuracy paired with better
closed-loop/gradient-fidelity outcomes -- is the candidate mechanism:
prediction MSE is not what the finite-budget solver actually needs;
gradient fidelity is.

## 2. Two-CSTR cross-check: the dissociation does NOT replicate

Direct recomputation from `results/interim/logs/two_cstr_lgrad.json`
(10 seeds, independently re-verified in this session, matches values
already used in `project_paper_h_status.md`'s prior entries):

| quantity | value_only | value+grad |
|---|---|---|
| mean test MSE | 0.006078 | **0.004313** |
| test MSE lower in 10/10 seeds | -- | **yes, value+grad** |
| mean gradient alignment (cosine) | 0.307 | 0.827 |

**On two-CSTR, `value+grad` has BOTH lower test MSE AND better gradient
alignment than `value_only`, on all 10 seeds.** This is the OPPOSITE of
the single-CSTR pattern (where E's MSE was worse, not better). The two
quantities move together on two-CSTR -- they cannot be dissociated as
distinguishing factors there, because nothing in the two-CSTR data shows
a case where MSE and gradient fidelity disagree.

**Conclusion: the single-CSTR MSE/gradient-fidelity dissociation is a
single-CSTR-specific observation. It does not generalize to two-CSTR and
must not be cited as the explanation for two-CSTR's robustness
advantage.** The honest, defensible statement is co-occurrence, not
unique attribution -- both metrics improved together on two-CSTR, so this
experiment cannot separate "better MSE caused the robustness" from
"better gradient fidelity caused the robustness" from "both, for
unrelated reasons."

## 3. Checkpoint-linkage caveat (disclosed, not resolved further)

`scripts/two_cstr_realism.py` (the script generating the manuscript's
robustness table `tab:oper`/`fig:realism`) retrains a fresh model for
every `(seed, config)` pair inside its own loop
(`two_cstr_realism.py:108-110`, confirmed by direct read) and does not
persist `test_mse` or a state-dict/init hash. It therefore uses the SAME
training recipe as `two_cstr_lgrad.json` (same `T.train`, same seeds,
same `lam_val`/`lam_grad` per config) but is not a checkpoint-identical
rerun -- the two files' models share a recipe, not a literal identity.

**This caveat does not weaken the Section 2 conclusion.** Since Section
2 already finds NO dissociation to attribute on two-CSTR (both metrics
move together), a checkpoint-level link to `tab:oper`'s specific models
would only matter if we were trying to make a POSITIVE mechanism claim
requiring exact model identity. We are not making one -- the response is
explicitly "co-occurred, not uniquely attributed," which holds under the
same-recipe (not same-checkpoint) evidence already in hand. A dedicated
compatibility audit (retrain with state-hash+MSE persisted, verify
against `tab:oper`'s literal checkpoints) is not undertaken here as it
would not change this conclusion; it remains a possible follow-up if
future work wants a POSITIVE causal claim on two-CSTR specifically,
which is not attempted in this response.

## 4. What this means for the R3.6 response

**Prohibited**: extending the single-CSTR MSE/gradient-fidelity
dissociation to explain two-CSTR's robustness result -- the mechanism
observed on one benchmark is not shown to hold on the other, and the
data available (both metrics improving together on two-CSTR) actively
argues against using it there.

**Correct, defensible framing**: report the single-CSTR dissociation as
a benchmark-specific descriptive finding (already scoped this way per
`docs/JPC_LAMJAC_GRID_EXTENSION_DECISION_RECORD_2026_09_03.md`). For
two-CSTR, state plainly that the robustness improvement co-occurred with
improvements in both prediction accuracy and gradient fidelity, so the
experiment supports the EMPIRICAL robustness finding but does not
uniquely attribute its cause to gradient consistency specifically.

## 5. Suggested response-letter language

> Two-CSTR robustness improvement co-occurred with improvements in both
> prediction accuracy and action-gradient fidelity; therefore, the
> experiment supports empirical robustness but does not uniquely
> attribute it to gradient consistency. A distinguishing mechanism
> (residual-prediction accuracy trading off against solver-facing
> gradient fidelity) was observed on the single-CSTR benchmark, where the
> two properties moved in opposite directions across ten seeds; it is
> reported there only, and is not offered as an explanation for the
> two-CSTR result.

## 6. Provenance

Single-CSTR numbers: `results/ablation_execution_2026_08_31/` (10-seed
M2 files, `test_mse`/`oracle_summary`), previously verified, re-cited
here unchanged. Two-CSTR numbers: `results/interim/logs/
two_cstr_lgrad.json`, independently recomputed directly in this session
(mean test MSE, per-seed comparison count, mean alignment) rather than
recalled from a prior turn's summary. `two_cstr_realism.py:108-110` read
directly to confirm the fresh-retrain-per-seed/no-persisted-MSE claim.

## 7. Status

This is the last planned experimental item (per the user's 2026-09-03
scoping decision). No further sweeps are planned. Next: update
`docs/JPC_REVIEW_RESPONSE_MATRIX_2026_09_01.md` (R3.6 row, ranked-gaps
summary), then move to manuscript/response-letter drafting (pending the
still-outstanding R1/R2/R3 review-quote re-extraction from source).
