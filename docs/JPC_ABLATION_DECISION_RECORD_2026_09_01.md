# JPC Go/No-Go Ablation — Decision Record

Status: FIXED, based on the real, frozen-protocol execution. This record
is descriptive of a completed real result; it is not itself part of the
Freeze Manifest's hash-locked file set and may be extended (never
retroactively altered) as follow-up work referenced in Section "Next
steps" below is completed.

## 0. Provenance

- Frozen protocol: `docs/JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_2026_08_31.md`
  (v7), hash-locked by `docs/JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_FREEZE_MANIFEST_2026_08_31.json`
  (self-hash `19bd6cab7a036642006b8627722161fcfa7686fe95bd3a7e34e22672f503242a`,
  12 files, smoke 167/167).
- Real execution: `scripts/run_real_protocol_launch_2026_09_01.py` ->
  `run_real_protocol()`, all 10 pre-registered seeds, wall-clock 2165.4s.
- Query Manifest (M1): `results/ablation_execution_2026_08_31/ablation_query_manifest_2026_08_31.json`,
  self-hash `9d20952e81a2f0e384e8e43318ef16b4a381d2def5dd0fd0f4ee426e7a423635`
  (n_train/n_val/n_test = 10500/3500/6000, matching the pre-registered
  52.5%/17.5%/30% split).
- Results Manifest (M3): `results/ablation_execution_2026_08_31/ablation_results_manifest_a28bb79aa5ca_2026_08_31.json`,
  self-hash `aff8803fd4455227d5cf680e4106f75bc64522ee2e76ee2c656eade8c113fe74`.
- Independent re-verification: `run_real_protocol()` was called a second
  time (resume-or-verify path) immediately after completion. It re-hashed
  every checkpoint and the noise-realized tables, re-derived pool/clean
  content hashes, re-checked the freeze chain, re-derived M3's statistics
  and all three companion files fresh, and reproduced byte-identical
  M1/M2(x10)/M3 self-hashes. Took 48.4s (verification only, no retraining).
  All 12 frozen files were independently re-hashed via `shasum -a 256`
  (outside the Python code path) after the real run and matched the
  Freeze Manifest exactly.

## 1. Headline result (pre-registered `outcome` field, transcribed verbatim from M3)

```
query_matched_effect:        SUPPORTED
target_specificity:          NOT_SPECIFIC
non_groupsort_replication:   GROUPSORT_SPECIFIC
loss_interaction:            INCONCLUSIVE
mse_attribution:             UNRESOLVED
mse_attribution_sub_label:   MSE_MATCH_INFEASIBLE
manuscript_action:           NARROW_TO_SUBSET
```

Primary contrast (E vs B, equal query budget): point estimate = 0.34,
95% CI = [0.14, 0.56], exact paired permutation p = 0.015625.

This directly answers Reviewer 3's most dangerous critique ("is the
improvement just extra query information, not the gradient-matching
loss?"): under an EQUAL query budget, E (residual action-gradient
consistency) still beats B (value-only) by +0.34 success rate,
significantly.

The scope this result supports must be narrowed to: **GroupSort-LCNN,
compared against the one-sided-residual value-only baseline B**. It does
not establish an architecture-independent or Jacobian-matching-specific
claim (see Sections 2, 5, 6 below).

## 2. Decomposition (all figures independently re-derived from the real M2/M3 data on 2026-09-01, not merely re-stated)

| step | conditions | mean closed-loop success (10 seeds) | reading |
|---|---|---|---|
| A -> B | 0.06 -> 0.66 | adding the extra (perturbation) transitions to an otherwise value-only baseline explains most of the FROM-SCRATCH gap by itself | information alone does most of the initial work |
| B -> E | 0.66 -> 1.00 | after equalizing query information, L_grad still adds real closed-loop gain | the query-matched, gradient-specific increment is real and is the +0.34 headline |
| B -> C | 0.66 -> 0.92 | a symmetric-residual loss (no explicit gradient term) gets most of the way to E on the SAME equal query budget | L_grad's gain over B is not exclusively attributable to the gradient term; symmetric-residual framing alone recovers most of it |

Per-seed success vectors (10 seeds, seed 0-9):
- A: `[0,0,0,0,0,0,0,0,0,0.6]` (mean 0.06)
- B: `[0.6,0.6,1.0,1.0,0.8,0.8,1.0,0.0,0.6,0.2]` (mean 0.66)
- C: `[1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.2,1.0,1.0]` (mean 0.92)
- D: `[0.8,0.2,1.0,0.4,1.0,0.4,1.0,1.0,0.6,1.0]` (mean 0.74)
- E: `[1.0]*10` (mean 1.00)
- F: `[1.0]*10` (mean 1.00, byte-identical to E's vector -- see below)
- B-FNN: `[1.0]*10` (mean 1.00)
- E-FNN: `[1.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0]` (mean 0.80)

**E and F are both exactly 1.0 on all 10 seeds.** Confirmed this is a
metric-ceiling saturation of the 5-initial-condition binary success
endpoint, not a code-duplication bug -- their MSE and oracle gradient
diagnostics differ (verified: `test_mse` and `oracle_summary` are
distinct per-condition entries in every seed's M2). A ceiling-free
continuous metric (settling time, terminal V, control TV, constraint
violation) is needed to actually separate E from F; see "Next steps" #2.

**B-FNN = 1.00, E-FNN = 0.80.** On the standard (non-GroupSort) Softsign
FNN, the value-only baseline B-FNN was already at ceiling (1.00), and
adding the gradient term (E-FNN) moved success DOWN to 0.80 (seeds 4 and
5 dropped from 1.0 to 0.0). The effect not only fails to replicate on
FNN, it reverses direction. `non_groupsort_replication = GROUPSORT_SPECIFIC`
is the correct, literal reading: this result must not be presented as
architecture-independent.

**D's `lambda_jac` selection hit the grid's upper bound (0.5) on ALL 10
seeds** (grid was `[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]`, confirmed by
direct read of `lambda_jac_selected` in every seed's M2). D is therefore
tuned against its own boundary in this run; a claim that E is uniquely
better than Jacobian-matching (D) is not supportable until the grid is
extended past 0.5 (see "Next steps" #3). Consistent with this, the
pre-registered `target_specificity` gate itself returned `NOT_SPECIFIC`
(E vs D: point 0.26, CI [0.08, 0.46], raw p = 0.0625, Holm-adjusted
p = 0.3125 -- not significant after correction).

**MSE-match: 5/10 seeds `MATCHED`, 5/10 `MSE_MATCH_INFEASIBLE_FOR_SEED`**
(seeds 1,3,4,5,6 matched; seeds 0,2,7,8,9 infeasible). Mean test MSE:
B = 2.6855e-4, E = 7.0824e-4 (E is ~2.64x WORSE). Despite this, E's
closed-loop success is higher than B's. This is direct evidence that
lower one-step prediction error does not guarantee better closed-loop
planning -- but because the match was infeasible for half the seeds, the
MSE-INDEPENDENT causal attribution question (`mse_attribution`) remains
formally `UNRESOLVED`, not affirmatively resolved either way.

**Oracle gradient-quality diagnostics, E vs B (10-seed means of per-seed
medians, independently re-derived):**
- median cosine(true, learned gradient): B = 0.937, E = 0.993 (+0.056)
- wrong-sign fraction (`frac_neg`): B = 5.69%, E = 0.25% (-5.44 pp)
- median relative gradient error: B = 0.600, E = 0.195 (-0.406, i.e. E's
  gradient error is roughly a third of B's)

These are internally consistent with the closed-loop result: E's
first-order gradient quality is uniformly better than B's, in exactly
the direction the mechanism (residual action-gradient consistency)
predicts.

**Compute-cost side-by-side, E vs C (both reach comparable closed-loop
success, 1.00 vs 0.92):** E used a mean of 2100 optimizer steps and 3.88s
training wall-clock; C used 10250 steps and 14.29s -- roughly 5x more.
E compresses the same perturbation information into a derivative target
on the base sample rather than training on every augmented transition
explicitly, reaching equal-or-better success at a fraction of the
training compute. This is a descriptive (not pre-registered, not
gating) practical-value comparison for L_grad, not a causal-uniqueness
claim (C's success is not below E's by a margin that would license one).

## 3. Defensible manuscript-level conclusion (current evidence ceiling)

> Under an equal offline identification-query budget, residual
> action-gradient consistency significantly improved GroupSort-LCNN's
> finite-budget closed-loop planning success and its action-gradient
> fidelity, even though its one-step prediction MSE was worse than the
> value-only baseline's. This effect did not replicate on a standard
> (non-GroupSort) FNN architecture, and mechanism-uniqueness relative to
> a symmetric-residual baseline and an action-Jacobian-matching baseline
> is not established by the current data.

Consequently: the broad architecture-independent claim must be dropped,
and Proposition 1 must be narrowed to a local, directional statement
(Section 4).

## 4. Proposition 1 — narrowed statement

Proposition 1's sole remaining job: **if the learned gradient error is
small enough, `-grad(g_hat)` is a strict local descent direction for the
true residual `g`.** Nothing about projected Adam, GGN, or multi-step
rollout convergence follows from it alone; that gap is filled empirically
by the gradient-audit and closed-loop results above, not by extending the
theorem's scope.

Recommended title: **"Local descent-direction preservation."**

Recommended statement:

> Under inactive projection and a sufficiently accurate residual action
> gradient, the learned negative-gradient direction is a strict local
> descent direction for the true residual. This result is a local
> directional statement; it does not establish convergence of projected
> Adam or Gauss-Newton, guarantee that a finite-budget multi-step solver
> reaches a Lyapunov-feasible input, or by itself prove closed-loop
> stability.

Recommended replacement for the "Consequence for closed-loop stability"
section, retitled **"Relation to the Lyapunov decrease condition and
scope"**:

> If the realized controller returns an input satisfying `g(x,u) <= 0`,
> the usual Lyapunov decrease condition follows. Proposition 1 does not
> guarantee that the finite-budget solver finds such an input; it only
> explains why reducing action-gradient error can improve the local
> search direction. Whether that directional improvement translates into
> realized closed-loop success is evaluated empirically.

This reframes Proposition 1 as the precise theoretical bridge to the
observed gradient-fidelity and planning-success improvement, rather than
an overreaching stability claim -- more defensible under review than a
theorem whose scope the actual experiments (projected Adam, multi-step,
GGN) already exceed.

## 5. Strategic handling of the four contested claims

The four items below must NOT each be defended outright, nor each be
simply conceded as "cannot claim" -- separating theoretical guarantee /
mechanism / empirical result / scope of applicability is the stronger
structure than either extreme.

1. **Proposition does not guarantee projected Adam / GGN / multi-step.**
   Keep the theorem narrow (Section 4); bridge to the actual solver via
   empirical diagnostics already available or addable without retraining
   (projection-active fraction, accepted-step true-residual decrease,
   predicted-vs-realized decrease agreement -- the same diagnostic class
   used in the GGN extension work, see [[project_paper_h_status]]).
2. **Proposition alone does not prove asymptotic stability.** State only
   the conditional relation to the standard Lyapunov decrease condition
   (Section 4's second box). The paper's actual contribution is
   finite-budget planner reliability, not an unconditional stability
   theorem.
3. **Failure to generalize to all architectures is not fatal.** B-FNN was
   already at ceiling (1.00, wrong-sign ~0.89%) before L_grad was added;
   there was little room to correct, and L_grad moved it DOWN to 0.80.
   Frame as a scope condition, not a failure: *L_grad is a targeted
   regularizer for baselines with a real action-gradient sign/error
   problem, not a universal prescription for architectures whose
   gradients are already well-behaved.*
4. **L_grad's uniqueness cannot be claimed.** C (symmetric residual, no
   explicit gradient term) also shows a +0.26 signal over B. L_grad's
   demonstrated practical differentiator is NOT exclusivity of mechanism
   but efficiency: E reaches success 1.00 in ~2100 steps / 3.88s vs. C's
   0.92 in ~10250 steps / 14.29s (~5x) -- compressing the same
   perturbation information into a base-sample derivative target instead
   of training on every augmented transition explicitly. Present this as
   a descriptive result (not a pre-registered/gating endpoint).

## 6. Recommended paper evidence chain

1. Proposition: local directional rationale (Section 4).
2. Gradient audit: sign / cosine / relative-error improvement (Section 2,
   oracle diagnostics).
3. Actual-solver audit: whether correct decrease is realized under
   projection and finite budget (not yet run; see "Next steps" #1-type
   diagnostics referenced in Section 5 item 1).
4. Closed-loop evidence: query-matched E-B improvement (Section 1-2).
5. Boundary conditions: symmetric-residual alternative (C), FNN
   non-replication, MSE-match infeasibility (Section 2).

This presents the paper not as "a study that failed to prove everything"
but as one that withdrew an overreaching universal claim and precisely
characterized failure mechanisms and the actual scope of validity.

## 7. Next steps (recommended order; none require retraining unless noted)

1. **This document** — the current results, fixed as of 2026-09-01,
   without modification to any of the 12 frozen files.
2. Re-evaluate the ALREADY-SAVED checkpoints for ceiling-free continuous
   metrics (terminal V, constraint violations, control total variation,
   a failure-penalized settling-time metric) to actually separate E from
   F (both saturate at 1.00 on the binary 5-IC endpoint). No retraining.
3. A separate, pre-registered follow-up extending `LAM_JAC_GRID` beyond
   0.5 for condition D, since all 10 seeds selected the grid's current
   upper bound.
4. Report the MSE-match infeasibility (5/10 seeds) as-is; if warranted,
   design a separate bidirectional-matching follow-up rather than
   concealing or re-litigating the current infeasible result.
5. Only after 2-4 (or an explicit decision to defer them) begin the
   reviewer-response and manuscript-text revision described in Sections
   3-6 above.

Items 2-4 are new work not yet started or scoped in detail; each should
get its own explicit go-ahead before implementation.

## 8. Precision addendum (2026-09-01, post-review)

Four wording/sequencing corrections to Sections 1-7 above, identified on
a second review pass. Recorded as an addendum rather than an in-place
edit, per this project's provenance convention (Sections 0-7 remain
exactly as originally written and self-consistent with the M3 data they
transcribe).

1. **"Significantly improved ... action-gradient fidelity" (Section 3)
   overstates the gradient-metric claim.** Verified against M3: the
   pre-registered statistical family (`secondary_family` =
   `{C_vs_B, EFNN_vs_BFNN, E_vs_D, F_vs_C, F_vs_E, MSE_match,
   interaction}`) contains no CI/p-value for the oracle metrics (`align_cos`,
   `rel_grad_error`, `frac_neg`) — those are Section 6's "final-evaluation-
   only," descriptive quantities by design, never bootstrap/permutation-
   tested. Only the closed-loop success contrasts (E vs B, etc.) carry a
   significance test. Section 3's sentence must be split: "...
   significantly improved GroupSort-LCNN's finite-budget closed-loop
   planning success, and substantially improved its action-gradient
   fidelity (Section 2's oracle diagnostics, descriptive, not
   significance-tested) ...". The same conflation should be avoided
   wherever Section 2's oracle numbers are cited as "improvement" in
   manuscript prose -- state them as consistent-with-mechanism
   observations, not as a second statistically-tested result.

2. **The E-vs-C compute-cost comparison (Section 2, Section 5 item 4) collapses
   two different ratios into one "~5x."** Recomputed exactly from the
   same M2 figures already cited (steps: 10250/2100; wall-clock:
   14.284653662890197/3.8812478333711624): optimizer steps ratio =
   **4.88x**, wall-clock ratio = **3.68x**. These must be reported
   separately (e.g. "~4.9x fewer optimizer steps and ~3.7x less
   wall-clock") -- they are not the same quantity and do not happen to
   coincide here.

3. **"GroupSort-specific" is the correct pre-registered outcome-map label
   but must not become a manuscript-prose claim about the GroupSort
   architecture class.** `non_groupsort_replication = GROUPSORT_SPECIFIC`
   is a valid name for what the protocol's Section 7 gate tests, and
   Section 3's own manuscript-conclusion sentence already gets this right
   ("did not replicate on a standard (non-GroupSort) FNN architecture").
   Sections 2 and 5, however, drift toward the stronger, architecture-class
   framing ("this result must not be presented as architecture-independent";
   "failure to generalize to all architectures"). Only ONE alternative
   architecture (Softsign FNN) was tested, so the only defensible
   manuscript claim is: *the effect did not replicate on the tested
   Softsign FNN* -- not a positive claim that the effect is specific to
   GroupSort's structural properties, which would require testing
   additional non-GroupSort architectures to support.

4. **Next steps (Section 7) has two sequencing problems.**
   - The "actual-solver audit" named in Section 6's evidence chain (item
     3: projection-active fraction, accepted-step true-residual decrease,
     predicted-vs-realized decrease agreement) has no corresponding entry
     in Section 7 -- it was only implicit inside item 2's checkpoint
     re-evaluation. It should be its own explicit next step, run
     TOGETHER with the ceiling-free continuous-metric re-evaluation
     (both use the same saved checkpoints, no retraining), and treated as
     at least as high priority as the D `lambda_jac` extension (Section 7
     item 3) -- it is the most direct additional evidence for Reviewers 1
     and 2 specifically (Section 5 item 1's stated bridge from the
     narrowed Proposition to the actual solver).
   - Gating ALL manuscript-text work behind items 2-4 (former Section 7
     item 5) is unsafe against the 2026-09-20 resubmission deadline.
     Everything in Sections 3-6 that does NOT depend on new results --
     Proposition 1's narrowing (Section 4), the Sobolev/derivative-matching
     related-work repositioning, terminology cleanup (wrong-sign
     definition, "rollout"/"single shooting" definitions), the
     reproducibility table, the query-fairness explanation for Reviewer 3,
     and a comment-by-comment reviewer response matrix -- should start in
     PARALLEL with items 2-4, not after them.

**Revised recommended order** (supersedes Section 7's ordering only; Section 7's
item list and content stand as the record of what was originally proposed):

1. Together, no retraining: ceiling-free continuous metrics (former item 2)
   AND the actual-solver audit (projection-active fraction, accepted-step
   decrease, predicted-vs-realized decrease agreement) -- the most direct
   additional evidence for Reviewers 1 and 2.
2. D's `lambda_jac > 0.5` extension, as a separate pre-registered follow-up.
3. For MSE-match, default to reporting the current 5/10-infeasible result
   honestly rather than immediately designing a new large bidirectional-
   matching experiment; revisit only if the honest report proves
   insufficient for the response.
4. In parallel with 1-3, starting now: Proposition 1 narrowing, related-work
   repositioning, terminology cleanup, reproducibility table, the
   query-fairness explanation, and the reviewer response matrix.
5. Insert the state-constraint case study, oracle/converged-NLP reference,
   and noise-sensitivity results into the manuscript at their appropriate
   locations once available.

Items 1-3 above are new work, not yet started or scoped in detail; each
still needs its own explicit go-ahead before implementation. Item 4's
components are results-independent and can start immediately once
authorized.

## 9. Precision addendum, round 2 (2026-09-01, post second review)

Two further corrections to Section 8, again appended rather than edited
in place, per this document's own stated policy (Section header: "may be
extended, never retroactively altered").

1. **"4.9x fewer optimizer steps" / "3.7x less wall-clock" (Section 8
   item 2) is a mathematically ambiguous construction ("N times fewer").**
   The underlying numbers are unchanged (steps ratio 10250/2100 = 4.881,
   wall-clock ratio 14.284653662890197/3.8812478333711624 = 3.680,
   independently reconfirmed here), but the phrasing must be one of:
   - "C required 4.88 times as many optimizer steps and 3.68 times as
     much training wall-clock as E."
   - "E used 79.5% fewer optimizer steps and 72.8% less training
     wall-clock than C." (reconfirmed: 1 - 2100/10250 = 0.7951;
     1 - 3.8812.../14.2846... = 0.7283.)
   Either is acceptable; the ambiguous "Nx fewer/less" form from Section 8
   item 2 and Section 5 item 4 should not be carried into the manuscript.

2. **Neither the "actual-solver audit" NOR the ceiling-free continuous
   metrics are post-hoc aggregations over already-saved artifacts --
   verified by direct code inspection, correcting an unchecked hedge in
   the first draft of this item.** `budget_sweep_solver.closed_loop()`
   (`scripts/budget_sweep_solver.py` lines 52-87) already computes and
   RETURNS `final_V`, `max_V`, `violations` (count of Lyapunov-decrease
   increases), and `tv` (control total variation) on every call -- exactly
   the ceiling-free metrics named in Section 2/Section 7 item 2. But
   `alib.safe_closed_loop_success()` (`src/cstr/ablation_lib.py` line
   378) immediately discards everything except the boolean `success`
   field, and the driver's `_closed_loop_success_rate()`
   (`scripts/ablation_driver_2026_08_31.py` line 253) further collapses
   that boolean across initial conditions into a single mean float --
   confirmed by direct read of both functions. **None of `final_V`,
   `max_V`, `violations`, or `tv`, per-IC or otherwise, were persisted
   anywhere in the real run's M2 or M3.** Recovering them, and likewise
   producing the projection-active-fraction / accepted-step-decrease /
   predicted-vs-realized-decrease diagnostics R1.2/R2.M1 ask for, both
   require a NEW instrumented closed-loop execution against the already-
   trained, already-checkpointed models (no retraining of the models
   themselves, but a real new run nonetheless, since the original run
   never captured these fields) -- not a query against existing
   manifests. The two are therefore the SAME class of follow-up work, not
   a "free" post-hoc item paired with a "real" new-experiment item as
   Section 8 item 4 implied. Both should be planned and executed with the
   same discipline as the main ablation: a scoped protocol (which fields
   to log per rollout step/IC, over which conditions/seeds, at what
   config), its own artifact/manifest chain or equivalent provenance
   record, and its own execution log.
