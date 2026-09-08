# JPC Revision Go/No-Go Ablation Protocol

Status: DRAFT, NOT FROZEN, v7 (2026-09-01). Implementation and a
synthetic smoke suite exist (Section 0.3), but the document itself is
still not frozen. It supersedes the informal "Tier 1 item 1" sketch in
`JPC_MAJOR_REVISION_RESPONSE_STRATEGY_2026_08_31.md` v2 with a fully
specified, pre-registered design.

**Revision history**: v1 (2026-08-31) was a self-authored first draft.
A review of v1 found 9 concrete issues (7 P0, 2 P1), all independently
re-verified against actual code before being incorporated into v2 — see
"v1 -> v2 corrections" further below. A second review of v2 found 5 more
P0 issues and 2 P1 issues before v3 — see "v2 -> v3 corrections". A third
review (two passes, treated as one round) found 11 more P0 issues and 7
more P1 issues before v4 — see "v3 -> v4 corrections". A fourth review
found 4 more P0 issues and 4 more P1 issues before v5 — see "v4 -> v5
corrections", including the FNN-freeze function mismatch. A fifth review
found the last confirmed factual error: `g_scale`/`ug_scale` are SCALAR
and SEED-INDEPENDENT (confirmed by direct read of `lgrad_experiment.py:
73-78`: `.std()` with no `dim` argument flattens all elements, and both
are computed from the pool's clean, seed-independent arrays), not shape
`(2,)`/per-seed as v5 incorrectly stated — plus two M2/M3 manifest
keyset gaps and a bootstrap-RNG reuse ambiguity, all fixed in this v6;
see "v5 -> v6 corrections". The reviewer's own assessment at v6: the
design was consistent and expected to be the last design-level revision
before implementation. Implementation then proceeded (`src/cstr/
ablation_lib.py`, `ablation_io.py`, `scripts/ablation_driver_2026_08_31.py`,
`scripts/ablation_synthetic_smoke_2026_08_31.py`), and two further code-
review rounds against the REAL implementation (not the design text) found
a total of 19 real bugs/gaps — most were pure implementation defects
requiring no design change (an augmented-residual-target bug, a missing
orchestration entry point, incomplete manifests, a missing resume path,
an incomplete finite-value contract, incomplete oracle reporting, a
missing per-lambda contrast computation, and others — see the
implementation-status memory record, not reproduced here since this
document specifies the DESIGN, not the code). Exactly two findings
required an actual addendum to this design document, both incorporated
into this v7 — see "v6 -> v7 corrections" at the end of this document.
Two
of the v1->v2 fixes (Section 8's checkpoint
cadence and Section 4's FNN activation/width) required new facts pulled
from the codebase that changed the plan materially: no training script in
this repo currently writes checkpoints to disk at any cadence (confirmed
by exhaustive grep), and the existing `OneStepFNN` baseline is ReLU-only
at hidden width 64, not tanh/Softsign at width 40 as an earlier draft
assumed. Both are now specified as new implementation work in this
protocol, not reuse of existing behavior. v1's two originally-open facts
(the base data split ratio and the closed-loop IC set) remain confirmed
against `generate_lcnn_paper_cstr_data.py` and `budget_sweep_solver.py`,
written into Sections 2 and 7.

This document still requires a human design-review pass before it is
frozen. Per Section 0's revised sequencing, "frozen" now means: frozen
together with its implementing code and a synthetic-smoke-test result,
not frozen as a text-only design (see Section 0).

## 0. Scope and precondition

This protocol covers exactly one thing: the equal-query-budget causal
ablation that decides whether the paper's central claim (residual
action-gradient consistency, not extra information or architecture, causes
the finite-budget recovery) survives. It does not cover the FD-sensitivity
sweep, the constraint-Jacobian case study, the oracle/converged-NLP
baseline, or any manuscript rewriting — those remain separate, later steps
in `JPC_MAJOR_REVISION_RESPONSE_STRATEGY_2026_08_31.md`, gated on this
protocol's outcome.

**Hard precondition — simplified in v4, the previous wording had two
sentences in tension (wall-clock-only gating vs. "should still avoid").**
Design review and implementation and the Section 0.3 synthetic smoke test
may proceed regardless of Paper M's state, since none of them touch real
data or run real training. But EVERY real step beyond that — Section 2's
real query-pool construction, any real training, and any real closed-loop
run, timed or not — requires Paper M's background real data-generation
process (`paper_m_data_scaling_pilot`, PID tracked in that project's own
`PAUSE_AND_RESUME_2026_08_31.md`) to be confirmed either complete or
stopped first. There is no "non-timed runs are exempt" carve-out: even an
untimed real training run contends for the same CPU Paper M's generation
needs, and this protocol's own primary/secondary contrasts do not need to
start one day earlier than Paper M's completion.

**Freeze sequence (corrected in v2 of this protocol)**: freezing a
text-only design before writing any code is unsafe — Paper M's own
Data-Generation Protocol needed a Revision 8/9/10 hardening cycle
precisely because implementation surfaced defects (a missing `mkdir`,
a non-resumable code edit) that a design review alone did not catch. The
correct order for this protocol is:

1. Design review of this document (in progress; this is a v1->v2 pass).
2. Implementation of the code this protocol requires (Sections 2-8, 10) —
   including the two genuinely new pieces this review surfaced: a disk
   checkpoint cadence (Section 8) and a Softsign, width-40 FNN variant
   (Section 4, exact 1922-parameter match to the LCNN), neither of which
   exists in the codebase today.
3. A synthetic/control-flow smoke test: tiny synthetic data, 1-2 seeds,
   a handful of epochs, exercising every code path (query-count
   assertions, the `lambda_Jac` grid, the MSE-match grid and its
   `MSE_MATCH_INFEASIBLE` path, the checkpoint cadence, the outcome-map
   arithmetic in Section 9) without touching the real 20k-point pool, the
   real plant simulator beyond what the synthetic stub needs, or any real
   gradient oracle. This is exactly the role Paper M's `*_synthetic_e2e_
   smoke_*.py` scripts already play for that protocol.
4. Freeze: this document, the implementing code, and the synthetic-smoke
   result together, as one hash-locked manifest (Section 10) — mirroring
   Paper M's convention where the freeze covers protocol + driver + I/O
   helpers + smoke result jointly, not the protocol document alone.
5. Only after that freeze does Section 2's real query-pool construction
   and any real training begin. Nothing in steps 1-3 touches the real
   identification source, the real 20k-point pool, or any real gradient
   oracle — only the freeze in step 4 unlocks real execution.

## 1. Plant and scope of the go/no-go run

Single CSTR only (two-state, two-input), matching the manuscript's own
"CSTR isolates the mechanism" role. The two-CSTR process is explicitly
OUT OF SCOPE for this protocol; it is committed to only if this
protocol's `QUERY_MATCHED_EFFECT` outcome (Section 9, Step 1) is
`SUPPORTED` and time remains. **Corrected in v3**: this used to cite a
stale `MECHANISM_SUPPORTED` label and the wrong section number — Section
9 now defines the actual outcome names.

**Primary closed-loop endpoint — added in v4, this was entirely undefined
before.** Confirmed by direct read of `budget_sweep_solver.py`'s
`closed_loop(seq, norm, x0, horizon, budget, steps, lr, rho_u, success_V,
un_lo, un_hi)` and its call site in `lgrad_experiment.py:119-121`, and
cross-checked against the manuscript's own headline configuration
(`paper_h_jpc.tex`: "a horizon-3, budget-20 first-order step," line 493;
"finite-budget first-order MPC (budget~20)," line 554; "budget~20, ten
seeds," line 776). The PRIMARY endpoint for every condition, seed, and
initial condition in this protocol is exactly the manuscript's own
headline configuration, fixed as follows and not re-tuned:
- `horizon = 3`, `budget = 20` (first-order Adam solver iterations per
  control step), `steps = 120` (closed-loop simulation length),
  `lr = 0.1` (Adam learning rate on the horizon input sequence),
  `rho_u = 0.01` (input-magnitude regularization in the horizon
  objective), `success_V = 2.0` (success threshold on the true Lyapunov
  value at the final step).
- Warm start: the input sequence is initialized to zero (normalized mean
  input) only at the very first closed-loop step; every subsequent step
  warm-starts from the previous step's optimized sequence, shifted by one
  and repeating the last entry (`un = cat([un[1:], un[-1:]])`,
  `budget_sweep_solver.py:82`) — a receding-horizon warm start, not a
  cold restart every step.
- Projection: after every solver iteration, the normalized input sequence
  is clamped elementwise to `[un_lo, un_hi]` (`budget_sweep_solver.py:76`)
  — this is the "projection-active" regime Proposition 1 does not cover
  (see the response-strategy document's Tier 1 item 2).
- Success criterion: `final_V <= success_V` at the end of the 120-step
  rollout for a given initial condition (`budget_sweep_solver.py:87`);
  the per-seed, per-condition success RATE (Section 7) is the fraction of
  the 5 fixed initial conditions (Section 7.2) meeting this criterion.
- `budget = 3` and the wider `B in {3,10,20,40,100}` sweep are reported as
  DESCRIPTIVE/secondary robustness context only (matching the
  manuscript's own budget-robustness figure, lines 625-628) — they are
  NEVER used for the primary or Holm-corrected secondary contrasts in
  Section 7, which are defined exclusively at `budget=20`.

## 2. Query pool and data roles

1. **Reuse the existing fixed pool, do not draw a new one.**
   `data/processed/lcnn_paper_cstr_onestep_20k.npz` already exists
   (confirmed on disk, 512,737 bytes, with a `.json` metadata sidecar),
   is exactly the pool `generate_lcnn_paper_cstr_data.py` produces by
   default, and already contains co-saved `train_idx`/`val_idx`/`test_idx`
   arrays indexing directly into its own `XU`/`Y` arrays (confirmed:
   `generate_lcnn_paper_cstr_data.py:133-168`), generated deterministically
   from `--seed` (default 0). The split ratio is confirmed from that same
   generation code: `n_train = round(0.525*N)`, `n_val = round(0.175*N)`,
   `n_test = N - n_train - n_val` (52.5% / 17.5% / 30%, `N=20000`).
   This protocol hash-locks and reuses this exact file rather than drawing
   a new pool, so a data change is never conflated with the method change
   under test. Note: `lgrad_experiment.py`/`budget_sweep_solver.py`
   currently only consume `train_idx`/`test_idx` from this file and ignore
   `val_idx` — this protocol is the first to require the file's own
   `val_idx` split for real validation-based selection (Section 5.3,
   Section 8), which the file already supports without regeneration.
2. **For every point in the TRAIN split only, generate the `2m=4` extra
   perturbation transitions — coordinate system corrected in v5.**
   Confirmed by direct read of `precompute_true_ugrad`
   (`lgrad_experiment.py:43-57`): `eps=1e-3` is applied in NORMALIZED
   action coordinates, `un +/- eps*e_j` where `un = (u - u_mean)/u_std`,
   NOT as a physical perturbation `u +/- eps*e_j` as earlier drafts of
   this document could be read to imply. The perturbed normalized action
   is then transformed back to physical units, `uu = un_perturbed *
   u_std + u_mean`, and it is THIS physical `uu` that is passed to
   `sim.step` to obtain the perturbation successor. Because `u_std`
   differs by orders of magnitude across the two CSTR inputs (`C_{A0}`
   vs. `Q`), a fixed normalized `eps` corresponds to a very different
   PHYSICAL step size per input dimension — this protocol uses the SAME
   normalized-coordinate convention as the existing code, not a
   physical-unit `eps`, and states this explicitly so the query pool is
   reproducible from this document alone.
3. These `2m` extra transitions per training point are the single shared
   "extra query pool." Every condition below either uses it or does not;
   no condition is allowed to draw additional identification-source
   queries beyond this shared pool. Validation and test splits are NEVER
   perturbed and never touched by any training objective.
4. **Boundary handling for `un +/- eps*e_j` — corrected in v3 for pairing,
   corrected in v5 for which coordinate system the check happens in.** A
   central finite difference over input dimension `j` needs BOTH
   `un+eps*e_j` and `un-eps*e_j`; per item 2, the boundary-validity check
   is performed on the PHYSICAL transformed action `uu = un_perturbed *
   u_std + u_mean` against the PHYSICAL input box (`INPUT_LO`/`INPUT_HI`
   from `budget_sweep_solver.py`) — never on `un +/- eps` against a
   normalized box, since `INPUT_LO`/`INPUT_HI` are physical-unit bounds.
   If either transformed physical action for dimension `j` falls outside
   the physical box for a given training point, BOTH transitions for that
   dimension `j` are dropped together for that point — never just the one
   that violated the box, since a one-sided remainder cannot form
   condition D/E/F's central-FD label, and B/C must drop the identical
   pair (not just one transition) to keep every query-consuming
   condition's total query count equal. The set of dropped (point,
   dimension) pairs is a property of the shared pool (fixed once, by the
   training points' locations, `eps`, and the normalized-to-physical
   transform), identical for every condition that consumes the pool, and
   its count is recorded and reported.
5. **Noise contract — corrected in v4: the primary comparison must use
   CLEAN B/C augmented data, not noisy, to actually close R3's fairness
   point.** v3 disclosed an asymmetry (noisy successor-state targets for
   B/C, clean FD-derived gradient labels for D/E/F) but disclosure alone
   does not remove the confound: under v3's design, E-B compares BOTH
   "gradient term on vs. off" AND "clean derivative-source label vs. noisy
   ordinary-transition label" at once, so a positive E-B result could
   still be attributed to label cleanliness rather than to `L_grad`
   itself. Fixed: the PRIMARY B/C cells use the CLEAN perturbation/
   auxiliary successor states (no noise) as their "ordinary transition"
   targets — putting every query-consuming condition (B, C, D, E, F) on
   equally clean identification-source information, differing ONLY in
   what they are trained to predict (successor states vs. gradients) and
   with what loss (Section 3's loss-family grid). This protocol still
   defines exactly two immutable tables, but their ROLE changes:
   - **CLEAN_QUERY_TABLE**: the base pool's `XU`/`Y` arrays (unchanged,
     as stored in `lcnn_paper_cstr_onestep_20k.npz`) plus the `2m`
     perturbation successor states and the Section 2.7 auxiliary-law
     successor states, ALL computed once via a direct, noiseless call to
     the true identification source (`sim.step`, matching how
     `precompute_true_ugrad` already computes perturbation successors at
     `lgrad_experiment.py:43-57`). This table is shared, seed-independent,
     identical across every condition, and is now what conditions B/C's
     PRIMARY augmented targets are drawn from (not noise-realized).
   - **NOISE_REALIZED_TABLE** (one per seed): built from CLEAN_QUERY_TABLE
     by applying the EXACT existing `rng = np.random.default_rng(seed +
     1000)` / Cauchy-noise procedure (`lgrad_experiment.py:60-71`) to the
     base pool's `train_idx` rows exactly as today — this is what EVERY
     condition's base (non-augmented) training points still use, matching
     the existing manuscript's own noisy-training-data convention, which
     this protocol does not change. **A separate, explicitly NON-primary,
     NON-gating SENSITIVITY ARM** additionally extends this same Cauchy
     procedure to B/C's augmented targets (i.e. exactly v3's design),
     reported alongside the primary clean-B/C result to characterize how
     much (if any) of the effect depends on the augmented data's noise
     level — but this arm never enters Section 7's primary or Holm-
     corrected secondary contrasts, and never gates Section 9's outcome
     map. FD-derived gradient labels (`true_ug` for E/F, `∇_u f` for D)
     remain CLEAN in both the primary design and the sensitivity arm,
     matching existing codebase precedent (`precompute_true_ugrad` never
     adds noise).
6. **Common scaling contract — corrected in v6: v5's shape/seed-
   dependence claims for `g_scale`/`ug_scale` were wrong.** Confirmed by
   direct read of `lgrad_experiment.py:73-78`: `g_scale = g_true_all
   [train_idx].std()` and `ug_scale = true_ug_t[train_idx].std()` both
   call PyTorch `.std()` with NO `dim` argument, which flattens ALL
   elements (both the point axis and, for `ug_scale`, the 2-input axis)
   into a single scalar — `ug_scale` is a SCALAR, not shape `(2,)` as v5
   stated. Further, `g_true_all` is computed from the pool's CLEAN `Y`/
   `Yphi_all` (line 73: `pr.Vt(torch.tensor(Y)) - pr.Vt(torch.tensor
   (Yphi_all))`, no noise-realized `Yn` involved), and `true_ug_t` comes
   from `precompute_true_ugrad`, which is likewise computed once from the
   clean pool and never touches per-seed noise. Since `train_idx` itself
   is also the SAME fixed set for every seed (Section 2.1), BOTH
   `g_scale` and `ug_scale` are IDENTICAL across every seed — they are
   SEED-INDEPENDENT, not seed-dependent as v5 stated. The exact 7-item
   keyset is: `xm` (shape `(4,)`, `float32`, seed-independent, from the
   pool file), `xs` (shape `(4,)`, `float32`, seed-independent), `ym`
   (shape `(2,)`, `float32`, seed-independent), `ys` (shape `(2,)`,
   `float32`, seed-independent), `g_scale` (SCALAR, `float32`, floor 1.0,
   SEED-INDEPENDENT), `ug_scale` (SCALAR, `float32`, floor 1.0,
   SEED-INDEPENDENT), and `disp` (shape `(2,)`, `float32`, median absolute
   deviation of `Yn[train_idx]` — the ONLY one of the 7 that is genuinely
   per-seed, since it is computed from the seed's noise-realized `Yn`).
   `xm`/`xs`/`ym`/`ys`/`g_scale`/`ug_scale` are all fixed ONCE (the first
   four when the pool file was generated, the latter two the first time
   they are computed from the pool's clean arrays — identical for every
   seed thereafter, never recomputed per seed); `disp` alone is computed
   per seed, from the base (un-augmented) `train_idx` rows of that seed's
   NOISE_REALIZED_TABLE, never from the 5x-larger augmented dataset. All 7
   are reused unchanged across every condition (A-F, both FNN cells) at a
   given seed — for the 6 seed-independent constants, that also means
   unchanged ACROSS every seed. Recomputing `disp` on an
   augmented dataset would silently rescale the loss itself between
   conditions, confounding the loss-family/gradient-term comparison with
   a scale change; fixing it from the base rows keeps every condition's
   loss numerically comparable at the same `lam_val`/`lam_g`/`lam_Jac`.
   **Addendum (v7): `ug_scale`'s valid-pair-only convention, pre-
   registered explicitly.** The reference formula quoted above
   (`true_ug_t[train_idx].std()`) is inherited from `lgrad_experiment.py`,
   which has NO boundary-drop/invalid-pair concept at all -- every entry
   of its `true_ug` is a genuine FD value. This protocol's Section 2.4
   boundary-drop rule introduces invalid `(point, dimension)` pairs that
   `build_clean_query_table` fills with a `0.0` placeholder (never a real
   gradient value). Applying the reference formula literally (`.std()`
   over ALL entries, placeholders included) would bias `ug_scale`
   downward toward the artificial zeros, which the reference formula's
   original context never had to account for. `ug_scale` is therefore
   computed over ONLY the valid entries (`true_ug[valid_mask]`), a
   deliberate, disclosed departure from a literal reading of the
   inherited formula, made necessary by a structural feature (the
   boundary-drop mask) this protocol adds that the reference
   implementation does not have. `g_scale` has no analogous issue (its
   `g_true_all` has no placeholder entries) and is computed exactly per
   the reference formula, unchanged.
7. **Auxiliary-law queries are common infrastructure, not part of the
   "extra" pool**: computing the residual training target
   `g(x,u) = V(f(x,u)) - V(f(x,Phi(x)))` requires evaluating the true
   identification source once at the auxiliary action `Phi(x)` for every
   training point, identically for every condition including reference
   condition A (it is already part of how the existing manuscript results
   were produced). This query is therefore counted in the "base" query
   total for every condition alike and does not affect the equal-EXTRA-
   query claim in item 9 below, but MUST still appear in the absolute
   total query count reported per condition, for full transparency.
8. **Boundary-valid mask and B/C's augmented objective — new in v4.** The
   Section 2.4 boundary-drop rule produces, per training point, a fixed
   0-2 count of VALID perturbation dimensions (0, 1, or 2 of the 2 input
   dimensions may be dropped as a `(point, dimension)` pair). This valid
   set is a single, shared mask applied identically to every
   query-consuming condition: D/E/F's Jacobian/gradient losses (Section 3)
   average ONLY over valid `(point, dimension)` pairs (the denominator is
   the valid count, not a fixed `2m=4` per point), and B/C's augmented
   training set includes ONLY the successor-state transitions from valid
   pairs (the same pairs D/E/F use, so every query-consuming condition's
   total query count, item 9, stays equal). **B/C's augmented points
   receive BOTH existing loss terms, exactly as ordinary training rows
   do**: the Cauchy data-fit loss (`cauchy_nll`, using the item 6 shared
   `disp`) against their clean successor-state target, AND the
   residual-value loss (`E_+`/symmetric, Section 3, using the item 6
   shared `g_scale`) against their own auxiliary-law-derived residual
   target — this is the most faithful reading of "ordinary `(x,u)->x'`
   transition data," since it is what a base training point already
   receives, and a weaker reading (data-fit loss only) would make B/C's
   augmented points a different KIND of training signal than D/E/F's
   augmented points, reopening a fairness gap.
9. Record, per condition, the exact count of identification-source queries
   consumed (base one-step queries, including the auxiliary-law query in
   item 7, plus any valid perturbation queries per item 8), and assert
   programmatically that the query-consuming conditions (B, C, D, E, F
   below) consume an IDENTICAL total query count to each other. Condition
   A (external reference) is explicitly exempt and must be labeled as
   using a smaller budget everywhere it is reported.

## 3. Conditions

All conditions except the FNN interaction block (Section 4) use the fixed
GroupSort-LCNN architecture at the manuscript's existing CSTR width/depth/
Lipschitz-bound settings, identical optimizer settings, and identical
random seeds across conditions (seed controls both data subsampling where
applicable and network initialization, matched pairwise across
conditions).

| ID | Loss family | Gradient term | Extra queries used as | Query budget |
|----|------------|----------------|------------------------|--------------|
| A (reference, not in primary contrast) | one-sided `E_+` | off | not used | base only |
| B | one-sided `E_+` | off | ordinary `(x,u)->x'` transitions | equal |
| C | symmetric least-squares residual | off | ordinary `(x,u)->x'` transitions | equal |
| D | one-sided `E_+` (Lyapunov value target unchanged) | off; instead `∂f/∂u` Sobolev term | FD-derived `∂f/∂u` labels | equal |
| E | one-sided `E_+` | on (`L_grad`, proposed) | FD-derived `∇_u V` labels | equal |
| F | symmetric least-squares residual | on (`L_grad`, proposed) | FD-derived `∇_u V` labels | equal |

This is a full 2 (loss family: one-sided/symmetric) x 2 (gradient term:
off/on) grid — cells B, C, E, F — plus D as a fifth, orthogonal
equal-query condition (alternative derivative target, not the
Lyapunov-specific one), plus A as an external, smaller-budget reference
point reported but never used in the primary or secondary causal
contrasts. This directly closes the gap in v2, which tested symmetric
loss only in combination with the gradient term (F) and never in
isolation (C) — R3's actual claim is that the one-sided penalty may hurt
the baseline BY ITSELF, which only cell C, contrasted against B, can test.

**Symmetric residual loss, exact formula — corrected in v4, the v3
formula dropped the existing `g_scale` normalization.** Confirmed by
direct read of `lgrad_experiment.py:96`, the existing one-sided term is
`lam_val * relu((g_true - g_hat) / g_scale).pow(2).mean()`, i.e. the
residual is normalized by `g_scale` (Section 2.6's shared, base-row-only
scale constant) BEFORE squaring. The symmetric variant removes ONLY the
`relu(...)` clamp, keeping the exact same normalization and weight:
`E_{sym}(\theta) = lam\_val \cdot \text{mean}\big((({g(x_i,u_i) -
\hat g_\theta(x_i,u_i)}) / g\_scale)^2\big)`. Dropping `g_scale` (as v3's
raw `(g-\hat g)^2` did) would change the loss FAMILY along two axes at
once — sidedness AND scale — making a fixed `lam_val=0.003` mean
something different for C/F than for B/E, which would confound the very
loss-form comparison this cell exists to isolate.

**Action-Jacobian Sobolev term (condition D), exact formula**:
`L_{Jac}(\theta) = \frac{1}{N}\sum_i \| \nabla_u \hat f_\theta(x_i,u_i) -
\nabla_u f(x_i,u_i) \|_F^2`, i.e. the Frobenius-norm error between the
learned and true action-Jacobian of the ONE-STEP DYNAMICS map (not of the
Lyapunov residual). This is a strictly different, lower-level target than
`L_grad`, which targets `\nabla_u V(\hat f)`, a scalar's gradient. Both
`\nabla_u \hat f_\theta` and `\nabla_u f` must be computed and compared in
the SAME normalized coordinates already used throughout the codebase
(`xm`/`xs`/`ym`/`ys`), not raw physical units, for the loss weight
calibration in Section 5 to be meaningful. **Reviewer-response framing,
added in v4**: `L_Jac` is explicitly the ACTION-Jacobian `\partial f /
\partial u` only, matching the `2m` perturbation pool's information
content exactly. The STATE-Jacobian `\partial f/\partial x` is
deliberately out of scope for this equal-query protocol: obtaining it
would require `2n` additional state perturbations per point, a strictly
larger and different query budget than the `2m` action perturbations
every other query-consuming condition uses — this must be stated
explicitly in the manuscript response so "full Jacobian Sobolev" is never
implied for condition D.

## 4. Architecture interaction block

Motivated by the reviewer's specific question (R3 point 3: does the
GroupSort/Björck construction itself manufacture the value-only failure
the paper then "solves"?). **Corrected in v3**: v2 said this block was
"not folded into the primary/secondary contrast family in Section 7,"
which directly contradicted Section 7.4, which DOES include `E-FNN vs.
B-FNN` in the Holm-corrected secondary family (needed because
`NON_GROUPSORT_REPLICATION` — renamed in v4 from `ARCHITECTURE_
GENERALIZATION`, Section 9 Step 2 — is a statistical gating judgment and
must go through the same family-wise error control as the other
secondary contrasts). The correct statement: the raw per-condition
success rates and parameter/architecture bookkeeping for `B-FNN`/`E-FNN`
are reported as a distinct descriptive block in this section, but the
`E-FNN vs. B-FNN` CONTRAST itself is one of the 7 members (v4) of
Section 7.4/7.5's Holm-corrected secondary family, not a separate,
uncorrected test.

- Two additional conditions: `B-FNN` (value-only-augmented, standard
  unconstrained feedforward network) and `E-FNN` (value+`L_grad`, same
  standard feedforward network).
- **Confirmed by direct read of `src/cstr/models_onestep.py`: this block
  cannot reuse an existing class as-is.** `OneStepFNN` (lines 131-144) is
  ReLU-only, has no `activation` constructor argument, and is used
  elsewhere at `hidden=64` (e.g. `cstr_sns_core_ablation.py`), while the
  GroupSort-LCNN used throughout every CSTR script is `hidden=40, n_layers=2`
  (`OneStepLCNN`, confirmed at `lgrad_experiment.py:151` and consistently
  across `probe_action_gradient.py`, `ggn_mpc_probe.py`, and others).
  `OneStepSmoothSpectral` (lines 175-216) does have a tanh/Softsign
  `acts` dict but is itself Björck-orthonormal-constrained, so it cannot
  be used as the "standard, unconstrained" comparator either. **This
  protocol therefore requires implementing a new, small variant**: copy
  `OneStepFNN`'s structure but add an `activation` argument restricted to
  `{"tanh", "softsign"}`, and fix the choice to **Softsign** specifically
  (not "tanh or Softsign" as an open choice) — Softsign is already the
  activation family used by the manuscript's existing smooth-Softsign LCNN
  comparator (Table 1), so this avoids introducing a third, never-before-
  used activation family into the paper. ReLU is explicitly excluded (it
  would introduce a new nonsmoothness confound absent from both GroupSort
  and the existing smooth comparator). No orthonormal (Björck) constraint,
  no GroupSort, no output Lipschitz bound.
- **Width fixed to 40, and parameter count IS exactly equal — corrected
  in v3.** Verified by direct construction from `models_onestep.py`:
  `OneStepLCNN(hidden=40, n_layers=2)` is `BjorckLinear(4,40)` (160+40
  params) -> GroupSort (0 params) -> `BjorckLinear(40,40)` (1600+40) ->
  GroupSort (0) -> `MaxNormLinear(40,2)` wrapping `nn.Linear(40,2)`
  (80+2) = **1922 parameters exactly**. A `OneStepFNN`-shaped network at
  `hidden=40` is `nn.Linear(4,40)` (160+40) -> ReLU -> `nn.Linear(40,40)`
  (1600+40) -> ReLU -> `nn.Linear(40,2)` (80+2) = **1922 parameters
  exactly** — identical, because `BjorckLinear`/`MaxNormLinear` wrap a
  weight+bias pair of the same shape as `nn.Linear` and add no extra
  parameters of their own. **The v2 text's claim that Björck/GroupSort
  layers "have structurally different parameter counts even at identical
  width/depth" was wrong** — there is no such difference here, so the
  10%-tolerance/width-adjustment fallback is removed: this protocol
  ASSERTS exact parameter-count equality (1922 == 1922) at `hidden=40,
  n_layers=2` for both networks, programmatically checked, not merely
  approximated.
- **Initialization — corrected in v3 to state the actual, disclosed
  difference rather than an unachievable "same family" claim.**
  `BjorckLinear.__init__` applies `nn.init.orthogonal_` to its raw weight
  before every forward pass's Björck projection (confirmed:
  `models_onestep.py:67`), while `OneStepFNN`'s (and the new FNN
  variant's) `nn.Linear` layers use PyTorch's ordinary default init
  (`nn.Linear.reset_parameters`'s Kaiming-uniform-based scheme). Because
  the two architectures parameterize their weights differently
  (orthogonal-then-Björck-projected vs. plain dense), there is no
  well-defined single "initialization family" that is literally identical
  across both — claiming one would be false precision. What IS matched,
  and is the honest, implementable rule: both networks are constructed
  with the SAME PyTorch RNG seed immediately before instantiation (so the
  draw order and seed state are identical inputs to each network's own,
  architecture-native default init), and both networks' actual
  initialization procedures are reported verbatim in the results
  artifact as a disclosed difference, not hidden behind a "matched init"
  label.
- Uses the SAME equal-query pool as conditions B/E (this block does not
  need its own separate query-budget bookkeeping beyond what Section 2
  already establishes, since it reuses exactly conditions B and E's data
  roles on a different architecture).
- **Exact matched initialization for the B-FNN/E-FNN pair specifically —
  strengthened in v4**: `B-FNN` and `E-FNN` at a given seed are
  constructed with the IDENTICAL PyTorch RNG draw (same seed, same
  instantiation order) so their initial weights are bit-identical before
  training diverges — any difference between them is then attributable
  only to the training objective, not to independent random init draws.
  This is a strictly tighter requirement than the seed-per-run matching
  already stated for LCNN-vs-FNN comparisons above.
- **Freeze/materialize adapter — corrected in v4: the v4 "already safe by
  construction" claim checked the WRONG function and is FALSE.** The
  primary evaluation path (`lgrad_experiment.py:118`: `seq = bs.freeze
  (model)`) does not call `models_onestep.py`'s `materialize_lcnn_for_
  inference()` at all — it calls `budget_sweep_solver.py:39`'s OWN
  `freeze(model)`, confirmed by direct read to unconditionally do
  `for layer in model.body: ...` and `model.out.materialized_linear()`.
  `OneStepFNN` (and the new FNN variant) has neither a `.body` nor an
  `.out` attribute — it has a single `.net` Sequential — so `bs.freeze
  (fnn_model)` raises `AttributeError` immediately, and every FNN-block
  closed-loop evaluation in Section 7 would fail at exactly this call.
  **Fixed**: this protocol requires implementing a small generalized
  freeze adapter, e.g. `freeze_any(model)`, that dispatches on model type
  — for `OneStepLCNN`/`OneStepSmoothSpectral` it calls the EXISTING
  `bs.freeze(model)` unchanged (no behavior change for any existing
  condition), and for the new FNN variant it returns `model` itself
  (already `.eval()`'d, `requires_grad_(False)`'d, with no Björck layers
  needing materialization, so no transformation is needed — its `.net`
  Sequential already matches the calling convention `seq(xun)` that
  `closed_loop()` expects). Every call site that currently calls
  `bs.freeze(model)` for closed-loop evaluation (Section 1's primary
  endpoint, Section 6's oracle, any budget-sweep descriptive run) must
  route through `freeze_any` instead once the FNN block exists. The
  Section 0.3 synthetic smoke test must exercise this THROUGH the actual
  `closed_loop()` entry path on a tiny synthetic FNN model (not just call
  a materialization function in isolation), since it was exactly this
  kind of gap between "the function I checked" and "the function actually
  called at runtime" that produced this bug in the first place.

## 5. Jacobian-target normalization and loss-weight calibration

1. All Jacobian/gradient targets (for `L_grad` and `L_Jac`) are computed
   and compared in normalized coordinates, using the plant's existing
   `xm`/`xs`/`ym`/`ys` normalization constants — the same convention
   `probe_action_gradient.py`'s `diagnose()` and `cartpole_lgrad.py`'s
   `precompute_true_ugrad()` already use. No new normalization scheme is
   introduced.
2. `\lambda_g` (the `L_grad` weight) is fixed at the manuscript's existing,
   already-tuned CSTR dose-response optimum (`\lambda_g \approx 0.05`,
   Fig. 5/`fig:dose`), not re-tuned in this protocol — re-tuning it here
   would bias the comparison in favor of the proposed method. Likewise,
   `lam_val` (the `E_+`/symmetric-residual weight, confirmed by direct
   read of `lgrad_experiment.py:96` at its existing default `0.003`) is
   held at this same fixed value for EVERY condition that has a residual-
   value term (A, B, C, D, E, F and both FNN cells) — including condition
   C/F's symmetric variant, even though removing the one-sided clamp
   changes the term's typical magnitude. Not re-tuning it, for the same
   reason `\lambda_g` is not re-tuned: the ablation tests the effect of
   the loss FORM and the gradient term, not of a re-optimized weight.
3. `\lambda_{Jac}` (condition D's weight) does NOT have an existing tuned
   value and must not be picked arbitrarily. Pre-registered coarse
   log-spaced grid: `{0.001, 0.005, 0.01, 0.05, 0.1, 0.5}`. **Selection
   rule, corrected in v2**: choose the value minimizing validation-split
   one-step MSE only; ties (numerically identical minimum MSE, which is
   not expected in practice) are broken by the smallest `\lambda_{Jac}`
   value in the grid — a deterministic rule that uses no additional
   information. The wrong-sign/cosine gradient-fidelity metric (Section 6)
   is NOT used anywhere in this selection, even as a tie-breaker: that
   metric requires the true plant's gradient at validation points, which
   would give condition D an oracle-informed selection step that
   conditions B/C/E/F do not receive under their own (MSE-only) selection
   rules, quietly breaking the equal-information design this whole
   protocol exists to enforce. The gradient-fidelity oracle is instead
   applied identically, post-hoc, to every condition's already-selected
   final model, purely as a reported metric — see Section 6. Closed-loop
   success is NEVER used to select `\lambda_{Jac}` either, to avoid
   cherry-picking on the very metric the ablation is testing.
   **Full-grid transparency — new in v4**: selecting `\lambda_{Jac}` by
   validation MSE alone, while the only fair rule available, could in
   principle land on a value with poor Jacobian fidelity, which would
   make condition D look artificially worse and bias `TARGET_SPECIFICITY`
   (Section 9) toward `SUPPORTED`. Rather than changing the selection rule
   (which would reopen the oracle-asymmetry problem this rule was fixed
   to close), this protocol requires reporting EVERY candidate in the
   pre-registered `\lambda_{Jac}` grid's validation MSE AND its resulting
   E-D contrast outcome in the results artifact — so a reviewer or the
   authors can see whether the selected value was an outlier or
   representative of the grid, without changing which value is used for
   the pre-registered primary/secondary computation. **Addendum (v7):**
   "resulting E-D contrast outcome" means the SAME statistical form as
   every other contrast in this protocol — a paired (E minus that
   `\lambda_{Jac}` candidate's model) closed-loop success-rate point
   estimate, seed-level bootstrap CI, and exact permutation p-value
   (Section 7.3's method, same fixed bootstrap seed) — computed for EVERY
   grid candidate, not only the selected one. These per-candidate
   contrasts are purely descriptive/diagnostic (never Holm-corrected as
   part of the 7-member secondary family, never gating any Section 9
   judgment) — they exist only so a reader can see the grid's sensitivity.

4. **Training hyperparameters and the "primary model" convention — new in
   v3, confirmed by direct read of `lgrad_experiment.py`'s `train()`
   signature and body.** Every condition (A-F and both FNN cells) trains
   with the SAME existing defaults: `epochs=50`, `warmup=30`, batch size
   `256`, Adam at learning rate `2e-3` (lines 61, 81, 85). The value-
   residual term (`lam_val`) and the gradient term (`lam_grad`/
   `lam_Jac`), where present for a given condition, activate only for
   `ep > warmup` (i.e. epoch 31 onward), matching the existing code's own
   `if lam_val > 0 and ep > warmup` / `if lam_grad > 0 and ep > warmup`
   gates (lines 91, 97) — this protocol does not change that schedule.
   **Primary model selection**: confirmed by direct read that
   `lgrad_experiment.py`'s `train()` returns `model.eval()` after the
   full 50-epoch loop with no best-validation-checkpoint selection
   anywhere — the existing manuscript results already use the FINAL-EPOCH
   model, not a validation-best one. This protocol keeps that convention
   for every condition's PRIMARY model (the one whose closed-loop success
   rate feeds the primary and secondary contrasts in Section 7): the
   model evaluated is always the epoch-50 model, for every condition,
   with no cherry-picking. The NEW disk-checkpoint cadence and validation-
   based selection introduced in Section 8 are used ONLY to build
   condition B's MSE-matched candidate for the separate MSE-match
   secondary contrast — they never change which model is used as
   condition B's (or any other condition's) primary evaluated model.
5. **RNG stream separation and initial-state hashing — new in v5.**
   Confirmed by direct read of `train()` (`lgrad_experiment.py:64,84`):
   the existing code already uses two SEPARATE generators — PyTorch's
   global RNG via `torch.manual_seed(seed)` (governs weight
   initialization) and a numpy `rng = np.random.default_rng(seed+1000)`
   (governs BOTH the base-pool Cauchy noise draw, line 69, AND the
   per-epoch minibatch shuffle order, line 84, from the SAME stream in
   sequence). This protocol's Section 2.5 sensitivity arm adds a NEW
   noise draw (extending Cauchy noise to B/C's augmented targets) that
   did not exist before — if that draw is taken from the SAME `rng`
   stream used for shuffling, it would consume extra random numbers
   BEFORE the shuffle draw for conditions that need it (B/C's sensitivity
   arm) but not for conditions that don't (A, D, E, F, and B/C's own
   PRIMARY clean cells), silently desynchronizing minibatch order across
   conditions for a reason unrelated to the scientific manipulation.
   **Fixed**: three numpy generators, independently seeded per training
   seed `k`, are used: `np.random.default_rng(k+1000)` for base-pool
   Cauchy noise (unchanged from existing code), `np.random.default_rng
   (k+4000)` for the Section 2.5 sensitivity-arm's augmented-target
   noise (NEW, used only by the non-primary sensitivity-arm runs), and
   `np.random.default_rng(k+5000)` for minibatch shuffle order (NEW —
   previously entangled with the noise stream, now split out so adding or
   removing a noise draw can never change shuffle order). PyTorch's
   `torch.manual_seed(k)` for weight initialization is unchanged. **Every
   condition's model, at a given seed, records a hash of its
   `state_dict()` immediately after construction (before any training
   step)** — for the GroupSort-LCNN-based conditions (A-F), this hash
   must be IDENTICAL across all 6 at a given seed (same architecture,
   same `torch.manual_seed(k)`); a mismatch is a protocol-execution
   failure requiring investigation before training proceeds, not a
   silent continuation.

## 6. Metrics, and the label-quality vs. model-gradient-quality distinction

This section exists because a prior draft conflated two different
quantities that must be kept separate. **This section's oracle is a
shared, final-evaluation-only grant**: it is computed once, identically,
on every condition's already-finalized model (A-F and both FNN cells) at
the same held-out TEST points, after all training and hyperparameter
selection (including condition D's `\lambda_{Jac}` choice, Section 5.3)
is already complete. It is never used inside any training loop or any
model/hyperparameter selection step for any condition — that is what
keeps it from re-opening the equal-information concern this protocol
exists to close.

- `reviewer_supplement_cstr.json`'s `label_quality()` block
  (`rel_err_median`, `rel_err_p90`, `cos_median`, `wrong_sign_frac` under
  `fd_clean`/`fd_noisy`) compares one finite-difference LABEL estimate
  against a cleaner reference FD estimate — it characterizes LABEL noise,
  not any trained model.
- The metric this protocol (and the revised manuscript) actually needs is
  the TRAINED MODEL's gradient fidelity: cosine similarity and relative
  error between the learned surrogate's gradient and the true plant's
  gradient, at held-out test points. `probe_action_gradient.py`'s
  `diagnose()` already computes `align_cos_med`/`align_frac_neg` this way
  (confirmed: `cos(true_grad, g_hat_grad)` where `g_hat_grad` comes from
  autograd through the trained model and `true_grad` from central finite
  differences of the true simulator) — this part can be reused as-is.
- The relative gradient error norm, `\|\nabla_u \hat g - \nabla_u g\| /
  \|\nabla_u g\|`, for the trained model is **not currently computed
  anywhere** and must be added as a new field inside `diagnose()` (the
  arrays `true_grad` and `g_hat_grad` already exist in memory at that
  point, so this is a small addition, not a new pipeline) — it must not be
  conflated with or substituted by `label_quality()`'s `rel_err_median`.
- **Near-zero true-gradient exclusion**: both the existing cosine
  computation and the new relative-error computation must exclude points
  where `\|\nabla_u g\|` (or `\|\nabla_u \hat g\|` for cosine) falls below
  a fixed threshold (the existing `diagnose()` code already uses `1e-9`
  for cosine's denominator guard). Report, for every condition, the
  fraction of evaluated points excluded by this guard — a condition that
  silently excludes a large fraction is hiding information, and the
  fraction itself must appear in the results table.
- **Fixed oracle test-point set — new in v4.** Confirmed by direct read
  of `probe_action_gradient.py:100-103`: `diagnose(..., seed=0)` draws its
  800-point subset via `rng = np.random.default_rng(seed)`; `rng.choice
  (test_idx, size=800, replace=False)` — since `eval_model()` calls
  `diagnose(..., seed=seed)` with the SAME per-condition training seed,
  different seeds (0-9) each get a DIFFERENT 800-point subset, conflating
  training-seed variance with oracle-sampling variance in the Section 6
  metrics (though this does not affect Section 7's closed-loop success
  rate, which never calls `diagnose()`). Fixed: this protocol draws ONE
  fixed 800-index subset of `test_idx`, using a separate meta-seed
  independent of the 10 primary training seeds, hash-locks it, and passes
  it explicitly to every `diagnose()` call for every condition and every
  seed (requiring a small new `fixed_idx` override parameter on
  `diagnose()`, since it currently only accepts a `seed` int) — so every
  reported oracle metric in Section 6 is evaluated on the exact same test
  points for every condition/seed, isolating training variance from test-
  point-sampling variance. **Meta-seed fixed numerically in v5**: the
  oracle's fixed 800-index subset is drawn with `np.random.default_rng
  (90210)` (an arbitrary literal, chosen only to be numerically distinct
  from every training seed 0-9 and from the bootstrap seed below — its
  specific value carries no meaning and must not be re-derived or
  changed after seeing results).

## 7. Seeds, initial conditions, primary/secondary contrasts

1. **Seeds — fixed as an explicit literal list in v5, not just "10
   seeds."** `SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` for the primary
   battery (conditions A-F), matching the manuscript's existing two-CSTR
   ablation seed count and its own `[0,1,2,...]` convention
   (`lgrad_experiment.py:40`'s `SEEDS = [0, 1, 2]` extended to 10). The
   Section 4 architecture-interaction block uses the same `SEEDS` list
   (paired with B/E by seed index — i.e. `B-FNN`/`E-FNN` at seed `k` are
   compared against `B`/`E` at the SAME seed `k`, never cross-seed). This
   protocol does not cover the FD-sensitivity sweep; that sweep (a
   separate, later step) uses a minimum of 5 seeds, not 3, per the
   corrected recommendation. **Bootstrap RNG convention, precision-fixed
   in v6**: every seed-level bootstrap CI in this protocol (item 3 below,
   and every Section 7.4 secondary contrast) instantiates a FRESH
   `np.random.default_rng(31415)` (the same literal seed, distinct from
   the training seeds and the Section 6 oracle meta-seed, fixed before
   any real run and never re-derived from results) independently for
   EACH contrast — not one stateful generator created once and
   sequentially consumed across contrasts, which would make later
   contrasts' resamples depend on how many draws earlier contrasts
   happened to take. A fresh same-seeded generator per contrast means
   every contrast's first 10,000 resample-index draws are identical in
   principle (deterministic given `n=10`), which is exactly the
   reproducibility property this protocol wants. As a second, independent
   safeguard (belt-and-suspenders, since RNG-implementation details can
   still surprise), the ACTUAL generated resample-index arrays for every
   contrast are hash-locked into the Results Manifest (Section 10, M3) —
   so reproducibility is verified by hash, not merely asserted by
   convention.
2. **Initial conditions**: the SAME fixed 5-point closed-loop
   initial-condition set already used throughout the manuscript's CSTR
   evaluation, confirmed by direct read of `budget_sweep_solver.py`
   (`ICS`, deviation coordinates `[ΔC_A, ΔT]`):
   `[0.0, 20.0]`, `[0.0, -20.0]`, `[0.25, 10.0]`, `[-0.25, -10.0]`,
   `[0.45, 0.0]` — held identical across every condition and seed in this
   protocol, never re-sampled.
3. **Primary contrast (the single confirmatory test, no multiple-comparison
   correction applied to it since it is the sole pre-registered
   confirmatory hypothesis)**: closed-loop success rate,
   condition E (`value + L_grad`, one-sided) minus condition B
   (`value-only-augmented`, one-sided) — same loss family, isolates the
   gradient-matching term alone under an equal query budget. Statistical
   unit: per-seed closed-loop success rate (fraction over the fixed IC
   set), paired by seed. **Fully specified statistical contract, v5**:
   - A seed-level bootstrap **95% two-sided percentile-method CI**,
     10,000 resamples, using the single fixed `np.random.default_rng
     (31415)` stream (Section 7.1) — percentile method means the
     `[2.5th, 97.5th]` percentiles of the 10,000 bootstrap-resampled paired
     differences, not a normal-approximation interval.
   - An exact paired permutation (sign-flip) test over the 10 seeds,
     feasible exactly by enumerating all `2^10=1024` sign patterns. The
     test statistic is the **two-sided absolute mean paired difference**,
     `|mean(d_1,...,d_10)|` where `d_i` is seed `i`'s paired difference;
     the p-value is the fraction of the 1024 sign-flip patterns whose
     `|mean|` is `>=` the observed `|mean|` — this is the SAME two-sided
     convention used for every one of Section 7.4/7.5's 7 secondary
     contrasts, so no contrast uses a different sidedness or statistic
     than any other.
   Both are reported side by side; the bootstrap CI is what Section 9's
   outcome map is defined on (unchanged as the pre-registered decision
   rule), and the permutation p-value is reported as a corroborating,
   assumption-light cross-check. If the two disagree on significance, both
   numbers are reported as-is and flagged in the write-up — the CI-based
   rule in Section 9 still governs the outcome, so this cannot be used to
   pick whichever test is more favorable after the fact.
4. **Secondary contrasts — corrected in v4, the family was mislabeled
   "5" while already enumerating 6, and was missing a 7th, decisive cell
   (Holm–Bonferroni corrected as a family of exactly 7, fixed before
   running)**:
   1. E vs. D (Lyapunov-specific gradient vs. generic action-Jacobian
      target).
   2. C vs. B (does the one-sided loss hurt the baseline by itself,
      isolated from the gradient term).
   3. F vs. E (does loss sidedness matter when the gradient term is on).
   4. **F vs. C — new in v4, the family's most direct test of R3's actual
      question.** Tests whether `L_grad` helps WITHIN the symmetric loss
      family, isolated from one-sidedness entirely. Without this
      contrast, the family cannot separately confirm "the gradient term
      itself helps" under symmetric loss — it can only infer it indirectly
      through the interaction contrast (item 5), which conflates the two
      effects rather than testing F vs. C directly.
   5. **Interaction contrast, `(F-E) - (C-B)`**: tests whether the
      gradient term's effect (E-B vs. F-C) itself depends on loss
      sidedness. Computed from the same paired per-seed quantities
      already collected for contrasts 2-4 (no new data), with its own
      seed-level bootstrap CI. **Corrected in v4 (the v3 wording here was
      itself an instance of the absence-of-evidence/evidence-of-absence
      error Section 9 already fixes elsewhere): a CI including zero is
      `NO_DETECTED_INTERACTION`, not evidence of independence** — genuine
      independence is only established by the `EQUIVALENT` equivalence
      test in Section 9, not by this raw CI read.
   6. E-FNN vs. B-FNN (does the effect replicate off the constrained
      architecture on a Softsign FNN — Section 9 reports this as
      `NON_GROUPSORT_REPLICATION`, not "architecture-independent," since a
      single alternative architecture cannot support the broader claim).
   7. The MSE-matched comparison (Section 8) as the 7th secondary
      contrast, not a replacement for the primary one.
5. **Per-contrast p-values and Holm adjustment — corrected in v4 for the
   family-of-7 and for the MSE-match infeasibility interaction.** Every
   one of the 7 secondary contrasts computes BOTH a seed-level bootstrap
   CI (same 95% two-sided percentile method, same fixed `np.random.
   default_rng(31415)` stream, 10,000 resamples, as the primary contrast,
   Section 7.3) AND an exact paired permutation p-value (same two-sided
   `|mean paired difference|` statistic as the primary contrast — no
   contrast in this family uses a different sidedness or bootstrap seed
   than any other). Contrasts 1-6 always use the
   full `2^10=1024` sign-flip enumeration over all 10 seeds (they never
   depend on MSE-match feasibility). **Contrast 7 (MSE-match) is
   different, fixed as follows to keep the family size constant
   regardless of outcome**: if the MSE-match comparison is fully
   `MSE_MATCH_INFEASIBLE` (Section 8.5, more than 3/10 seeds), contrast
   7's p-value is fixed at `p=1` (i.e. treated as showing no detectable
   difference, the maximally conservative value, never significant) and
   is still included in the Holm family at that value — it is NEVER
   dropped, and the family is NEVER re-sized to 6 post-hoc, since a
   result-dependent family size would make the correction itself
   result-dependent. If MSE-match is feasible but some seeds are
   individually `MSE_MATCH_INFEASIBLE_FOR_SEED` (at most 3 of 10), the
   permutation test for contrast 7 ONLY uses `2^n_valid` sign patterns
   over the `n_valid <= 10` seeds with a matched checkpoint — this
   reduced-`n` computation applies ONLY to contrast 7, never to contrasts
   1-6 or the primary contrast, all of which always use the full 10
   seeds. The 7 resulting p-values (with contrast 7's p=1 substitution
   where applicable) are Holm-Bonferroni adjusted as one family (sorted
   ascending, sequentially compared against `0.05/7, 0.05/6, ...,
   0.05/1`). Section 9's `TARGET_SPECIFICITY`, `NON_GROUPSORT_
   REPLICATION`, `LOSS_INTERACTION`, and `MSE_ATTRIBUTION` judgments are
   decided by these Holm-adjusted p-values (and the paired point-estimate
   direction), not by an uncorrected per-contrast CI read.
6. Every contrast's confidence interval and corrected p-value (where
   applicable) must be reported numerically in the results artifact; no
   contrast is summarized only as "the gap closed" or "the gap survived"
   without the interval.
7. **Failure and completeness contract — new in v4, previously
   unspecified.** (a) Non-finite training (`NaN`/`Inf` loss or weights at
   any epoch) for a given (condition, seed) aborts that run and records a
   hard `TRAINING_DIVERGED` failure — it is never silently retried with a
   different seed or excluded from reporting. (b) A simulator exception
   or non-finite rollout during closed-loop evaluation for a given
   (condition, seed, IC) is recorded as `success=False` (0) for that IC,
   contributing normally to the success-RATE denominator — it is never
   excluded from the rate calculation, since silently excluding hard
   failures would upward-bias every condition's success rate. (c) The
   primary and secondary contrasts (items 3-6) require ALL 10 seeds
   complete for EVERY condition in the primary battery (A-F) and the FNN
   block (B-FNN, E-FNN) before Section 9's outcome computation may run —
   a missing or failed primary seed is a protocol-execution failure
   requiring investigation and, if needed, a fully re-run seed (never a
   silent drop to `n=9` with contrasts rescaled).

## 8. MSE-matched comparison: explicit rule and failure handling

1. Purpose: condition E already reports lower one-step **test** MSE (not
   validation MSE — corrected in v3: `lgrad_experiment.py:108-111`
   confirms the existing `mse` return value is computed on `test_idx`,
   not `val_idx`) than condition A/B in existing results; an unconstrained
   comparison cannot rule out "lower MSE alone, not gradient consistency,
   explains the closed-loop gain." This comparison isolates that
   possibility by forcing condition B's evaluated checkpoint to have
   VALIDATION MSE (a distinct quantity from the existing manuscript's
   reported test MSE, computed on the previously-unused `val_idx` split,
   Section 2.1) close to condition E's own validation MSE — never test
   MSE, to keep the test split untouched by any selection decision.
2. **Selection uses the VALIDATION split only** (the `val_idx` split
   already present in `lcnn_paper_cstr_onestep_20k.npz`, confirmed to
   exist but currently unused by `lgrad_experiment.py`/`budget_sweep_
   solver.py` — this protocol is the first to require it for real
   validation-based selection). Closed-loop results are never inspected
   before the checkpoint is selected, and the selection rule is fixed
   before any closed-loop evaluation runs.
3. **Candidate grid for condition B — corrected in v2, this cadence does
   not yet exist and must be implemented, not "matched" to existing
   behavior.** Confirmed by exhaustive grep (`torch\.save`, `checkpoint`,
   `.pt`/`.pth`) across `scripts/` and `src/`: no CSTR training script in
   this codebase writes model weights to disk at any cadence. The only
   existing pattern is in-memory-only best-validation-MSE tracking
   (`train_lcnn_paper_cstr_model.py:140,150`: `best_state` kept as a
   Python dict, never written to disk), and `lgrad_experiment.py`/
   `budget_sweep_solver.py` do not even do that — they train once and
   evaluate the final epoch. This protocol therefore specifies a NEW
   cadence to implement: save condition B's (and, for symmetry and
   auditability, every other condition's) model `state_dict` to disk
   every 10 epochs and at the final epoch, using the SAME atomic
   no-overwrite convention already established for Paper M
   (`atomic_write_*_no_overwrite`-style, adapted for `.pt` files) so that
   partial/interrupted runs cannot silently corrupt a candidate. The
   candidate grid is exactly this saved-checkpoint set. If no candidate
   falls inside the tolerance band defined below, fall back to a
   secondary grid varying only regularization strength (weight decay) over
   `{0, 1e-5, 1e-4, 1e-3, 1e-2}` while holding every other hyperparameter
   fixed, retrained once per value (each retrain also checkpointed at the
   same 10-epoch cadence).
4. **Tolerance band and deterministic tie-break — tie-break rule added in
   v3.** A condition-B checkpoint is accepted as "matched" if its
   validation MSE lies within \pm10% (relative) of condition E's own
   validation MSE for the same seed. If more than one candidate (across
   both the epoch-cadence grid and, if needed, the weight-decay grid)
   falls inside the band, selection is fully deterministic, applied in
   this fixed order: (a) smallest absolute relative difference to
   condition E's validation MSE; (b) if still tied, smallest weight-decay
   value (preferring the unregularized/default-grid candidate); (c) if
   still tied, earliest checkpoint epoch. No candidate is ever chosen by
   inspecting anything other than validation MSE and this fixed tie-break
   order. **New in v5**: once selected, the matched checkpoint's TEST-
   split MSE (`test_idx`) is ALSO computed and recorded (never used for
   selection, only reported) — this is what Section 9's `MSE_ATTRIBUTION`
   `VALIDATION_MATCH_DID_NOT_GENERALIZE` check reads.
5. **Failure handling**: if no checkpoint in either grid falls inside the
   tolerance band for a given seed, that seed is recorded with outcome
   `MSE_MATCH_INFEASIBLE_FOR_SEED` and excluded ONLY from the MSE-matched
   secondary contrast (not from the primary contrast or any other
   condition's evaluation). If more than 3 of the 10 seeds hit this
   outcome, the MSE-matched comparison as a whole is reported as
   `MSE_MATCH_INFEASIBLE`, with the raw per-seed attempt log published
   rather than silently dropped. **Cross-reference, added in v4**: per
   Section 7.5, `MSE_MATCH_INFEASIBLE` does NOT remove this contrast from
   the 7-member Holm family — it is retained at a fixed conservative
   `p=1`. Per Section 9, `MSE_MATCH_INFEASIBLE` also does not count as
   support for the full claim: it yields `MSE_ATTRIBUTION = UNRESOLVED`,
   which blocks the full-claim conjunction the same way a failed test
   would, just for a different reason.
6. **Disclosure, not matching, for training compute**: optimizer steps and
   training wall-clock time are reported for every condition (the
   augmented dataset is roughly 5x larger for the CSTR's `2m=4`
   perturbations per point), but are NOT a quantity this protocol attempts
   to equalize — the reviewers' fairness concern is about identification-
   source query access (Section 2), not training compute, and equalizing
   training compute in addition would conflate two different resources.

## 9. Outcome map (fixed before running, governs manuscript narrowing)

**Corrected in v2**: a single `MECHANISM_SUPPORTED` label decided by the
primary contrast alone was over-scoped. E-B alone only establishes that,
in the GroupSort-LCNN, `L_grad` beats an equal-query augmented
value-only baseline — it does not by itself establish that the effect is
architecture-independent or specific to the Lyapunov-residual gradient
target rather than any action-Jacobian target. The outcome is therefore
decomposed into four named judgments; only their conjunction licenses the
manuscript's full claim.

**Step 1 — `QUERY_MATCHED_EFFECT` (the primary decision, decided ONLY by
the E-B contrast)**. Pre-registered minimal-meaningful-effect threshold:
0.2 absolute success-rate difference. **Corrected in v3: the three
categories below were not exhaustive** — a CI that includes zero with a
point estimate in `(-0.1, 0]` fell into none of them. Fixed by evaluating
the categories in a FIXED ORDER (NEGATIVE first, then SUPPORTED, and
UNCLEAR as the exhaustive remainder), so every possible (CI, point
estimate) pair lands in exactly one category:

1. **`NEGATIVE`** (checked first): the CI's upper bound is below 0 (a
   statistically significant reversal favoring condition B), OR the point
   estimate is at most -0.1 regardless of significance (a pre-registered
   materiality floor for a reversal, set to account for `n=10` seeds
   having limited power to reach significance even on a real reversal).
2. **`SUPPORTED`** (checked only if not `NEGATIVE`): the CI's lower bound
   exceeds 0 (statistically resolved) AND the point estimate is at least
   0.2 (practically meaningful). Both conditions reported explicitly, not
   merged.
3. **`UNCLEAR`** (everything else — the exhaustive remainder, not a third
   independently-defined condition): covers a real-but-small effect (CI
   lower bound > 0, point estimate in `[0, 0.2)`), a positive-but-
   unresolved effect (CI includes zero, point estimate in `(0, 0.2)` or
   `>=0.2`), and the previously-unclassified near-zero/slightly-negative
   region (CI includes zero, point estimate in `(-0.1, 0]`).

The exact permutation test from Section 7.3 is reported alongside for
every one of these three cases as a corroborating check, but the category
above is decided by the bootstrap CI / point-estimate rule only, fixed
before running.

**Step 2 — further decompositions, computed and reported ONLY if Step 1
is `SUPPORTED`** (if Step 1 is `UNCLEAR` or `NEGATIVE`, these are still
computed for transparency but are labeled exploratory-only and do not
change the manuscript action below). **Corrected in v3: gating now uses
the Holm-adjusted secondary-family p-value and direction, not a raw
"CI excludes zero" read, since these ARE members of Section 7.4/7.5's
7-member Holm-corrected family** — an uncorrected per-contrast CI read
would inflate the family-wise false-positive rate exactly the way Holm
correction exists to prevent:

- **`TARGET_SPECIFICITY`** (E vs. D): `SUPPORTED` if that contrast's
  Holm-adjusted p-value is `<= 0.05` AND the point estimate favors E;
  otherwise `NOT_SPECIFIC` — the generic action-Jacobian target cannot be
  ruled out as equally effective, and the manuscript may not claim the
  Lyapunov-residual gradient is specifically what matters.
- **`NON_GROUPSORT_REPLICATION`** (E-FNN vs. B-FNN) — **renamed in v4
  from `ARCHITECTURE_GENERALIZATION`, since a single alternative
  architecture (one Softsign FNN) cannot support a general
  "architecture-independent" claim, only a narrower "replicates on this
  one non-GroupSort architecture" claim.** `SUPPORTED` if that contrast's
  Holm-adjusted p-value is `<= 0.05` AND the point estimate favors
  E-FNN; otherwise `GROUPSORT_SPECIFIC` — the effect does not clearly
  replicate off the constrained GroupSort-LCNN on the one alternative
  architecture tested, and the manuscript's claim must be scoped to
  GroupSort-LCNN specifically (never to "architecture-independent" even
  if `SUPPORTED`, since one replication is evidence, not generality).
- **`LOSS_INTERACTION`** (from the `(F-E)-(C-B)` interaction contrast,
  Section 7.4 item 5) — **corrected in v3, and Section 7.4's own
  description of this contrast corrected to match in v4: a CI including
  zero is absence of evidence for an interaction, not evidence of its
  absence; claiming "independent" needs an equivalence test, not a null
  result.** Three outcomes, not two:
  - `INTERACTING`: the Holm-adjusted p-value is `<= 0.05` (a detected
    nonzero interaction) — the gradient-term effect and loss-sidedness
    are entangled and must be reported jointly, not as separable claims.
  - `EQUIVALENT`: not `INTERACTING`, AND the bootstrap CI lies entirely
    within a pre-registered equivalence margin of `[-0.1, 0.1]` —
    **justification corrected in v4: this is HALF of the primary 0.2
    minimal-effect threshold, not "one-tenth" as v3 incorrectly stated
    (0.1 is 1/2 of 0.2, not 1/10)**. Half the primary threshold is chosen
    as the largest interaction magnitude still small enough not to
    matter for the manuscript's claim relative to the effect size the
    claim itself requires — this is the only case in which the
    manuscript may state the gradient-term effect is loss-family-
    independent, since it is now backed by a genuine equivalence bound,
    not merely a failure to reject.
  - `INCONCLUSIVE`: neither of the above (a non-significant point
    estimate whose CI is not tight enough to fall inside the equivalence
    margin either) — reported as-is; the manuscript may not claim either
    independence or interaction from this contrast.
  This judgment is reported but does NOT gate the conjunction below — it
  characterizes robustness to loss-family choice, it is not itself part
  of the core causal claim.
- **`MSE_ATTRIBUTION`** (Section 8's MSE-matched comparison) — **new in
  v4, closing a gap where the MSE-matched contrast was computed but never
  connected to the full claim; made exhaustive and numerically defined in
  v5 (v4's "indistinguishable from zero" was not a number, and the
  selected model's TEST-set generalization was never checked).** If
  condition E already reports lower one-step test MSE than B/A (confirmed
  in existing results), the full causal claim ("gradient consistency
  causes the recovery") is not supportable if an MSE-MATCHED condition-B
  model closes the SAME closed-loop gap — that would mean lower MSE
  alone, not gradient consistency, explains it. Evaluated in this FIXED
  order, so every case lands in exactly one category:
  1. **`UNRESOLVED`** (checked first): the MSE-match comparison itself is
     `MSE_MATCH_INFEASIBLE` (Section 8.5) — an unresolved test is NOT
     treated as support, per the fixed `p=1` Holm substitution (Section
     7.5) already making it non-significant by construction.
  2. **`VALIDATION_MATCH_DID_NOT_GENERALIZE` -> `UNRESOLVED`** (checked
     next, only if not already `UNRESOLVED` above): for every seed's
     selected MSE-matched condition-B checkpoint (Section 8.4), also
     compute its TEST-split MSE (the same `test_idx` quantity the
     existing manuscript already reports for the final-epoch model) and
     compare it to condition E's own test MSE for that seed. If more than
     3 of the (up to) 10 matched seeds have a test-MSE relative
     difference outside `+/-10%` of condition E's (i.e. the
     validation-MSE match did not carry over to test MSE), the whole
     `MSE_ATTRIBUTION` judgment is `UNRESOLVED` with sub-label
     `VALIDATION_MATCH_DID_NOT_GENERALIZE` — a validation-only match that
     does not generalize to held-out test data is not a meaningful
     MSE-matched comparison, and this is reported distinctly from
     `MSE_MATCH_INFEASIBLE` in the writeup.
  3. **`SUPPORTED`** (checked next): the MSE-matched contrast's
     Holm-adjusted p-value is `<= 0.05` AND the point estimate still
     favors E over the MSE-matched B by at least the SAME 0.2
     minimal-meaningful-effect threshold used everywhere else in this
     protocol (Section 9's primary threshold) — i.e. E's advantage
     survives MSE-matching both statistically and practically.
  4. **`NOT_SUPPORTED`** (the exhaustive remainder — every case not
     covered by 1-3 lands here, not left unclassified): includes the
     MSE-matched B statistically closing the gap (Holm-adjusted p-value
     `<= 0.05` favoring B), a point estimate below the 0.2 threshold even
     if nominally significant, and a CI/point-estimate pattern
     genuinely indistinguishable from zero (CI includes zero) — in every
     one of these cases, lower MSE alone cannot be ruled out as
     explaining part or all of the effect, and the full claim does not
     receive `MSE_ATTRIBUTION` support.

**Manuscript action — corrected in v4 to require FOUR gating judgments,
not three, adding `MSE_ATTRIBUTION`**:

- Step 1 = `SUPPORTED` AND `TARGET_SPECIFICITY` = `SUPPORTED` AND
  `NON_GROUPSORT_REPLICATION` = `SUPPORTED` AND `MSE_ATTRIBUTION` =
  `SUPPORTED`: the full claim (gradient-consistency to the
  Lyapunov-residual target, not information, architecture, or lower MSE
  alone, causes the recovery) stands, scoped to the GroupSort-LCNN with
  the Softsign FNN as a replication data point, never as
  "architecture-independent" in general. `LOSS_INTERACTION` is reported
  alongside as additional characterization (only an `EQUIVALENT` result
  licenses also claiming the effect is loss-family-independent). Proceed
  with the already-agreed Tier-1/2 items in the response-strategy
  document.
- Step 1 = `SUPPORTED` but one or more of `TARGET_SPECIFICITY`,
  `NON_GROUPSORT_REPLICATION`, `MSE_ATTRIBUTION` did not support (i.e.
  is `NOT_SPECIFIC`/`GROUPSORT_SPECIFIC`/`NOT_SUPPORTED`/`UNRESOLVED`):
  the claim narrows to exactly the subset that held (e.g. "the effect is
  real and query-matched, but we cannot rule out that a generic
  action-Jacobian target works equally well," and/or "...demonstrated
  specifically for the GroupSort-LCNN architecture," and/or "...and we
  cannot rule out that lower predictive error alone explains part of the
  gain"), stated explicitly per-judgment in the manuscript, never rounded
  up to the broader claim.
- Step 1 = `UNCLEAR`: **corrected in v3 — "a more query-efficient way to
  use the same information" is itself an unearned positive claim that
  this outcome does not support.** State instead, throughout the
  manuscript (abstract, introduction, discussion, conclusion), that the
  equal-query comparison between `L_grad` and an augmented value-only
  baseline was inconclusive at this sample size, and qualify every claim
  of gradient-consistency as the operative mechanism accordingly, before
  any further Tier-1/2 work proceeds.
- Step 1 = `NEGATIVE`: the manuscript's central causal contribution does
  not survive this test. Stop and discuss fundamental reframing with the
  user before any further drafting — not resolved by a wording change
  alone.

No outcome category, threshold, or manuscript action above may be
selected, adjusted, or reinterpreted after the results are seen; this
section is frozen exactly as written, together with the rest of this
protocol and its implementing code (per Section 0's freeze sequence),
before Section 2's real query-pool construction begins.

## 10. Artifacts, manifest, and no-overwrite convention

Mirrors the convention already established and hash-verified for Paper
M's Data Generation Protocol in this workspace (external freeze manifest,
`canonical_sha256` self-verification, atomic no-overwrite writes, dated
namespace per run). **Restructured in v4**: a single flat Runtime
Manifest conflates artifacts with very different lifetimes (a fixed query
pool that never changes once built, vs. per-seed training results, vs.
the final statistical computation) — mirroring Paper M's OWN convention
of multiple numbered, individually-hash-verified Artifacts (its Artifacts
1-4) is safer here too. This protocol therefore uses a 4-stage manifest
chain, each stage hash-locking its own exact keyset and recording its
immediate predecessor's hash (so a later stage cannot silently run
against a different, unverified earlier stage):

1. **Freeze Manifest** (Section 0.4): hash-locks this document, the
   implementing code (Sections 2-8), and the synthetic-smoke-test result
   — produced once, before any real execution, exactly as already
   specified.
2. **Query Manifest (M1)**: hash-locks the CLEAN_QUERY_TABLE (Section
   2.5), the boundary-drop/valid-pair mask (Section 2.4/2.8), and the
   per-condition query-count assertions (Section 2.9) — built once, seed-
   independent, records the Freeze Manifest's hash as its predecessor.
   No training may read from an unhashed or predecessor-mismatched Query
   Manifest.
3. **Training Manifest (M2) — keyset expanded in v6.** Hash-locks, per
   seed: the NOISE_REALIZED_TABLE (Section 2.5); the 7-item shared
   scaling keyset (Section 2.6: `xm`,`xs`,`ym`,`ys`,`g_scale`,`ug_scale`
   — seed-independent, hash-locked once and reused — and `disp`, hashed
   per seed); the `\lambda_{Jac}` full-grid selection trace (Section 5.3);
   the checkpoint file hashes (Section 8.3); each condition's final
   (epoch-50) and MSE-match-candidate model hashes (including the matched
   candidate's test MSE, Section 8.4); **the RNG stream mapping (new in
   v6, Section 5.5)** — for seed `k`, the three literal generator seeds
   actually used (`k+1000` base noise, `k+4000` sensitivity-arm noise,
   `k+5000` shuffle order) and the resulting per-(condition,seed) initial
   `state_dict()` hash (required identical across conditions A-F at a
   given seed, per Section 5.5). Records M1's hash as its predecessor.
   Resume-or-verify: if a Training Manifest partially exists for a seed,
   re-verify its recorded hashes against the actual files before
   extending it; never regenerate an already-hashed checkpoint to make a
   resume look clean.
4. **Results Manifest (M3) — keyset expanded in v6.** Hash-locks: the raw
   per-seed per-condition metrics (Section 6); the 800-index fixed oracle
   test set (Section 6); the primary and all 7 secondary contrasts'
   bootstrap CIs and permutation p-values (Section 7), **including the
   actual generated bootstrap resample-index arrays themselves (new in
   v6, per Section 7.3's fresh-generator-per-contrast convention)** — not
   merely the resulting CI numbers, so a re-verifier can recompute each
   contrast's CI byte-for-byte from the hashed resample indices; the Holm
   adjustment table; and the final Step-1/Step-2 outcome computation
   including `MSE_ATTRIBUTION` (Section 9) — records M2's hash as its
   predecessor, is the terminal artifact, and is never used as an input
   to any further stage of this protocol.

Every stage is written atomically to a dated results namespace and never
overwritten in place; a provenance mismatch at any stage (a hash that
does not match its predecessor, or a file present that is not in the
manifest that should have produced it) halts execution rather than
proceeding past it, matching the discipline already used for Paper M's
data-generation artifacts.

## Not yet done, listed for the record

**Status as of the 2026-09-01 update to this section (after the sixth
review round)**: this document is still NOT frozen, and no real query
pool has been drawn, no condition has been trained, and no metric has
been computed under this protocol against real data. What has changed
since the paragraph below was originally written (when nothing existed
yet) is that the full implementation and a synthetic/control-flow smoke
suite now exist and have been through SIX independent, adversarial
code-review rounds against the actual code (not just this design
document): `src/cstr/ablation_lib.py` (the computational library:
query-table construction, training loops for all conditions including
the FNN block and the sensitivity arm, the D full-grid selection, the
statistics/outcome-map machinery, and finite/divergence guards),
`src/cstr/ablation_io.py` (independent atomic, self-verifying,
tamper-evident I/O — no-overwrite publish, content hashing, and
partial-publication-safe "verify-if-exists" writers for companion
files), `scripts/ablation_driver_2026_08_31.py` (the 4-stage manifest
chain — Freeze -> Query Manifest M1 -> per-seed Training Manifest M2 ->
Results Manifest M3 — plus the real `run_seed`/`run_full_protocol`/
`run_real_protocol` orchestration, with full resume-or-verify semantics:
reusing an existing M1/M2/M3 re-verifies the actual underlying evidence,
including re-hashing every checkpoint, re-deriving pool/clean/scale
content hashes, re-checking the freeze chain, checking each manifest's
exact `kind` and exact payload keyset, and re-deriving M3's statistics
AND its companion files' content fresh rather than trusting a self-hash
or a byte-hash alone), and `scripts/ablation_synthetic_smoke_2026_08_31.py`
(**167 checks, all passing**, confirmed by a fresh full run after the
sixth round's fixes). Each review round found real bugs in the actual
implementation (not cosmetic issues) and required fixes plus expanded
smoke coverage before the next round; examples include a wrong
residual-value target for augmented rows (silently reusing the base
point's successor instead of the perturbed row's own), train-local vs.
pool-global indexing errors, several late-binding default-argument bugs
that silently ignored smoke-time hyperparameter overrides, a completely
unimplemented resume-or-verify path (any second call to the driver
crashed instead of resuming), a fifth-round simulator-identity split risk
(the `sim` object passed into training/oracle code could silently differ
from the module-level global `budget_sweep_solver.SIM` actually used by
the real `closed_loop()` call, now asserted against by construction), an
M3 reuse path that checked companion-file hashes but not whether M3's
own recorded statistical content matched a fresh recomputation (fixed via
`_assert_aggregated_matches_m3_payload`), an inability to resume after a
crash mid-run before a seed's M2 existed without colliding with that
crashed attempt's already-written checkpoint files (fixed via
per-attempt-numbered checkpoint subdirectories), and — in the sixth,
most recent round — the canonical `run_real_protocol()` entry point
computing real simulator/pool/query-table data BEFORE checking the
freeze gate rather than after (fixed: the freeze check is now the first
statement in the function, before any import or data access, and the
same entry point now also refuses any seed set other than the full
pre-registered 10 seeds, with arbitrary subsets reachable only through
the generic smoke/test-only `run_full_protocol`), each of M1/M2/M3's
exact payload keyset going unchecked despite the document's own "each
stage hash-locks its exact keyset" promise (fixed via `expected_keys` in
`read_and_verify_json_artifact`), and M3's companions (raw-metrics JSON,
oracle-fixed-idx npz, raw-resample-arrays npz) being checked only for
byte-drift against their own recorded hash, never for whether their
content was ever semantically correct (fixed via
`_assert_m3_companions_match_fresh_evidence`, which re-derives all three
fresh from the resumed M2 data / a fresh oracle-index computation / the
same deterministic bootstrap-reseed contract the statistics themselves
rely on). All P0/P1 findings from all six rounds are now fixed and have
dedicated regression coverage in the smoke suite; the sixth review round
found no new defects in the training or statistical logic itself. The
sixth reviewer's own characterization: what remains is narrow
execution/evidence-contract scope, not protocol redesign. Remaining
before freeze, per Section 0's sequence: (1) a further human review round
against this updated implementation and smoke result (the user has not
authorized freeze as of the sixth review); (2) only then the joint
hash-locked Freeze Manifest of document + code + smoke result; (3) only
then the Query Manifest (M1) and any real training. The original
v1-drafting-time paragraph's two open items (Section 2's exact
train/validation/test split ratio and Section 7's exact closed-loop
initial-condition set) were resolved during the v1->v2 review and are
written into Sections 2 and 7; see the correction-history sections below
for the full v1->v7 review trail (9, 7, 18, 8, 1+2, and further v6->v7
issues respectively).

## v1 -> v2 corrections

A review of v1 found 9 concrete issues (7 P0, 2 P1), all independently
re-verified against actual code before being incorporated:

1. **The outcome map's `MECHANISM_SUPPORTED` label was decided by the E-B
   contrast alone**, which cannot support the architecture-independent,
   Lyapunov-specific claim the manuscript wants. Fixed: Section 9 now
   decomposes the decision into `QUERY_MATCHED_EFFECT` (primary, E-B
   only), plus `TARGET_SPECIFICITY` (E vs. D), `LOSS_INTERACTION`
   (interaction contrast), and `ARCHITECTURE_GENERALIZATION` (E-FNN vs.
   B-FNN, gating), with the full claim requiring all three gating
   judgments to support it.
2. **Condition D's `\lambda_{Jac}` tie-break used a validation wrong-sign
   fraction**, which requires a true-gradient oracle at validation points
   that no other condition's selection rule uses — an undisclosed
   information asymmetry favoring D. Fixed: Section 5.3 now selects
   `\lambda_{Jac}` by validation MSE only, ties broken deterministically
   by smallest `\lambda`; the gradient-fidelity oracle moved to Section 6
   as a shared, final-evaluation-only quantity applied identically, post-
   hoc, to every condition.
3. **The MSE-match candidate grid (Section 8) assumed an "existing
   checkpoint cadence" that does not exist.** Confirmed by exhaustive grep
   across `scripts/` and `src/`: no CSTR training script writes model
   weights to disk at any cadence; only in-memory best-validation-state
   tracking exists, and not even that in `lgrad_experiment.py`/`budget_
   sweep_solver.py`. Fixed: Section 8 now specifies a new 10-epoch disk
   checkpoint cadence to implement, applied to every condition.
4. **The freeze-before-implementation order was unsafe** — Paper M's own
   Revision 8/9/10 cycle shows implementation surfaces defects a design
   review alone misses. Fixed: Section 0 now states an explicit design
   review -> implement -> synthetic smoke -> freeze (doc+code+smoke
   jointly) -> real execution sequence; Section 10 corrected to match.
5. **Section 2 proposed drawing a new 20,000-point base pool**, which
   would conflate a data change with the method change under test.
   Confirmed by direct inspection: `data/processed/lcnn_paper_cstr_
   onestep_20k.npz` already exists, already contains `train_idx`/
   `val_idx`/`test_idx` indexing its own arrays, and is exactly what
   `generate_lcnn_paper_cstr_data.py` produces by default. Fixed: Section
   2.1 now hash-locks and reuses this exact file; only the training-split
   perturbation queries are newly generated.
6. **The query/noise contract was incomplete**: the auxiliary-law query
   needed for the residual target was not counted anywhere; no rule fixed
   how Cauchy noise is injected into B/C's augmented targets relative to
   the base pool; no rule handled `u +/- eps*e_j` falling outside the
   input box. Fixed: Section 2.4-2.6 add an explicit boundary-drop rule
   (recorded, never clipped), a common-random-number noise rule keyed on
   `(point, direction, seed)`, and confirms auxiliary-law queries are
   common infrastructure reported in the absolute total but excluded from
   the equal-EXTRA-query claim.
7. **The FNN interaction block's activation was left as "tanh or
   Softsign"** (an open choice) and assumed an existing width-matched
   baseline. Confirmed by direct read of `src/cstr/models_onestep.py`:
   `OneStepFNN` is ReLU-only with no `activation` argument, used elsewhere
   at `hidden=64`, while the GroupSort-LCNN is `hidden=40`; no tanh/
   Softsign FNN variant exists anywhere. Fixed: Section 4 now fixes the
   choice to Softsign specifically (matching the manuscript's existing
   smooth comparator) at width 40, and states this requires a new small
   class, not reuse of `OneStepFNN`.
8. **[P1] No interaction contrast tested whether the gradient term's
   effect depends on loss sidedness.** Fixed: Section 7.4 adds the
   `(F-E)-(C-B)` interaction contrast to the secondary family (now 5
   contrasts, Holm-corrected), feeding `LOSS_INTERACTION` in Section 9.
9. **[P1] The primary decision rule conflated statistical significance
   with practical magnitude, and "materially reversed" was not
   quantified.** Fixed: Section 9's `QUERY_MATCHED_EFFECT` now separates
   the CI-excludes-zero test from the point-estimate >= 0.2 test, defines
   `NEGATIVE` with a concrete -0.1 materiality floor in addition to CI
   reversal, and Section 7.3 adds an exact paired permutation test (1024
   sign patterns over 10 seeds) as a reported cross-check alongside a
   10,000-resample bootstrap (up from the 2,000-resample minimum).

## v2 -> v3 corrections

A second review of v2 found 5 more P0 issues and 2 more P1 issues, all
independently re-verified against actual code before being incorporated:

1. **The noise contract (Section 2.5) described a general per-transition
   noise rule that did not match how noise is actually injected.**
   Confirmed by direct read of `lgrad_experiment.py:60-71`: the existing
   pipeline loads clean `Y` targets and adds Cauchy noise ONLY to
   `train_idx` rows, in-memory, once per `train()` call, via a single
   `rng = np.random.default_rng(seed + 1000)` stream — there was no
   existing "per-transition seed" mechanism to extend. Fixed: Section 2.5
   now defines two immutable tables (a shared, seed-independent
   `CLEAN_QUERY_TABLE` and a per-seed `NOISE_REALIZED_TABLE` built by the
   EXACT existing Cauchy procedure, extended in a fixed draw order to
   B/C's augmented targets), and states explicitly that FD-derived
   gradient labels (D/E/F) are never noise-injected in this protocol,
   matching existing codebase precedent.
2. **Boundary dropping (Section 2.4) allowed dropping a single
   perturbation direction**, which cannot happen without breaking central-
   FD's requirement of both `+eps` and `-eps` per dimension, and would
   have let B/C keep one query that D/E/F could not use. Fixed: dropping
   now happens in `(point, dimension)` pairs, both signs together,
   identically for every condition.
3. **The outcome map's three Step-1 categories were not exhaustive** — a
   CI including zero with a point estimate in `(-0.1, 0]` matched none of
   `SUPPORTED`/`UNCLEAR`/`NEGATIVE`. Fixed: Section 9 now evaluates
   `NEGATIVE` first, then `SUPPORTED`, with `UNCLEAR` as the explicit
   exhaustive remainder. Also fixed the `UNCLEAR` manuscript action's
   "more query-efficient" language, which was itself an unearned positive
   claim not supported by an inconclusive result — replaced with "the
   equal-query comparison was inconclusive."
4. **Holm-Bonferroni correction and the gating judgments were
   disconnected**: `TARGET_SPECIFICITY` and `ARCHITECTURE_GENERALIZATION`
   were decided by a raw "CI excludes zero" read despite being members of
   a Holm-corrected family, which would inflate the family-wise error
   rate exactly the way Holm correction exists to prevent; and a CI
   including zero was treated as evidence FOR independence in
   `LOSS_INTERACTION`, which is an absence-of-evidence/evidence-of-absence
   error. Fixed: Section 7.4/7.5 now specifies that every secondary
   contrast computes both a bootstrap CI and a paired permutation
   p-value, Holm-adjusted as one family; Section 9's gating judgments use
   the Holm-adjusted p-value and direction; `LOSS_INTERACTION` is now a
   three-way `INTERACTING`/`EQUIVALENT`/`INCONCLUSIVE` judgment, with
   `EQUIVALENT` requiring the CI to lie entirely within a pre-registered
   `[-0.1, 0.1]` equivalence margin, not merely a non-significant result.
5. **Checkpoint/model selection was underspecified in three ways**: no
   rule fixed which epoch is each condition's "primary" evaluated model;
   no deterministic tie-break existed for multiple in-band MSE-match
   candidates; and Section 8's purpose statement misattributed the
   existing manuscript's reported MSE as "validation MSE." Confirmed by
   direct read of `lgrad_experiment.py`: the existing `mse` return value
   is computed on `test_idx` (lines 108-111), not `val_idx`; `train()`
   always returns the final-epoch (epoch-50) model with no best-
   validation-checkpoint logic; extra loss terms activate only for
   `ep > warmup` (epoch 31+). Fixed: Section 5.4 now states every
   condition's PRIMARY model is the existing final-epoch convention
   (unchanged by the new checkpoint cadence, which is used only to build
   condition B's MSE-match candidates); Section 8.1 now correctly calls
   the existing reported quantity "test MSE" and clarifies the new
   validation-based selection is a distinct quantity; Section 8.4 adds a
   fully deterministic 3-level tie-break (smallest relative MSE
   difference, then smallest weight decay, then earliest epoch).
6. **[P1] The FNN parameter-count claim was factually wrong.** v2 claimed
   Björck/GroupSort layers "have structurally different parameter counts
   even at identical width/depth" and specified a 10%-tolerance fallback.
   Confirmed wrong by direct construction from `models_onestep.py`: at
   `hidden=40, n_layers=2`, `OneStepLCNN` and an `OneStepFNN`-shaped
   network both have EXACTLY 1922 parameters (`BjorckLinear`/
   `MaxNormLinear` wrap a weight+bias pair of the same shape as
   `nn.Linear`, adding none of their own). Fixed: Section 4 now asserts
   exact parameter-count equality programmatically instead of a tolerance
   band, and separately states the actual, disclosed initialization
   difference (`nn.init.orthogonal_` for `BjorckLinear` vs. PyTorch's
   default `nn.Linear` init for the FNN) rather than an unachievable
   "same initialization family" claim.
7. **[P1] Several internal wording conflicts remained.** Section 1 cited
   a stale `MECHANISM_SUPPORTED` outcome name at the wrong section number
   (fixed: now cites `QUERY_MATCHED_EFFECT` = `SUPPORTED`, Section 9);
   Section 4 claimed the FNN block was "not folded into the primary/
   secondary contrast family in Section 7," directly contradicting
   Section 7.4's inclusion of `E-FNN vs. B-FNN` (fixed: Section 4 now
   states only the descriptive per-condition bookkeeping is separate, the
   contrast itself is a Holm-family member); Section 0 still said "tanh/
   Softsign...FNN variant" after Section 4 fixed the choice to Softsign
   alone (fixed to say "Softsign"); Section 8's purpose statement called
   the existing reported MSE "validation MSE" when it is test MSE (fixed,
   see item 5 above).

## v3 -> v4 corrections

Two review passes on v3 (treated as one round) found 11 more P0 issues
and 7 more P1 issues, all independently re-verified against actual code
or the manuscript before being incorporated:

1. **The primary closed-loop endpoint was completely undefined** — a
   success rate with no fixed horizon, optimizer budget, steps, learning
   rate, warm-start, or projection rule, while the codebase's own
   `budget_sweep_solver.py`/`lgrad_experiment.py` mix `B=3` and `B=20` and
   `lr=0.1`/`0.2` across scripts. Confirmed by direct read of
   `closed_loop()`'s call site (`lgrad_experiment.py:119-121`) and the
   manuscript's own headline framing (`paper_h_jpc.tex` lines 493, 554,
   776: "horizon-3, budget-20"). Fixed: Section 1 now fixes
   `horizon=3, budget=20, steps=120, lr=0.1, rho_u=0.01, success_V=2.0`,
   shift-repeat warm start, and per-iteration box-clamp projection as the
   PRIMARY endpoint, with `budget=3` and the wider budget sweep demoted to
   descriptive/secondary context only.
2. **Noise asymmetry remained a live fairness confound even after v3's
   disclosure**: noisy B/C augmented targets vs. clean D/E/F gradient
   labels meant E-B still compared "gradient term on/off" AND "clean vs.
   noisy label" at once. Fixed: Section 2.5 now makes the PRIMARY B/C
   cells use CLEAN successor states (equal information quality to D/E/F),
   demoting the noisy-augmented version to a separate, explicitly
   non-primary, non-gating sensitivity arm.
3. **No common scaling contract existed** for `xm`/`xs`/`ym`/`ys`, the
   Cauchy `disp`, `g_scale`, and `ug_scale` — recomputing them on B/C's
   5x-larger augmented dataset would silently rescale the loss between
   conditions. Confirmed by direct read of `lgrad_experiment.py:74-78`
   that these are computed once from the base `train_idx` rows only.
   Fixed: Section 2.6 now fixes this and requires the same base-row-only
   constants shared across every condition at a given seed.
4. **The symmetric-loss formula (Section 3) dropped the existing
   `g_scale` normalization**, confirmed wrong against `lgrad_
   experiment.py:96`'s `relu((g_true-g_hat)/g_scale).pow(2)`. Fixed: the
   symmetric variant now removes ONLY the `relu` clamp, keeping
   `/g_scale`, so `lam_val=0.003` means the same thing for C/F as for
   B/E.
5. **No boundary-valid mask or B/C augmented-objective definition
   existed**: which loss terms apply to B/C's augmented points, and how
   D/E/F's loss denominator handles dropped pairs, were both unspecified.
   Fixed: Section 2.8 defines a shared valid-pair mask (denominator for
   D/E/F; inclusion set for B/C) and requires B/C's augmented points
   receive BOTH the Cauchy data-fit loss and the residual-value loss,
   matching how base training points are already treated.
6. **The 2x2 grid was missing the `F vs. C` contrast**, the family's most
   direct test of whether `L_grad` helps under symmetric loss specifically
   (isolated from sidedness) — without it, R3's actual question can only
   be inferred indirectly through the interaction contrast. Fixed: added
   as Section 7.4 item 4.
7. **The secondary family was mislabeled "5" while already enumerating 6
   contrasts**, and item 6 makes it 7. Fixed: Section 7.4/7.5 now state
   and use a family of exactly 7 (E-D, C-B, F-E, F-C, interaction,
   E-FNN-vs-B-FNN, MSE-match), Holm-adjusted against `0.05/7...0.05/1`.
8. **The MSE-matched comparison was computed but never connected to the
   full causal claim** — a positive MSE-matched-B result (B closing the
   gap once MSE-matched) would undermine "gradient consistency causes
   it," but nothing in Section 9 checked for this. Fixed: added
   `MSE_ATTRIBUTION` as a 4th required gating judgment (`SUPPORTED` /
   `NOT_SUPPORTED` / `UNRESOLVED`), now required alongside
   `TARGET_SPECIFICITY` and `NON_GROUPSORT_REPLICATION` for the full
   claim.
9. **MSE-match infeasibility conflicted with the fixed-family Holm
   contract and the `2^10` exact-permutation claim**: excluding seeds
   changes `n`, and a fully infeasible comparison has no p-value to
   Holm-adjust, which would make the family size result-dependent. Fixed:
   Section 7.5 now fixes the family at 7 always, substitutes a
   conservative `p=1` for a fully `MSE_MATCH_INFEASIBLE` contrast (never
   dropping it), and uses `2^n_valid` (not `2^10`) ONLY for that one
   contrast when some seeds are individually excluded.
10. **The gradient-fidelity oracle (Section 6) resampled its 800-point
    test subset per training seed**, confirmed by direct read of
    `probe_action_gradient.py:100-103` (`rng = np.random.default_rng
    (seed)`), conflating training-seed variance with test-point-sampling
    variance across seeds. Fixed: Section 6 now hash-locks ONE fixed
    800-index subset (independent meta-seed) applied to every
    condition/seed, requiring a new `fixed_idx` parameter on `diagnose()`.
11. **No failure/completeness contract existed** for non-finite training,
    simulator exceptions, or missing seeds. Fixed: Section 7.7 now
    requires hard `TRAINING_DIVERGED` recording (never silent retry),
    closed-loop failures recorded as `success=False` (never excluded from
    the rate), and full 10-seed completeness for every primary-battery
    condition before Section 9 may run.
12. **[P1] Selecting `\lambda_{Jac}` by validation MSE alone could in
    principle land on a poor-fidelity value that biases `TARGET_
    SPECIFICITY` toward `SUPPORTED`.** Fixed: Section 5.3 now requires
    reporting the full pre-registered `\lambda_{Jac}` grid's validation
    MSE and resulting E-D outcome for transparency, without changing the
    pre-registered MSE-only selection rule itself.
13. **[P1] Condition D's scope (`\partial f/\partial u` only, not a "full
    Jacobian") needed an explicit reviewer-response sentence.** Fixed:
    Section 3 now states this explicitly, including that `\partial
    f/\partial x` would need `2n` additional queries out of scope here.
14. **[P1] A single alternative architecture (one Softsign FNN) cannot
    support an "architecture-independent" claim.** Fixed: renamed
    `ARCHITECTURE_GENERALIZATION` to `NON_GROUPSORT_REPLICATION` (Section
    9), scoped its `SUPPORTED` manuscript language to "replicates on this
    one non-GroupSort architecture," and strengthened Section 4 to
    require bit-identical initialization between the paired `B-FNN`/
    `E-FNN` run at a given seed.
15. **[P1] The equivalence margin's justification had an arithmetic
    error**: `0.1` is HALF of the primary `0.2` threshold, not
    "one-tenth" as v3 stated. Fixed: Section 9 now justifies `[-0.1,0.1]`
    correctly as half the primary threshold.
16. **[P1] Section 7.4's own description of the interaction contrast
    still had the pre-v3-fix wrong framing** ("CI including zero supports
    ...loss-family-independent"), inconsistent with Section 9's corrected
    3-way `EQUIVALENT`/`INTERACTING`/`INCONCLUSIVE` scheme. Fixed: Section
    7.4 item 5 now says a CI including zero is `NO_DETECTED_INTERACTION`,
    matching Section 9.
17. **[P1] Whether the FNN variant would correctly pass through the
    LCNN-specific inference-materialization path was unresolved.**
    Confirmed by direct read of `materialize_lcnn_for_inference()`
    (`models_onestep.py:242-243`): it already returns any non-
    `OneStepLCNN` model unchanged, so this is safe by construction, not a
    gap. Fixed: Section 4 documents this and adds a synthetic-smoke-test
    check that verifies (not just asserts) output/Jacobian identity for
    the FNN variant.
18. **[P1] A single flat Runtime Manifest conflated artifacts of very
    different lifetimes** (fixed query pool vs. per-seed training vs.
    final statistics). Fixed: Section 10 restructured into a 4-stage
    chain (Freeze Manifest -> Query Manifest M1 -> Training Manifest M2
    -> Results Manifest M3), each recording its predecessor's hash,
    mirroring Paper M's own multi-Artifact convention.

**Separately**: Section 0's Paper M precondition had two sentences in
tension (wall-clock-only gating vs. "should still avoid" heavy parallel
compute). Simplified: implementation and synthetic smoke may proceed
regardless of Paper M's state; every real query/training/closed-loop step
requires Paper M complete or stopped first, with no non-timed-run
carve-out.

## v4 -> v5 corrections

A fourth review found 4 more P0 issues and 4 more P1 issues, all
independently re-verified against actual code before being incorporated.
No new experimental axis was added — every fix closes an execution-
contract gap in the design already agreed at v4:

1. **v4's FNN "freeze/materialize pass-through is already safe" claim was
   FALSE — it checked the wrong function.** Confirmed by direct read:
   the primary evaluation path (`lgrad_experiment.py:118`) calls
   `budget_sweep_solver.py:39`'s `freeze(model)`, which unconditionally
   accesses `model.body`/`model.out.materialized_linear()` — `OneStepFNN`
   has neither, so every FNN-block closed-loop evaluation would crash
   with `AttributeError`. `models_onestep.py`'s `materialize_lcnn_for_
   inference()`, which v4 checked, is a DIFFERENT function not used on
   this path. Fixed: Section 4 now specifies a `freeze_any()` adapter
   (existing `bs.freeze()` for LCNN/SmoothSpectral, identity pass-through
   for the FNN variant) and requires the synthetic smoke test to exercise
   it through the actual `closed_loop()` entry path, not a materialization
   function in isolation.
2. **The FD perturbation `eps=1e-3` coordinate system was ambiguous**,
   readable as a physical `u +/- eps` when the actual code
   (`precompute_true_ugrad`) applies it in NORMALIZED action space then
   transforms back to physical units — a real difference since `u_std`
   varies by orders of magnitude between the CSTR's two inputs, changing
   which points the boundary-valid mask drops. Fixed: Section 2.2/2.4 now
   state the normalized-then-transform convention explicitly and require
   the boundary check on the PHYSICAL transformed action.
3. **The statistical contract left the CI confidence level, bootstrap
   RNG seed, and permutation-test sidedness/statistic unspecified** —
   different reasonable choices could produce different significance
   calls from the same data. Fixed: Section 7.1/7.3/7.5 now fix a 95%
   two-sided percentile-method CI, a single literal bootstrap seed
   (`np.random.default_rng(31415)`) reused for every contrast, a
   two-sided `|mean paired difference|` permutation statistic, and a
   literal oracle meta-seed (`np.random.default_rng(90210)`), uniformly
   across the primary and all 7 secondary contrasts.
4. **`MSE_ATTRIBUTION` (added in v4) was not exhaustively/numerically
   defined**: "indistinguishable from zero" had no number, and a
   validation-matched model's TEST-MSE generalization was never checked.
   Fixed: Section 9 now evaluates `MSE_ATTRIBUTION` in a fixed order
   (`UNRESOLVED` if infeasible, `UNRESOLVED`/`VALIDATION_MATCH_DID_NOT_
   GENERALIZE` if the matched checkpoint's test MSE drifts outside
   +/-10% of E's for more than 3/10 seeds, `SUPPORTED` if Holm-significant
   AND the point estimate is `>= 0.2`, `NOT_SUPPORTED` as the exhaustive
   remainder); Section 8.4 now also records the matched checkpoint's test
   MSE (reporting only, never used for selection).
5. **[P1] Seeds, the oracle meta-seed, and the bootstrap seed were named
   but not fixed to literal values** ("10 seeds," "a separate meta-seed").
   Fixed: Section 7.1 now fixes `SEEDS = [0,...,9]` as an explicit list
   and the two other seeds as the literals in item 3 above.
6. **[P1] Section 2.6's "four shared scale constants" undercounted the
   actual keyset by 3.** Fixed: corrected to the exact 7-item keyset
   (`xm`,`xs`,`ym`,`ys`,`disp`,`g_scale`,`ug_scale`) with names, shapes,
   and dtypes, and Section 10's M2 description updated to match.
7. **[P1] No RNG-stream separation existed between weight initialization,
   base-pool noise, minibatch shuffle order, and the new sensitivity-arm
   noise draw** — adding the sensitivity arm's extra noise draw to the
   existing shared noise/shuffle stream would silently change shuffle
   order only for the conditions that need it, a confound unrelated to
   the scientific manipulation. Fixed: Section 5.5 now specifies three
   independent numpy generators (base noise, sensitivity-arm noise,
   shuffle order) plus a required per-(condition,seed) initial-`state_
   dict` hash check across the GroupSort-LCNN conditions (A-F).
8. **[P1] The strategy document's Verdict section still recommended
   narrowing to "a cheaper way to use the same information"** if the
   go/no-go result is inconclusive — inconsistent with this protocol's
   own v3/v4 fix that `UNCLEAR` supports no positive framing, only
   "inconclusive." Fixed in `JPC_MAJOR_REVISION_RESPONSE_STRATEGY_
   2026_08_31.md`'s Verdict section. (The reviewer's other cited
   instances of old outcome names/`tanh/Softsign` in that document are
   inside its own "v2 -> v3 corrections" historical section, describing
   what was true AT THAT TIME — left unchanged as accurate period record,
   consistent with how this protocol document treats its own history.)

## v5 -> v6 corrections

A fifth review found the last confirmed factual error plus two
manifest/RNG precision gaps — no new experimental axis, purely closing
execution-contract details:

1. **`g_scale` and `ug_scale`'s shape and seed-dependence were both
   wrong in v5.** v5 stated `ug_scale` has shape `(2,)` and that both are
   computed "per seed." Confirmed wrong by direct read of `lgrad_
   experiment.py:73-78`: `.std()` is called with NO `dim` argument on
   both, which flattens ALL elements into a scalar — so `ug_scale` is a
   SCALAR, not `(2,)`. Further, both are computed from the pool's CLEAN
   `Y`/`Yphi_all`/`true_ug` arrays (never the seed's noise-realized `Yn`)
   at the pool's fixed, seed-independent `train_idx` — so both are
   IDENTICAL across every seed, i.e. SEED-INDEPENDENT, computed once and
   reused, exactly like `xm`/`xs`/`ym`/`ys`. Only `disp` (shape `(2,)`)
   is genuinely per-seed, since it alone depends on the seed's
   noise-realized `Yn`. Fixed: Section 2.6 now states the correct
   6-seed-independent + 1-per-seed split, and Section 10's M2 keyset
   updated to match (this also matters for M2's schema/hashing, since a
   scalar and a shape-`(2,)` array hash and broadcast differently).
2. **[P1] Section 5.5's RNG-stream mapping and the per-seed initial-
   state-dict hash were specified as a training-time requirement but
   never added to M2's hash-locked keyset**, so a resumed or re-verified
   Training Manifest could not actually confirm they were honored. Fixed:
   Section 10's M2 now explicitly hash-locks the RNG stream mapping
   (`k+1000`/`k+4000`/`k+5000`) and the per-(condition,seed) initial
   `state_dict` hash.
3. **[P1] The bootstrap RNG contract ("a single fixed seed... reused
   identically for every contrast") was ambiguous** between "one
   stateful generator consumed sequentially across contrasts" (which
   would make later contrasts depend on how many draws earlier ones
   took) and "the same seed value re-instantiated fresh per contrast."
   Fixed: Section 7.1 now specifies a FRESH `np.random.default_rng
   (31415)` instantiated independently for each contrast, and — as a
   second, hash-based safeguard — Section 10's M3 now hash-locks the
   actual generated bootstrap resample-index arrays for every contrast,
   not just the resulting CI numbers.

## v6 -> v7 corrections

Two rounds of code review against the REAL implementation (not this
design document) found 19 total issues. Seventeen were pure
implementation defects with no design implication (fixed in code only:
an augmented-residual-target bug, a missing production orchestration
entry point, M1/M2/M3 recording only derived counts/summaries instead of
the actual underlying data, an unverified freeze-hash parameter, a
freeze-keyset check too weak to catch a stripped required file, an
un-implemented resume-or-verify path, MSE-match fallback checkpoints
missing from the hash closure, a finite-value check that covered only
the loss and not post-optimizer-step parameters, incomplete oracle
metric reporting, a missing per-lambda-candidate contrast computation, a
missing freeze-closure file, and a smoke-gate check that only read
`all_passed` without checking internal consistency). Exactly two findings
required a change to THIS document, both incorporated above:

1. **`ug_scale`'s exclusion of invalid (dropped-pair) placeholder
   entries was implemented but never pre-registered in this document.**
   Section 2.6 stated the reference formula (`.std()` over all of
   `true_ug_t[train_idx]`) without addressing that this protocol's own
   Section 2.4 boundary-drop rule introduces zero-placeholder entries the
   reference formula's original context never had. Fixed: Section 2.6
   now has an explicit addendum stating and justifying the valid-pair-
   only convention as a deliberate, disclosed departure from a literal
   reading of the inherited formula.
2. **Section 5.3's "resulting E-D contrast outcome" for the `\lambda_
   {Jac}` full grid did not specify what statistical form that outcome
   should take**, which meant an implementation reporting only raw
   success rates per candidate (with no CI or p-value) would not
   obviously violate the text. Fixed: Section 5.3 now states explicitly
   that every grid candidate gets the same point-estimate/CI/permutation-
   p-value treatment as every other contrast in this protocol, and that
   these per-candidate contrasts are descriptive only (never Holm-
   corrected, never gating).
