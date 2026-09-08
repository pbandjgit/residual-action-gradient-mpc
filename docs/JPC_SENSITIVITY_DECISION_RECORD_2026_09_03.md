# N-family / N-scale / FD / L-grad sensitivity — Decision Record (2026-09-03)

## 0. Provenance

- Protocol: [`docs/JPC_SENSITIVITY_PROTOCOL_2026_09_03.md`](JPC_SENSITIVITY_PROTOCOL_2026_09_03.md) (v5).
- Freeze Manifest: [`docs/JPC_SENSITIVITY_PROTOCOL_FREEZE_MANIFEST_2026_09_03.json`](JPC_SENSITIVITY_PROTOCOL_FREEZE_MANIFEST_2026_09_03.json),
  self-hash `14bf236e2cf0f7131f052fdbb2c1ac1a51524ff60b27d26229163980d716cb87`
  (19 files via programmatic transitive local-import closure, independently
  `shasum -a 256`-verified for the 4 new files).
- Compatibility reconstruction: `results/sensitivity_compatibility_reconstruction_2026_09_03.json`,
  self-hash `61b7eb1112f992b40e1a4cc108126ecdfca89bba6e3f4a0d9e83803cffc9a917`,
  all 5 checks passed (`val_mse`/`test_mse`/`init_state_hash`/final-checkpoint-
  state-dict-hash/`clean`-rebuild byte-identical), reconstruction discarded
  after verification.
- Setup Manifest + Results Manifest: `results/ablation_sensitivity_2026_09_03/`,
  Results Manifest self-hash `e7c27a465060c3e4d74785cc5af5d29095f7b088d06107bbfc4b1591dc25faaa`
  (independently re-derived from raw payload).
- Contrast aggregation: `results/ablation_sensitivity_2026_09_03/sensitivity_contrast_aggregation_2026_09_03.json`,
  self-hash `65ca65ec4a4a29e9dec8c39ff8b83b1f49dee19b868c3ae8bfac98c17c0f5277`.
- Original `ablation_execution_2026_08_31/` and `solver_audit_execution_2026_09_01/`
  were **not modified** — read-only, hash-verified.

## 1. Execution summary

80/80 evidence models `TRAINED` (0 `TRAINING_DIVERGED`), 80/80
`evaluation_outcome=COMPLETE` (0 `EVALUATION_FAILED`, 0
`INCOMPLETE_ROLLOUT`), 80/80 `init_state_hash` matched the frozen
baseline for the same seed. Evaluator-compatibility gate (B and E's
frozen seed-0 checkpoints, all 5 ICs each, `isclose` on `final_V`/`max_V`,
exact on `settling_time`/`success` against Battery 1) passed. Real wall
clock: 1478.5 s.

Aggregation followed Section 4's fixed hierarchy: seed-level summary
first, then paired bootstrap (`alib.bootstrap_ci`, `BOOTSTRAP_SEED=31415`,
`N_BOOTSTRAP=10000`, fresh RNG per contrast) on `variant - reference`
across the 5 seeds. Reference per axis: N-family/N-scale/L-grad against
the frozen 2026-08-31 baseline; FD against the new common-support
`eps=1e-3` reference (never the frozen baseline, since the frozen E used
its own native, less-restrictive mask).

## 2. FD step (R2.M3, resolves the real `eps=1e-3` vs `eps=0.02` discrepancy)

Closed-loop success = 1.0 on all 5 seeds at **every** tested eps
(`1e-4`, `1e-3`-common-support-reference, `1e-2`, and `0.02` — the exact
value `reviewer_supplement_cstr.py` uses). **The main-text/supplement FD-step
discrepancy does not drive the core closed-loop conclusion.**

**Not all endpoints are eps-invariant, and this must not be overstated.**
At `eps=0.02`, `rel_grad_error_median` differs from the `1e-3`
common-support reference: point estimate `-0.0104`, 95% CI
`[-0.0217, -0.0025]` — the CI excludes zero (improvement direction).
`align_cos_median`/`test_mse_endpoint`/`wrong_sign_fraction`/
`final_V_mean`/`max_V_mean`/`settling_time_mean` all have CIs spanning
zero at this eps. **Correct statement**: closed-loop recovery success is
insensitive to the tested FD step range, including the exact supplement
value; the gradient-fidelity metric is not perfectly invariant, and one
endpoint (relative gradient error) shows a real, CI-excluding-zero
improvement at the coarser supplement step. "All metrics unchanged"
is prohibited language.

## 3. `lambda_grad` (R2.M3): non-monotonic, practical optimum near baseline

`lambda_grad=0.025`: closed-loop success diff = `0.0`, CI `[0,0]`
(indistinguishable from baseline `0.05`). `lambda_grad=0.1`: success
diff = `-0.2`, CI `[-0.6, 0.0]` — driven by a single seed (seed 4) whose
`val_mse` (0.032) is roughly 30-50x every other seed at this level
(others: 0.0006-0.0012) and whose closed-loop success is exactly `0.0`.
**Correct statement**: the gradient-consistency term's benefit is not
monotonically increasing in its weight — a real practical optimum exists
near the baseline value (`0.05`), and doubling it (`0.1`) can destabilize
training for at least one seed on this benchmark. "Larger gradient
penalty is better" is prohibited language; "effective within an
appropriate weight range" is the accurate framing.

## 4. Noise family (R2.m8): closed-loop recovery success is robust; B's weakness is noise-sensitive

E's closed-loop success = 1.0 on all 5 seeds under noiseless, Gaussian
(MAD-matched, sigma=0.29652044), correlated-Gaussian (rho=0.7, common
random numbers with Gaussian), and Cauchy (baseline) — diff=0, CI=[0,0]
in every case. **This directly answers R2.m8's "show the failure isn't
specific to heavy-tailed corruption": E's recovery success does not
depend on the noise family assumption, within the four tested models.**

B is noise-sensitive: **noiseless** significantly improves B (diff
`+0.20`, CI `[0.04, 0.36]`, 5/5 success, `test_mse` collapsing to
~1e-6-4e-5 from the baseline's noisier fit). Gaussian and
correlated-Gaussian (same corruption magnitude, different shape/
correlation) show no significant effect on B (CIs include zero).
**Correct statement**: B's baseline weakness is substantially sensitive
to the presence and magnitude of label corruption, not specifically to
its heavy-tailed shape — supports, but does not by itself fully
establish (this experiment did not vary EVERY possible confound), the
account that noise magnitude, not tail shape, is what matters for B.
"Entirely due to noise" is prohibited language.

## 5. Noise scale (strategy doc Tier 2 item 6): binary success is robust, continuous quality is not

At `scale=0.4` (2x baseline), E's closed-loop success remains exactly
1.0 (diff=0, CI=[0,0]) — but the continuous endpoints degrade with CIs
excluding zero: `final_V_mean` `+0.097` CI `[0.025, 0.200]`,
`settling_time_mean` `+0.36` CI `[0.16, 0.56]`, `test_mse_endpoint`
`+0.000495` CI `[0.000365, 0.000629]`, `rel_grad_error_median` `+0.0558`
CI `[0.0015, 0.110]`. **Correct statement**: within the tested noise
family/scale range, BINARY recovery success is robust, but continuous
solution quality is NOT scale-invariant — larger corruption measurably
degrades settling speed, terminal Lyapunov value, prediction accuracy,
and gradient fidelity even while the pass/fail outcome is unaffected.
"Performance is invariant to noise" is prohibited language; the accurate
claim is scoped to the binary recovery endpoint only.

B at `scale=0.4` degrades significantly (diff `-0.32`, CI
`[-0.56, -0.08]`), with high seed-to-seed variance (raw: 0.2, 0.0, 0.4,
1.0, 0.8) — consistent with, not independently proving, the "noise
magnitude matters for B" account from Section 4.

## 6. Scope discipline

This experiment varied the **label-corruption process** and **FD step
size** and **gradient-loss weight** — the training loss itself
(Cauchy NLL) was held fixed on every model, every axis, every level. It
therefore does **not** establish that Cauchy-family loss is superior to
any alternative loss family; the defensible claim is narrower: *E's
closed-loop recovery advantage is not confined to Cauchy-distributed
label corruption specifically*.

## 7. Statistical framing

Per the protocol's own non-gating status (5-seed, descriptive, no
`SUPPORTED`/`NOT_SUPPORTED` outcome map applied): report CI-excludes-zero
findings as **"the descriptive 95% CI excluded zero,"** never as
"statistically significant" (that phrase implies a pre-registered
hypothesis test with a controlled error rate, which this protocol
explicitly does not claim).

## 8. Suggested response-letter language

> Across the tested finite-difference steps and label-noise families,
> the proposed condition retained perfect recovery success in all five
> seeds. This robustness was not absolute: doubling the Cauchy corruption
> scale degraded several continuous endpoints despite leaving the binary
> success rate unchanged. The results therefore support robustness of the
> recovery conclusion, rather than invariance of all performance
> measures.

## 9. Review-response status

- **R2.M3** (FD-step/noise/gradient-weight sensitivity): **RESOLVED** —
  all three requested sensitivity dimensions real-executed and reported,
  including both favorable (FD-step/noise-family robustness of the
  binary recovery endpoint) and unfavorable (lambda_grad=0.1 dose-response
  failure mode, continuous-endpoint degradation under larger noise scale)
  findings.
- **R2.m8** (justify Cauchy choice; add noiseless/Gaussian/correlated
  cases): **RESOLVED within the tested corruption models** — the
  requested noiseless/Gaussian/correlated-Gaussian conditions were built
  and real-executed; E's recovery-success robustness holds across all
  four, not established as universal across every conceivable noise
  process.

## 10. Next steps

Update `docs/JPC_REVIEW_RESPONSE_MATRIX_2026_09_01.md` (R2.M3, R2.m8 rows
and the ranked-gaps summary). Per the user's direction, the remaining
experimental scope is narrowed to two checks only: R3.4's MPC-necessity
evidence (freshness check of the Sontag comparison against the CURRENT
JPC single-CSTR setup, not reuse of stale EAAI-era numbers) and R3.6's
robustness-mechanism question (two-CSTR cross-check of the MSE/
gradient-fidelity dissociation observed on single-CSTR). No further
large sweeps are planned; the active-temperature-constraint/CLBF study
and non-ceiling FNN benchmark remain correctly deferred/optional.
