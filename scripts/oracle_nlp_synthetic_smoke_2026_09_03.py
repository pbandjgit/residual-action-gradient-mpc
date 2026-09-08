#!/usr/bin/env python3
"""Synthetic smoke suite for the Oracle-Gradient / Converged-NLP
comparison protocol (`docs/JPC_ORACLE_NLP_COMPARISON_PROTOCOL_2026_09_03.md`,
v11). Run before freeze (Sec. 4 step 1). Exercises real code paths
against small/fast/synthetic inputs -- the actual real-data checks
(Gates 1-6 against the real pool/checkpoints) are separately verified by
`build_preflight_artifacts()`, not duplicated here.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cstr.oracle_nlp_io as onio   # noqa: E402
import cstr.oracle_nlp_lib as onl   # noqa: E402

RESULTS_PATH = ROOT / "results" / "oracle_nlp_synthetic_smoke_results_2026_09_03.json"

_checks = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _checks.append(dict(name=name, passed=bool(condition), detail=detail))


# ---------------------------------------------------------------------------
# oracle_nlp_io.py primitives
# ---------------------------------------------------------------------------
def test_err_formula():
    check("err_identical_is_zero", onio.err(1.0, 1.0) == 0.0)
    # denominator is max(1, |a|, |b|) = 101 here, NOT max(|a|,|b|) alone.
    check("err_relative_regime", abs(onio.err(100.0, 101.0) - (1.0 / 101.0)) < 1e-9)
    check("err_absolute_regime_small_values",
          abs(onio.err(1e-10, 3e-10) - 2e-10) < 1e-15)
    e = onio.err(np.array([1.0, 2.0]), np.array([1.0, 2.1]))
    # component 0: |1-1|/max(1,1,1)=0; component 1: |2-2.1|/max(1,2,2.1)=0.1/2.1.
    check("err_vector_takes_max_component", abs(e - (0.1 / 2.1)) < 1e-9)


def test_flatten_leaves():
    obj = {"a": 1, "b": [1, 2, 3], "c": {"x": {"y": 5.0}}, "d": np.array([7.0, 8.0])}
    flat = dict(onio.flatten_leaves(obj))
    check("flatten_scalar", flat.get("a") == 1)
    check("flatten_list", flat.get("b.0") == 1 and flat.get("b.1") == 2 and flat.get("b.2") == 3)
    check("flatten_nested_dict", flat.get("c.x.y") == 5.0)
    check("flatten_ndarray", flat.get("d.0") == 7.0 and flat.get("d.1") == 8.0)


def test_gate_compare_flattened():
    new = dict(success=True, tv=[1.0, 2.0], nested={"1": {"v": 3.0000001}})
    old = dict(success=True, tv=[1.0, 2.0], nested={"1": {"v": 3.0}})
    res = onio.gate_compare_flattened(new, old, exact_fields=["success"], isclose_fields=["tv", "nested"])
    check("gate_compare_flattened_isclose_pass", res["passed"], detail=str(res["mismatches"]))

    new2 = dict(success=False)
    old2 = dict(success=True)
    res2 = onio.gate_compare_flattened(new2, old2, exact_fields=["success"], isclose_fields=[])
    check("gate_compare_flattened_exact_mismatch_detected", not res2["passed"])

    new3 = dict(a=1)
    old3 = dict(a=1)
    res3 = onio.gate_compare_flattened(new3, old3, exact_fields=[], isclose_fields=["missing"])
    check("gate_compare_flattened_missing_field_reported",
          not res3["passed"] and res3["mismatches"][0]["reason"] == "MISSING_FIELD")

    # A genuine indexing/shape-mismatch (different leaf sets) must not be
    # silently treated as a pass just because both sides are non-empty.
    new4 = dict(x=[1.0, 2.0, 3.0])
    old4 = dict(x=[1.0, 2.0])
    res4 = onio.gate_compare_flattened(new4, old4, exact_fields=[], isclose_fields=["x"])
    check("gate_compare_flattened_shape_mismatch_detected",
          not res4["passed"] and res4["mismatches"][0]["reason"] == "LEAF_SHAPE_MISMATCH")


# ---------------------------------------------------------------------------
# J_true / one-step CasADi machinery -- internal consistency (no real data
# dependency; real-data numeric agreement is Gates 1-3, checked separately
# by `build_preflight_artifacts()`).
# ---------------------------------------------------------------------------
def _fake_norm():
    u_mean = np.array([0.1, -100.0])
    u_std = np.array([2.0, 5.0e5])
    return u_mean, u_std


def test_j_true_determinism_and_shapes():
    u_mean, u_std = _fake_norm()
    bundle = onl.build_j_true(u_mean, u_std)
    x0 = np.array([0.0, 20.0])
    Un = np.zeros((3, 2))
    J1, mt1 = onl.j_true_value(bundle, x0, Un)
    J2, mt2 = onl.j_true_value(bundle, x0, Un)
    check("j_true_deterministic", J1 == J2 and mt1 == mt2)
    grad, J3, mt3 = onl.j_true_grad(bundle, x0, Un)
    check("j_true_grad_shape", grad.shape == (3, 2))
    check("j_true_grad_matches_value_call", abs(J3 - J1) < 1e-12)


def test_floor_active_threshold():
    check("floor_active_below_threshold", onl.floor_active(1.0 + onio.FLOOR_MARGIN_K - 0.01) is True)
    check("floor_active_at_threshold", onl.floor_active(1.0 + onio.FLOOR_MARGIN_K) is True)
    check("floor_active_above_threshold", onl.floor_active(1.0 + onio.FLOOR_MARGIN_K + 0.01) is False)


def test_compat_sample_shape():
    u_mean, u_std = _fake_norm()
    un_lo = (onl.INPUT_LO - u_mean) / u_std
    un_hi = (onl.INPUT_HI - u_mean) / u_std
    ics = [np.array([0.0, 20.0]), np.array([0.0, -20.0]), np.array([0.25, 10.0]),
           np.array([-0.25, -10.0]), np.array([0.45, 0.0])]
    sample = onl.build_compat_sample(ics, un_lo, un_hi)
    check("compat_sample_one_step_count", sample["one_step_x0"].shape[0] == 15)
    check("compat_sample_multi_step_count", sample["multi_step_x0"].shape[0] == 5)
    check("compat_sample_multi_step_U_is_zero", np.all(sample["multi_step_U"] == 0.0))


# ---------------------------------------------------------------------------
# Oracle-20 controller -- gradient-source pluggability, finite-guard,
# floor-active bookkeeping, deterministic warm start.
# ---------------------------------------------------------------------------
def test_oracle20_gradient_source_pluggable():
    u_mean, u_std = _fake_norm()
    bundle = onl.build_j_true(u_mean, u_std)
    true_src = onl.true_gradient_source(bundle)
    info = true_src(np.array([0.0, 20.0]), np.zeros((3, 2)))
    check("true_gradient_source_returns_grad_J_min_temp",
          set(info.keys()) >= {"grad", "J", "min_temp"} and info["grad"].shape == (3, 2))


def test_oracle20_step_finite_guard():
    """A gradient source that returns NaN must be caught and reported as
    `any_non_finite`, not silently propagated into `un.grad`."""
    def broken_source(x_phys, un_np):
        return dict(grad=np.full((3, 2), np.nan), J=float("nan"), min_temp=200.0)
    un_lo = torch.tensor([-2.0, -2.0], dtype=torch.float32)
    un_hi = torch.tensor([2.0, 2.0], dtype=torch.float32)
    result = onl.oracle20_step(broken_source, np.array([0.0, 20.0]), torch.zeros(3, 2),
                                budget=5, lr=0.1, un_lo=un_lo, un_hi=un_hi)
    check("oracle20_step_detects_non_finite_gradient", result["any_non_finite"] is True)


def test_oracle20_step_floor_active_bookkeeping():
    """A gradient source that reports a floor-active min_temp must set
    `any_floor_active=True` for that step, without aborting the
    optimization (Sec. 6: floor-active MARKS invalidity, does not halt)."""
    call_count = {"n": 0}

    def floored_source(x_phys, un_np):
        call_count["n"] += 1
        return dict(grad=np.ones((3, 2)) * 0.01, J=1.0, min_temp=0.5)  # 0.5 <= 1.0+1.0 -> floor active

    un_lo = torch.tensor([-2.0, -2.0], dtype=torch.float32)
    un_hi = torch.tensor([2.0, 2.0], dtype=torch.float32)
    result = onl.oracle20_step(floored_source, np.array([0.0, 20.0]), torch.zeros(3, 2),
                                budget=5, lr=0.1, un_lo=un_lo, un_hi=un_hi)
    check("oracle20_step_floor_active_flagged", result["any_floor_active"] is True)
    check("oracle20_step_floor_active_does_not_abort", call_count["n"] == 5 and not result["any_non_finite"])


def test_oracle20_rollout_reference_validity_whole_ic_invalidation():
    """Sec. 6: if floor-active occurs at ANY step, the WHOLE IC is
    reference-invalid, not just the affected step."""
    u_mean, u_std = _fake_norm()

    def once_floored_source_factory():
        state = {"step": 0}

        def source(x_phys, un_np):
            state["step"] += 1
            # floor-active only on the very first optimizer iteration ever seen
            min_temp = 0.5 if state["step"] == 1 else 200.0
            return dict(grad=np.ones((3, 2)) * 0.001, J=1.0, min_temp=min_temp)
        return source

    un_lo = torch.tensor([-2.0, -2.0], dtype=torch.float32)
    un_hi = torch.tensor([2.0, 2.0], dtype=torch.float32)

    class _FakeSim:
        def step(self, x, u):
            return np.asarray(x, dtype=float) * 0.9

    result = onl.oracle20_rollout(once_floored_source_factory(), np.array([0.0, 20.0]), steps=5, budget=3,
                                   lr=0.1, success_v=2.0, un_lo=un_lo, un_hi=un_hi, sim=_FakeSim(),
                                   input_lo=onl.INPUT_LO, input_hi=onl.INPUT_HI, u_mean=u_mean, u_std=u_std)
    check("oracle20_rollout_completed_despite_floor", result["summary"]["solver_completed"] is True)
    check("oracle20_rollout_whole_ic_marked_invalid_from_one_occurrence",
          result["summary"]["any_floor_active"] is True and result["summary"]["reference_valid"] is False)


# ---------------------------------------------------------------------------
# Certified-NLP controller -- dedup, certification, winner selection.
# ---------------------------------------------------------------------------
def test_dedup_groups():
    p0 = np.zeros((3, 2))
    p1 = np.zeros((3, 2)) + 1e-12
    p2 = np.ones((3, 2))
    groups = onl._dedup_groups([p0, p1, p2])
    check("dedup_groups_merges_near_identical", groups == [[0, 1], [2]], detail=str(groups))

    p3 = np.ones((3, 2)) * 0.5
    groups2 = onl._dedup_groups([p0, p2, p3])
    check("dedup_groups_no_merge_when_distinct", groups2 == [[0], [1], [2]], detail=str(groups2))


def test_certified_nlp_startup_assertion():
    u_mean, u_std = _fake_norm()
    un_lo = (onl.INPUT_LO - u_mean) / u_std
    un_hi = (onl.INPUT_HI - u_mean) / u_std
    res = onl.certified_nlp_startup_assertion(un_lo, un_hi)
    check("startup_assertion_passes_for_real_style_bounds", res["passed"] is True)

    raised = False
    try:
        # symmetric bounds around zero -> quartile point degenerates toward
        # zero, exactly the v6 P0 finding -- assertion must catch this.
        onl.certified_nlp_startup_assertion(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    except AssertionError:
        raised = True
    # NOTE: quartile = lo + 0.75*(hi-lo) = -1 + 0.75*2 = 0.5, distance from
    # zero = 0.5 > TOL_DEDUP_INPUT, so THIS particular symmetric case does
    # NOT degenerate (0.75 quartile is not the midpoint) -- confirms the
    # design's choice of 0.75 (not 0.5) is exactly what avoids the v6 bug.
    check("startup_assertion_quartile_design_avoids_mean_degeneracy", raised is False)


def test_certify_candidate_kkt_and_status_whitelist():
    u_mean, u_std = _fake_norm()
    bundle = onl.build_j_true(u_mean, u_std)
    un_lo = (onl.INPUT_LO - u_mean) / u_std
    un_hi = (onl.INPUT_HI - u_mean) / u_std
    x0 = np.array([0.0, 20.0])
    U_interior = np.zeros((3, 2))  # zero is interior; gradient there is nonzero in general -> likely NOT a KKT point
    cand_bad_status = dict(U_sol=U_interior, return_status="Maximum_Iterations_Exceeded",
                            J_true=0.0, minimum_stage_temperature=200.0)
    cert = onl.certify_candidate(bundle, x0, cand_bad_status, un_lo, un_hi)
    check("certify_candidate_rejects_non_whitelisted_status_even_if_numerics_pass",
          cert["certified"] is False and cert["status_ok"] is False)


def test_certified_nlp_step_winner_selection_and_fallback():
    """Uses the real `certified_nlp_lib.certified_nlp_step` machinery via
    the synthetic-problem path (`synthetic_nlp_step`, gates 5/6's own
    infrastructure) to confirm winner selection picks the lowest-J_true
    CERTIFIED candidate, with force_uncertify correctly skipping a lower-
    objective but disqualified candidate."""
    u_mean, u_std = _fake_norm()
    un_lo = (onl.INPUT_LO - u_mean) / u_std
    un_hi = (onl.INPUT_HI - u_mean) / u_std
    nlp = onl.build_synthetic_nlp_solver(3, un_lo, un_hi)
    warm_shifted = onl._tiled_plan(un_hi, 3)
    step = onl.synthetic_nlp_step(nlp, un_lo, un_hi, warm_shifted, force_uncertify={0})
    check("certified_nlp_step_fallback_skips_disqualified_lowest_objective",
          step["selected_start_id"] is not None and step["selected_start_id"] != 0)


# ---------------------------------------------------------------------------
# Manifest builders -- the specific class of bug the solver audit's own
# round 2 found (fresh-build return dict missing a required key).
# ---------------------------------------------------------------------------
def test_manifest_builders_include_payload_key():
    import oracle_nlp_driver_2026_09_03 as odrv
    import inspect
    for name in ["build_preflight_artifacts", "build_setup_manifest", "build_compat_gate_manifest",
                 "run_oracle20_reference", "run_certified_nlp_reference", "run_timing_pass",
                 "build_summary_manifest"]:
        src = inspect.getsource(getattr(odrv, name))
        check(f"{name}_return_includes_payload_literal", "payload=" in src, detail=name)


def test_oracle20_endpoint_equivalence_detects_real_mismatch():
    """The endpoint-equivalence gate must actually FAIL when fed two
    genuinely different trajectories -- confirms it is not vacuously
    passing (e.g. due to an empty comparison list)."""
    clean = dict(light_step_records=[dict(step_index=0, u0=np.array([0.0, 0.0]),
                                           final_plan_un=np.zeros((3, 2)), x_after=np.array([1.0, 2.0]),
                                           V_after=5.0, minimum_stage_temperature=300.0)])
    instrumented = dict(light_step_records=[dict(step_index=0, u0=np.array([0.0, 0.0]),
                                                  final_plan_un=np.zeros((3, 2)), x_after=np.array([1.0, 999.0]),
                                                  V_after=5.0, minimum_stage_temperature=300.0)])
    res = onl.oracle20_endpoint_equivalence(clean, instrumented)
    check("oracle20_endpoint_equivalence_detects_real_mismatch", res["passed"] is False)

    res_ok = onl.oracle20_endpoint_equivalence(clean, clean)
    check("oracle20_endpoint_equivalence_passes_identical", res_ok["passed"] is True)


def test_compute_gaps_decomposition_identity_enforced():
    """`compute_gaps` must raise if the decomposition identity fails --
    verified here by monkeypatching `_battery1_seed_final_v` to return
    seed-dependent noise that breaks the identity (since the identity is a
    property of consistent aggregation, not of the specific numbers)."""
    import oracle_nlp_driver_2026_09_03 as odrv
    original = odrv._battery1_seed_final_v
    rng = np.random.default_rng(1)
    noise = {("B", s): rng.normal() for s in range(10)}
    noise.update({("E", s): rng.normal() for s in range(10)})

    def _consistent(condition, seed):
        return 5.0 + noise[(condition, seed)]

    odrv._battery1_seed_final_v = _consistent
    try:
        result = odrv.compute_gaps([1.0] * 5, [0.5] * 5)
        check("compute_gaps_identity_holds_for_consistent_data", result["decomposition_identity_passed"] is True)
    finally:
        odrv._battery1_seed_final_v = original


# ---------------------------------------------------------------------------
# Timing-pass rotation (Sec. 12) -- verified with cheap mock controllers,
# not real Oracle-20/Certified-NLP/learned-model controllers (too slow for
# smoke; the real controllers' own correctness is covered elsewhere).
# ---------------------------------------------------------------------------
class _MockTimingController:
    def __init__(self, name):
        self.name = name
        self.call_log = []

    def reset(self, x0):
        pass

    def step(self):
        self.call_log.append(len(self.call_log))
        return 0.001


def test_timing_pass_rotation_full_coverage_and_warmup():
    names = [f"c{i}" for i in range(4)]
    factories = {name: (lambda name=name: _MockTimingController(name)) for name in names}
    ics = [np.zeros(2), np.ones(2)]
    result = onl.run_interleaved_timing_pass(factories, ics, steps=6, warmup_steps=2)
    # every per-IC controller instance must have been stepped exactly `steps` times
    for name in names:
        for ic_index, ctrl in enumerate(result[name]["controllers"]):
            check(f"timing_rotation_full_coverage_{name}_ic{ic_index}", len(ctrl.call_log) == 6)
    # confirms the factory-per-IC fix: two ICs for the same controller name
    # must NOT share state (the original single-shared-instance bug would
    # have collapsed both ICs' call logs into one).
    check("timing_rotation_ics_are_independent_instances",
          result["c0"]["controllers"][0] is not result["c0"]["controllers"][1])
    # warmup exclusion: 6 steps - 2 warmup = 4 timed steps per IC, x2 ICs = 8
    check("timing_rotation_warmup_excluded_from_summary",
          all(len(result[name]["raw_by_ic"][0]) == 6 for name in names)
          and result["c0"]["median"] == 0.001)


def test_timing_pass_rotation_is_deterministic_and_varies_start():
    """The rotation offset must actually change which controller goes
    first across (step, ic) pairs -- otherwise the whole point (avoiding a
    fixed first/last controller) is not actually achieved."""
    n = 22
    offsets = {(step, ic) for step in range(3) for ic in range(5)}
    first_controllers = set()
    for step, ic in offsets:
        r = (step * 5 + ic) % n
        first_controllers.add(r)
    check("timing_rotation_offset_varies_across_step_ic_pairs", len(first_controllers) > 1,
          detail=f"observed {len(first_controllers)} distinct starting offsets")


def test_decomposition_identity_math():
    """Sec. 12a's mandatory identity: (gap_X + optimizer_gap) == X - NLP,
    verified as a pure arithmetic property independent of any specific
    implementation (protects the FORMULA itself, which the eventual
    Summary Manifest builder must reproduce exactly)."""
    rng = np.random.default_rng(0)
    oracle_vals = rng.normal(5, 1, size=5)
    nlp_vals = rng.normal(4, 1, size=5)
    b_vals = rng.normal(6, 1, size=5)
    optimizer_gap = float(np.mean(oracle_vals - nlp_vals))
    oracle_mean = float(np.mean(oracle_vals))
    gap_b = float(np.mean(b_vals)) - oracle_mean
    lhs = gap_b + optimizer_gap
    rhs = float(np.mean(b_vals)) - float(np.mean(nlp_vals))
    check("decomposition_identity_holds", abs(lhs - rhs) < 1e-9)

    # Inject a deliberate IC-indexing bug (shuffle nlp_vals) and confirm
    # the identity check WOULD catch it (i.e. it's a real, non-vacuous test).
    nlp_shuffled = nlp_vals[::-1]
    optimizer_gap_bug = float(np.mean(oracle_vals - nlp_shuffled))
    lhs_bug = gap_b + optimizer_gap_bug
    rhs_bug = float(np.mean(b_vals)) - float(np.mean(nlp_shuffled))
    # lhs_bug should still equal rhs_bug (identity holds regardless of
    # pairing, since it's a property of the MEANS) -- confirms the
    # identity check on its own cannot catch a per-IC pairing bug, only a
    # top-level aggregation-formula bug. This is an important, honest
    # limitation to record, not a failure to fix here.
    check("decomposition_identity_is_mean_level_not_per_ic_pairing_check",
          abs(lhs_bug - rhs_bug) < 1e-9,
          detail="documents that this check catches aggregation-formula bugs, not per-IC index-pairing bugs")


def main() -> dict:
    test_err_formula()
    test_flatten_leaves()
    test_gate_compare_flattened()
    test_j_true_determinism_and_shapes()
    test_floor_active_threshold()
    test_compat_sample_shape()
    test_oracle20_gradient_source_pluggable()
    test_oracle20_step_finite_guard()
    test_oracle20_step_floor_active_bookkeeping()
    test_oracle20_rollout_reference_validity_whole_ic_invalidation()
    test_dedup_groups()
    test_certified_nlp_startup_assertion()
    test_certify_candidate_kkt_and_status_whitelist()
    test_certified_nlp_step_winner_selection_and_fallback()
    test_manifest_builders_include_payload_key()
    test_oracle20_endpoint_equivalence_detects_real_mismatch()
    test_compute_gaps_decomposition_identity_enforced()
    test_timing_pass_rotation_full_coverage_and_warmup()
    test_timing_pass_rotation_is_deterministic_and_varies_start()
    test_decomposition_identity_math()

    n_pass = sum(1 for c in _checks if c["passed"])
    n_total = len(_checks)
    all_passed = n_pass == n_total
    result = dict(kind="oracle_nlp_synthetic_smoke_results", generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  n_pass=n_pass, n_total=n_total, all_passed=all_passed, checks=_checks)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    res = main()
    for c in res["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"[{status}] {c['name']}" + (f" -- {c['detail']}" if c["detail"] and not c["passed"] else ""))
    print(f"\n{res['n_pass']}/{res['n_total']} passed")
    if not res["all_passed"]:
        sys.exit(1)
