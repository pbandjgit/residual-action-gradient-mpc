#!/usr/bin/env python3
"""Synthetic (+ limited real-recipe) smoke test for the single-CSTR
query-matched solver audit + two-plant state-constraint minimum audit
(`src/cstr/solver_audit_lib.py`, `src/cstr/solver_audit_io.py`,
`scripts/solver_audit_driver_2026_09_01.py`), per
`docs/JPC_SOLVER_AUDIT_PROTOCOL_2026_09_01.md` v7, Section 0.

Mirrors `ablation_synthetic_smoke_2026_08_31.py`'s own conventions
(`check()`/`CHECKS`, `FakeSim`, monkeypatching `budget_sweep_solver.SIM`,
tiny horizon/budget/steps) wherever a check needs a cheap fabricated
plant. Two exceptions, both deliberate: (1) the state-constraint-
statistics check uses the REAL single-CSTR simulator directly (cheap --
pure numpy/scipy RK4, no data files, no CasADi -- needed for
`to_absolute_state()`/`xs_abs`/`dt_hr` fidelity); (2) the two-CSTR Gate 4
check runs ONE real reconstruction (seed 0, `value_only`) against
`two_cstr_lgrad.py`'s actual training recipe and compares it to the
ALREADY-LOGGED `results/interim/logs/two_cstr_lgrad.json` values -- the
single highest-risk untested path (a reimplementation-fidelity bug here
would only otherwise surface after a much more expensive real Battery 5
run), accepted as the one genuinely non-trivial-cost smoke check.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cstr.ablation_lib as alib                       # noqa: E402
import cstr.solver_audit_io as sio                      # noqa: E402
import cstr.solver_audit_lib as slib                    # noqa: E402
import solver_audit_driver_2026_09_01 as drv             # noqa: E402
import ablation_synthetic_smoke_2026_08_31 as asmoke     # noqa: E402

RESULTS_DIR = ROOT / "results"
SCRATCH = RESULTS_DIR / "_solver_audit_smoke_scratch_2026_09_01"

CHECKS = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append({"name": name, "passed": bool(condition), "detail": detail})
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))


def fresh_scratch() -> Path:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    return SCRATCH


# ---------------------------------------------------------------------------
# Section A: gate / tolerance-contract unit tests
# ---------------------------------------------------------------------------
def test_gate_compare_dict_tolerance_contract():
    new = dict(a=True, b=3, x=1.0000001, y=5.0)
    old_close = dict(a=True, b=3, x=1.0000002, y=5.0)
    r = sio.gate_compare_dict(new, old_close, exact_keys=("a", "b"), isclose_keys=("x", "y"))
    check("gate_compare_dict passes when float fields are within isclose tolerance", r["passed"], str(r))

    old_far = dict(a=True, b=3, x=1.1, y=5.0)
    r2 = sio.gate_compare_dict(new, old_far, exact_keys=("a", "b"), isclose_keys=("x", "y"))
    check("gate_compare_dict fails when a float field exceeds isclose tolerance", not r2["passed"])

    old_exact_mismatch = dict(a=True, b=4, x=1.0000001, y=5.0)
    r3 = sio.gate_compare_dict(new, old_exact_mismatch, exact_keys=("a", "b"), isclose_keys=("x", "y"))
    check("gate_compare_dict fails on ANY exact-field mismatch, even integer-close", not r3["passed"])


def test_battery1_sanity_field_map_and_mismatch():
    instrumented = dict(success=True, lyapunov_increase_count=2, final_V=1.5, max_V=3.0, tv=0.2)
    original = dict(success=True, violations=2, final_V=1.5, max_V=3.0, tv=0.2)
    r = slib.battery1_sanity_compare(instrumented, original)
    check("battery1_sanity_compare correctly maps violations <-> lyapunov_increase_count and passes "
          "on matching values", r["passed"], str(r))

    original_diff_success = dict(original); original_diff_success["success"] = False
    r2 = slib.battery1_sanity_compare(instrumented, original_diff_success)
    check("battery1_sanity_compare fails when success differs", not r2["passed"])


# ---------------------------------------------------------------------------
# Section B: FD boundary-handling contract (Section 3.5), vs. a smooth
# synthetic map with a KNOWN analytic gradient (not the real CSTR).
# ---------------------------------------------------------------------------
class _QuadraticPlant:
    """`step(x, u) = x + A @ u + 0.5 * diag(c) @ (u**2)` -- smooth, and
    `V(step(x,u)) = step(x,u)^T P step(x,u)` has a hand-differentiable
    gradient w.r.t. `u`, so the FD helper's accuracy can be checked
    directly against an analytic reference."""

    def __init__(self):
        self.A = np.array([[1.3, 0.2], [0.1, 0.9]])
        self.c = np.array([0.05, 0.03])

    def step(self, x, u):
        x = np.asarray(x, float); u = np.asarray(u, float)
        return x + self.A @ u + 0.5 * self.c * (u ** 2)


def _analytic_v_grad(plant: "_QuadraticPlant", x: np.ndarray, u: np.ndarray, u_mean, u_std) -> np.ndarray:
    """d/d(u_n) [V(step(x,u))], u = u_n*u_std + u_mean, via the chain rule
    (P from `slib.P`)."""
    y = plant.step(x, u)
    dydu = plant.A + np.diag(plant.c * u)          # (2,2) d y / d u (physical)
    dydun = dydu * u_std[None, :]                   # chain rule through u = un*u_std + u_mean
    dVdy = 2.0 * (slib.P @ y)
    return dVdy @ dydun


def test_fd_boundary_handling_contract():
    plant = _QuadraticPlant()
    x = np.array([0.3, -0.2])
    u_mean = np.array([0.0, 0.0]); u_std = np.array([1.0, 1.0])

    def v_of_un_factory(un0):
        def v_fn(un):
            u_phys = un * u_std + u_mean
            y = plant.step(x, u_phys)
            return float(y @ slib.P @ y)
        return v_fn

    # (1) interior point -> central FD, close to analytic.
    u_n0 = np.array([0.0, 0.0])
    v_fn = v_of_un_factory(u_n0)
    grad, diag = slib.true_v_gradient_with_boundary_handling(v_fn, u_n0, np.array([-1.0, -1.0]),
                                                              np.array([1.0, 1.0]))
    analytic = _analytic_v_grad(plant, x, u_n0 * u_std + u_mean, u_mean, u_std)
    check("FD interior point uses central formula for both coords",
          all(d["kind"] == "central" for d in diag), str(diag))
    check("central FD matches the analytic gradient to 1e-4",
          np.allclose(grad, analytic, atol=1e-4), f"grad={grad} analytic={analytic}")

    # (2) near the UPPER edge -> one-sided-forward (toward interior), still accurate.
    eps = slib.FD_EPS
    u_n0_hi = np.array([1.0 - 0.5 * eps, 0.0])
    v_fn_hi = v_of_un_factory(u_n0_hi)
    grad_hi, diag_hi = slib.true_v_gradient_with_boundary_handling(v_fn_hi, u_n0_hi, np.array([-1.0, -1.0]),
                                                                    np.array([1.0, 1.0]))
    analytic_hi = _analytic_v_grad(plant, x, u_n0_hi * u_std + u_mean, u_mean, u_std)
    # Near the UPPER edge, the only available interior clearance is in the
    # DECREASING direction -- i.e. the "backward" one-sided formula (which
    # steps toward -eps/-2eps), not "forward".
    check("FD near the upper box edge switches coord 0 to one_sided_backward (interior = decreasing)",
          diag_hi[0]["kind"] == "one_sided_backward", str(diag_hi))
    check("one-sided FD near the upper edge matches the analytic gradient to 1e-3",
          np.allclose(grad_hi, analytic_hi, atol=1e-3), f"grad={grad_hi} analytic={analytic_hi}")

    # (3) near the LOWER edge -> one-sided-forward (interior = increasing).
    u_n0_lo = np.array([-1.0 + 0.5 * eps, 0.0])
    v_fn_lo = v_of_un_factory(u_n0_lo)
    grad_lo, diag_lo = slib.true_v_gradient_with_boundary_handling(v_fn_lo, u_n0_lo, np.array([-1.0, -1.0]),
                                                                    np.array([1.0, 1.0]))
    analytic_lo = _analytic_v_grad(plant, x, u_n0_lo * u_std + u_mean, u_mean, u_std)
    check("FD near the lower box edge switches coord 0 to one_sided_forward (interior = increasing)",
          diag_lo[0]["kind"] == "one_sided_forward", str(diag_lo))
    check("one-sided-backward FD matches the analytic gradient to 1e-3",
          np.allclose(grad_lo, analytic_lo, atol=1e-3), f"grad={grad_lo} analytic={analytic_lo}")

    # (4) box smaller than 2*eps -> excluded, reason fd_clearance_unavailable.
    tiny_lo = np.array([-0.5 * eps, -1.0]); tiny_hi = np.array([0.5 * eps, 1.0])
    u_n0_tiny = np.array([0.0, 0.0])
    v_fn_tiny = v_of_un_factory(u_n0_tiny)
    grad_tiny, diag_tiny = slib.true_v_gradient_with_boundary_handling(v_fn_tiny, u_n0_tiny, tiny_lo, tiny_hi)
    check("FD excludes a coordinate with no 2*eps clearance on either side, reason fd_clearance_unavailable",
          diag_tiny[0]["kind"] == "excluded" and diag_tiny[0]["reason"] == "fd_clearance_unavailable"
          and np.isnan(grad_tiny[0]), str(diag_tiny))
    check("FD never silently downgrades an excluded coordinate to a lower-order estimate (stays NaN)",
          np.isnan(grad_tiny[0]))


# ---------------------------------------------------------------------------
# Section C: settling time / state-constraint statistics
# ---------------------------------------------------------------------------
def test_settling_time():
    check("settling_time = 0 when V(x0) already <= success_V and stays there",
          slib.compute_settling_time([1.0, 0.5, 0.3], 2.0, 3) == 0)
    check("settling_time = steps+1 sentinel when V never drops below success_V",
          slib.compute_settling_time([5.0, 4.0, 3.5], 2.0, 3) == 4)
    check("settling_time = steps+1 sentinel when V drops below then EXCEEDS again later",
          slib.compute_settling_time([5.0, 1.0, 6.0], 2.0, 3) == 4)
    check("settling_time = first permanent-below index when V settles mid-rollout",
          slib.compute_settling_time([5.0, 4.0, 1.0, 1.0, 1.0], 2.0, 4) == 2)


class _FakeSingleCSTRSimForStats:
    """Minimal object exposing exactly what `state_constraint_stats` needs
    (`xs_abs`, `dt_hr`) without constructing the real ODE simulator, for a
    hand-computable state-constraint check."""

    dt_hr = 1e-3
    xs_abs = np.array([1.95, 402.0])


def test_state_constraint_stats_hand_computed():
    sim = _FakeSingleCSTRSimForStats()
    # x_traj: 3 points, temperature deviation (index 1) = [10, 80, 40] -> cap=70.
    x_traj = [np.array([0.0, 10.0]), np.array([0.0, 80.0]), np.array([0.0, 40.0])]
    us = [np.array([1.0, 100.0]), np.array([3.5, 100.0])]  # 2nd actuator never saturates
    input_lo = np.array([-3.5, -5.0e5]); input_hi = np.array([3.5, 5.0e5])
    stats = slib.state_constraint_stats(x_traj, us, sim, input_lo, input_hi, cap_deviation_K=70.0)
    idx1 = stats["per_temperature_index"]["1"]  # string key (JSON round-trip safe, Section 3.6)
    check("max_temperature_deviation_K hand-computed correctly", idx1["max_temperature_deviation_K"] == 80.0)
    check("max_temperature_absolute_K = deviation + xs_abs[1]", idx1["max_temperature_absolute_K"] == 482.0)
    check("cap_absolute_K = cap_deviation_K + xs_abs[1]", idx1["cap_absolute_K"] == 472.0)
    # frequency/integral denominator is the 2 POST-STEP samples (x_traj[1:]
    # = [80, 40]), t=0's value (10) excluded -- 1 of those 2 exceeds cap.
    check("temperature_violation_frequency counts exactly 1 of 2 POST-STEP points > cap "
          "(t=0 excluded from this denominator)",
          abs(idx1["temperature_violation_frequency"] - 0.5) < 1e-12)
    expected_integral = sim.dt_hr * max(0.0, 80.0 - 70.0)
    check("temperature_violation_integral hand-computed correctly",
          abs(idx1["temperature_violation_integral"] - expected_integral) < 1e-12)
    check("input_saturation_frequency detects actuator-0 at the upper bound in exactly 1/2 steps",
          abs(stats["input_saturation_frequency"]["0"] - 0.5) < 1e-12)
    check("input_saturation_frequency reports 0 for an actuator that never saturates",
          stats["input_saturation_frequency"]["1"] == 0.0)
    check("temperature_cap_satisfied_posthoc is False when the cap was exceeded",
          stats["temperature_cap_satisfied_posthoc"] is False)
    # `us` here = [[1.0, 100.0], [3.5, 100.0]] -- actuator-0's SECOND value
    # (3.5) sits EXACTLY at input_hi[0]=3.5. A boundary value is still
    # box-FEASIBLE (the constraint is `<=`, not `<`) -- `input_box_
    # satisfied` must be True here, while `input_never_saturated` (the
    # SEPARATE "was it ever pinned at the bound" diagnostic, which the
    # earlier, incorrect version of this code called "input_box_
    # satisfied") must be False, since actuator-0 WAS at bound once.
    check("input_box_satisfied is True for a value sitting EXACTLY at the boundary "
          "(a boundary value is legitimately box-feasible, not a violation)",
          stats["input_box_satisfied"] is True, str(stats["input_box_satisfied"]))
    check("input_never_saturated is False when an actuator was ever at its bound "
          "(the separate diagnostic the earlier code incorrectly called 'input_box_satisfied')",
          stats["input_never_saturated"] is False)

    us_never_saturated = [np.array([1.0, 100.0]), np.array([2.0, 100.0])]
    stats_never_sat = slib.state_constraint_stats(x_traj, us_never_saturated, sim, input_lo, input_hi,
                                                   cap_deviation_K=70.0)
    check("input_box_satisfied is True when every actuator stays strictly within the box",
          stats_never_sat["input_box_satisfied"] is True)
    check("input_never_saturated is True when NO actuator was ever at its bound for the whole rollout",
          stats_never_sat["input_never_saturated"] is True)

    # Genuinely OUT-OF-RANGE input (bypassing the normal clip-before-call
    # convention deliberately, to test `state_constraint_stats` itself in
    # isolation): `input_box_satisfied` must be False, and -- since this
    # value is nowhere NEAR either bound -- `input_never_saturated` would
    # be True, which is exactly why the two fields must never be conflated.
    us_out_of_range = [np.array([1.0, 100.0]), np.array([10.0, 100.0])]  # 10.0 > input_hi[0]=3.5
    stats_oob = slib.state_constraint_stats(x_traj, us_out_of_range, sim, input_lo, input_hi,
                                            cap_deviation_K=70.0)
    check("input_box_satisfied is False for a genuinely out-of-range input, even though it is "
          "nowhere near the boundary (would have been wrongly True under the old, incorrect "
          "saturation-based definition)", stats_oob["input_box_satisfied"] is False, str(stats_oob))

    # NaN input -> genuinely FAILS box feasibility (NaN satisfies no
    # inequality) -- `False`, not a silently-passing `True`. `None` is
    # reserved for the "no applied actions at all" case (tested next),
    # which is a different situation (no data to evaluate) from "data
    # exists and it is invalid."
    us_nan = [np.array([float("nan"), 100.0]), np.array([1.0, 100.0])]
    stats_nan = slib.state_constraint_stats(x_traj, us_nan, sim, input_lo, input_hi, cap_deviation_K=70.0)
    check("input_box_satisfied is False (not silently True) when an applied action is NaN",
          stats_nan["input_box_satisfied"] is False, str(stats_nan["input_box_satisfied"]))

    stats_empty = slib.state_constraint_stats(x_traj, [], sim, input_lo, input_hi, cap_deviation_K=70.0)
    check("input_box_satisfied is None (undefined) when there are no applied actions at all",
          stats_empty["input_box_satisfied"] is None)


# ---------------------------------------------------------------------------
# Section D: counterfactual-true-objective formula (Section 6 item 7)
# ---------------------------------------------------------------------------
def test_counterfactual_true_objective_matches_manual():
    plant = _QuadraticPlant()
    xm = torch.tensor([0.0, 0.0, 0.0, 0.0]); xs = torch.tensor([1.0, 1.0, 1.0, 1.0])
    ym = torch.tensor([0.0, 0.0]); ys = torch.tensor([1.0, 1.0])
    x0 = np.array([0.2, -0.1])
    horizon, rho_u = 2, 0.01
    U = np.array([[0.1, -0.2], [0.05, 0.15]])
    got = slib.counterfactual_true_objective(plant, x0, U, horizon, rho_u, xm, xs, ym, ys)

    xt = x0.copy(); expected = 0.0
    for h in range(horizon):
        y = plant.step(xt, U[h])
        expected += float(y @ slib.P @ y) + rho_u * float((U[h] ** 2).sum())
        xt = y
    check("counterfactual_true_objective matches a manual step-by-step recomputation",
          abs(got - expected) < 1e-9, f"got={got} expected={expected}")


# ---------------------------------------------------------------------------
# Section E: Battery 1/2 rollout vs. the FROZEN closed_loop() (Gate 1/
# Battery-1-sanity), using a monkeypatched FakeSim.
# ---------------------------------------------------------------------------
def test_single_cstr_rollout_matches_frozen_closed_loop():
    import budget_sweep_solver as bs
    from disambiguate_landscape_vs_gradient import make_norm
    old_sim = bs.SIM
    try:
        bs.SIM = asmoke.FakeSim()
        # `single_cstr_rollout`'s summary needs `sim.dt_hr` (Section 3.2's
        # `settling_time_hours`) -- the real simulator always has it;
        # `FakeSim` (from the frozen ablation smoke suite, never edited
        # here) does not, so it is set on this INSTANCE only.
        bs.SIM.dt_hr = 1e-3
        pool = asmoke.make_fake_pool(seed=21)
        torch.manual_seed(5)
        model = alib.build_model("E", use_fnn=False)
        seq = alib.freeze_any(model)
        data_tuple = (pool["XU"], pool["Y"], pool["xm"], pool["xs"], pool["ym"], pool["ys"],
                      pool["train_idx"], pool["test_idx"])
        norm, un_lo, un_hi = make_norm(data_tuple)
        x0 = np.array([0.1, 5.0])
        horizon, budget, steps, lr, rho_u, success_V = 2, 3, 4, 0.1, 0.01, 2.0
        input_lo, input_hi = bs.INPUT_LO, bs.INPUT_HI

        original = bs.closed_loop(seq, norm, x0, horizon, budget, steps, lr, rho_u, success_V, un_lo, un_hi)
        rollout_light = slib.single_cstr_rollout(seq, norm, x0, horizon, budget, steps, lr, rho_u, success_V,
                                                 un_lo, un_hi, bs.SIM, input_lo, input_hi,
                                                 record_iterations=False)
        r_light = slib.battery1_sanity_compare(rollout_light["summary"], original)
        check("Battery 1's light instrumented rollout matches the frozen closed_loop() exactly "
              "(Battery-1 sanity check)", r_light["passed"], str(r_light))

        rollout_detailed = slib.single_cstr_rollout(seq, norm, x0, horizon, budget, steps, lr, rho_u, success_V,
                                                     un_lo, un_hi, bs.SIM, input_lo, input_hi,
                                                     record_iterations=True)
        r_detailed = slib.gate1_compare(rollout_detailed["summary"], original)
        check("Battery 2's detailed (per-iteration-logging) rollout ALSO matches the frozen "
              "closed_loop() exactly (Gate 1) -- logging does not perturb the optimization",
              r_detailed["passed"], str(r_detailed))

        check("Battery 2's step_records cover every completed control step",
              len(rollout_detailed["step_records"]) == rollout_detailed["summary"]["n_steps_completed"])
        all_budget_matched = all(len(sr["iterations"]) == budget for sr in rollout_detailed["step_records"])
        check("every logged control step has exactly `budget` iteration records", all_budget_matched)
        n_clamped_nonneg = all(it["n_clamped_coords"] >= 0 for sr in rollout_detailed["step_records"]
                               for it in sr["iterations"])
        check("n_clamped_coords is always non-negative", n_clamped_nonneg)
    finally:
        bs.SIM = old_sim


def test_residual_bridge_enrichment_real_sim():
    """Section 3.5's `saturated_sontag_np` (used to compute Phi(x)) needs
    `sim._dynamics()`/`sim.analytic_jacobian()`, which the fabricated
    `FakeSim` above does not implement -- this specific enrichment check
    uses the REAL single-CSTR simulator (cheap: pure numpy/scipy RK4, no
    data files, no CasADi) rather than extending `FakeSim`'s interface,
    which would risk silently drifting from the real Sontag-feedback
    contract it exists to exercise."""
    import budget_sweep_solver as bs
    from cstr.lcnn_paper_simulator import create_lcnn_paper_cstr
    from disambiguate_landscape_vs_gradient import make_norm
    old_sim = bs.SIM
    try:
        real_sim = create_lcnn_paper_cstr(shifted=True)
        bs.SIM = real_sim
        pool = asmoke.make_fake_pool(seed=41)
        torch.manual_seed(9)
        model = alib.build_model("E", use_fnn=False)
        seq = alib.freeze_any(model)
        data_tuple = (pool["XU"], pool["Y"], pool["xm"], pool["xs"], pool["ym"], pool["ys"],
                      pool["train_idx"], pool["test_idx"])
        norm, un_lo, un_hi = make_norm(data_tuple)
        x0 = np.array([0.1, 5.0])
        horizon, budget, steps, lr, rho_u = 2, 3, 2, 0.1, 0.01
        rollout = slib.single_cstr_rollout(seq, norm, x0, horizon, budget, steps, lr, rho_u, 2.0,
                                           un_lo, un_hi, real_sim, bs.INPUT_LO, bs.INPUT_HI,
                                           record_iterations=True)
        xm, xs, ym, ys = norm
        un_lo_np, un_hi_np = un_lo.numpy(), un_hi.numpy()
        sr0 = rollout["step_records"][0]
        slib.enrich_step_record_with_residual_bridge(seq, real_sim, sr0, un_lo_np, un_hi_np, xm, xs, ym, ys,
                                                      bs.INPUT_LO, bs.INPUT_HI)
        slib.enrich_step_record_with_counterfactual(real_sim, sr0, horizon, rho_u, xm, xs, ym, ys)
        rb0 = sr0["iterations"][0]["residual_bridge"]
        cosine_ok = rb0["excluded"] or (-1.0 - 1e-6 <= rb0["cosine"] <= 1.0 + 1e-6)
        check("residual_bridge cosine (real sim) is either NaN(excluded) or within [-1,1]", cosine_ok, str(rb0))
        check("residual_bridge g_hat/g_true (real sim) are finite floats",
              np.isfinite(rb0["g_hat"]) and np.isfinite(rb0["g_true"]))
        check("residual_bridge raw_inner_product (real sim) is a finite float",
              np.isfinite(rb0["raw_inner_product"]))
        it0 = sr0["iterations"][0]
        check("counterfactual_true_objective_before/after (real sim) are both populated as finite "
              "floats after enrichment (before/after symmetry with GGN, Section 3.4 item 6)",
              np.isfinite(it0["counterfactual_true_objective_before"])
              and np.isfinite(it0["counterfactual_true_objective_after"]))
    finally:
        bs.SIM = old_sim


# ---------------------------------------------------------------------------
# Section F: GGN Gate 3 (implementation-equivalence)
# ---------------------------------------------------------------------------
def test_ggn_gate3_matches_original():
    import budget_sweep_solver as bs
    import ggn_mpc_probe as ggn
    from disambiguate_landscape_vs_gradient import make_norm
    old_bs_sim = bs.SIM
    old_ggn_sim = ggn.SIM
    try:
        fake = asmoke.FakeSim()
        bs.SIM = fake
        ggn.SIM = fake
        pool = asmoke.make_fake_pool(seed=31)
        torch.manual_seed(7)
        model = alib.build_model("B", use_fnn=False)
        seq = alib.freeze_any(model)
        data_tuple = (pool["XU"], pool["Y"], pool["xm"], pool["xs"], pool["ym"], pool["ys"],
                      pool["train_idx"], pool["test_idx"])
        norm, un_lo, un_hi = make_norm(data_tuple)
        x0 = np.array([0.1, 5.0])
        horizon, budget, steps, rho_u, success_V = 2, 2, 3, 0.01, 2.0

        original = ggn.ggn_closed_loop(seq, norm, x0, horizon, budget, steps, rho_u, success_V,
                                       un_lo, un_hi, "V")
        # `log_iterations=True` -- full diagnostic data, its OWN timing is
        # only ever a "with_logging_overhead" diagnostic field.
        instrumented = slib.ggn_closed_loop_instrumented(seq, norm, x0, horizon, budget, steps, rho_u,
                                                          success_V, un_lo, un_hi, "V", fake,
                                                          log_iterations=True)
        r = slib.gate3_compare(instrumented, original)
        check("GGN instrumented rollout (log_iterations=True) matches the ORIGINAL "
              "ggn_mpc_probe.ggn_closed_loop() (Gate 3: success exact, final_V/accepted_frac isclose)",
              r["passed"], str(r))
        check("GGN instrumented rollout's mean_counterfactual_dec_applied is a finite float "
              "(Section 3.4 item 6 fix)", np.isfinite(instrumented["mean_counterfactual_dec_applied"]))
        check("log_iterations=True does NOT populate the 'core' timing key (never claim "
              "contaminated time is core)", "rollout_solver_core_wall_clock_seconds" not in instrumented)
        check("log_iterations=True populates the 'with_logging_overhead' diagnostic timing key",
              "rollout_instrumented_wall_clock_with_logging_overhead_seconds" in instrumented)

        check("GGN step_records has exactly one entry per completed control step",
              len(instrumented["step_records"]) == steps)
        check("every GGN step_record has exactly `budget` per-iteration entries",
              all(len(sr["iterations"]) == budget for sr in instrumented["step_records"]))
        # Timing-contamination fix, part 1: even within the log_iterations=
        # True pass, the counterfactual audit computation must be timed
        # SEPARATELY from the surrounding per-iteration bookkeeping.
        core_sum = sum(sr["solve_core_wall_clock_seconds"] for sr in instrumented["step_records"])
        audit_sum = sum(sr["solve_audit_wall_clock_seconds"] for sr in instrumented["step_records"])
        with_overhead = instrumented["rollout_instrumented_wall_clock_with_logging_overhead_seconds"]
        check("GGN's diagnostic timing excludes audit time from its own 'core' component "
              "(sum of per-step core times matches the raw array)",
              abs(core_sum - sum(with_overhead["raw"])) < 1e-9)
        check("GGN rollout_instrumented_total_wall_clock_seconds = core + audit, exactly",
              abs(instrumented["rollout_instrumented_total_wall_clock_seconds"] - (core_sum + audit_sum)) < 1e-9)
        if instrumented["accepted_frac"] > 0:
            check("GGN audit wall-clock is strictly positive whenever at least one update was accepted "
                  "(the counterfactual computation genuinely ran and was genuinely timed)",
                  instrumented["rollout_audit_counterfactual_wall_clock_seconds"] > 0.0)

        # Timing-contamination fix, part 2 (this round's actual fix): a
        # SEPARATE `log_iterations=False` pass is the genuinely clean
        # "core" timing source -- no per-iteration bookkeeping AT ALL,
        # not even the (already audit-excluded) dict/list construction.
        clean = slib.ggn_closed_loop_instrumented(seq, norm, x0, horizon, budget, steps, rho_u,
                                                  success_V, un_lo, un_hi, "V", fake,
                                                  log_iterations=False)
        r_clean = slib.gate3_compare(clean, original)
        check("GGN's log_iterations=False clean-timing pass ALSO matches the ORIGINAL exactly "
              "(same core math, zero bookkeeping)", r_clean["passed"], str(r_clean))
        check("log_iterations=False populates the 'core' timing key, not the overhead one",
              "rollout_solver_core_wall_clock_seconds" in clean
              and "rollout_instrumented_wall_clock_with_logging_overhead_seconds" not in clean)
        check("log_iterations=False produces no step_records (zero logging overhead by construction)",
              clean["step_records"] is None)
        check("log_iterations=False reports zero audit wall-clock (no counterfactual computation ran)",
              clean["rollout_audit_counterfactual_wall_clock_seconds"] == 0.0)
    finally:
        bs.SIM = old_bs_sim
        ggn.SIM = old_ggn_sim


# ---------------------------------------------------------------------------
# Section F2: Battery 1/2 dedup contract (Section 2) -- B/E's Battery-1-
# level record must be MERGED from Battery 2, never independently re-run.
# ---------------------------------------------------------------------------
def test_battery1_battery2_dedup_contract(scratch: Path):
    out_dir = scratch / "dedup_out"
    out_dir.mkdir(parents=True)
    fake_battery2_payload = dict(
        checkpoint_path="/tmp/fake_E.pt", checkpoint_sha256="deadbeef",
        gate2_result=dict(passed=True, mismatches=[]),
        per_ic_summary=[
            dict(initial_condition_index=i, summary=dict(success=(i < 3)),
                 state_constraint=dict(dummy=True))
            for i in range(5)
        ],
    )
    merged = drv.battery1_record_from_battery2("E", 0, fake_battery2_payload, out_dir, "0" * 64)
    self_hash, payload = sio.read_and_verify_json_artifact(
        merged["path"], expected_kind="solver_audit_battery1_record",
        expected_keys=drv.EXPECTED_BATTERY1_RECORD_PAYLOAD_KEYS)
    check("battery1_record_from_battery2 tags the merged record's source correctly",
          payload["source"] == "merged_from_battery2")
    check("battery1_record_from_battery2 correctly aggregates the merged seed-level success rate "
          "from Battery 2's per_ic_summary (3 of 5 ICs succeeded)",
          abs(payload["seed_level_success_rate"] - 0.6) < 1e-12)
    check("battery1_record_from_battery2 propagates Battery 2's own checkpoint identity, not a "
          "re-resolved one", payload["checkpoint_path"] == "/tmp/fake_E.pt")

    assert "E" not in drv.BATTERY1_OWN_CONDITIONS and "B" not in drv.BATTERY1_OWN_CONDITIONS
    check("BATTERY1_OWN_CONDITIONS excludes B and E (Battery 1 never independently re-runs them)",
          set(drv.BATTERY1_OWN_CONDITIONS) == set(drv.ALL_CONDITIONS) - set(drv.DETAILED_CONDITIONS))


def test_configure_deterministic_single_thread():
    slib.configure_deterministic_single_thread()
    check("configure_deterministic_single_thread() actually sets torch to the pinned thread count",
          torch.get_num_threads() == sio.TIMING_TORCH_NUM_THREADS)


# ---------------------------------------------------------------------------
# Section G: checkpoint lookup
# ---------------------------------------------------------------------------
def test_checkpoint_lookup(scratch: Path):
    ckdir = scratch / "fake_checkpoints"
    ckdir.mkdir(parents=True)
    torch.manual_seed(0)
    model = alib.build_model("E", use_fnn=False)
    path = ckdir / "E_seed0_ep050.pt"
    sio.atomic_write_torch_no_overwrite(path, model.state_dict())
    m2_payload = dict(checkpoint_sha256={str(path): sio.file_sha256(path)})
    found = slib.find_checkpoint_path(m2_payload, "E", 0)
    check("find_checkpoint_path locates the correct checkpoint by naming convention", found == str(path))

    path.write_bytes(path.read_bytes() + b"\x00")  # corrupt
    drifted = False
    try:
        slib.find_checkpoint_path(m2_payload, "E", 0)
    except RuntimeError as e:
        drifted = "CHECKPOINT_DRIFT" in str(e)
    check("find_checkpoint_path detects checkpoint content drift against the recorded hash", drifted)


def test_real_single_cstr_battery123_e2e_reduced_scale(scratch: Path):
    """The single highest-value check in this suite: runs the ACTUAL
    driver-level `run_battery1_condition_seed`/`run_battery2_condition_
    seed`/`run_battery3_condition_seed` functions against REAL seed-0
    checkpoints and the REAL simulator (not FakeSim, not a fabricated
    payload) -- proving the full nested JSON payload these functions
    build (rollout summaries, state-constraint stats, gate results,
    residual-bridge/counterfactual enrichment) actually self-verifies on
    a WRITE, not just a hand-fabricated small dict. This is exactly the
    class of check that caught the `state_constraint_stats` int-dict-key
    JSON round-trip bug (only surfaced via Battery 5's real write path)
    -- exercising the single-CSTR path for real closes the same blind
    spot there. `drv.STEPS`/`drv.BUDGET` are monkeypatched to tiny values
    purely to keep this specific check fast; restored afterward."""
    ablation_m2_path = drv.adrv.NAMESPACE / "ablation_training_manifest_seed0_2026_08_31.json"
    if not ablation_m2_path.exists():
        check("real single-CSTR Battery 1/2/3 E2E skipped: real ablation seed-0 M2 not present",
              True, "SKIPPED")
        return
    import budget_sweep_solver as bs
    old_steps, old_budget = drv.STEPS, drv.BUDGET
    out_dir = scratch / "real_e2e_out"
    out_dir.mkdir(parents=True)
    try:
        drv.STEPS, drv.BUDGET = 3, 3
        b1_A = drv.run_battery1_condition_seed("A", 0, ablation_m2_path, bs.SIM, out_dir, "0" * 64)
        check("real Battery 1 (condition A, seed 0, real checkpoint+simulator) writes and "
              "self-verifies successfully", isinstance(b1_A["self_hash"], str) and len(b1_A["self_hash"]) == 64)

        b2_E = drv.run_battery2_condition_seed("E", 0, ablation_m2_path, bs.SIM, out_dir, "0" * 64)
        _, b2_payload = sio.read_and_verify_json_artifact(
            b2_E["path"], expected_kind="solver_audit_battery2_record",
            expected_keys=drv.EXPECTED_BATTERY2_RECORD_PAYLOAD_KEYS)
        check("real Battery 2 (condition E, seed 0) writes and self-verifies successfully, "
              "including real residual-bridge/counterfactual-enriched per-iteration arrays",
              True)
        check("real Battery 2's Gate 1 passes for every IC against the real frozen closed_loop() "
              "(both sides run at the SAME reduced steps/budget -- a genuine apples-to-apples check)",
              all(r["passed"] for r in b2_payload["gate1_results"]), str(b2_payload["gate1_results"]))
        # Gate 2 is NOT expected to pass here: it compares against M2's
        # `closed_loop_success`, which was measured at the REAL
        # steps=120/budget=20 endpoint -- this test deliberately runs at
        # steps=3/budget=3 for speed, so the two operating points are not
        # comparable. Only the gate's MACHINERY (well-formed result,
        # no crash) is checked at reduced scale; Gate 2's actual pass/
        # fail is only meaningful at the real endpoint (exercised
        # separately, at full scale, once this file is frozen).
        check("real Battery 2's Gate 2 machinery runs and returns a well-formed result "
              "(pass/fail itself is not meaningful at this test's reduced steps/budget)",
              isinstance(b2_payload["gate2_result"], dict) and "passed" in b2_payload["gate2_result"])

        b3_E = drv.run_battery3_condition_seed("E", 0, ablation_m2_path, bs.SIM, out_dir, "0" * 64)
        _, b3_payload = sio.read_and_verify_json_artifact(
            b3_E["path"], expected_kind="solver_audit_battery3_record",
            expected_keys=drv.EXPECTED_BATTERY3_RECORD_PAYLOAD_KEYS)
        check("real Battery 3's Gate 3 passes for every IC against the real ORIGINAL ggn_closed_loop()",
              all(r["passed"] for r in b3_payload["gate3_results"]), str(b3_payload["gate3_results"]))

        merged_e = drv.battery1_record_from_battery2("E", 0, b2_payload, out_dir, "0" * 64)
        check("real Battery 1's merged B/E record (dedup contract) writes and self-verifies successfully",
              isinstance(merged_e["self_hash"], str))

        # Exercise the Section 5 Summary builders end-to-end on REAL data
        # (not a fabricated payload) -- this is what caught the missing
        # `payload=` key in every fresh-build return dict (Section 5's
        # aggregation helpers all read `record["payload"]`, which a
        # freshly-built, non-resumed record previously never had).
        b1_C = drv.run_battery1_condition_seed("C", 0, ablation_m2_path, bs.SIM, out_dir, "0" * 64)
        b1_F = drv.run_battery1_condition_seed("F", 0, ablation_m2_path, bs.SIM, out_dir, "0" * 64)
        b1_BFNN = drv.run_battery1_condition_seed("B-FNN", 0, ablation_m2_path, bs.SIM, out_dir, "0" * 64)
        b1_EFNN = drv.run_battery1_condition_seed("E-FNN", 0, ablation_m2_path, bs.SIM, out_dir, "0" * 64)
        b1_D = drv.run_battery1_condition_seed("D", 0, ablation_m2_path, bs.SIM, out_dir, "0" * 64)
        b2_B = drv.run_battery2_condition_seed("B", 0, ablation_m2_path, bs.SIM, out_dir, "0" * 64)
        b3_B = drv.run_battery3_condition_seed("B", 0, ablation_m2_path, bs.SIM, out_dir, "0" * 64)
        _, b2_b_payload = sio.read_and_verify_json_artifact(
            b2_B["path"], expected_kind="solver_audit_battery2_record",
            expected_keys=drv.EXPECTED_BATTERY2_RECORD_PAYLOAD_KEYS)
        merged_b = drv.battery1_record_from_battery2("B", 0, b2_b_payload, out_dir, "0" * 64)

        battery1_records = {("A", 0): b1_A, ("B", 0): merged_b, ("C", 0): b1_C, ("D", 0): b1_D,
                            ("E", 0): merged_e, ("F", 0): b1_F, ("B-FNN", 0): b1_BFNN, ("E-FNN", 0): b1_EFNN}
        battery2_records = {("B", 0): b2_B, ("E", 0): b2_E}
        battery3_records = {("B", 0): b3_B, ("E", 0): b3_E}

        for name, rec in list(battery1_records.items()) + list(battery2_records.items()) + list(
                battery3_records.items()):
            if "payload" not in rec:
                raise AssertionError(f"record {name} is missing 'payload' -- the exact bug class this "
                                     "test exists to catch")
        check("every real battery record dict (fresh-built, not resumed) includes a 'payload' key",
              True)

        b1_summary = drv.build_battery1_summary_manifest(battery1_records, [0], out_dir, "0" * 64)
        _, b1_summary_payload = sio.read_and_verify_json_artifact(
            b1_summary["path"], expected_kind="solver_audit_battery1_summary",
            expected_keys=drv.EXPECTED_BATTERY1_SUMMARY_PAYLOAD_KEYS)
        check("Battery 1 Summary Manifest builds on REAL data and reports all 4 contrast pairs "
              "for every metric in BATTERY1_METRIC_SPECS",
              set(b1_summary_payload["contrasts"].keys()) == set(drv.BATTERY1_METRIC_SPECS)
              and all(set(v.keys()) == set(drv.BATTERY1_CONTRAST_PAIRS)
                      for v in b1_summary_payload["contrasts"].values()))

        b2_summary = drv.build_battery2_summary_manifest(battery2_records, [0], out_dir, "0" * 64)
        _, b2_summary_payload = sio.read_and_verify_json_artifact(
            b2_summary["path"], expected_kind="solver_audit_battery2_summary",
            expected_keys=drv.EXPECTED_BATTERY2_SUMMARY_PAYLOAD_KEYS)
        check("Battery 2 Summary Manifest builds on REAL data and reports E_vs_B for every metric "
              "in BATTERY2_METRIC_FNS", set(b2_summary_payload["contrasts"].keys()) == set(drv.BATTERY2_METRIC_FNS)
              and all("E_vs_B" in v for v in b2_summary_payload["contrasts"].values()))

        b3_summary = drv.build_battery3_summary_manifest(battery3_records, [0], out_dir, "0" * 64)
        _, b3_summary_payload = sio.read_and_verify_json_artifact(
            b3_summary["path"], expected_kind="solver_audit_battery3_summary",
            expected_keys=drv.EXPECTED_BATTERY3_SUMMARY_PAYLOAD_KEYS)
        check("Battery 3 Summary Manifest builds on REAL data (seed-first aggregation)",
              set(b3_summary_payload["descriptive"].keys()) == set(drv.DETAILED_CONDITIONS))

        # Resume-or-verify at the Summary level too.
        b1_summary2 = drv.build_battery1_summary_manifest(battery1_records, [0], out_dir, "0" * 64)
        check("Battery 1 Summary Manifest resume-or-verify returns the SAME hash on a second call",
              b1_summary2["self_hash"] == b1_summary["self_hash"])
    finally:
        drv.STEPS, drv.BUDGET = old_steps, old_budget

    check("checkpoint_prefix_for_condition('B-FNN', ...) == 'BFNN'",
          slib.checkpoint_prefix_for_condition("B-FNN") == "BFNN")
    check("checkpoint_prefix_for_condition('E-FNN', ...) == 'E-FNN'",
          slib.checkpoint_prefix_for_condition("E-FNN") == "E-FNN")
    check("checkpoint_prefix_for_condition('D', ...) uses the seed's selected lambda_jac",
          slib.checkpoint_prefix_for_condition("D", dict(lambda_jac_selected=0.05)) == "D_lamjac0.05")


# ---------------------------------------------------------------------------
# Section H: two-CSTR Battery 5's ACTUAL 4-phase driver flow -- ONE real
# reconstruction (seed 0, value_only) against the already-logged
# two_cstr_lgrad.json, run through the SAME phase1/2 -> phase4 functions
# `run_real_two_cstr_minimum_audit` uses (not the lower-level lib
# function directly), so this also exercises: phase 3's "freeze the
# checkpoint set" contract, phase 4's RELOAD from the frozen checkpoint
# file (never the in-memory training object), and the reload-equivalence
# gate against the original `two_cstr_lgrad.closed_loop()`.
# ---------------------------------------------------------------------------
def test_two_cstr_battery5_four_phase_flow(scratch: Path):
    ckdir = scratch / "two_cstr_checkpoints"
    ckdir.mkdir(parents=True)
    out_dir = scratch / "battery5_out"
    out_dir.mkdir(parents=True)

    recon = drv.battery5_phase1_and_2_reconstruct_and_gate("value_only", 0, out_dir, ckdir)
    check("Battery 5 phases 1-2 (real reconstruction, seed=0/value_only) pass Gate 4 against "
          "the already-logged two_cstr_lgrad.json", recon["gate4"]["passed"], str(recon["gate4"]))
    ckpt_path = recon["replica"]["checkpoint_path"]
    check("Battery 5 phase 1's checkpoint file exists on disk (phase 3's freeze target)",
          Path(ckpt_path).exists())

    # Phase 1's checkpoint-resume path: reruns reconstruct_two_cstr_replica
    # for the SAME (config, seed) -- must reuse the existing checkpoint
    # (loaded, not retrained) and recompute metrics fresh against it.
    recon_resumed = drv.battery5_phase1_and_2_reconstruct_and_gate("value_only", 0, out_dir, ckdir)
    check("reconstruct_two_cstr_replica's checkpoint-resume path reproduces IDENTICAL metrics from "
          "the reused checkpoint (deterministic reload, not a fresh retrain)",
          recon_resumed["replica"]["metrics"] == recon["replica"]["metrics"])

    fake_setup_hash = "0" * 64
    reconstructions = {("value_only", 0): recon}
    replica_manifest = drv.build_replica_manifest(reconstructions, out_dir, fake_setup_hash)
    replica_manifest_verified = drv.verify_replica_manifest(replica_manifest["path"])
    check("Replica Manifest (phase 3's REAL freeze artifact) builds and self-verifies",
          replica_manifest_verified["self_hash"] == replica_manifest["self_hash"])
    check("Replica Manifest re-verifies the referenced checkpoint's current hash",
          Path(ckpt_path).exists())

    record = drv.battery5_phase4_instrumented_audit(
        "value_only", 0, out_dir, replica_manifest_verified["self_hash"], replica_manifest_verified["payload"])
    self_hash, payload = sio.read_and_verify_json_artifact(
        record["path"], expected_kind="solver_audit_battery5_record",
        expected_keys=drv.EXPECTED_BATTERY5_RECORD_PAYLOAD_KEYS)
    check("Battery 5 phase 4's record round-trips with the exact expected keyset",
          self_hash == record["self_hash"])
    check("Battery 5 phase 4's reload-equivalence gate passes for every IC (instrumented rollout "
          "on the RELOADED checkpoint matches the original two_cstr_lgrad.closed_loop())",
          all(r["passed"] for r in payload["reload_equivalence_gate_results"]),
          str(payload["reload_equivalence_gate_results"]))
    check("Battery 5 phase 4's state_constraint_audit reports BOTH reactor temperature indices",
          all(set(row["state_constraint"]["per_temperature_index"].keys()) == {"1", "3"}
              for row in payload["state_constraint_audit"]))

    # Resume-or-verify: calling phase 4 again for the SAME (config, seed)
    # must reuse the existing record rather than re-running the audit.
    record2 = drv.battery5_phase4_instrumented_audit(
        "value_only", 0, out_dir, replica_manifest_verified["self_hash"], replica_manifest_verified["payload"])
    check("battery5_phase4_instrumented_audit resume-or-verify returns the SAME hash on a second call",
          record2["self_hash"] == record["self_hash"])

    # Resume-or-verify at the Replica Manifest level too.
    replica_manifest2 = drv.build_replica_manifest(reconstructions, out_dir, fake_setup_hash)
    check("build_replica_manifest resume-or-verify returns the SAME hash on a second call",
          replica_manifest2["self_hash"] == replica_manifest["self_hash"])

    # Battery 5 Summary Manifest (10-seed descriptive mean/min/max) --
    # exercised here with the single (config, seed) pair this test built.
    b5_records = {("value_only", 0): record}
    b5_summary = drv.build_battery5_summary_manifest(b5_records, [0], out_dir, fake_setup_hash)
    _, b5_summary_payload = sio.read_and_verify_json_artifact(
        b5_summary["path"], expected_kind="solver_audit_battery5_summary",
        expected_keys=drv.EXPECTED_BATTERY5_SUMMARY_PAYLOAD_KEYS)
    check("Battery 5 Summary Manifest builds and reports descriptive stats for value_only",
          "value_only" in b5_summary_payload["descriptive"])


# ---------------------------------------------------------------------------
# Section I: manifest I/O round-trips (reusing the REAL, already-verified
# ablation Freeze Manifest / M1 / seed-0 M2 as inputs -- no fabrication
# needed, since those real artifacts already exist on disk).
# ---------------------------------------------------------------------------
def test_setup_manifest_round_trip(scratch: Path):
    import hashlib
    ablation_freeze_path = drv.adrv.FREEZE_MANIFEST_PATH
    ablation_m1_path = drv.adrv.NAMESPACE / "ablation_query_manifest_2026_08_31.json"
    # Uses the REAL full 10-seed set (`alib.SEEDS`), matching the seed-set
    # tag the real go/no-go ablation's actual M3 was built for -- a
    # smaller/different seed subset would have no matching real M3 on
    # disk and this check would only ever hit its SKIPPED branch.
    all_seeds = list(alib.SEEDS)
    ablation_m2_paths = {
        s: drv.adrv.NAMESPACE / f"ablation_training_manifest_seed{s}_2026_08_31.json" for s in all_seeds
    }
    seed_set_tag = hashlib.sha256(repr(sorted(all_seeds)).encode()).hexdigest()[:12]
    ablation_m3_path = drv.adrv.NAMESPACE / f"ablation_results_manifest_{seed_set_tag}_2026_08_31.json"
    if not (ablation_freeze_path.exists() and ablation_m1_path.exists() and ablation_m3_path.exists()
            and all(p.exists() for p in ablation_m2_paths.values())):
        check("setup manifest round-trip skipped: real ablation artifacts not present on this machine",
              True, "SKIPPED")
        return
    pool = alib.load_pool()
    fake_solver_audit_freeze_hash = "1" * 64
    setup = drv.build_setup_manifest(ablation_freeze_path, ablation_m1_path, ablation_m2_paths,
                                     ablation_m3_path, pool, fake_solver_audit_freeze_hash, out_dir=scratch,
                                     dest_path=scratch / "setup_manifest.json")
    verified = drv.verify_setup_manifest(setup["path"])
    check("Setup Manifest self-verification round-trip succeeds", verified["self_hash"] == setup["self_hash"])
    check("Setup Manifest payload has the exact expected keyset",
          set(verified["payload"].keys()) == drv.EXPECTED_SETUP_MANIFEST_PAYLOAD_KEYS)
    check("Setup Manifest pins the real ablation's actual M3 hash",
          verified["payload"]["predecessor_ablation_m3_sha256"] == adrv_verify_m3_hash(ablation_m3_path))
    check("Setup Manifest pins THIS protocol's own solver-audit freeze hash (traceability, round-5 fix)",
          verified["payload"]["predecessor_solver_audit_freeze_sha256"] == fake_solver_audit_freeze_hash)
    check("Setup Manifest resolved_checkpoints covers every condition for every requested seed",
          set(verified["payload"]["resolved_checkpoints"].keys()) == set(drv.ALL_CONDITIONS)
          and all(set(v.keys()) == {str(s) for s in all_seeds}
                  for v in verified["payload"]["resolved_checkpoints"].values()))

    # Resume-or-verify: a second call must NOT recompute (no FileExistsError,
    # no re-verification work beyond self-hash) and must return the SAME hash.
    setup2 = drv.build_setup_manifest(ablation_freeze_path, ablation_m1_path, ablation_m2_paths,
                                      ablation_m3_path, pool, fake_solver_audit_freeze_hash, out_dir=scratch,
                                      dest_path=scratch / "setup_manifest.json")
    check("build_setup_manifest resume-or-verify returns the SAME hash on a second call "
          "(no FileExistsError, no silent rebuild)", setup2["self_hash"] == setup["self_hash"])
    check("build_setup_manifest resume-or-verify marks the second call as resumed",
          setup2.get("resumed") is True)

    # A DIFFERENT solver-audit freeze hash (simulating the protocol having
    # been re-frozen since this Setup Manifest was built) must be rejected
    # on resume, not silently reused.
    stale_detected = False
    try:
        drv.build_setup_manifest(ablation_freeze_path, ablation_m1_path, ablation_m2_paths,
                                 ablation_m3_path, pool, "2" * 64, out_dir=scratch,
                                 dest_path=scratch / "setup_manifest.json")
    except RuntimeError as e:
        stale_detected = "RESUME_STALE_SETUP_MANIFEST" in str(e)
    check("build_setup_manifest resume rejects a Setup Manifest whose recorded solver-audit "
          "freeze hash no longer matches the CURRENT freeze (protocol re-frozen since)", stale_detected)
    return setup


def adrv_verify_m3_hash(m3_path: Path) -> str:
    return drv.adrv.verify_results_manifest(m3_path)["self_hash"]


def test_two_cstr_setup_manifest_round_trip(scratch: Path):
    fake_hash = "3" * 64
    setup = drv.build_two_cstr_setup_manifest(fake_hash, out_dir=scratch,
                                              dest_path=scratch / "two_cstr_setup_manifest.json")
    verified = drv.verify_two_cstr_setup_manifest(setup["path"])
    check("two-CSTR Setup Manifest self-verification round-trip succeeds",
          verified["self_hash"] == setup["self_hash"])
    check("two-CSTR Setup Manifest payload has the exact expected keyset",
          set(verified["payload"].keys()) == drv.EXPECTED_TWO_CSTR_SETUP_MANIFEST_PAYLOAD_KEYS)
    check("two-CSTR Setup Manifest pins THIS protocol's own solver-audit freeze hash",
          verified["payload"]["predecessor_solver_audit_freeze_sha256"] == fake_hash)
    check("two-CSTR Setup Manifest records a full software/hardware environment "
          "(previously entirely missing, round-5 P1 fix)",
          {"os_platform", "python_version", "numpy_version", "torch_version",
           "cpu_architecture", "processor"} <= set(verified["payload"]["environment"].keys()))
    check("two-CSTR Setup Manifest records the actual compute device and thread count",
          verified["payload"]["compute_device"] == "cpu"
          and verified["payload"]["torch_num_threads"] == sio.TIMING_TORCH_NUM_THREADS)

    # Stale-freeze rejection, mirroring the single-CSTR Setup Manifest's own check.
    stale_detected = False
    try:
        drv.build_two_cstr_setup_manifest("4" * 64, out_dir=scratch,
                                          dest_path=scratch / "two_cstr_setup_manifest.json")
    except RuntimeError as e:
        stale_detected = "RESUME_STALE_TWO_CSTR_SETUP_MANIFEST" in str(e)
    check("two-CSTR Setup Manifest resume rejects a stale solver-audit freeze hash", stale_detected)


def test_battery_record_schemas_round_trip(scratch: Path):
    fabricated = dict(
        kind="solver_audit_battery1_record", generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_setup_manifest_sha256="deadbeef", condition="E", seed=0, source="battery1_independent",
        checkpoint_path="/tmp/fake.pt", checkpoint_sha256="deadbeef", per_ic=[],
        seed_level_success_rate=1.0, gate2_result=dict(passed=True, mismatches=[]),
    )
    path = scratch / "battery1_fake.json"
    self_hash = sio.write_with_self_verification(path, fabricated)
    reread_hash, payload = sio.read_and_verify_json_artifact(
        path, expected_kind="solver_audit_battery1_record",
        expected_keys=drv.EXPECTED_BATTERY1_RECORD_PAYLOAD_KEYS)
    check("Battery 1 record schema round-trips and matches its exact expected keyset",
          reread_hash == self_hash and set(payload.keys()) == drv.EXPECTED_BATTERY1_RECORD_PAYLOAD_KEYS)


# ---------------------------------------------------------------------------
# Section J: round-3 fixes -- signed objective-decrease, survivorship
# aggregation unit, incomplete-evidence contrast status, Summary
# companion tamper detection.
# ---------------------------------------------------------------------------
def test_objective_decrease_metrics_distinguish_improvement_from_regression():
    arrays_improve = {
        "ic0_j_learned": np.array([10.0, 8.0, 6.0]), "ic0_j_learned_after": np.array([8.0, 6.0, 5.0]),
        "ic0_counterfactual_true_objective_before": np.array([10.0, 8.0, 6.0]),
        "ic0_counterfactual_true_objective_after": np.array([8.0, 6.0, 5.0]),
    }
    dec = drv._b2_mean_signed_decrease(arrays_improve, 0, "j_learned", "j_learned_after")
    check("signed objective-decrease is POSITIVE when the objective genuinely improved every iteration "
          "(before > after)", dec is not None and dec > 0, str(dec))

    arrays_regress = {
        "ic0_j_learned": np.array([5.0, 6.0, 7.0]), "ic0_j_learned_after": np.array([6.0, 7.0, 9.0]),
    }
    dec_bad = drv._b2_mean_signed_decrease(arrays_regress, 0, "j_learned", "j_learned_after")
    check("signed objective-decrease is NEGATIVE when the objective got WORSE every iteration "
          "(the old abs-diff metric could not distinguish this from improvement)",
          dec_bad is not None and dec_bad < 0, str(dec_bad))

    # The LAST iteration's decrease must be counted (the old stepwise-
    # abs-diff metric silently dropped it, since it diffed CONSECUTIVE
    # raw values and the last iteration has no "next" to diff against).
    arrays_single = {"ic0_j_learned": np.array([9.0]), "ic0_j_learned_after": np.array([1.0])}
    dec_single = drv._b2_mean_signed_decrease(arrays_single, 0, "j_learned", "j_learned_after")
    check("signed objective-decrease is computed even for a step with a SINGLE logged iteration "
          "(before/after pair, not a consecutive-value diff)",
          dec_single is not None and abs(dec_single - 8.0) < 1e-9, str(dec_single))


def test_survivorship_uses_overall_rate_not_max_per_seed_gap():
    seeds = list(range(10))
    # Condition A: every seed completes all 5 ICs. Condition B: exactly
    # ONE IC fails in exactly ONE seed (per-seed completion 4/5=0.8 for
    # that one seed, 1.0 for the other 9) -- per-seed gap for that ONE
    # seed is 0.2, but the OVERALL completion-rate gap across all 50 ICs
    # is (50-49)/50 = 0.02, well under the 0.10 threshold.
    completion_a = {str(s): 1.0 for s in seeds}
    completion_b = {str(s): (0.8 if s == 0 else 1.0) for s in seeds}
    metric_a = {str(s): 1.0 for s in seeds}
    metric_b = {str(s): 1.0 for s in seeds}
    result, _ = drv.contrast_with_survivorship_rule(metric_a, metric_b, completion_a, completion_b, seeds)
    check("survivorship rule uses the OVERALL per-condition completion rate (0.02 gap here), "
          "NOT the max per-seed gap (which would have been 0.2 and wrongly triggered a downgrade)",
          not result["survivorship_downgraded_to_descriptive"], str(result))
    check("completion_rate_gap is computed correctly as the overall (mean-of-means) difference",
          abs(result["completion_rate_gap"] - 0.02) < 1e-9, str(result["completion_rate_gap"]))

    # A genuinely large OVERALL gap must still downgrade.
    completion_b_large_gap = {str(s): 0.5 for s in seeds}
    result2, _ = drv.contrast_with_survivorship_rule(metric_a, metric_b, completion_a,
                                                      completion_b_large_gap, seeds)
    check("survivorship rule still downgrades when the OVERALL completion-rate gap genuinely "
          "exceeds the 0.10 threshold", result2["survivorship_downgraded_to_descriptive"])


def test_safe_evaluate_contrast_incomplete_evidence():
    diffs_with_nan = np.array([0.1, 0.2, float("nan"), 0.15, 0.05, 0.1, 0.2, 0.1, 0.15, 0.1])
    result, resample_idx = drv._safe_evaluate_contrast(diffs_with_nan)
    check("a non-finite paired difference produces an EXPLICIT "
          "CONTRAST_UNAVAILABLE_INCOMPLETE_EVIDENCE status, not a silently-computed NaN CI",
          result["status"] == "CONTRAST_UNAVAILABLE_INCOMPLETE_EVIDENCE", str(result))
    check("the incomplete-evidence result reports NaN point/CI/p-value explicitly "
          "(never a spurious finite-looking number)",
          all(np.isnan(result[k]) for k in ("point_estimate", "ci_lower", "ci_upper", "p_value")))
    check("the incomplete-evidence result discloses n_valid_seeds < n_seeds",
          result["n_valid_seeds"] == 9 and result["n_seeds"] == 10)
    check("no resample_idx array is produced when the contrast is unavailable", resample_idx is None)

    diffs_all_finite = np.array([0.1, 0.2, 0.1, 0.15, 0.05, 0.1, 0.2, 0.1, 0.15, 0.1])
    result_ok, resample_idx_ok = drv._safe_evaluate_contrast(diffs_all_finite)
    check("a fully-finite paired-difference array is evaluated normally with status OK",
          result_ok["status"] == "OK" and np.isfinite(result_ok["point_estimate"]))
    check("a fully-finite contrast DOES produce a resample_idx array", resample_idx_ok is not None)


def test_summary_companion_npz_tamper_detection(scratch: Path):
    out_dir = scratch / "summary_tamper_out"
    out_dir.mkdir(parents=True)
    seeds = [0]
    fabricated_records = {}
    for condition in ("A", "B", "C", "D", "E", "F", "B-FNN", "E-FNN"):
        payload = dict(
            kind="solver_audit_battery1_record", generated_at_utc=datetime.now(timezone.utc).isoformat(),
            predecessor_setup_manifest_sha256="0" * 64, condition=condition, seed=0,
            source="battery1_independent", checkpoint_path="/tmp/fake.pt", checkpoint_sha256="0" * 64,
            per_ic=[dict(initial_condition_index=i,
                        summary=dict(success=True, settling_time=5, final_V=1.0, max_V=2.0, tv=0.1,
                                    lyapunov_increase_count=0, solver_completed=True,
                                    input_tv_per_actuator_physical=[0.1, 0.1],
                                    input_tv_per_actuator_normalized=[0.1, 0.1],
                                    rollout_solver_core_wall_clock_seconds=dict(median=0.01, p95=0.02, raw=[0.01])),
                        state_constraint=dict(
                            per_temperature_index={"1": dict(max_temperature_deviation_K=10.0,
                                                             max_temperature_absolute_K=412.0,
                                                             temperature_violation_frequency=0.0,
                                                             temperature_violation_integral=0.0)},
                            input_saturation_frequency={"0": 0.0, "1": 0.0},
                            input_box_satisfied=True, input_never_saturated=True,
                            temperature_cap_satisfied_posthoc=True),
                        sanity_check_vs_frozen_closed_loop=None)
                    for i in range(5)],
            seed_level_success_rate=1.0, gate2_result=dict(passed=True, mismatches=[]),
        )
        path = out_dir / f"solver_audit_battery1_{condition}_seed0_2026_09_01.json"
        self_hash = sio.write_with_self_verification(path, payload)
        fabricated_records[(condition, 0)] = dict(path=path, self_hash=self_hash, payload=payload)

    summary = drv.build_battery1_summary_manifest(fabricated_records, seeds, out_dir, "0" * 64)
    _, summary_payload = sio.read_and_verify_json_artifact(
        summary["path"], expected_kind="solver_audit_battery1_summary",
        expected_keys=drv.EXPECTED_BATTERY1_SUMMARY_PAYLOAD_KEYS)

    # Delete the resample-arrays companion NPZ and confirm resume detects it.
    resample_npz = ROOT / summary_payload["resample_arrays_path"]
    resample_npz.unlink()
    deleted_detected = False
    try:
        drv.build_battery1_summary_manifest(fabricated_records, seeds, out_dir, "0" * 64)
    except RuntimeError as e:
        deleted_detected = "RESUME_SUMMARY_COMPANION_MISSING" in str(e)
    check("Summary Manifest resume detects a DELETED companion NPZ (resample arrays), "
          "not just a changed input-record hash", deleted_detected)

    # Restore a TAMPERED (different-content) version and confirm drift is caught.
    sio.atomic_write_npz_no_overwrite(resample_npz, {"tampered": np.array([1, 2, 3])})
    tamper_detected = False
    try:
        drv.build_battery1_summary_manifest(fabricated_records, seeds, out_dir, "0" * 64)
    except RuntimeError as e:
        tamper_detected = "RESUME_SUMMARY_COMPANION_DRIFT" in str(e) or "RESUME_SUMMARY_COMPANION_MISSING" in str(e)
    check("Summary Manifest resume detects a TAMPERED companion NPZ (different content, same path)",
          tamper_detected)


# ---------------------------------------------------------------------------
# Section K: round-4 fixes -- instrumented-total timing undercounting,
# Battery-5 worst-case dilution, GGN singular-solve exclusion.
# ---------------------------------------------------------------------------
def test_apply_measured_audit_wall_clock_uses_logged_pass_total():
    # Clean pass: fast (no bookkeeping). Logged pass: slower (includes the
    # new j_learned_after evaluation) -- the bug this round fixed used the
    # CLEAN pass's total instead of the LOGGED pass's own total, silently
    # undercounting the real instrumented cost.
    summary = {
        "rollout_solver_core_wall_clock_seconds": dict(median=0.01, p95=0.02, raw=[0.01, 0.01, 0.01]),
        "rollout_instrumented_wall_clock_with_logging_overhead_seconds": dict(
            median=0.05, p95=0.06, raw=[0.05, 0.05, 0.05]),
    }
    slib.apply_measured_audit_wall_clock(summary, audit_wall_clock_seconds=0.2)
    expected_total = 0.05 * 3 + 0.2
    check("rollout_instrumented_total_wall_clock_seconds uses the LOGGED pass's own total "
          "(0.15s), not the clean pass's smaller total (0.03s) -- the round-4 undercounting fix",
          abs(summary["rollout_instrumented_total_wall_clock_seconds"] - expected_total) < 1e-12,
          str(summary["rollout_instrumented_total_wall_clock_seconds"]))
    check("rollout_audit_counterfactual_wall_clock_seconds is set to exactly the measured audit time",
          summary["rollout_audit_counterfactual_wall_clock_seconds"] == 0.2)
    check("the clean pass's own core timing is left untouched (still reported separately as 'core')",
          summary["rollout_solver_core_wall_clock_seconds"]["raw"] == [0.01, 0.01, 0.01])


def test_battery3_singular_solve_included_in_damping_mean():
    fake_summary = dict(step_records=[
        dict(step_index=0, iterations=[
            dict(iteration_index=0, accepted=True, lam_before=0.01, accepted_alpha=1.0,
                singular_solve=False),
            dict(iteration_index=1, accepted=False, lam_before=100.0, accepted_alpha=None,
                singular_solve=True),  # the hardest damping case -- must not be dropped
            dict(iteration_index=2, accepted=False, lam_before=0.02, accepted_alpha=None,
                singular_solve=False),
        ]),
    ])
    stats = drv._battery3_ic_iteration_stats(fake_summary)
    check("mean_lam_before INCLUDES the singular-solve iteration's lam_before "
          "(the hardest damping case, previously silently dropped)",
          abs(stats["mean_lam_before"] - np.mean([0.01, 100.0, 0.02])) < 1e-9, str(stats))
    check("singular_solve_frac is reported separately (1 of 3 iterations here)",
          abs(stats["singular_solve_frac"] - (1 / 3)) < 1e-12)
    check("rejection_frac denominator still EXCLUDES the singular-solve iteration "
          "(1 rejection of 2 non-singular attempts)",
          abs(stats["rejection_frac"] - 0.5) < 1e-12, str(stats))


def test_battery5_worst_case_not_diluted_by_seed_mean(scratch: Path):
    out_dir = scratch / "b5_worst_case_out"
    out_dir.mkdir(parents=True)

    def _row(ic_idx, dev1, dev2):
        return dict(initial_condition_index=ic_idx,
                    summary=dict(final_V=1.0, max_V=2.0, success=True, solver_completed=True),
                    state_constraint=dict(
                        per_temperature_index={
                            "1": dict(max_temperature_deviation_K=dev1, max_temperature_absolute_K=dev1 + 402.0,
                                     temperature_violation_frequency=0.0, temperature_violation_integral=0.0),
                            "3": dict(max_temperature_deviation_K=dev2, max_temperature_absolute_K=dev2 + 402.0,
                                     temperature_violation_frequency=0.0, temperature_violation_integral=0.0),
                        },
                        input_saturation_frequency={"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0},
                        input_box_satisfied=True, input_never_saturated=True,
                        temperature_cap_satisfied_posthoc=True))

    # Seed 0: one IC spikes to 200K deviation, the other 4 stay at 10K --
    # seed-level MEAN = (200+10*4)/5 = 48K, hiding the genuine 200K worst case.
    payload0 = dict(state_constraint_audit=[_row(0, 200.0, 10.0)] + [_row(i, 10.0, 10.0) for i in range(1, 5)])
    payload1 = dict(state_constraint_audit=[_row(i, 10.0, 10.0) for i in range(5)])
    records = {
        ("value_only", 0): dict(payload=payload0, self_hash="0" * 64),
        ("value_only", 1): dict(payload=payload1, self_hash="1" * 64),
    }
    summary = drv.build_battery5_summary_manifest(records, [0, 1], out_dir, "0" * 64)
    _, payload = sio.read_and_verify_json_artifact(
        summary["path"], expected_kind="solver_audit_battery5_summary",
        expected_keys=drv.EXPECTED_BATTERY5_SUMMARY_PAYLOAD_KEYS)
    seed_first_max = payload["descriptive"]["value_only"]["max_temperature_deviation_reactor1"]["max"]
    check("the seed-first descriptive max (mean-of-5-ICs-per-seed, then max-over-seeds) is DILUTED "
          "and does NOT report the genuine 200K spike", abs(seed_first_max - 48.0) < 1e-9, str(seed_first_max))
    worst = payload["worst_case"]["value_only"]["reactor1"]["worst_max_temperature_deviation_K"]
    check("the SEPARATE worst_case field reports the TRUE worst-case 200K spike across all 10 rollouts, "
          "not diluted by seed-level averaging", abs(worst - 200.0) < 1e-9, str(worst))
    check("worst_case reports n_rollouts = 10 (2 seeds x 5 ICs)",
          payload["worst_case"]["value_only"]["reactor1"]["n_rollouts"] == 10)
    worst_abs = payload["worst_case"]["value_only"]["reactor1"]["worst_max_temperature_absolute_K"]
    check("worst_case ALSO reports the absolute-K maximum (missing from the summary entirely before "
          "this round's fix)", abs(worst_abs - (200.0 + 402.0)) < 1e-9, str(worst_abs))


def test_freeze_manifest_drift_detection(scratch: Path):
    old_required = drv.REQUIRED_UPSTREAM_FILES
    old_smoke_path = drv.SMOKE_RESULTS_PATH
    fake_dep = scratch / "fake_dep.py"
    fake_dep.write_text("X = 1\n")
    fake_smoke_results = scratch / "fake_smoke_results.json"
    sio.atomic_write_json_no_overwrite(fake_smoke_results, dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(), n_pass=1, n_total=1, all_passed=True,
        checks=[{"name": "x", "passed": True, "detail": ""}]))
    try:
        drv.REQUIRED_UPSTREAM_FILES = [fake_dep, fake_smoke_results]
        drv.SMOKE_RESULTS_PATH = fake_smoke_results
        freeze_path = drv.build_freeze_manifest(dest_path=scratch / "fake_freeze.json")
        verified_hash = drv.verify_freeze_manifest(freeze_path)
        check("solver-audit Freeze Manifest builds and verifies on a passing fabricated smoke record",
              isinstance(verified_hash, str) and len(verified_hash) == 64)

        fake_dep.write_text("X = 2\n")  # mutate a covered file after freezing
        drifted = False
        try:
            drv.verify_freeze_manifest(freeze_path)
        except RuntimeError as e:
            drifted = "FREEZE_DRIFT" in str(e)
        check("solver-audit Freeze Manifest verification detects post-freeze file drift", drifted)
    finally:
        drv.REQUIRED_UPSTREAM_FILES = old_required
        drv.SMOKE_RESULTS_PATH = old_smoke_path


# ---------------------------------------------------------------------------
def main():
    scratch = fresh_scratch()
    try:
        test_gate_compare_dict_tolerance_contract()
        test_battery1_sanity_field_map_and_mismatch()
        test_fd_boundary_handling_contract()
        test_settling_time()
        test_state_constraint_stats_hand_computed()
        test_counterfactual_true_objective_matches_manual()
        test_single_cstr_rollout_matches_frozen_closed_loop()
        test_residual_bridge_enrichment_real_sim()
        test_ggn_gate3_matches_original()
        test_battery1_battery2_dedup_contract(scratch)
        test_configure_deterministic_single_thread()
        test_checkpoint_lookup(scratch)
        test_real_single_cstr_battery123_e2e_reduced_scale(scratch)
        test_two_cstr_battery5_four_phase_flow(scratch)
        test_setup_manifest_round_trip(scratch)
        test_two_cstr_setup_manifest_round_trip(scratch)
        test_battery_record_schemas_round_trip(scratch)
        test_objective_decrease_metrics_distinguish_improvement_from_regression()
        test_survivorship_uses_overall_rate_not_max_per_seed_gap()
        test_safe_evaluate_contrast_incomplete_evidence()
        test_summary_companion_npz_tamper_detection(scratch)
        test_apply_measured_audit_wall_clock_uses_logged_pass_total()
        test_battery3_singular_solve_included_in_damping_mean()
        test_battery5_worst_case_not_diluted_by_seed_mean(scratch)
        test_freeze_manifest_drift_detection(scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    n_pass = sum(1 for c in CHECKS if c["passed"])
    n_total = len(CHECKS)
    print(f"\n{n_pass}/{n_total} checks passed")

    out_path = RESULTS_DIR / "solver_audit_synthetic_smoke_results_2026_09_01.json"
    if out_path.exists():
        out_path.unlink()
    sio.atomic_write_json_no_overwrite(out_path, {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_pass": n_pass, "n_total": n_total,
        "all_passed": n_pass == n_total,
        "checks": CHECKS,
    })

    if n_pass != n_total:
        raise SystemExit(f"FAILED: {n_total - n_pass} check(s) did not pass")


if __name__ == "__main__":
    main()
