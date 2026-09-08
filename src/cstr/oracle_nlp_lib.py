"""Core evaluator logic for the Oracle-Gradient / Converged-NLP
comparison protocol (`docs/JPC_ORACLE_NLP_COMPARISON_PROTOCOL_2026_09_03.md`,
v11). Zero edits to any of the 12 go/no-go-ablation frozen files or the
11 solver-audit frozen files -- this module only imports/reuses them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import casadi as ca
import numpy as np
import torch

from cstr.lcnn_paper_simulator import LCNNPaperCSTRParams, LCNNPaperCSTRSimulator
from cstr.solver_audit_lib import compute_settling_time
from cstr.oracle_nlp_io import (
    ALPHA_KKT,
    FD_EPS,
    FLOOR_MARGIN_K,
    GATE12_ERR_TOL,
    GATE3_ERR_TOL,
    TOL_DEDUP_INPUT,
    TOL_KKT,
    TOL_PRIMAL,
    TOL_TIE_OBJECTIVE,
    err,
    flatten_leaves,
    gate_compare_flattened,
)

# ---------------------------------------------------------------------------
# Sec. 3 -- shared literal constants, confirmed by direct code read against
# scripts/solver_audit_driver_2026_09_01.py:61 and budget_sweep_solver.py.
# ---------------------------------------------------------------------------
P_MAT = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float64)
INPUT_LO = np.array([-3.5, -5.0e5], dtype=np.float64)
INPUT_HI = np.array([3.5, 5.0e5], dtype=np.float64)
RHO_U = 0.01
SUCCESS_V = 2.0
HORIZON = 3
STEPS = 120
BUDGET = 20
LR = 0.1
DT_HR = 1e-3
INTEGRATION_SUBSTEPS = 100

_PARAMS = LCNNPaperCSTRParams()


# ---------------------------------------------------------------------------
# Sec. 6 -- J_true: explicit inline unroll of 300 RK4 integration steps
# (100 substeps/control-step x HORIZON=3), each with 4 RHS/stage
# evaluations (k1-k4) = 1,200 RHS evaluations total, matching
# `LCNNPaperCSTRSimulator._rk4_single`/`step()` exactly, including the
# `temp = ca.fmax(T, 1.0)` floor (`rhs_absolute`, `lcnn_paper_simulator.py:117`
# has `max(float(x_abs[1]), 1.0)` -- NOT fully smooth, must be represented).
# ---------------------------------------------------------------------------
def _rhs_ca(x: ca.MX, u: ca.MX) -> tuple:
    """Shifted-coordinate RHS, mirroring `rhs_absolute`/`_dynamics` exactly.
    Returns `(dx, temp_abs)` -- `temp_abs` is the RAW (pre-floor) absolute
    temperature, needed for the floor-inactivity invariant (Sec. 6): the
    floor is only ever a numerical guard against a pathological sub-1K
    excursion, so tracking how close `temp_abs` comes to 1.0 across every
    RHS evaluation is exactly what "was the fmax kink touched" means."""
    p = _PARAMS
    ca_abs = x[0] + p.CAs
    temp_abs = x[1] + p.Ts
    ca0_abs = u[0] + p.CA0s
    q_abs = u[1] + p.Qs
    temp_floored = ca.fmax(temp_abs, 1.0)
    k = p.k0 * ca.exp(-p.E / (p.R * temp_floored))
    reaction = k * ca_abs * ca_abs
    dca = (p.F / p.V) * (ca0_abs - ca_abs) - reaction
    dtemp = (
        (p.F / p.V) * (p.T0 - temp_abs)
        - (p.dH / (p.rhoL * p.Cp)) * reaction
        + q_abs / (p.rhoL * p.Cp * p.V)
    )
    return ca.vertcat(dca, dtemp), temp_abs


def _rk4_step_ca(x: ca.MX, u: ca.MX, h: float) -> tuple:
    """One RK4 integration step (k1-k4), mirroring `_rk4_single` exactly.
    Returns `(x_next, min_temp_over_4_stages)`."""
    k1, t1 = _rhs_ca(x, u)
    k2, t2 = _rhs_ca(x + 0.5 * h * k1, u)
    k3, t3 = _rhs_ca(x + 0.5 * h * k2, u)
    k4, t4 = _rhs_ca(x + h * k3, u)
    x_next = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    min_temp = ca.fmin(ca.fmin(t1, t2), ca.fmin(t3, t4))
    return x_next, min_temp


def _one_control_step_ca(x: ca.MX, u: ca.MX, dt_hr: float = DT_HR,
                          substeps: int = INTEGRATION_SUBSTEPS) -> tuple:
    """100 RK4 integration steps = one `sim.step()` call, mirroring
    `LCNNPaperCSTRSimulator.step()` exactly. Returns `(y, min_temp_over_
    400_evals)`."""
    h = dt_hr / substeps
    y = x
    min_temp = None
    for i in range(substeps):
        y, mt = _rk4_step_ca(y, u, h)
        min_temp = mt if i == 0 else ca.fmin(min_temp, mt)
    return y, min_temp


def build_one_step_function() -> ca.Function:
    """Sec. 5.3 gate 1: `f(x, u) -> (y, min_temp)`, shifted coordinates,
    for comparison against `LCNNPaperCSTRSimulator.step()`."""
    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u", 2)
    y, min_temp = _one_control_step_ca(x, u)
    return ca.Function("one_step", [x, u], [y, min_temp])


@dataclass
class JTrueBundle:
    """`J_true(x0, U_n) -> (J, min_temp)` plus its exact AD gradient
    `grad_J_true(x0, U_n) -> (grad, J, min_temp)`, built ONCE (Sec. 9's
    object-construction contract) and shared by Oracle-20 and
    Certified-NLP. `U_n` has shape `(HORIZON, 2)` (normalized input);
    `x0` has shape `(2,)` (physical/shifted state, matching
    `counterfactual_true_objective`'s own `x_phys` argument -- no state
    normalization is used anywhere in the true objective, only the
    learned model's forward pass needs that)."""

    j_fn: ca.Function
    grad_fn: ca.Function
    u_mean: np.ndarray
    u_std: np.ndarray
    horizon: int


def build_j_true(u_mean: np.ndarray, u_std: np.ndarray, horizon: int = HORIZON,
                  rho_u: float = RHO_U) -> JTrueBundle:
    """Builds the shared `J_true`/gradient bundle. `u_mean`/`u_std` are the
    pool's input normalization constants (`xm[2:]`/`xs[2:]`, Sec. 3)."""
    x0 = ca.MX.sym("x0", 2)
    Un = ca.MX.sym("Un", horizon, 2)
    u_mean_ca = ca.DM(np.asarray(u_mean, dtype=np.float64))
    u_std_ca = ca.DM(np.asarray(u_std, dtype=np.float64))
    P_ca = ca.DM(P_MAT)

    xt = x0
    J = ca.MX(0.0)
    min_temp_all = None
    for h in range(horizon):
        un_h = Un[h, :].T  # (1,2) row -> (2,1) column
        u_phys = un_h * u_std_ca + u_mean_ca
        y, min_temp_step = _one_control_step_ca(xt, u_phys)
        J = J + ca.mtimes([y.T, P_ca, y]) + rho_u * ca.sumsqr(un_h)
        xt = y
        min_temp_all = min_temp_step if h == 0 else ca.fmin(min_temp_all, min_temp_step)

    j_fn = ca.Function("J_true", [x0, Un], [J, min_temp_all])
    grad_J = ca.gradient(J, Un)
    grad_fn = ca.Function("J_true_grad", [x0, Un], [grad_J, J, min_temp_all])
    return JTrueBundle(j_fn=j_fn, grad_fn=grad_fn, u_mean=np.asarray(u_mean, dtype=np.float64),
                        u_std=np.asarray(u_std, dtype=np.float64), horizon=horizon)


def j_true_value(bundle: JTrueBundle, x0: np.ndarray, Un: np.ndarray) -> tuple:
    """Returns `(J: float, min_temp: float)`."""
    J, min_temp = bundle.j_fn(np.asarray(x0, dtype=np.float64).reshape(2),
                               np.asarray(Un, dtype=np.float64).reshape(bundle.horizon, 2))
    return float(J), float(min_temp)


def j_true_grad(bundle: JTrueBundle, x0: np.ndarray, Un: np.ndarray) -> tuple:
    """Returns `(grad: (HORIZON,2) float64 ndarray, J: float, min_temp: float)`."""
    grad, J, min_temp = bundle.grad_fn(np.asarray(x0, dtype=np.float64).reshape(2),
                                        np.asarray(Un, dtype=np.float64).reshape(bundle.horizon, 2))
    grad_np = np.asarray(grad).reshape(bundle.horizon, 2)
    return grad_np, float(J), float(min_temp)


def floor_active(min_temp: float, margin: float = FLOOR_MARGIN_K) -> bool:
    """Sec. 6: `floor_active = (minimum_stage_temperature <= 1.0 + margin)`."""
    return bool(min_temp <= 1.0 + margin)


# ---------------------------------------------------------------------------
# Sec. 5.1 -- fixed compatibility sample. Index list/content fixed at
# design time, never selected post-hoc.
# ---------------------------------------------------------------------------
def build_compat_sample(ics: list, un_lo: np.ndarray, un_hi: np.ndarray) -> dict:
    """5 ICs x 3 fixed control vectors (zero, un_lo, un_hi) = 15 one-step
    test points; 5 ICs each paired with the all-zero U_n plan = 5
    multi-step test points."""
    ics_arr = np.stack([np.asarray(ic, dtype=np.float64) for ic in ics])  # (5,2)
    control_vectors = np.stack([np.zeros(2), np.asarray(un_lo, dtype=np.float64),
                                 np.asarray(un_hi, dtype=np.float64)])  # (3,2)
    one_step_x0 = np.repeat(ics_arr, 3, axis=0)  # (15,2)
    one_step_u = np.tile(control_vectors, (5, 1))  # (15,2)
    multi_step_x0 = ics_arr  # (5,2)
    multi_step_U = np.zeros((5, HORIZON, 2), dtype=np.float64)
    return dict(
        one_step_x0=one_step_x0, one_step_u=one_step_u,
        multi_step_x0=multi_step_x0, multi_step_U=multi_step_U,
        un_lo=np.asarray(un_lo, dtype=np.float64), un_hi=np.asarray(un_hi, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Sec. 5.3 gates 1-3.
# ---------------------------------------------------------------------------
def gate1_symbolic_vs_numeric_one_step(one_step_fn: ca.Function, sim: LCNNPaperCSTRSimulator,
                                        sample: dict) -> dict:
    """`err<=GATE12_ERR_TOL` on all 15 fixed one-step points."""
    worst = 0.0
    per_point = []
    for i in range(sample["one_step_x0"].shape[0]):
        x0 = sample["one_step_x0"][i]
        u0 = sample["one_step_u"][i]
        y_ca, _ = one_step_fn(x0, u0)
        y_ca = np.asarray(y_ca).reshape(2)
        y_np = sim.step(x0, u0)
        e = err(y_ca, y_np)
        per_point.append(e)
        worst = max(worst, e)
    return dict(passed=bool(worst <= GATE12_ERR_TOL), measured_error=worst, per_point_error=per_point)


def gate2_symbolic_vs_numeric_j_true(bundle: JTrueBundle, counterfactual_true_objective_fn,
                                      sim: LCNNPaperCSTRSimulator, sample: dict, xm, xs, ym, ys) -> dict:
    """`err<=GATE12_ERR_TOL` on all 5 fixed multi-step points, against
    `counterfactual_true_objective()` (`solver_audit_lib.py:135`)."""
    worst = 0.0
    per_point = []
    for i in range(sample["multi_step_x0"].shape[0]):
        x0 = sample["multi_step_x0"][i]
        Un = sample["multi_step_U"][i]
        J_ca, _ = j_true_value(bundle, x0, Un)
        J_np = counterfactual_true_objective_fn(sim, x0, Un.reshape(-1), HORIZON, RHO_U, xm, xs, ym, ys)
        e = err(J_ca, J_np)
        per_point.append(e)
        worst = max(worst, e)
    return dict(passed=bool(worst <= GATE12_ERR_TOL), measured_error=worst, per_point_error=per_point)


def _central_fd_gradient(bundle: JTrueBundle, x0: np.ndarray, Un: np.ndarray,
                          eps: float = FD_EPS) -> np.ndarray:
    grad = np.zeros_like(Un, dtype=np.float64)
    for h in range(Un.shape[0]):
        for a in range(Un.shape[1]):
            up = Un.copy(); up[h, a] += eps
            dn = Un.copy(); dn[h, a] -= eps
            Jp, _ = j_true_value(bundle, x0, up)
            Jm, _ = j_true_value(bundle, x0, dn)
            grad[h, a] = (Jp - Jm) / (2 * eps)
    return grad


def j_true_learned_style_value(bundle: JTrueBundle, x0: np.ndarray, Un: np.ndarray) -> float:
    """`J_true` evaluated at an arbitrary plan, mirroring
    `learned_objective_value()`'s role for B/E (`solver_audit_lib.py:151`)
    -- used to compute `j_true_after` (post-clamp) for a genuine signed
    before/after decrease, Sec. 12a's `final_V`-based analysis notwithstanding
    (this is a per-ITERATION diagnostic, distinct from the rollout-level
    `final_V`)."""
    J, _ = j_true_value(bundle, x0, Un)
    return J


# ---------------------------------------------------------------------------
# Sec. 7 -- Oracle-20 controller. Structurally identical to
# `_adam_mpc_step`/`single_cstr_rollout` (`solver_audit_lib.py:274-440`):
# same optimizer, same LR/BUDGET/HORIZON, same clamp-after-step order, same
# deterministic zero-init + shift-repeat warm start -- the ONLY difference
# is the gradient source (CasADi AD of `J_true` instead of autograd
# through a learned model).
# ---------------------------------------------------------------------------
def true_gradient_source(bundle: JTrueBundle):
    """Sec. 7's actual gradient source: CasADi exact AD of `J_true`.
    Returns a callable `(x_phys, un_np) -> dict(grad, J, min_temp)`."""
    def _source(x_phys: np.ndarray, un_np: np.ndarray) -> dict:
        grad, J_val, min_temp = j_true_grad(bundle, x_phys, un_np)
        return dict(grad=grad, J=J_val, min_temp=min_temp)
    return _source


def learned_gradient_source(seq, norm, horizon: int = HORIZON, rho_u: float = RHO_U):
    """Sec. 5.3 Gate 4: TEMPORARILY swaps Oracle-20's gradient source for
    autograd through a LEARNED model `seq` (e.g. condition B's frozen
    seed-0 checkpoint) -- mirrors `_adam_mpc_step`'s own inner-loop J
    computation exactly (`solver_audit_lib.py:274-325`), so that feeding
    this into `oracle20_step` should reproduce `single_cstr_rollout`'s
    own frozen result bit-for-bit (within `values_match()` tolerance).
    `min_temp` is `None` (the learned model has no floor concept) --
    `oracle20_step` treats a `None` min_temp as never floor-active."""
    xm, xs, ym, ys = norm
    xm2, xs2 = xm[:2], xs[:2]
    Pt_torch = torch.tensor(P_MAT, dtype=torch.float32)

    def _source(x_phys: np.ndarray, un_np: np.ndarray) -> dict:
        un_t = torch.as_tensor(un_np, dtype=torch.float32).requires_grad_(True)
        x0n = torch.tensor((np.asarray(x_phys, dtype=float) - xm2.numpy()) / xs2.numpy(),
                            dtype=torch.float32).view(1, 2)
        xt = x0n
        J = torch.zeros(())
        for h in range(horizon):
            xun = torch.cat([xt, un_t[h].view(1, 2)], dim=1)
            y = seq(xun) * ys + ym
            J = J + (y @ Pt_torch * y).sum() + rho_u * (un_t[h] ** 2).sum()
            xt = (y - xm2) / xs2
        J.backward()
        grad_np = un_t.grad.detach().numpy().copy()
        return dict(grad=grad_np, J=float(J.detach()), min_temp=None)
    return _source


def oracle20_step(gradient_source, x_phys: np.ndarray, un_warm_start: torch.Tensor,
                   budget: int, lr: float, un_lo: torch.Tensor, un_hi: torch.Tensor,
                   j_true_bundle: Optional[JTrueBundle] = None, record_iterations: bool = False) -> dict:
    """One control step's projected-Adam solve. Mirrors `_adam_mpc_step`
    exactly except: (1) the gradient comes from `gradient_source(x_phys,
    un_np) -> dict(grad, J, min_temp)`, manually assigned to `un.grad`,
    replacing `J.backward()`'s implicit autograd population (Gate 4 swaps
    this for a learned-model source to validate the harness itself,
    Sec. 5.3); (2) `x_phys` is NOT normalized anywhere the true gradient
    source is used (J_true operates in physical/shifted coordinates
    directly, Sec. 6) -- the learned-model source normalizes internally,
    matching `_adam_mpc_step`. `j_true_bundle`, if given, is used ONLY to
    compute the `j_true_after` diagnostic for instrumented logging (Sec.
    7), independent of which gradient source drove the update -- when
    `None` (as under Gate 4's learned-gradient swap), that diagnostic is
    simply omitted."""
    un = un_warm_start.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([un], lr=lr)
    iteration_records = [] if record_iterations else None
    any_floor_active = False
    any_non_finite = False
    last_min_temp = None
    for _it in range(budget):
        un_before = un.detach().clone() if record_iterations else None
        opt.zero_grad()
        grad_info = gradient_source(x_phys, un.detach().numpy())
        grad_np, J_val, min_temp = grad_info["grad"], grad_info["J"], grad_info.get("min_temp")
        if not (np.all(np.isfinite(grad_np)) and np.isfinite(J_val)):
            any_non_finite = True
            break
        last_min_temp = min_temp
        step_floor_active = floor_active(min_temp) if min_temp is not None else False
        if step_floor_active:
            any_floor_active = True
        un.grad = torch.as_tensor(grad_np, dtype=torch.float32)
        opt.step()
        un_raw = un.detach().clone() if record_iterations else None
        with torch.no_grad():
            un.clamp_(un_lo, un_hi)
        if record_iterations:
            un_projected = un.detach().clone()
            j_true_after = (j_true_learned_style_value(j_true_bundle, x_phys, un_projected.numpy())
                             if j_true_bundle is not None else None)
            iteration_records.append(dict(
                un_before=un_before.numpy().copy(), j_true=J_val, j_true_after=j_true_after,
                un_raw=un_raw.numpy().copy(), un_projected=un_projected.numpy().copy(),
                minimum_stage_temperature=min_temp, floor_active=step_floor_active,
            ))
    u0n = un[0].detach().numpy().copy()
    return dict(u0n=u0n, un_final=un.detach().clone(), any_floor_active=any_floor_active,
                any_non_finite=any_non_finite, iteration_records=iteration_records,
                last_minimum_stage_temperature=last_min_temp)


def oracle20_rollout(gradient_source, x0: np.ndarray, steps: int, budget: int, lr: float,
                      success_v: float, un_lo: torch.Tensor, un_hi: torch.Tensor,
                      sim: LCNNPaperCSTRSimulator, input_lo: np.ndarray, input_hi: np.ndarray,
                      u_mean: np.ndarray, u_std: np.ndarray, horizon: int = HORIZON,
                      j_true_bundle: Optional[JTrueBundle] = None, record_iterations: bool = False) -> dict:
    """Mirrors `single_cstr_rollout` (`solver_audit_lib.py:335-440`)
    exactly, operating on physical/shifted state throughout (no state
    normalization when `gradient_source` is `true_gradient_source`, Sec.
    6). Deterministic zero-init + shift-repeat warm start (Sec. 3) -- no
    seed axis. `gradient_source` is pluggable (Sec. 5.3 Gate 4: swap in
    `learned_gradient_source(...)` to validate this harness against a
    frozen B/E rollout before trusting `true_gradient_source(bundle)`'s
    output)."""
    x = np.asarray(x0, dtype=float)
    un = torch.zeros(horizon, 2)
    Vtraj = [float(x @ P_MAT @ x)]
    x_traj = [x.copy()]
    us = []
    step_wall_clock = []
    step_records = [] if record_iterations else None
    # Populated regardless of `record_iterations` -- the "minimum needed to
    # determine floor_active post-hoc" the clean pass still requires for the
    # endpoint-equivalence gate (Sec. 7) and reference/gap extraction
    # (Sec. 12a), WITHOUT the heavy per-iteration Adam logs `step_records`
    # carries only when `record_iterations=True`.
    light_step_records = []
    solver_completed = True
    failure_reason = None
    failed_at_step = None
    any_floor_active_rollout = False
    try:
        for step_i in range(steps):
            warm_start_un = un.detach().clone() if record_iterations else None
            t0 = _now()
            result = oracle20_step(gradient_source, x, un, budget, lr, un_lo, un_hi,
                                    j_true_bundle=j_true_bundle, record_iterations=record_iterations)
            t1 = _now()
            step_wall_clock.append(t1 - t0)
            if result["any_non_finite"]:
                solver_completed = False
                failure_reason = "ORACLE20_NON_FINITE_GRADIENT_OR_OBJECTIVE"
                failed_at_step = len(us)
                break
            if result["any_floor_active"]:
                any_floor_active_rollout = True
            u0n = result["u0n"]
            un_next = result["un_final"]
            u0 = np.clip(u0n * u_std + u_mean, input_lo, input_hi)
            x_before = x.copy()
            V_before = float(x_before @ P_MAT @ x_before)
            x = sim.step(x, u0)
            V_after = float(x @ P_MAT @ x)
            x_traj.append(x.copy())
            Vtraj.append(V_after)
            us.append(u0)
            light_step_records.append(dict(
                step_index=step_i, u0=u0, final_plan_un=un_next.numpy().copy(), x_after=x.copy(),
                V_after=V_after, minimum_stage_temperature=result["last_minimum_stage_temperature"],
                floor_active=result["any_floor_active"],
            ))
            if record_iterations:
                step_records.append(dict(
                    step_index=step_i, x_before=x_before, u0=u0,
                    warm_start_un=warm_start_un.numpy().copy(), final_plan_un=un_next.numpy().copy(),
                    x_after=x.copy(), V_before=V_before, V_after=V_after,
                    realized_decrease=V_before - V_after,
                    solve_wall_clock_seconds=t1 - t0,
                    any_floor_active=result["any_floor_active"],
                    iterations=result["iteration_records"],
                ))
            un = torch.cat([un_next[1:], un_next[-1:]])
    except Exception as e:  # pragma: no cover -- mirrors solver_audit_lib's SIMULATOR_FAILURE_EXCEPTIONS pattern
        solver_completed = False
        failure_reason = repr(e)
        failed_at_step = len(us)

    n_steps_completed = len(us)
    lyapunov_increase_count = int(sum(Vtraj[i + 1] > Vtraj[i] + 1e-9 for i in range(len(Vtraj) - 1)))
    us_arr = np.asarray(us) if us else np.zeros((0, 2))
    tv = float(np.sum(np.abs(np.diff(us_arr / (input_hi - input_lo), axis=0)))) if len(us_arr) > 1 else 0.0
    settling_time = compute_settling_time(Vtraj, success_v, steps) if solver_completed else steps + 1

    # Sec. 10: reference validity requires ALL 120 steps completed, no
    # non-finite iterate anywhere, and non-floor-active throughout.
    reference_valid = bool(solver_completed and n_steps_completed == steps and not any_floor_active_rollout)

    summary = dict(
        final_V=(Vtraj[-1] if solver_completed else float("nan")),
        max_V=float(max(Vtraj)),
        success=bool(solver_completed and Vtraj[-1] <= success_v),
        lyapunov_increase_count=lyapunov_increase_count,
        tv=tv,
        settling_time=settling_time,
        solver_completed=solver_completed,
        failure_reason=failure_reason,
        failed_at_step=failed_at_step,
        n_steps_completed=n_steps_completed,
        any_floor_active=any_floor_active_rollout,
        reference_valid=reference_valid,
        rollout_solver_core_wall_clock_seconds=dict(
            median=float(np.median(step_wall_clock)) if step_wall_clock else float("nan"),
            p95=float(np.percentile(step_wall_clock, 95)) if step_wall_clock else float("nan"),
            raw=step_wall_clock,
        ),
    )
    return dict(summary=summary, x_traj=x_traj, us=us, Vtraj=Vtraj, step_records=step_records,
                light_step_records=light_step_records)


def _now() -> float:
    import time
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Sec. 8-9 -- Certified-NLP controller. U-only single shooting, IPOPT,
# 3 fixed deterministic starts, generic per-step dedup, self-computed
# certification (status whitelist AND primal feasibility AND KKT AND
# finite AND non-floor-active).
# ---------------------------------------------------------------------------
IPOPT_OPTIONS = {
    "print_time": 0,
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",
    "ipopt.max_iter": 500,
    "ipopt.tol": 1e-8,
    "ipopt.acceptable_tol": 1e-6,
    "ipopt.acceptable_iter": 15,
    "ipopt.hessian_approximation": "exact",
    "ipopt.linear_solver": "mumps",
    "ipopt.mu_strategy": "adaptive",
}


@dataclass
class CertifiedNLPSolver:
    opti: ca.Opti
    x0p: ca.MX
    Un: ca.MX
    un_lo: np.ndarray
    un_hi: np.ndarray
    horizon: int


def build_certified_nlp_solver(bundle: JTrueBundle, un_lo: np.ndarray, un_hi: np.ndarray) -> CertifiedNLPSolver:
    """Sec. 9's object-construction contract: `Opti` built exactly ONCE,
    reused via `set_value`/`set_initial` for every candidate/step/IC. Same
    `J_true` formula as `build_j_true`, tied to the Opti's own `x0p`
    parameter / `Un` variable rather than `build_j_true`'s standalone
    `ca.Function` (a KKT-residual check reuses `bundle.grad_fn` separately,
    numerically, after the solve -- no need to duplicate the gradient
    inside the Opti graph itself)."""
    horizon = bundle.horizon
    opti = ca.Opti()
    x0p = opti.parameter(2)
    Un = opti.variable(horizon, 2)
    u_mean_ca = ca.DM(bundle.u_mean)
    u_std_ca = ca.DM(bundle.u_std)
    P_ca = ca.DM(P_MAT)
    xt = x0p
    J = ca.MX(0.0)
    for h in range(horizon):
        un_h = Un[h, :].T
        u_phys = un_h * u_std_ca + u_mean_ca
        y, _ = _one_control_step_ca(xt, u_phys)
        J = J + ca.mtimes([y.T, P_ca, y]) + RHO_U * ca.sumsqr(un_h)
        xt = y
    opti.minimize(J)
    un_lo_arr = np.asarray(un_lo, dtype=np.float64)
    un_hi_arr = np.asarray(un_hi, dtype=np.float64)
    for h in range(horizon):
        for a in range(2):
            opti.subject_to(opti.bounded(float(un_lo_arr[a]), Un[h, a], float(un_hi_arr[a])))
    opti.solver("ipopt", IPOPT_OPTIONS)
    return CertifiedNLPSolver(opti=opti, x0p=x0p, Un=Un, un_lo=un_lo_arr, un_hi=un_hi_arr, horizon=horizon)


def certified_nlp_startup_assertion(un_lo: np.ndarray, un_hi: np.ndarray) -> dict:
    """Sec. 8: checked ONCE at protocol startup, BEFORE any rollout.
    Covers ONLY the two step-invariant fixed vectors (zero, quartile) --
    NOT all three starts pairwise, since warm-shifted is expected to
    coincide with zero at a rollout's first step by design."""
    un_lo = np.asarray(un_lo, dtype=np.float64)
    un_hi = np.asarray(un_hi, dtype=np.float64)
    zero_vec = np.zeros(2)
    quartile_vec = un_lo + 0.75 * (un_hi - un_lo)
    dist = float(np.max(np.abs(zero_vec - quartile_vec)))
    zero_inside = bool(np.all(zero_vec > un_lo) and np.all(zero_vec < un_hi))
    quartile_inside = bool(np.all(quartile_vec > un_lo) and np.all(quartile_vec < un_hi))
    dist_ok = bool(dist > TOL_DEDUP_INPUT)
    passed = bool(zero_inside and quartile_inside and dist_ok)
    if not passed:
        raise AssertionError(
            f"CERTIFIED_NLP_STARTUP_ASSERTION_FAILED: zero_inside={zero_inside} "
            f"quartile_inside={quartile_inside} dist={dist} (tol={TOL_DEDUP_INPUT})"
        )
    return dict(passed=passed, zero_vec=zero_vec, quartile_vec=quartile_vec, dist=dist)


def _tiled_plan(vec: np.ndarray, horizon: int) -> np.ndarray:
    return np.tile(np.asarray(vec, dtype=np.float64).reshape(1, 2), (horizon, 1))


def certified_nlp_candidate_starts(un_lo: np.ndarray, un_hi: np.ndarray, warm_shifted_plan: np.ndarray,
                                    horizon: int) -> list:
    """Sec. 8: `[zero, warm-shifted, upper-quartile]`, index order fixed
    (0=zero, 1=warm-shifted, 2=upper-quartile) for reproducible tie-break."""
    zero_plan = _tiled_plan(np.zeros(2), horizon)
    quartile_vec = np.asarray(un_lo, dtype=np.float64) + 0.75 * (np.asarray(un_hi) - np.asarray(un_lo))
    quartile_plan = _tiled_plan(quartile_vec, horizon)
    return [zero_plan, np.asarray(warm_shifted_plan, dtype=np.float64).reshape(horizon, 2), quartile_plan]


def _dedup_groups(plans: list, tol: float = TOL_DEDUP_INPUT) -> list:
    """Sec. 8: generic pairwise dedup at every control step (not just the
    first). Returns a list of index groups; only the first (lowest-index)
    member of each group is actually solved."""
    n = len(plans)
    assigned = [False] * n
    groups = []
    for i in range(n):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if np.max(np.abs(plans[i] - plans[j])) < tol:
                group.append(j)
                assigned[j] = True
        groups.append(group)
    return groups


def solve_one_candidate(nlp: CertifiedNLPSolver, bundle: JTrueBundle, x0_val: np.ndarray, plan: np.ndarray) -> dict:
    nlp.opti.set_value(nlp.x0p, np.asarray(x0_val, dtype=np.float64).reshape(2))
    nlp.opti.set_initial(nlp.Un, np.asarray(plan, dtype=np.float64).reshape(nlp.horizon, 2))
    try:
        sol = nlp.opti.solve()
        stats = sol.stats()
        U_sol = np.asarray(sol.value(nlp.Un)).reshape(nlp.horizon, 2)
    except RuntimeError:
        stats = nlp.opti.debug.stats()
        U_sol = np.asarray(nlp.opti.debug.value(nlp.Un)).reshape(nlp.horizon, 2)
    return_status = stats.get("return_status")
    iter_count = stats.get("iter_count")
    iterations = stats.get("iterations", {}) or {}
    J_val, min_temp = j_true_value(bundle, x0_val, U_sol)
    return dict(U_sol=U_sol, return_status=return_status, iter_count=iter_count,
                alpha_pr=iterations.get("alpha_pr"), alpha_du=iterations.get("alpha_du"),
                J_true=J_val, minimum_stage_temperature=min_temp)


def certify_candidate(bundle: JTrueBundle, x0_val: np.ndarray, candidate: dict,
                       un_lo: np.ndarray, un_hi: np.ndarray) -> dict:
    """Sec. 9: self-computed certification. `return_status` whitelist is
    NECESSARY but not by itself sufficient; the KKT residual is the
    authoritative stationarity criterion."""
    U_sol = candidate["U_sol"]
    finite_ok = bool(np.all(np.isfinite(U_sol)) and np.isfinite(candidate["J_true"]))
    within_box = bool(np.all(U_sol >= un_lo - TOL_PRIMAL) and np.all(U_sol <= un_hi + TOL_PRIMAL)) if finite_ok else False
    primal_ok = bool(finite_ok and within_box)
    if finite_ok:
        grad, _, _ = j_true_grad(bundle, x0_val, U_sol)
        proj = np.clip(U_sol - ALPHA_KKT * grad, un_lo, un_hi)
        kkt_residual = float(np.max(np.abs(U_sol - proj)))
    else:
        kkt_residual = float("inf")
    kkt_ok = bool(kkt_residual <= TOL_KKT)
    is_floor_active = floor_active(candidate["minimum_stage_temperature"])
    status_ok = bool(candidate["return_status"] == "Solve_Succeeded")
    would_be_certified_if_status_relaxed = bool(primal_ok and kkt_ok and finite_ok and not is_floor_active)
    certified = bool(status_ok and would_be_certified_if_status_relaxed)
    return dict(certified=certified, status_ok=status_ok, primal_ok=primal_ok, kkt_ok=kkt_ok,
                kkt_residual=kkt_residual, finite_ok=finite_ok, floor_active=is_floor_active,
                would_be_certified_if_status_relaxed=would_be_certified_if_status_relaxed)


def certified_nlp_step(nlp: CertifiedNLPSolver, bundle: JTrueBundle, x0_val: np.ndarray,
                        warm_shifted_plan: np.ndarray, record_iterations: bool = False) -> dict:
    """One control step: solve (post-dedup) candidates, certify each,
    select the lowest-`J_true` CERTIFIED, non-floor-active winner (tie-
    break `TOL_TIE_OBJECTIVE`, lowest start index wins)."""
    plans = certified_nlp_candidate_starts(nlp.un_lo, nlp.un_hi, warm_shifted_plan, nlp.horizon)
    groups = _dedup_groups(plans)
    candidates = [None] * len(plans)
    starts_deduplicated = []
    for group in groups:
        rep = group[0]
        sol = solve_one_candidate(nlp, bundle, x0_val, plans[rep])
        cert = certify_candidate(bundle, x0_val, sol, nlp.un_lo, nlp.un_hi)
        for idx in group:
            entry = dict(sol)
            entry.update(cert)
            entry["start_id"] = idx
            entry["dedup_source_start_id"] = rep
            candidates[idx] = entry
        if len(group) > 1:
            starts_deduplicated.append(list(group))
    eligible = [c for c in candidates if c["certified"]]
    winner = None
    if eligible:
        best_J = min(c["J_true"] for c in eligible)
        tied = [c for c in eligible if abs(c["J_true"] - best_J) <= TOL_TIE_OBJECTIVE]
        winner = min(tied, key=lambda c: c["start_id"])
    return dict(candidates=candidates, starts_deduplicated=starts_deduplicated,
                selected_start_id=(winner["start_id"] if winner is not None else None),
                winner=winner)


def certified_nlp_rollout(nlp: CertifiedNLPSolver, bundle: JTrueBundle, x0: np.ndarray, steps: int,
                           sim: LCNNPaperCSTRSimulator, input_lo: np.ndarray, input_hi: np.ndarray,
                           u_mean: np.ndarray, u_std: np.ndarray, success_v: float,
                           record_iterations: bool = False) -> dict:
    """Sec. 10: first uncertified step (no winner) stops the rollout
    immediately, no fallback action. Reference validity (per IC): all
    `steps` control steps completed AND every step's selected winner
    passed full certification (implied by "a winner existed at every
    step," since only certified candidates are winner-eligible)."""
    horizon = nlp.horizon
    x = np.asarray(x0, dtype=float)
    warm_shifted_plan = np.zeros((horizon, 2))
    Vtraj = [float(x @ P_MAT @ x)]
    x_traj = [x.copy()]
    us = []
    step_wall_clock = []
    step_records = [] if record_iterations else None
    light_step_records = []
    solver_completed = True
    failure_reason = None
    failed_at_step = None
    try:
        for step_i in range(steps):
            t0 = _now()
            step_result = certified_nlp_step(nlp, bundle, x, warm_shifted_plan,
                                              record_iterations=record_iterations)
            t1 = _now()
            step_wall_clock.append(t1 - t0)
            if step_result["winner"] is None:
                solver_completed = False
                failure_reason = "CERTIFIED_NLP_NO_CERTIFIED_CANDIDATE"
                failed_at_step = len(us)
                break
            winner = step_result["winner"]
            U_sol = winner["U_sol"]
            u0n = U_sol[0]
            u0 = np.clip(u0n * u_std + u_mean, input_lo, input_hi)
            x_before = x.copy()
            V_before = float(x_before @ P_MAT @ x_before)
            x = sim.step(x, u0)
            V_after = float(x @ P_MAT @ x)
            x_traj.append(x.copy())
            Vtraj.append(V_after)
            us.append(u0)
            light_step_records.append(dict(
                step_index=step_i, u0=u0, final_plan_un=U_sol.copy(), x_after=x.copy(), V_after=V_after,
                minimum_stage_temperature=winner["minimum_stage_temperature"],
                floor_active=winner["floor_active"], selected_start_id=winner["start_id"],
                J_true=winner["J_true"], certified=winner["certified"],
            ))
            if record_iterations:
                step_records.append(dict(
                    step_index=step_i, x_before=x_before, u0=u0, x_after=x.copy(),
                    V_before=V_before, V_after=V_after, realized_decrease=V_before - V_after,
                    selected_start_id=winner["start_id"], J_true=winner["J_true"],
                    minimum_stage_temperature=winner["minimum_stage_temperature"],
                    floor_active=winner["floor_active"],
                    candidates=step_result["candidates"], starts_deduplicated=step_result["starts_deduplicated"],
                    solve_wall_clock_seconds=t1 - t0,
                ))
            warm_shifted_plan = np.concatenate([U_sol[1:], U_sol[-1:]], axis=0)
    except Exception as e:  # pragma: no cover
        solver_completed = False
        failure_reason = repr(e)
        failed_at_step = len(us)

    n_steps_completed = len(us)
    lyapunov_increase_count = int(sum(Vtraj[i + 1] > Vtraj[i] + 1e-9 for i in range(len(Vtraj) - 1)))
    us_arr = np.asarray(us) if us else np.zeros((0, 2))
    tv = float(np.sum(np.abs(np.diff(us_arr / (input_hi - input_lo), axis=0)))) if len(us_arr) > 1 else 0.0
    settling_time = compute_settling_time(Vtraj, success_v, steps) if solver_completed else steps + 1
    reference_valid = bool(solver_completed and n_steps_completed == steps)

    summary = dict(
        final_V=(Vtraj[-1] if solver_completed else float("nan")),
        max_V=float(max(Vtraj)),
        success=bool(solver_completed and Vtraj[-1] <= success_v),
        lyapunov_increase_count=lyapunov_increase_count,
        tv=tv,
        settling_time=settling_time,
        solver_completed=solver_completed,
        failure_reason=failure_reason,
        failed_at_step=failed_at_step,
        n_steps_completed=n_steps_completed,
        reference_valid=reference_valid,
        rollout_solver_core_wall_clock_seconds=dict(
            median=float(np.median(step_wall_clock)) if step_wall_clock else float("nan"),
            p95=float(np.percentile(step_wall_clock, 95)) if step_wall_clock else float("nan"),
            raw=step_wall_clock,
        ),
    )
    return dict(summary=summary, x_traj=x_traj, us=us, Vtraj=Vtraj, step_records=step_records,
                light_step_records=light_step_records)


def gate3_ad_vs_fd_gradient(bundle: JTrueBundle, sample: dict) -> dict:
    """`err<=GATE3_ERR_TOL` on all 5 fixed points, elementwise on the
    6-dim gradient. Both AD and FD must be entirely finite -- a
    non-finite value on either side is an automatic gate failure."""
    worst = 0.0
    per_point = []
    for i in range(sample["multi_step_x0"].shape[0]):
        x0 = sample["multi_step_x0"][i]
        Un = sample["multi_step_U"][i]
        grad_ad, _, _ = j_true_grad(bundle, x0, Un)
        grad_fd = _central_fd_gradient(bundle, x0, Un)
        if not (np.all(np.isfinite(grad_ad)) and np.all(np.isfinite(grad_fd))):
            return dict(passed=False, measured_error=float("inf"),
                        reason="NON_FINITE_GRADIENT", point_index=i)
        e = err(grad_ad, grad_fd)
        per_point.append(e)
        worst = max(worst, e)
    return dict(passed=bool(worst <= GATE3_ERR_TOL), measured_error=worst, per_point_error=per_point)


# ---------------------------------------------------------------------------
# Sec. 5.3 gate 4 -- Oracle-20 harness reproduction, summary + per-step.
# ---------------------------------------------------------------------------
GATE4_SUMMARY_EXACT_FIELDS = ("success", "solver_completed", "failed_at_step",
                               "n_steps_completed", "lyapunov_increase_count")
GATE4_SUMMARY_ISCLOSE_FIELDS = ("final_V", "max_V", "settling_time", "tv")


def _frozen_perstep_from_npz(npz: dict, ic_index: int, step_i: int, horizon: int = HORIZON) -> dict:
    """Reconstructs one step's `(x_before, u0, final_plan_un, V_before,
    V_after_realized)` from Battery 2's flattened per-step NPZ arrays
    (`solver_audit_battery2_{condition}_seed{seed}_iterations_2026_09_01.npz`)."""
    p = f"ic{ic_index}_perstep_"
    stored_step_index = int(npz[p + "step_index"][step_i])
    if stored_step_index != step_i:
        raise RuntimeError(
            f"GATE4_NPZ_STEP_ALIGNMENT_FAILED: requested step_i={step_i} but the NPZ's own "
            f"{p}step_index[{step_i}]={stored_step_index} -- the flattened per-step arrays are "
            f"not in the assumed step order, per-step comparison would be silently misaligned"
        )
    x_before = np.array([npz[p + "x_before_0"][step_i], npz[p + "x_before_1"][step_i]])
    u0 = np.array([npz[p + "u0_0"][step_i], npz[p + "u0_1"][step_i]])
    final_plan_un = np.array([[npz[p + f"final_plan_un_h{h}_d{d}"][step_i] for d in range(2)]
                               for h in range(horizon)])
    return dict(x_before=x_before, u0=u0, final_plan_un=final_plan_un,
                V_before=float(npz[p + "V_before"][step_i]),
                V_after_realized=float(npz[p + "V_after_realized"][step_i]))


def gate4_oracle20_harness_reproduction(seq, norm, sim: LCNNPaperCSTRSimulator, x0: np.ndarray,
                                         frozen_summary: dict, frozen_npz: dict, ic_index: int,
                                         un_lo: torch.Tensor, un_hi: torch.Tensor,
                                         input_lo: np.ndarray, input_hi: np.ndarray,
                                         u_mean: np.ndarray, u_std: np.ndarray,
                                         budget: int = BUDGET, lr: float = LR, steps: int = STEPS,
                                         success_v: float = SUCCESS_V) -> dict:
    """Sec. 5.3 gate 4: Oracle-20's harness, TEMPORARILY fed autograd
    through a learned model (`seq`), must reproduce the frozen rollout at
    BOTH the summary level AND the per-step level (fixed in v9: `x_after`
    is reconstructed via `sim.step(x_before, u0)` for all `steps`
    transitions, since Battery 2 never stored `x_after` and the "borrow
    the next record's x_before" trick has no analogue for the last step)."""
    learned_src = learned_gradient_source(seq, norm)
    res = oracle20_rollout(learned_src, x0, steps, budget, lr, success_v, un_lo, un_hi, sim,
                            input_lo, input_hi, u_mean, u_std, j_true_bundle=None, record_iterations=True)
    summary_cmp = gate_compare_flattened(res["summary"], frozen_summary,
                                          GATE4_SUMMARY_EXACT_FIELDS, GATE4_SUMMARY_ISCLOSE_FIELDS)
    per_step_mismatches = []
    n_steps_to_check = min(len(res["step_records"]), steps)
    for step_i in range(n_steps_to_check):
        rec = res["step_records"][step_i]
        frozen_step = _frozen_perstep_from_npz(frozen_npz, ic_index, step_i)
        x_after_ref = sim.step(frozen_step["x_before"], frozen_step["u0"])
        step_cmp = gate_compare_flattened(
            dict(u0=rec["u0"], final_plan_un=rec["final_plan_un"], x_after=rec["x_after"],
                 V_after_realized=rec["V_after"]),
            dict(u0=frozen_step["u0"], final_plan_un=frozen_step["final_plan_un"], x_after=x_after_ref,
                 V_after_realized=frozen_step["V_after_realized"]),
            exact_fields=(), isclose_fields=("u0", "final_plan_un", "x_after", "V_after_realized"),
        )
        if not step_cmp["passed"]:
            per_step_mismatches.append(dict(step_index=step_i, mismatches=step_cmp["mismatches"]))
    full_coverage = bool(n_steps_to_check == steps)
    passed = bool(summary_cmp["passed"] and not per_step_mismatches and full_coverage)
    return dict(passed=passed, summary_mismatches=summary_cmp["mismatches"],
                per_step_mismatches=per_step_mismatches, n_steps_checked=n_steps_to_check,
                full_coverage=full_coverage)


# ---------------------------------------------------------------------------
# Sec. 7 / Sec. 9 -- endpoint-equivalence gates between the clean and
# instrumented passes, using each rollout's `light_step_records` (populated
# regardless of `record_iterations`).
# ---------------------------------------------------------------------------
def oracle20_endpoint_equivalence(clean_result: dict, instrumented_result: dict) -> dict:
    """Sec. 7: `u0`, `final_plan_un`, `x_after`, `V_after`, and
    `minimum_stage_temperature` must match between the clean and
    instrumented passes for every control step -- not merely the
    rollout's final aggregate values."""
    clean_steps = clean_result["light_step_records"]
    instrumented_steps = instrumented_result["light_step_records"]
    if len(clean_steps) != len(instrumented_steps):
        return dict(passed=False, reason="STEP_COUNT_MISMATCH",
                    clean_steps=len(clean_steps), instrumented_steps=len(instrumented_steps))
    mismatches = []
    for step_i, (c, i) in enumerate(zip(clean_steps, instrumented_steps)):
        cmp = gate_compare_flattened(
            dict(u0=i["u0"], final_plan_un=i["final_plan_un"], x_after=i["x_after"],
                 V_after=i["V_after"], minimum_stage_temperature=i["minimum_stage_temperature"]),
            dict(u0=c["u0"], final_plan_un=c["final_plan_un"], x_after=c["x_after"],
                 V_after=c["V_after"], minimum_stage_temperature=c["minimum_stage_temperature"]),
            exact_fields=(), isclose_fields=("u0", "final_plan_un", "x_after", "V_after",
                                              "minimum_stage_temperature"),
        )
        if not cmp["passed"]:
            mismatches.append(dict(step_index=step_i, mismatches=cmp["mismatches"]))
    return dict(passed=len(mismatches) == 0, mismatches=mismatches)


def certified_nlp_endpoint_equivalence(clean_result: dict, instrumented_result: dict) -> dict:
    """Sec. 9: `certified`/`selected_start_id` must match EXACTLY; `J_true`,
    the winner's final plan, `x_after`, and `minimum_stage_temperature` via
    `values_match(exact=False)`, per Sec. 5.2a -- NOT the mixed `err()`."""
    clean_steps = clean_result["light_step_records"]
    instrumented_steps = instrumented_result["light_step_records"]
    if len(clean_steps) != len(instrumented_steps):
        return dict(passed=False, reason="STEP_COUNT_MISMATCH",
                    clean_steps=len(clean_steps), instrumented_steps=len(instrumented_steps))
    mismatches = []
    for step_i, (c, i) in enumerate(zip(clean_steps, instrumented_steps)):
        cmp = gate_compare_flattened(
            dict(certified=i["certified"], selected_start_id=i["selected_start_id"],
                 J_true=i["J_true"], final_plan_un=i["final_plan_un"], x_after=i["x_after"],
                 minimum_stage_temperature=i["minimum_stage_temperature"]),
            dict(certified=c["certified"], selected_start_id=c["selected_start_id"],
                 J_true=c["J_true"], final_plan_un=c["final_plan_un"], x_after=c["x_after"],
                 minimum_stage_temperature=c["minimum_stage_temperature"]),
            exact_fields=("certified", "selected_start_id"),
            isclose_fields=("J_true", "final_plan_un", "x_after", "minimum_stage_temperature"),
        )
        if not cmp["passed"]:
            mismatches.append(dict(step_index=step_i, mismatches=cmp["mismatches"]))
    return dict(passed=len(mismatches) == 0, mismatches=mismatches)


# ---------------------------------------------------------------------------
# Sec. 5.3 gates 5-6 -- Certified-NLP harness unit-regression + reproducibility,
# against a fixed SYNTHETIC problem `J_test(U) = ||U||^2` (hand-computable
# optimum: the box point nearest zero), independent of the true CSTR
# dynamics. Reuses the SAME certification/dedup/winner-selection code
# path as the real Certified-NLP controller (Sec. 8-9) -- only the
# objective/solver-building differs.
# ---------------------------------------------------------------------------
@dataclass
class SyntheticNLPSolver:
    opti: ca.Opti
    Un: ca.MX  # note: no x0 parameter needed -- J_test doesn't depend on x0
    horizon: int


def build_synthetic_nlp_solver(horizon: int, un_lo: np.ndarray, un_hi: np.ndarray,
                                ipopt_options: dict = None) -> SyntheticNLPSolver:
    opti = ca.Opti()
    Un = opti.variable(horizon, 2)
    opti.minimize(ca.sumsqr(Un))
    un_lo = np.asarray(un_lo, dtype=np.float64)
    un_hi = np.asarray(un_hi, dtype=np.float64)
    for h in range(horizon):
        for a in range(2):
            opti.subject_to(opti.bounded(float(un_lo[a]), Un[h, a], float(un_hi[a])))
    opti.solver("ipopt", ipopt_options if ipopt_options is not None else IPOPT_OPTIONS)
    return SyntheticNLPSolver(opti=opti, Un=Un, horizon=horizon)


def _synthetic_j_true_value(x0_ignored: np.ndarray, Un: np.ndarray) -> tuple:
    """`J_test(U) = ||U||^2`, no floor concept -- `min_temp` is always a
    large finite sentinel (never floor-active). Signature matches
    `JTrueBundle.j_fn(x0, Un)` exactly (`x0` ignored) so this can be
    plugged into `_SyntheticBundleAdapter` and drive the REAL
    `certify_candidate`/`j_true_value`/`j_true_grad` verbatim -- no
    duplicated certification logic for the synthetic problem."""
    return float(np.sum(np.asarray(Un, dtype=np.float64) ** 2)), 1e6


def _synthetic_j_true_grad(x0_ignored: np.ndarray, Un: np.ndarray) -> tuple:
    """Matches `JTrueBundle.grad_fn(x0, Un) -> (grad, J, min_temp)`."""
    Un = np.asarray(Un, dtype=np.float64)
    return 2.0 * Un, float(np.sum(Un ** 2)), 1e6


class _SyntheticBundleAdapter:
    """Duck-types `JTrueBundle`'s interface (`j_true_value`/`j_true_grad`
    call `bundle.j_fn(x0, Un)`/`bundle.grad_fn(x0, Un)` directly) so gates
    5-6 reuse `certify_candidate` VERBATIM against the synthetic `J_test`
    -- no separate copy of the certification logic to silently drift from
    the real one."""

    def __init__(self, horizon: int):
        self.horizon = horizon
        self.j_fn = _synthetic_j_true_value
        self.grad_fn = _synthetic_j_true_grad


def _synthetic_solve_one_candidate(nlp: SyntheticNLPSolver, plan: np.ndarray) -> dict:
    """Genuinely distinct from `solve_one_candidate` (not a duplicate to
    eliminate): `SyntheticNLPSolver` has no `x0p` parameter at all (`J_test`
    doesn't depend on `x0`), so `nlp.opti.set_value(...)` has nothing to
    set -- the two `Opti` object shapes are structurally different, not
    merely differently-named copies of the same logic."""
    nlp.opti.set_initial(nlp.Un, np.asarray(plan, dtype=np.float64).reshape(nlp.horizon, 2))
    try:
        sol = nlp.opti.solve()
        stats = sol.stats()
        U_sol = np.asarray(sol.value(nlp.Un)).reshape(nlp.horizon, 2)
    except RuntimeError:
        stats = nlp.opti.debug.stats()
        U_sol = np.asarray(nlp.opti.debug.value(nlp.Un)).reshape(nlp.horizon, 2)
    J_val, min_temp = _synthetic_j_true_value(None, U_sol)
    return dict(U_sol=U_sol, return_status=stats.get("return_status"), iter_count=stats.get("iter_count"),
                alpha_pr=None, alpha_du=None, J_true=J_val, minimum_stage_temperature=min_temp)


def synthetic_nlp_step(nlp: SyntheticNLPSolver, un_lo: np.ndarray, un_hi: np.ndarray,
                        warm_shifted_plan: np.ndarray,
                        force_status_override: dict = None, force_uncertify: set = None) -> dict:
    """Mirrors `certified_nlp_step`'s control flow exactly (same dedup +
    certify + winner-select shape), but against the synthetic `J_test`.
    `force_status_override` ({start_id: status_str}) and `force_uncertify`
    ({start_id}) let Gate 5's smoke suite deterministically exercise the
    non-whitelisted-status-rejection and uncertified-candidate-fallback
    paths without depending on IPOPT actually failing by chance."""
    force_status_override = force_status_override or {}
    force_uncertify = force_uncertify or set()
    adapter = _SyntheticBundleAdapter(nlp.horizon)
    plans = certified_nlp_candidate_starts(un_lo, un_hi, warm_shifted_plan, nlp.horizon)
    groups = _dedup_groups(plans)
    candidates = [None] * len(plans)
    starts_deduplicated = []
    for group in groups:
        rep = group[0]
        sol = _synthetic_solve_one_candidate(nlp, plans[rep])
        for idx in group:
            entry = dict(sol)
            if idx in force_status_override:
                entry["return_status"] = force_status_override[idx]
            cert = certify_candidate(adapter, np.zeros(2), entry, un_lo, un_hi)
            if idx in force_uncertify:
                cert = dict(cert)
                cert["certified"] = False
            entry.update(cert)
            entry["start_id"] = idx
            entry["dedup_source_start_id"] = rep
            candidates[idx] = entry
        if len(group) > 1:
            starts_deduplicated.append(list(group))
    eligible = [c for c in candidates if c["certified"]]
    winner = None
    if eligible:
        best_J = min(c["J_true"] for c in eligible)
        tied = [c for c in eligible if abs(c["J_true"] - best_J) <= TOL_TIE_OBJECTIVE]
        winner = min(tied, key=lambda c: c["start_id"])
    return dict(candidates=candidates, starts_deduplicated=starts_deduplicated,
                selected_start_id=(winner["start_id"] if winner is not None else None), winner=winner)


def gate5_certified_nlp_unit_regression(un_lo: np.ndarray, un_hi: np.ndarray, horizon: int = HORIZON) -> dict:
    """Sec. 5.3 gate 5: three separately-exercised synthetic smoke cases
    (not just a single success path)."""
    nlp = build_synthetic_nlp_solver(horizon, un_lo, un_hi)
    results = {}

    # (a) dedup path: warm_shifted == zero at the first control step, by construction.
    zero_plan = np.zeros((horizon, 2))
    step_a = synthetic_nlp_step(nlp, un_lo, un_hi, warm_shifted_plan=zero_plan)
    results["dedup_path"] = dict(
        passed=bool(step_a["starts_deduplicated"] == [[0, 1]] and step_a["selected_start_id"] is not None),
        starts_deduplicated=step_a["starts_deduplicated"], selected_start_id=step_a["selected_start_id"],
    )

    # (b) forced non-whitelisted return_status: start 0 (normally the
    # certified winner, since J_test's optimum at zero start is exactly 0)
    # is forced to a non-whitelisted status and must be rejected regardless
    # of numerically passing KKT/feasibility.
    warm_shifted_ok = _tiled_plan(un_hi, horizon)  # distinct from zero/quartile, avoids dedup here
    step_b = synthetic_nlp_step(nlp, un_lo, un_hi, warm_shifted_plan=warm_shifted_ok,
                                 force_status_override={0: "Maximum_Iterations_Exceeded"})
    cand0 = step_b["candidates"][0]
    results["status_rejection"] = dict(
        passed=bool((not cand0["certified"]) and (not cand0["status_ok"])),
        candidate0_certified=cand0["certified"], candidate0_status_ok=cand0["status_ok"],
    )

    # (c) uncertified-candidate fallback: force the lowest-J_true candidate
    # (start 0, zero start) to fail certification; the harness must select
    # the next-best CERTIFIED candidate, not the globally-lowest-objective one.
    step_c = synthetic_nlp_step(nlp, un_lo, un_hi, warm_shifted_plan=warm_shifted_ok,
                                 force_uncertify={0})
    results["uncertified_fallback"] = dict(
        passed=bool(step_c["selected_start_id"] is not None and step_c["selected_start_id"] != 0),
        selected_start_id=step_c["selected_start_id"],
    )

    passed = all(r["passed"] for r in results.values())
    return dict(passed=passed, cases=results)


def gate6_certified_nlp_reproducibility(un_lo: np.ndarray, un_hi: np.ndarray, horizon: int = HORIZON) -> dict:
    """Sec. 5.3 gate 6: solving the SAME fixed synthetic problem twice
    must reproduce `return_status`/`selected_start_id` EXACTLY and
    `J_test` via `values_match(exact=False)` -- NOT the mixed `err()`
    formula (Sec. 5.2a)."""
    nlp1 = build_synthetic_nlp_solver(horizon, un_lo, un_hi)
    nlp2 = build_synthetic_nlp_solver(horizon, un_lo, un_hi)
    warm_shifted = _tiled_plan(un_hi, horizon)
    step1 = synthetic_nlp_step(nlp1, un_lo, un_hi, warm_shifted_plan=warm_shifted)
    step2 = synthetic_nlp_step(nlp2, un_lo, un_hi, warm_shifted_plan=warm_shifted)
    mismatches = []
    if step1["selected_start_id"] != step2["selected_start_id"]:
        mismatches.append(dict(field="selected_start_id", new=step2["selected_start_id"], old=step1["selected_start_id"]))
    for c1, c2 in zip(step1["candidates"], step2["candidates"]):
        cmp = gate_compare_flattened(dict(return_status=c2["return_status"], J_true=c2["J_true"]),
                                      dict(return_status=c1["return_status"], J_true=c1["J_true"]),
                                      exact_fields=("return_status",), isclose_fields=("J_true",))
        mismatches.extend(cmp["mismatches"])
    return dict(passed=len(mismatches) == 0, mismatches=mismatches)


# ---------------------------------------------------------------------------
# Sec. 12 -- timing-pass controller abstraction. Each controller exposes
# `reset(x0)` + `step() -> wall_clock_seconds`, so the driver's nested
# `(step, ic)` loop with deterministic rotation (Sec. 12) can advance all
# 22 controllers uniformly without needing to know their internal solver
# type.
# ---------------------------------------------------------------------------
class Oracle20TimingController:
    kind = "oracle20"

    def __init__(self, bundle: JTrueBundle, un_lo: torch.Tensor, un_hi: torch.Tensor,
                 sim: LCNNPaperCSTRSimulator, input_lo: np.ndarray, input_hi: np.ndarray,
                 u_mean: np.ndarray, u_std: np.ndarray):
        self.grad_src = true_gradient_source(bundle)
        self.un_lo, self.un_hi = un_lo, un_hi
        self.sim, self.input_lo, self.input_hi = sim, input_lo, input_hi
        self.u_mean, self.u_std = u_mean, u_std
        self.horizon = bundle.horizon
        self.x = None
        self.un = None

    def reset(self, x0: np.ndarray) -> None:
        self.x = np.asarray(x0, dtype=float)
        self.un = torch.zeros(self.horizon, 2)

    def step(self) -> float:
        t0 = _now()
        result = oracle20_step(self.grad_src, self.x, self.un, BUDGET, LR, self.un_lo, self.un_hi)
        t1 = _now()
        u0 = np.clip(result["u0n"] * self.u_std + self.u_mean, self.input_lo, self.input_hi)
        self.x = self.sim.step(self.x, u0)
        un_next = result["un_final"]
        self.un = torch.cat([un_next[1:], un_next[-1:]])
        return t1 - t0


class CertifiedNLPTimingController:
    kind = "certified_nlp"

    def __init__(self, nlp: CertifiedNLPSolver, bundle: JTrueBundle, sim: LCNNPaperCSTRSimulator,
                 input_lo: np.ndarray, input_hi: np.ndarray, u_mean: np.ndarray, u_std: np.ndarray):
        self.nlp, self.bundle = nlp, bundle
        self.sim, self.input_lo, self.input_hi = sim, input_lo, input_hi
        self.u_mean, self.u_std = u_mean, u_std
        self.x = None
        self.warm_shifted_plan = None

    def reset(self, x0: np.ndarray) -> None:
        self.x = np.asarray(x0, dtype=float)
        self.warm_shifted_plan = np.zeros((self.nlp.horizon, 2))

    def step(self) -> float:
        t0 = _now()
        step_result = certified_nlp_step(self.nlp, self.bundle, self.x, self.warm_shifted_plan)
        t1 = _now()
        winner = step_result["winner"]
        if winner is None:
            return t1 - t0
        U_sol = winner["U_sol"]
        u0 = np.clip(U_sol[0] * self.u_std + self.u_mean, self.input_lo, self.input_hi)
        self.x = self.sim.step(self.x, u0)
        self.warm_shifted_plan = np.concatenate([U_sol[1:], U_sol[-1:]], axis=0)
        return t1 - t0


class LearnedModelTimingController:
    """Wraps `solver_audit_lib._adam_mpc_step` (a frozen, private function
    of a frozen file -- called as-is, never copied or edited) for a single
    B/E (condition, seed) controller's timing re-measurement (Sec. 12)."""
    kind = "learned"

    def __init__(self, condition: str, seed: int, seq, norm, un_lo: torch.Tensor, un_hi: torch.Tensor,
                 sim: LCNNPaperCSTRSimulator, input_lo: np.ndarray, input_hi: np.ndarray):
        from cstr.solver_audit_lib import _adam_mpc_step
        self._adam_mpc_step = _adam_mpc_step
        self.condition, self.seed = condition, seed
        self.seq, self.norm = seq, norm
        self.un_lo, self.un_hi = un_lo, un_hi
        self.sim, self.input_lo, self.input_hi = sim, input_lo, input_hi
        xm, xs, ym, ys = norm
        self.u_mean, self.u_std = xm[2:].numpy(), xs[2:].numpy()
        self.x = None
        self.un = None

    def reset(self, x0: np.ndarray) -> None:
        self.x = np.asarray(x0, dtype=float)
        self.un = torch.zeros(HORIZON, 2)
        # Trajectory tracking, ONLY for the B/E reproduction gate (Sec. 12) --
        # a fresh same-session timing re-execution must reproduce the frozen
        # Battery 1 raw per-IC summary, not merely provide a timing number.
        self.Vtraj = [float(self.x @ P_MAT @ self.x)]
        self.us = []

    def step(self) -> float:
        t0 = _now()
        u0n, un_next, _ = self._adam_mpc_step(self.seq, self.norm, self.x, self.un, HORIZON, BUDGET, LR,
                                               RHO_U, self.un_lo, self.un_hi, record_iterations=False)
        t1 = _now()
        u0 = np.clip(u0n * self.u_std + self.u_mean, self.input_lo, self.input_hi)
        self.x = self.sim.step(self.x, u0)
        self.un = torch.cat([un_next[1:], un_next[-1:]])
        self.Vtraj.append(float(self.x @ P_MAT @ self.x))
        self.us.append(u0)
        return t1 - t0

    def summary(self, success_v: float = SUCCESS_V, steps: int = STEPS) -> dict:
        """Sec. 12's B/E reproduction-gate comparand -- same schema as
        Battery 1's own `summary` dict."""
        us_arr = np.asarray(self.us) if self.us else np.zeros((0, 2))
        tv = float(np.sum(np.abs(np.diff(us_arr / (self.input_hi - self.input_lo), axis=0)))) if len(us_arr) > 1 else 0.0
        lyapunov_increase_count = int(sum(self.Vtraj[i + 1] > self.Vtraj[i] + 1e-9 for i in range(len(self.Vtraj) - 1)))
        return dict(
            final_V=self.Vtraj[-1], max_V=float(max(self.Vtraj)),
            success=bool(self.Vtraj[-1] <= success_v), lyapunov_increase_count=lyapunov_increase_count,
            tv=tv, solver_completed=True, failed_at_step=None, n_steps_completed=len(self.us),
            input_tv_per_actuator_physical=(np.sum(np.abs(np.diff(us_arr, axis=0)), axis=0).tolist()
                                             if len(us_arr) > 1 else [0.0, 0.0]),
            input_tv_per_actuator_normalized=(np.sum(np.abs(np.diff(us_arr / (self.input_hi - self.input_lo), axis=0)),
                                                       axis=0).tolist() if len(us_arr) > 1 else [0.0, 0.0]),
        )


def run_interleaved_timing_pass(controller_factories: dict, ics: list, steps: int = STEPS,
                                 warmup_steps: int = 2) -> dict:
    """Sec. 12: `controller_factories` is an ordered `{name: callable}`
    dict (fixed order, 22 entries in the real campaign), each callable
    returning a FRESH `TimingController` instance -- required because each
    of the 22 controllers needs 5 INDEPENDENT per-IC states running
    concurrently in the interleaved loop below, and a single controller
    object only holds one `(x, warm_start)` state at a time (a single
    shared instance reset in a loop over ICs would silently discard all
    but the last IC's state -- confirmed as a real bug during
    implementation and fixed by this factory-per-(name,ic) design).

    For each `(step, ic)` pair, computes the deterministic rotation
    `r = (step*n_ic + ic) % n_controllers` and advances every controller
    once, in rotated order, so no single controller systematically
    benefits from or is penalized by a warm cache/thermal state the
    others didn't also just experience. Returns `{name: {"raw_by_ic":
    [[...]]*n_ic, "median": ..., "p95": ..., "deadline_exceedance_
    fraction": ..., "controllers": [per-ic controller instances]}}`,
    warmup-excluded per Sec. 12. The returned per-ic controller instances
    let the caller extract a final trajectory summary (e.g. for the B/E
    reproduction gate) without re-running anything."""
    names = list(controller_factories.keys())
    n = len(names)
    n_ic = len(ics)
    instances = {name: [controller_factories[name]() for _ in range(n_ic)] for name in names}
    for name in names:
        for ic_index, x0 in enumerate(ics):
            instances[name][ic_index].reset(x0)
    timings = {name: [[] for _ in range(n_ic)] for name in names}
    for step_i in range(steps):
        for ic_index in range(n_ic):
            r = (step_i * n_ic + ic_index) % n
            rotated_order = [names[(r + k) % n] for k in range(n)]
            for name in rotated_order:
                wall_clock = instances[name][ic_index].step()
                timings[name][ic_index].append(wall_clock)

    result = {}
    for name in names:
        all_timed = []
        deadline_exceed = 0
        n_timed = 0
        for ic_index in range(n_ic):
            series = timings[name][ic_index][warmup_steps:] if len(timings[name][ic_index]) > warmup_steps else []
            all_timed.extend(series)
            deadline_exceed += sum(1 for t in series if t > 3.6)
            n_timed += len(series)
        result[name] = dict(
            raw_by_ic=timings[name],
            median=float(np.median(all_timed)) if all_timed else float("nan"),
            p95=float(np.percentile(all_timed, 95)) if all_timed else float("nan"),
            deadline_exceedance_fraction=(deadline_exceed / n_timed) if n_timed else float("nan"),
            controllers=instances[name],
        )
    return result
