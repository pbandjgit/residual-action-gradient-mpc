"""Core library for the single-CSTR query-matched solver audit + two-plant
state-constraint minimum audit
(`docs/JPC_SOLVER_AUDIT_PROTOCOL_2026_09_01.md`, v7).

Implements, in order: Section 1 (naming corrections), Section 2 (five
batteries), Section 3 (logging schema: rollout summary, per-control-step,
per-iteration, one-step residual/gradient bridge, state-constraint
statistics, failure handling), Section 4a (four compatibility gates),
Section 8 (two-CSTR minimum audit, four-phase reconstruction).

None of the 12 Freeze-Manifest-covered files (or, in spirit, the
non-frozen-but-still-reference `ggn_mpc_probe.py`/`two_cstr_lgrad.py`/
`two_cstr_series.py`/`two_cstr_constraint.py`) are edited here -- this
module only IMPORTS and calls their existing functions, or reimplements
control-flow-identical copies where per-iteration instrumentation is
genuinely required (Battery 2's Adam audit, Battery 3's GGN audit),
each verified against the original via a compatibility gate (Section 4a)
before its output is trusted.

Nothing in this module has real side effects at import time (no real
pool/simulator query happens merely by importing it) -- mirrors
`ablation_lib.py`'s own design-review -> implement -> smoke -> freeze ->
real-execution discipline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cstr.ablation_lib as alib      # noqa: E402
import cstr.solver_audit_io as sio    # noqa: E402

# ---------------------------------------------------------------------------
# Shared literals (single-CSTR Lyapunov matrix, FD step, timing/tolerance
# constants re-exported from solver_audit_io for a single source of truth).
# ---------------------------------------------------------------------------
P = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float64)
Pt_torch = torch.tensor(P, dtype=torch.float32)
FD_EPS = 1e-3
TIMING_WARMUP_STEPS = sio.TIMING_WARMUP_STEPS
ISCLOSE_RTOL = sio.ISCLOSE_RTOL
ISCLOSE_ATOL = sio.ISCLOSE_ATOL


def configure_deterministic_single_thread() -> None:
    """Section 3.2's timing methodology: single-threaded, for maximal
    reproducibility/comparability. Caller (driver/smoke) is responsible
    for calling this ONCE before any timed rollout, and for recording it
    in the Setup Manifest -- this module never calls it implicitly, since
    a library function silently mutating global torch state on import or
    on every call would be a surprising side effect."""
    torch.set_num_threads(sio.TIMING_TORCH_NUM_THREADS)


# ---------------------------------------------------------------------------
# Section 1: naming corrections
# ---------------------------------------------------------------------------
# `budget_sweep_solver.closed_loop()`'s `violations` field counts
# Lyapunov-INCREASE steps, not state-constraint violations -- renamed
# `lyapunov_increase_count` in every new output. A single `feasible`
# boolean is never produced by this module; `solver_completed` (numerical
# completion), per-actuator saturation stats, and
# `temperature_cap_satisfied_posthoc` are always separate fields.


# ---------------------------------------------------------------------------
# Checkpoint loading (real ablation's M2 payload is the sole source of
# truth for checkpoint paths/hashes, re-verified here rather than trusted)
# ---------------------------------------------------------------------------
CONDITION_CHECKPOINT_PREFIX = {
    "A": "A", "B": "B", "C": "C", "E": "E", "F": "F",
    "B-FNN": "BFNN", "E-FNN": "E-FNN",
}


def checkpoint_prefix_for_condition(condition: str, m2_payload: dict = None) -> str:
    if condition == "D":
        if m2_payload is None:
            raise ValueError("condition D's checkpoint filename depends on the seed's selected lambda_jac")
        return f"D_lamjac{m2_payload['lambda_jac_selected']}"
    return CONDITION_CHECKPOINT_PREFIX[condition]


def find_checkpoint_path(m2_payload: dict, prefix: str, seed: int, epoch: int = None) -> str:
    epoch = epoch if epoch is not None else alib.EPOCHS
    target = f"{prefix}_seed{seed}_ep{epoch:03d}.pt"
    matches = [p for p in m2_payload["checkpoint_sha256"] if Path(p).name == target]
    if len(matches) != 1:
        raise RuntimeError(
            f"CHECKPOINT_LOOKUP_AMBIGUOUS_OR_MISSING: target={target!r} matched "
            f"{len(matches)} paths in M2's checkpoint_sha256 (expected exactly 1)"
        )
    path = matches[0]
    expected_hash = m2_payload["checkpoint_sha256"][path]
    actual_hash = sio.file_sha256(Path(path))
    if actual_hash != expected_hash:
        raise RuntimeError(f"CHECKPOINT_DRIFT: {path} hash changed since M2 was built")
    return path


def load_condition_model(condition: str, seed: int, m2_payload: dict) -> nn.Module:
    use_fnn = condition.endswith("-FNN")
    prefix = checkpoint_prefix_for_condition(condition, m2_payload)
    path = find_checkpoint_path(m2_payload, prefix, seed)
    state = torch.load(path, map_location="cpu")
    model = alib.build_model(condition, use_fnn=use_fnn)
    model.load_state_dict(state)
    model.eval()
    return model


def load_condition_seq(condition: str, seed: int, m2_payload: dict):
    """Returns the FROZEN (materialized, inference-only) `seq` -- exactly
    what `budget_sweep_solver.closed_loop()`/`ggn_mpc_probe.ggn_closed_loop()`
    expect, via the existing `alib.freeze_any` adapter (reused, not
    reimplemented)."""
    return alib.freeze_any(load_condition_model(condition, seed, m2_payload))


# ---------------------------------------------------------------------------
# Section 6 item 7 -- counterfactual-true-objective, EXACT same functional
# form as the learned multi-step objective, SIM.step substituted for the
# learned surrogate. Shared by Battery 2 (Adam, item 5) and Battery 3
# (GGN, item 6).
# ---------------------------------------------------------------------------
def counterfactual_true_objective(sim, x_phys: np.ndarray, U_normalized, horizon: int, rho_u: float,
                                   xm: torch.Tensor, xs: torch.Tensor, ym: torch.Tensor,
                                   ys: torch.Tensor) -> float:
    del ym, ys  # the true-plant rollout uses SIM.step (already physical), not the learned de-normalization
    u_mean, u_std = xm[2:].numpy(), xs[2:].numpy()
    Uarr = np.asarray(U_normalized, dtype=np.float64).reshape(horizon, 2)
    xt = np.asarray(x_phys, dtype=float).copy()
    J = 0.0
    for h in range(horizon):
        u_phys = Uarr[h] * u_std + u_mean
        y = sim.step(xt, u_phys)
        J += float(y @ P @ y) + rho_u * float((Uarr[h] ** 2).sum())
        xt = y
    return J


def learned_objective_value(seq, x_phys: np.ndarray, un: np.ndarray, horizon: int, rho_u: float,
                             xm: torch.Tensor, xs: torch.Tensor, ym: torch.Tensor, ys: torch.Tensor) -> float:
    """EXACT same functional form as `_adam_mpc_step`'s per-iteration `J`
    (the learned multi-step objective Adam actually optimizes) -- but
    usable to evaluate J at an ARBITRARY plan, not only the one the
    in-progress optimization happens to be at. Needed because the
    original algorithm only ever evaluates J at `un_before` (the point
    the gradient is taken from); there is no "J after this iteration's
    update" anywhere in the original solve loop. Used to compute a
    genuine, SIGNED `J_before -> J_after_projected` decrease per
    iteration (Section 3.4 item 4), mirroring the counterfactual
    objective's own before/after treatment above."""
    xm2, xs2 = xm[:2], xs[:2]
    with torch.no_grad():
        x0n = torch.tensor((np.asarray(x_phys, dtype=float) - xm2.numpy()) / xs2.numpy(),
                            dtype=torch.float32).view(1, 2)
        un_t = torch.as_tensor(np.asarray(un, dtype=np.float32))
        xt = x0n
        J = torch.zeros(())
        for h in range(horizon):
            xun = torch.cat([xt, un_t[h].view(1, 2)], dim=1)
            y = seq(xun) * ys + ym
            J = J + (y @ Pt_torch * y).sum() + rho_u * (un_t[h] ** 2).sum()
            xt = (y - xm2) / xs2
    return float(J)


# ---------------------------------------------------------------------------
# Section 3.5 -- boundary-handling contract for the true-gradient FD.
# ---------------------------------------------------------------------------
def true_v_gradient_with_boundary_handling(v_fn, u_n0: np.ndarray, un_lo: np.ndarray, un_hi: np.ndarray,
                                            eps: float = FD_EPS) -> tuple:
    """`v_fn(u_n) -> float` = physical V(SIM.step(x, u_phys(u_n))). Central
    FD (`eps`) when `2*eps` interior clearance exists on both sides;
    otherwise the pinned 3-point second-order ONE-SIDED FD toward the
    interior (`(-3 f0 + 4 f(+eps) - f(+2eps)) / (2 eps)`, mirrored for the
    backward case), which needs `2*eps` clearance on the chosen side; if
    even that is unavailable, the coordinate is excluded with reason
    `fd_clearance_unavailable` -- never silently downgraded to a
    lower-order one-sided formula."""
    grad = np.full(u_n0.shape[0], np.nan)
    diag = []
    for j in range(u_n0.shape[0]):
        lo, hi = float(un_lo[j]), float(un_hi[j])
        can_plus_eps = (u_n0[j] + eps) <= hi
        can_minus_eps = (u_n0[j] - eps) >= lo
        if can_plus_eps and can_minus_eps:
            up = u_n0.copy(); up[j] += eps
            dn = u_n0.copy(); dn[j] -= eps
            grad[j] = (v_fn(up) - v_fn(dn)) / (2 * eps)
            diag.append(dict(coord=j, kind="central", excluded=False))
            continue
        can_forward_2eps = (u_n0[j] + 2 * eps) <= hi
        can_backward_2eps = (u_n0[j] - 2 * eps) >= lo
        if can_forward_2eps:
            p1 = u_n0.copy(); p1[j] += eps
            p2 = u_n0.copy(); p2[j] += 2 * eps
            f0, f1, f2 = v_fn(u_n0), v_fn(p1), v_fn(p2)
            grad[j] = (-3 * f0 + 4 * f1 - f2) / (2 * eps)
            diag.append(dict(coord=j, kind="one_sided_forward", excluded=False))
        elif can_backward_2eps:
            m1 = u_n0.copy(); m1[j] -= eps
            m2 = u_n0.copy(); m2[j] -= 2 * eps
            f0, f1, f2 = v_fn(u_n0), v_fn(m1), v_fn(m2)
            grad[j] = (3 * f0 - 4 * f1 + f2) / (2 * eps)
            diag.append(dict(coord=j, kind="one_sided_backward", excluded=False))
        else:
            grad[j] = float("nan")
            diag.append(dict(coord=j, kind="excluded", excluded=True, reason="fd_clearance_unavailable"))
    return grad, diag


def _vnp(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(y @ P @ y)


def learned_residual_value_and_grad(seq, x_phys: np.ndarray, u_n0: np.ndarray, phi_x: np.ndarray,
                                     xm: torch.Tensor, xs: torch.Tensor, ym: torch.Tensor,
                                     ys: torch.Tensor) -> tuple:
    """`g_hat(x, u_n0) = V(f_hat(x,u)) - V(f_hat(x,Phi(x)))` and its
    gradient w.r.t. `u_n0` (normalized), via autograd. Same functional
    form as `probe_action_gradient.diagnose()`'s residual-value formula."""
    x_mean, x_std = xm[:2].numpy(), xs[:2].numpy()
    u_mean, u_std = xm[2:].numpy(), xs[2:].numpy()
    xn_t = torch.tensor((x_phys - x_mean) / x_std, dtype=torch.float32).view(1, 2)
    un_t = torch.tensor(u_n0, dtype=torch.float32, requires_grad=True).view(1, 2)
    phin_t = torch.tensor((phi_x - u_mean) / u_std, dtype=torch.float32).view(1, 2)
    y_u = seq(torch.cat([xn_t, un_t], 1)) * ys + ym
    y_phi = seq(torch.cat([xn_t, phin_t], 1)) * ys + ym
    g = (y_u @ Pt_torch * y_u).sum() - (y_phi @ Pt_torch * y_phi).sum()
    (grad,) = torch.autograd.grad(g, un_t)
    return float(g.detach()), grad.detach().numpy().reshape(-1)


def true_residual_value_and_grad(sim, x_phys: np.ndarray, u_n0: np.ndarray, phi_x: np.ndarray,
                                  un_lo: np.ndarray, un_hi: np.ndarray,
                                  xm: torch.Tensor, xs: torch.Tensor) -> tuple:
    """`g(x, u_n0) = V(f(x,u)) - V(f(x,Phi(x)))` (true plant) and its
    gradient w.r.t. `u_n0`, via the boundary-aware FD above. The
    Phi(x)-subtraction term is `u`-independent, so its gradient
    contribution is exactly zero (`probe_action_gradient.py`'s own
    documented identity) -- the returned gradient equals the gradient of
    the raw `V(f(x,u))`, matching `diagnose_fixed_idx()`'s convention."""
    u_mean, u_std = xm[2:].numpy(), xs[2:].numpy()
    u_phys = u_n0 * u_std + u_mean
    g_true = _vnp(sim.step(x_phys, u_phys)) - _vnp(sim.step(x_phys, phi_x))

    def v_of_un(un):
        return _vnp(sim.step(x_phys, un * u_std + u_mean))

    grad_true, fd_diag = true_v_gradient_with_boundary_handling(v_of_un, u_n0, un_lo, un_hi)
    return g_true, grad_true, fd_diag


# ---------------------------------------------------------------------------
# Shared single-CSTR Adam inner-loop -- the numerically sensitive part,
# factored out ONCE so Battery 1's light rollout and Battery 2's
# per-iteration-logging rollout can never silently diverge from each
# other. Byte-for-byte the same math as `budget_sweep_solver.closed_loop()`
# (Gate 1/Gate 2 verify this against the ORIGINAL before any output here
# is trusted).
# ---------------------------------------------------------------------------
def _adam_mpc_step(seq, norm, x_phys: np.ndarray, un_warm_start: torch.Tensor, horizon: int, budget: int,
                    lr: float, rho_u: float, un_lo: torch.Tensor, un_hi: torch.Tensor,
                    record_iterations: bool = False):
    xm, xs, ym, ys = norm
    xm2, xs2 = xm[:2], xs[:2]
    un = un_warm_start.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([un], lr=lr)
    x0n = torch.tensor((x_phys - xm2.numpy()) / xs2.numpy(), dtype=torch.float32).view(1, 2)
    iteration_records = [] if record_iterations else None
    for _it in range(budget):
        un_before = un.detach().clone() if record_iterations else None
        opt.zero_grad()
        xt = x0n
        J = torch.zeros(())
        for h in range(horizon):
            xun = torch.cat([xt, un[h].view(1, 2)], dim=1)
            y = seq(xun) * ys + ym
            J = J + (y @ Pt_torch * y).sum() + rho_u * (un[h] ** 2).sum()
            xt = (y - xm2) / xs2
        j_value = float(J.detach()) if record_iterations else None
        J.backward()
        opt.step()
        if record_iterations:
            un_raw = un.detach().clone()
        with torch.no_grad():
            un.clamp_(un_lo, un_hi)
        if record_iterations:
            un_projected = un.detach().clone()
            n_clamped = int((~torch.isclose(un_raw, un_projected)).sum().item())
            # Section 3.5's Proposition-1 bridge is scoped to the FIRST
            # horizon block only (`un[0]`/`Delta u_0`) -- tracked
            # separately from `n_clamped_coords` (whole-horizon count,
            # Section 3.4 item 7) so `projection_active` downstream
            # reflects clamp activity on the SAME block the bridge
            # quantities are computed at, not an aggregate over blocks the
            # bridge never touches.
            n_clamped_block0 = int((~torch.isclose(un_raw[0], un_projected[0])).sum().item())
            # J at the PROJECTED (post-update, post-clamp) plan -- the
            # original algorithm never evaluates this (it only computes J
            # once per iteration, at `un_before`, to get the gradient) --
            # needed so `j_learned_before -> j_learned_after` gives a
            # genuine SIGNED decrease, mirroring the counterfactual
            # objective's own before/after pair.
            j_learned_after = learned_objective_value(seq, x_phys, un_projected.numpy(), horizon, rho_u,
                                                       xm, xs, ym, ys)
            iteration_records.append(dict(
                un_before=un_before.numpy().copy(), j_learned=j_value, j_learned_after=j_learned_after,
                un_raw=un_raw.numpy().copy(), un_projected=un_projected.numpy().copy(),
                n_clamped_coords=n_clamped, n_clamped_coords_block0=n_clamped_block0,
            ))
    u0n = un[0].detach().numpy().copy()
    return u0n, un.detach().clone(), iteration_records


# ---------------------------------------------------------------------------
# Battery 1 (+ Battery 4 piggybacked) / Battery 2 -- single unified
# rollout function. `record_iterations=False` -> Battery 1's light
# trajectory-level rollout (Section 2 item 1/4). `record_iterations=True`
# -> Battery 2's full per-control-step + per-iteration record (Section 2
# item 2, Section 3.3/3.4), B/E only.
# ---------------------------------------------------------------------------
def single_cstr_rollout(seq, norm, x0: np.ndarray, horizon: int, budget: int, steps: int, lr: float,
                         rho_u: float, success_V: float, un_lo: torch.Tensor, un_hi: torch.Tensor,
                         sim, input_lo: np.ndarray, input_hi: np.ndarray,
                         record_iterations: bool = False) -> dict:
    xm, xs, ym, ys = norm
    xm2, xs2 = xm[:2], xs[:2]
    x = np.asarray(x0, float)
    un = torch.zeros(horizon, 2)
    Vtraj = [float(x @ P @ x)]
    x_traj = [x.copy()]
    us = []
    step_wall_clock = []
    step_records = [] if record_iterations else None
    solver_completed = True
    failure_reason = None
    failed_at_step = None
    try:
        for step_i in range(steps):
            warm_start_un = un.detach().clone() if record_iterations else None
            t0 = time.perf_counter()
            u0n, un_next, iteration_records = _adam_mpc_step(
                seq, norm, x, un, horizon, budget, lr, rho_u, un_lo, un_hi,
                record_iterations=record_iterations)
            t1 = time.perf_counter()
            step_wall_clock.append(t1 - t0)
            u0 = np.clip(u0n * xs[2:].numpy() + xm[2:].numpy(), input_lo, input_hi)
            x_before = x.copy()
            V_before = float(x_before @ P @ x_before)
            if record_iterations:
                # Predicted next state/V (Section 3.3): the LEARNED
                # model's own forward rollout of the FINAL plan's first
                # action, applied at the current (pre-step) state -- for
                # comparison against the REALIZED next state below.
                with torch.no_grad():
                    x0n = torch.tensor((x_before - xm2.numpy()) / xs2.numpy(), dtype=torch.float32).view(1, 2)
                    y_pred = (seq(torch.cat([x0n, un_next[0].view(1, 2)], dim=1)) * ys + ym).numpy().reshape(-1)
                V_pred = float(y_pred @ P @ y_pred)
            x = sim.step(x, u0)
            V_after = float(x @ P @ x)
            x_traj.append(x.copy())
            Vtraj.append(V_after)
            us.append(u0)
            if record_iterations:
                step_records.append(dict(
                    step_index=step_i, x_before=x_before, u0=u0,
                    warm_start_un=warm_start_un.numpy().copy(), final_plan_un=un_next.numpy().copy(),
                    predicted_next_state=y_pred.copy(), predicted_V=V_pred,
                    V_before=V_before, V_after_realized=V_after,
                    realized_decrease=V_before - V_after,
                    solve_wall_clock_seconds=t1 - t0,
                    iterations=iteration_records,
                ))
            un = torch.cat([un_next[1:], un_next[-1:]])
    except alib.SIMULATOR_FAILURE_EXCEPTIONS as e:
        solver_completed = False
        failure_reason = repr(e)
        failed_at_step = len(us)

    n_steps_completed = len(us)
    lyapunov_increase_count = int(sum(Vtraj[i + 1] > Vtraj[i] + 1e-9 for i in range(len(Vtraj) - 1)))
    us_arr = np.asarray(us) if us else np.zeros((0, 2))
    tv = float(np.sum(np.abs(np.diff(us_arr / (input_hi - input_lo), axis=0)))) if len(us_arr) > 1 else 0.0
    tv_per_actuator_normalized = (
        np.sum(np.abs(np.diff(us_arr / (input_hi - input_lo), axis=0)), axis=0).tolist()
        if len(us_arr) > 1 else [0.0, 0.0]
    )
    tv_per_actuator_physical = (
        np.sum(np.abs(np.diff(us_arr, axis=0)), axis=0).tolist() if len(us_arr) > 1 else [0.0, 0.0]
    )
    # Section 3.7: on failure, settling_time is the fixed steps+1 sentinel,
    # NEVER imputed from a partial (pre-failure) trajectory.
    settling_time = compute_settling_time(Vtraj, success_V, steps) if solver_completed else steps + 1
    timed = step_wall_clock[TIMING_WARMUP_STEPS:] if len(step_wall_clock) > TIMING_WARMUP_STEPS else []

    summary = dict(
        final_V=(Vtraj[-1] if solver_completed else float("nan")),
        max_V=float(max(Vtraj)),
        success=bool(solver_completed and Vtraj[-1] <= success_V),
        lyapunov_increase_count=lyapunov_increase_count,
        tv=tv,
        input_tv_per_actuator_physical=tv_per_actuator_physical,
        input_tv_per_actuator_normalized=tv_per_actuator_normalized,
        settling_time=settling_time,
        settling_time_hours=(settling_time * sim.dt_hr if solver_completed else None),
        solver_completed=solver_completed,
        failure_reason=failure_reason,
        failed_at_step=failed_at_step,
        n_steps_completed=n_steps_completed,
        rollout_solver_core_wall_clock_seconds=dict(
            median=float(np.median(timed)) if timed else float("nan"),
            p95=float(np.percentile(timed, 95)) if timed else float("nan"),
            raw=step_wall_clock,
        ),
        # `record_iterations=False` (Battery 1/4): no extra audit
        # instrumentation runs, so this is genuinely 0.0. `record_iterations=
        # True` (Battery 2): the Section 3.4/3.5 enrichment happens AFTER
        # this function returns (it needs the caller's `sim`/box-bound
        # context) -- left as `None` here, a caller MUST measure and patch
        # both wall-clock fields below via `apply_measured_audit_wall_clock`
        # once enrichment is done, rather than silently reporting 0.0 for
        # instrumentation that did, in fact, run.
        rollout_audit_counterfactual_wall_clock_seconds=(0.0 if not record_iterations else None),
        rollout_instrumented_total_wall_clock_seconds=(
            float(np.sum(step_wall_clock)) if not record_iterations else None),
    )
    return dict(summary=summary, x_traj=x_traj, us=us, Vtraj=Vtraj, step_records=step_records)


def apply_measured_audit_wall_clock(summary: dict, audit_wall_clock_seconds: float) -> None:
    """Patches `summary`'s two audit-dependent timing fields (left `None`
    by `single_cstr_rollout(record_iterations=True)`, Section 3.2) once
    the caller has actually measured how long the Section 3.4/3.5
    enrichment took -- never a silent 0.0 for instrumentation that ran.

    `rollout_instrumented_total_wall_clock_seconds` MUST be built from
    the LOGGED (detailed) pass's OWN wall-clock total -- by the driver's
    calling convention, that total is stashed under `summary[
    "rollout_instrumented_wall_clock_with_logging_overhead_seconds"]`
    (the caller renames the detailed pass's own raw timing there BEFORE
    calling this function, and only then patches in the separate CLEAN
    pass's timing under `rollout_solver_core_wall_clock_seconds` for
    "core" reporting). Using the clean-pass total here instead would
    silently UNDERCOUNT the real instrumented cost: the logged pass's own
    per-iteration overhead now includes the Section 3.4 item 4
    `j_learned_after` evaluation, which happens INSIDE that pass's timed
    loop, not in this function's separately-measured post-hoc audit
    window."""
    logged_pass_total = float(np.sum(
        summary["rollout_instrumented_wall_clock_with_logging_overhead_seconds"]["raw"]))
    summary["rollout_audit_counterfactual_wall_clock_seconds"] = float(audit_wall_clock_seconds)
    summary["rollout_instrumented_total_wall_clock_seconds"] = logged_pass_total + float(audit_wall_clock_seconds)


def compute_settling_time(Vtraj: list, success_V: float, steps: int) -> int:
    """Section 6 item 1: first index `t` (0 = initial state, before any
    control) such that `V` drops below `success_V` and stays below it for
    the rest of the rollout. `steps + 1` sentinel if never achieved."""
    n = len(Vtraj)
    for t in range(n):
        if all(v <= success_V for v in Vtraj[t:]):
            return t
    return steps + 1


# ---------------------------------------------------------------------------
# Battery 1 sanity check (Section 2 item 4: "no compatibility gate is
# strictly required here, though a cheap sanity check ... is still worth
# including") -- implemented at the strongest, cheapest granularity: a
# full per-IC comparison against the ORIGINAL frozen `closed_loop()`.
# ---------------------------------------------------------------------------
BATTERY1_SANITY_EXACT_KEYS = ("success", "lyapunov_increase_count")
BATTERY1_SANITY_ISCLOSE_KEYS = ("final_V", "max_V", "tv")
BATTERY1_SANITY_FIELD_MAP = {}  # both sides already use this module's field names for these keys


def battery1_sanity_compare(instrumented_summary: dict, original_summary: dict) -> dict:
    """`original_summary` = `budget_sweep_solver.closed_loop()`'s own
    return dict (field `violations`, not yet renamed) -- mapped onto this
    module's `lyapunov_increase_count` per Section 1."""
    original_mapped = dict(original_summary)
    original_mapped["lyapunov_increase_count"] = original_mapped.pop("violations")
    return sio.gate_compare_dict(instrumented_summary, original_mapped,
                                  exact_keys=BATTERY1_SANITY_EXACT_KEYS,
                                  isclose_keys=BATTERY1_SANITY_ISCLOSE_KEYS)


# Gate 1 (controller-equivalence, Battery 2, Adam) -- same comparison,
# used for Battery 2's detailed-audit checkpoints/ICs specifically.
GATE1_EXACT_KEYS = BATTERY1_SANITY_EXACT_KEYS
GATE1_ISCLOSE_KEYS = BATTERY1_SANITY_ISCLOSE_KEYS


def gate1_compare(instrumented_summary: dict, original_summary: dict) -> dict:
    return battery1_sanity_compare(instrumented_summary, original_summary)


# Gate 2 (legacy-summary gate, Batteries 1/2) -- new evaluator's seed-level
# aggregate success (mean over the 5 ICs, matching
# `_closed_loop_success_rate()`'s own aggregation) vs. M2's recorded value.
def gate2_compare(new_seed_level_success_rate: float, m2_recorded_success_rate: float) -> dict:
    return sio.gate_compare_dict(
        dict(closed_loop_success=new_seed_level_success_rate),
        dict(closed_loop_success=m2_recorded_success_rate),
        exact_keys=(), isclose_keys=("closed_loop_success",))


# ---------------------------------------------------------------------------
# Section 3.5 enrichment: one-step residual/gradient bridge, computed
# post-hoc over Battery 2's already-collected `step_records`
# (`record_iterations=True`), B/E only, first horizon block only.
# ---------------------------------------------------------------------------
def enrich_step_record_with_residual_bridge(seq, sim, step_record: dict, un_lo_np: np.ndarray,
                                             un_hi_np: np.ndarray, xm: torch.Tensor, xs: torch.Tensor,
                                             ym: torch.Tensor, ys: torch.Tensor, input_lo: np.ndarray,
                                             input_hi: np.ndarray) -> None:
    """Mutates `step_record["iterations"]` in place, adding a
    `residual_bridge` dict to every iteration -- Phi(x) is computed ONCE
    per control step (it does not depend on `u_n`) and reused across all
    of that step's logged iterations."""
    from cstr.sontag import saturated_sontag_np
    x = step_record["x_before"]
    phi_x = saturated_sontag_np(sim, x, input_lo, input_hi, P)
    for it_rec in step_record["iterations"]:
        u_n0 = it_rec["un_before"][0]
        raw_step0 = it_rec["un_raw"][0] - u_n0
        projected_step0 = it_rec["un_projected"][0] - u_n0
        g_hat, grad_hat = learned_residual_value_and_grad(seq, x, u_n0, phi_x, xm, xs, ym, ys)
        g_true, grad_true, fd_diag = true_residual_value_and_grad(
            sim, x, u_n0, phi_x, un_lo_np, un_hi_np, xm, xs)
        norm_hat = float(np.linalg.norm(grad_hat))
        norm_true = float(np.linalg.norm(grad_true))
        excluded = (norm_hat <= 1e-9) or (norm_true <= 1e-9) or bool(np.any(np.isnan(grad_true)))
        cosine = float("nan") if excluded else float(np.dot(grad_hat, grad_true) / (norm_hat * norm_true))
        it_rec["residual_bridge"] = dict(
            g_hat=g_hat, g_true=g_true,
            grad_hat=grad_hat.tolist(), grad_true=grad_true.tolist(),
            cosine=cosine, raw_inner_product=float(np.dot(grad_hat, grad_true)), excluded=excluded,
            directional_derivative_true_raw_step=float(np.dot(grad_true, raw_step0)),
            directional_derivative_true_projected_step=float(np.dot(grad_true, projected_step0)),
            fd_diagnostics=fd_diag,
            # Scoped to the FIRST horizon block only (`n_clamped_coords_
            # block0`), matching the bridge's own Proposition-1 scope --
            # NOT the whole-horizon `n_clamped_coords`, which would flag
            # "projection active" from a LATER block's clamp that has no
            # bearing on this block's directional-derivative quantities.
            projection_active=int(it_rec["n_clamped_coords_block0"]) > 0,
        )


def enrich_step_record_with_counterfactual(sim, step_record: dict, horizon: int, rho_u: float,
                                            xm: torch.Tensor, xs: torch.Tensor, ym: torch.Tensor,
                                            ys: torch.Tensor) -> None:
    """Section 3.4 item 5: per-iteration counterfactual-true-objective, at
    BOTH `un_before` (the candidate plan entering this iteration) AND
    `un_projected` (the plan this iteration actually produced, after
    clamping) -- the before/after PAIR (not `un_before` alone) is what
    lets a reader reconstruct this iteration's own counterfactual
    decrease, mirroring GGN's own before/after treatment (Section 3.4
    item 6) rather than leaving Adam asymmetric with it. Both use the
    EXACT SAME functional form as `J_learned` (Section 6 item 7), rolled
    through the TRUE plant from the step's real state, never applied."""
    x = step_record["x_before"]
    for it_rec in step_record["iterations"]:
        it_rec["counterfactual_true_objective_before"] = counterfactual_true_objective(
            sim, x, it_rec["un_before"], horizon, rho_u, xm, xs, ym, ys)
        it_rec["counterfactual_true_objective_after"] = counterfactual_true_objective(
            sim, x, it_rec["un_projected"], horizon, rho_u, xm, xs, ym, ys)


# ---------------------------------------------------------------------------
# Section 3.6 -- state-constraint statistics (piggybacked on Battery 1's
# rollouts; Battery 5 reuses this for the two-CSTR audit with
# `temperature_state_indices=(1, 3)`).
# ---------------------------------------------------------------------------
def state_constraint_stats(x_traj: list, us: list, sim, input_lo: np.ndarray, input_hi: np.ndarray,
                            cap_deviation_K: float = 70.0, temperature_state_indices: tuple = (1,)) -> dict:
    x_arr = np.asarray(x_traj, dtype=float)
    dt_hr = sim.dt_hr
    # `x_traj` has `len(us) + 1` points: the initial state (t=0, before any
    # control) plus one POST-STEP state per applied action. max/absolute/
    # cap-satisfied deliberately use the FULL trajectory (t=0 included --
    # worst-case disclosure should cover the given IC too). frequency/
    # integral instead use only the `len(us)` post-step samples: each of
    # those corresponds to exactly one elapsed `dt_hr` control interval,
    # whereas t=0 has no such interval attributable to it -- including it
    # in a `dt_hr`-weighted sum would overcount elapsed time by one step.
    x_arr_post_step = x_arr[1:] if len(x_arr) > 1 else x_arr[0:0]
    per_index = {}
    max_devs = []
    for idx in temperature_state_indices:
        t_dev_full = x_arr[:, idx]
        t_dev_post_step = x_arr_post_step[:, idx] if len(x_arr_post_step) else np.zeros(0)
        cap_abs = float(sim.xs_abs[idx]) + cap_deviation_K
        t_abs_full = t_dev_full + float(sim.xs_abs[idx])
        max_dev = float(np.max(t_dev_full))
        max_devs.append(max_dev)
        # STRING key (never a raw int): a dict with int keys survives an
        # in-memory equality check but NOT a JSON write/read round-trip
        # (JSON object keys are always strings), which would make every
        # self-verifying write of a payload containing this dict fail
        # with a spurious round-trip mismatch -- matches this codebase's
        # own established convention (e.g. `lambda_jac_full_grid_closed_
        # loop={str(lam): v ...}` in `ablation_driver_2026_08_31.py`).
        per_index[str(int(idx))] = dict(
            max_temperature_deviation_K=max_dev,
            max_temperature_absolute_K=float(np.max(t_abs_full)),
            cap_deviation_K=float(cap_deviation_K),
            cap_absolute_K=cap_abs,
            temperature_violation_frequency=(
                float(np.mean(t_dev_post_step > cap_deviation_K)) if len(t_dev_post_step) else float("nan")),
            temperature_violation_integral=float(dt_hr * np.sum(np.maximum(0.0, t_dev_post_step - cap_deviation_K))),
        )
    us_arr = np.asarray(us, dtype=float) if len(us) else np.zeros((0, len(input_lo)))
    sat_tol = 1e-6 * (input_hi - input_lo)
    saturation_frequency = {}
    for i in range(len(input_lo)):
        if len(us_arr) == 0:
            saturation_frequency[str(i)] = float("nan")
        else:
            at_bound = ((np.abs(us_arr[:, i] - input_lo[i]) <= sat_tol[i]) |
                        (np.abs(us_arr[:, i] - input_hi[i]) <= sat_tol[i]))
            saturation_frequency[str(i)] = float(np.mean(at_bound))
    # Section 1 item 2: `solver_completed` (numerical completion),
    # `input_box_satisfied` (THIS field, corrected), and `temperature_
    # cap_satisfied_posthoc` are the THREE separate fields a single
    # `feasible` boolean used to conflate -- never merged back into one.
    #
    # `input_box_satisfied` is a genuine BOX-FEASIBILITY check -- `True`
    # iff every applied action satisfies `lo - tol <= u <= hi + tol` and
    # is finite. A value sitting exactly AT the boundary is still box-
    # FEASIBLE (the constraint is `<=`/`>=`, not `<`/`>`) -- corrected
    # from an earlier version that conflated this with "never saturated"
    # (which would have wrongly reported `False` for a value legitimately
    # sitting at the bound). In THIS codebase every applied action is
    # always clamped before use (Adam/GGN both clip), so this check is
    # trivially `True` in practice, but it is a REAL, independently
    # verified check here (not assumed), and remains correct/meaningful
    # if ever called on an un-clamped trajectory.
    #
    # `input_never_saturated` is the SEPARATE diagnostic the earlier,
    # incorrect `input_box_satisfied` definition actually computed --
    # `True` iff no actuator was EVER at its bound for the whole rollout
    # (every per-actuator saturation frequency, below, is exactly 0).
    # This is genuinely useful (whether the solver's solution needed to
    # be pinned at all) but is NOT a feasibility statement, so it must
    # not share the "satisfied" framing that implies constraint
    # violation would otherwise have occurred.
    if len(us_arr) == 0:
        input_box_satisfied = None
    else:
        finite_ok = np.all(np.isfinite(us_arr))
        within_box = np.all((us_arr >= (input_lo - sat_tol)) & (us_arr <= (input_hi + sat_tol)))
        input_box_satisfied = bool(finite_ok and within_box)
    sat_values = list(saturation_frequency.values())
    if not sat_values or any(not np.isfinite(v) for v in sat_values):
        input_never_saturated = None  # undefined (e.g. zero applied actions), never a false "False"
    else:
        input_never_saturated = bool(all(v == 0.0 for v in sat_values))
    return dict(
        per_temperature_index=per_index,
        temperature_cap_satisfied_posthoc=bool(max(max_devs) <= cap_deviation_K) if max_devs else None,
        input_box_satisfied=input_box_satisfied,
        input_never_saturated=input_never_saturated,
        input_saturation_frequency=saturation_frequency,
        denominator_note=(
            f"max_temperature_*/cap_satisfied use the FULL recorded trajectory "
            f"({len(x_traj)} points, including the initial state at t=0); "
            f"violation_frequency/violation_integral use only the "
            f"{max(len(x_traj) - 1, 0)} post-step samples (one per elapsed dt_hr "
            f"interval, t=0 excluded to avoid overcounting elapsed time by one step); "
            f"saturation frequency denominator is total applied control steps ({len(us)})"
        ),
    )


# ---------------------------------------------------------------------------
# Section 4a Gate 3 (GGN implementation-equivalence) + Battery 3 driver.
# ---------------------------------------------------------------------------
def ggn_closed_loop_instrumented(seq, norm, x0: np.ndarray, horizon: int, budget: int, steps: int,
                                  rho_u: float, success_V: float, un_lo: torch.Tensor, un_hi: torch.Tensor,
                                  objective: str, sim, log_iterations: bool = True) -> dict:
    """Mirrors `ggn_mpc_probe.ggn_closed_loop` EXACTLY (same
    Levenberg-Marquardt + backtracking algorithm, same literals) --
    verified via Gate 3 before its output is trusted.

    `log_iterations=True` (default): full per-iteration records (lam,
    accepted, alpha, pred_dec, per-accepted-update counterfactual-true-
    objective, Section 3.4 item 6) -- but Python dict/list-append
    bookkeeping at this granularity (up to `budget` per control step)
    itself measurably contaminates a naive wall-clock, on top of the
    counterfactual audit computation. `log_iterations=False`: skips ALL
    of that bookkeeping (still runs the IDENTICAL core LM/backtracking
    math, unchanged) -- this is the genuinely clean per-step timing
    source; callers should run BOTH: once with `log_iterations=False`
    for `rollout_solver_core_wall_clock_seconds`, once with `True` for
    the Section 3.4 diagnostic data (whose OWN timing is then only
    reported as `instrumented_wall_clock_with_logging_overhead_seconds`,
    never claimed as "core")."""
    import ggn_mpc_probe as ggn
    xm, xs, ym, ys = norm
    x = np.asarray(x0, float)
    Vtraj = [float(x @ P @ x)]
    n = horizon * 2
    U = torch.zeros(n)
    un_lo_f = un_lo.repeat(horizon)
    un_hi_f = un_hi.repeat(horizon)
    accepted = total_it = true_incr = 0
    pred_dec_applied, pred_dec_full, true_dec = [], [], []
    counterfactual_dec_applied = []
    step_records = [] if log_iterations else None
    step_core_wall_clock = []
    step_audit_wall_clock = []
    solver_completed = True
    failure_reason = None
    failed_at_step = None
    try:
        for step_i in range(steps):
            resfn = ggn.make_residual_fn(seq, norm, x, horizon, rho_u, objective)
            lam = 1e-2
            iteration_records = [] if log_iterations else None
            audit_time_this_step = 0.0
            t0 = time.perf_counter()
            for it_i in range(budget):
                total_it += 1
                U = U.detach()
                r = resfn(U)
                J = torch.autograd.functional.jacobian(resfn, U, vectorize=True, strategy="forward-mode")
                obj0 = 0.5 * float(r @ r)
                g = J.t() @ r
                JTJ = J.t() @ J
                H = JTJ + lam * torch.eye(n)
                try:
                    delta = torch.linalg.solve(H, -g)
                except RuntimeError:
                    lam *= 10
                    if log_iterations:
                        iteration_records.append(dict(iteration_index=it_i, accepted=False,
                                                      lam_before=lam / 10, singular_solve=True))
                    continue
                pred_full = float(-(g @ delta) - 0.5 * (delta @ JTJ @ delta))
                acc = False
                U_old = U
                pred_applied = 0.0
                accepted_alpha = None
                for alpha in (1.0, 0.5, 0.25, 0.125):
                    Ut = torch.clamp(U_old + alpha * delta, un_lo_f, un_hi_f)
                    rt = resfn(Ut)
                    if 0.5 * float(rt @ rt) < obj0:
                        dU = Ut - U_old
                        pred_applied = float(-(g @ dU) - 0.5 * (dU @ JTJ @ dU))
                        U = Ut
                        acc = True
                        accepted_alpha = alpha
                        break
                lam_before_update = lam
                cf_dec = None
                if acc:
                    accepted += 1
                    lam = max(lam * 0.5, 1e-4)
                    if log_iterations:
                        pred_dec_full.append(max(pred_full, 0.0))
                        pred_dec_applied.append(max(pred_applied, 0.0))
                        ta0 = time.perf_counter()
                        cf_before = counterfactual_true_objective(sim, x, U_old.detach().numpy(), horizon,
                                                                   rho_u, xm, xs, ym, ys)
                        cf_after = counterfactual_true_objective(sim, x, U.detach().numpy(), horizon,
                                                                  rho_u, xm, xs, ym, ys)
                        ta1 = time.perf_counter()
                        audit_time_this_step += ta1 - ta0
                        cf_dec = cf_before - cf_after
                        counterfactual_dec_applied.append(cf_dec)
                else:
                    lam = min(lam * 10, 1e4)
                if log_iterations:
                    iteration_records.append(dict(
                        iteration_index=it_i, accepted=acc, lam_before=lam_before_update,
                        accepted_alpha=accepted_alpha, pred_dec_full=max(pred_full, 0.0),
                        pred_dec_applied=(max(pred_applied, 0.0) if acc else None),
                        counterfactual_dec_applied=cf_dec, singular_solve=False,
                    ))
            t1 = time.perf_counter()
            step_core_wall_clock.append((t1 - t0) - audit_time_this_step)
            step_audit_wall_clock.append(audit_time_this_step)
            u0 = np.clip(U.view(horizon, 2)[0].numpy() * xs[2:].numpy() + xm[2:].numpy(),
                         ggn.INPUT_LO, ggn.INPUT_HI)
            Vbefore = float(x @ P @ x)
            x = sim.step(x, u0)
            Vafter = float(x @ P @ x)
            true_dec.append(Vbefore - Vafter)
            if Vafter > Vbefore + 1e-9:
                true_incr += 1
            Vtraj.append(Vafter)
            if log_iterations:
                step_records.append(dict(step_index=step_i, iterations=iteration_records,
                                         solve_core_wall_clock_seconds=step_core_wall_clock[-1],
                                         solve_audit_wall_clock_seconds=step_audit_wall_clock[-1]))
            U = torch.cat([U.view(horizon, 2)[1:], U.view(horizon, 2)[-1:]]).reshape(-1)
    except alib.SIMULATOR_FAILURE_EXCEPTIONS as e:
        solver_completed = False
        failure_reason = repr(e)
        failed_at_step = len(true_dec)

    timed = step_core_wall_clock[TIMING_WARMUP_STEPS:] if len(step_core_wall_clock) > TIMING_WARMUP_STEPS else []
    instrumented_timing_label = ("rollout_instrumented_wall_clock_with_logging_overhead_seconds" if log_iterations
                                 else "rollout_solver_core_wall_clock_seconds")
    result = dict(
        final_V=(Vtraj[-1] if solver_completed else float("nan")),
        max_V=float(max(Vtraj)),
        success=bool(solver_completed and Vtraj[-1] <= success_V),
        solver_completed=solver_completed,
        failure_reason=failure_reason, failed_at_step=failed_at_step,
        accepted_frac=accepted / max(total_it, 1),
        true_incr_frac=(true_incr / max(len(true_dec), 1)) if solver_completed else float("nan"),
        mean_pred_dec_applied=float(np.mean(pred_dec_applied)) if pred_dec_applied else 0.0,
        mean_pred_dec_full=float(np.mean(pred_dec_full)) if pred_dec_full else 0.0,
        mean_true_dec=float(np.mean(true_dec)) if true_dec else float("nan"),
        mean_counterfactual_dec_applied=(
            float(np.mean(counterfactual_dec_applied)) if counterfactual_dec_applied else 0.0),
        step_records=step_records,
        log_iterations=log_iterations,
    )
    # `log_iterations=False`'s timing is the trustworthy "core" number;
    # `log_iterations=True`'s timing is diagnostic-only (see docstring) --
    # deliberately given a DIFFERENT key name so a caller can never
    # accidentally read the contaminated number under the "core" key.
    result[instrumented_timing_label] = dict(
        median=float(np.median(timed)) if timed else float("nan"),
        p95=float(np.percentile(timed, 95)) if timed else float("nan"),
        raw=step_core_wall_clock,
    )
    if not log_iterations:
        result["rollout_audit_counterfactual_wall_clock_seconds"] = 0.0
        result["rollout_instrumented_total_wall_clock_seconds"] = float(np.sum(step_core_wall_clock))
    else:
        result["rollout_audit_counterfactual_wall_clock_seconds"] = float(np.sum(step_audit_wall_clock))
        result["rollout_instrumented_total_wall_clock_seconds"] = (
            float(np.sum(step_core_wall_clock)) + float(np.sum(step_audit_wall_clock)))
    return result


GATE3_EXACT_KEYS = ("success",)
GATE3_ISCLOSE_KEYS = ("final_V", "accepted_frac")


def gate3_compare(instrumented: dict, original: dict) -> dict:
    """`original` = `ggn_mpc_probe.ggn_closed_loop()`'s OWN unmodified
    return dict, called with the SAME checkpoint/seed/IC/budget."""
    return sio.gate_compare_dict(instrumented, original, exact_keys=GATE3_EXACT_KEYS,
                                  isclose_keys=GATE3_ISCLOSE_KEYS)


# ---------------------------------------------------------------------------
# Section 8 -- two-CSTR minimum audit (Battery 5), four phases.
# ---------------------------------------------------------------------------
TWO_CSTR_SEEDS = list(range(10))
TWO_CSTR_CONFIGS = {"value_only": (0.003, 0.0), "value+grad": (0.003, 0.2)}
TWO_CSTR_GATE4_EXACT_KEYS = ("b3", "b20", "frac_neg")
TWO_CSTR_GATE4_ISCLOSE_KEYS = ("test_mse", "align", "state_jac_med")


def _two_cstr_test_mse(model, data: tuple, norm: tuple) -> float:
    """Recomputes `two_cstr_lgrad.train()`'s own tail (test MSE on the
    held-out `test_idx` split) for an ALREADY-trained/loaded model --
    used identically whether `reconstruct_two_cstr_replica` just trained
    fresh or resumed from an existing checkpoint, so both paths converge
    on the SAME metrics-computation code rather than one trusting a
    stashed value from `train()`'s return and the other recomputing it a
    different way."""
    XU, Y, xm, xs, ym, ys, _, test_idx = data
    xm_t, xs_t, ym_t, ys_t = norm
    Xn = torch.tensor((XU - xm) / xs)
    model.eval()
    with torch.no_grad():
        pv = (model(Xn[test_idx]) * ys_t + ym_t).numpy()
    return float(np.mean(((pv - Y[test_idx]) / ys) ** 2))


def reconstruct_two_cstr_replica(config_name: str, seed: int, checkpoint_dir: Path) -> dict:
    """Phase 1: retrain EXACTLY `two_cstr_lgrad.py`'s recipe (same
    `configs` entries, same `SEEDS = list(range(10))`, same
    `torch.manual_seed(seed)`, `n_data=12000`/`epochs=70`), then
    `torch.save()` the result -- naming it `metric_compatible_replica`
    (Section 8: no original checkpoint was ever persisted by
    `two_cstr_lgrad.py`, so this is never called "the original model").

    Resume-or-verify: if the deterministic checkpoint path already exists
    (a prior attempt trained and saved it, then crashed/was interrupted
    before Gate 4 or the Replica Manifest completed), it is REUSED rather
    than retrained -- architecture/training are fully deterministic given
    (seed, config), so an existing file is trusted as the SAME replica a
    fresh run would produce, verified by recomputing every metric fresh
    against it (never by trusting a stashed value)."""
    import two_cstr_lgrad as T
    lam_val, lam_grad = TWO_CSTR_CONFIGS[config_name]
    data = T.make_data(n=12000)
    norm = T.make_norm(data)
    un_lo = torch.tensor((T.BOX_LO - data[2][4:]) / data[3][4:], dtype=torch.float32)
    un_hi = torch.tensor((T.BOX_HI - data[2][4:]) / data[3][4:], dtype=torch.float32)
    ckpt_path = Path(checkpoint_dir) / f"two_cstr_{config_name}_seed{seed}_metric_compatible_replica.pt"

    if ckpt_path.exists():
        model = T.build_lcnn(seed)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
    else:
        Phi, Yphi = T.precompute_phi(data[0])
        true_ug = T.precompute_true_ugrad(data[0], data[2], data[3])
        model, _ = T.train(seed, data, Phi, Yphi, true_ug, lam_val=lam_val, lam_grad=lam_grad, epochs=70)
        sio.atomic_write_torch_no_overwrite(ckpt_path, model.state_dict())

    seq = T.freeze(model)
    mse = _two_cstr_test_mse(model, data, norm)
    al, fn = T.align_med(seq, data, norm, seed=seed)
    b3 = float(np.mean([T.closed_loop(seq, norm, x0, 3, 3, 120, 0.2, 0.01, 2.0, un_lo, un_hi)["success"]
                        for x0 in T.ICS]))
    b20 = float(np.mean([T.closed_loop(seq, norm, x0, 3, 20, 120, 0.2, 0.01, 2.0, un_lo, un_hi)["success"]
                         for x0 in T.ICS]))
    jx = T.jac_state_med(seq, data, norm, seed=seed)
    metrics = dict(test_mse=mse, align=al, frac_neg=fn, b3=b3, b20=b20, state_jac_med=jx,
                   lam_val=lam_val, lam_grad=lam_grad)

    return dict(metrics=metrics, checkpoint_path=str(ckpt_path),
                checkpoint_sha256=sio.file_sha256(ckpt_path),
                model=model, seq=seq, data=data, norm=norm, un_lo=un_lo, un_hi=un_hi)


def load_two_cstr_legacy_metrics(config_name: str, seed: int,
                                  log_path: Path = None) -> dict:
    import json
    log_path = log_path if log_path is not None else (ROOT / "results" / "interim" / "logs" / "two_cstr_lgrad.json")
    d = json.loads(Path(log_path).read_text())
    return d["per_seed"][str(seed)][config_name]


def gate4_compare(replica_metrics: dict, legacy_metrics: dict) -> dict:
    return sio.gate_compare_dict(replica_metrics, legacy_metrics,
                                  exact_keys=TWO_CSTR_GATE4_EXACT_KEYS,
                                  isclose_keys=TWO_CSTR_GATE4_ISCLOSE_KEYS)


def two_cstr_rollout_instrumented(seq, norm, x0: np.ndarray, horizon: int, budget: int, steps: int,
                                   lr: float, rho_u: float, success_V: float, un_lo: torch.Tensor,
                                   un_hi: torch.Tensor) -> dict:
    """Phase 4: mirrors `two_cstr_lgrad.closed_loop()` exactly (`rho_c=0.0`
    -- no soft temperature penalty in the objective, the protocol's
    'clean apples-to-apples baseline' choice), adding trajectory capture
    for the Section 3.6 state-constraint statistics (state indices 1 and
    3, one per reactor)."""
    import two_cstr_lgrad as T
    xm, xs, ym, ys = norm
    xm4, xs4 = xm[:4], xs[:4]
    x = np.asarray(x0, float)
    x_traj = [x.copy()]
    us = []
    un = torch.zeros(horizon, 4)
    solver_completed = True
    failure_reason = None
    failed_at_step = None
    try:
        for _ in range(steps):
            un = un.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([un], lr=lr)
            x0n = torch.tensor((x - xm4.numpy()) / xs4.numpy(), dtype=torch.float32).view(1, 4)
            for _it in range(budget):
                opt.zero_grad()
                xt = x0n
                J = torch.zeros(())
                for h in range(horizon):
                    xun = torch.cat([xt, un[h].view(1, 4)], 1)
                    y = seq(xun) * ys + ym
                    J = J + (y @ T.Pt * y).sum() + rho_u * (un[h] ** 2).sum()
                    xt = (y - xm4) / xs4
                J.backward()
                opt.step()
                with torch.no_grad():
                    un.clamp_(un_lo, un_hi)
            u0 = un[0].detach().numpy() * xs[4:].numpy() + xm[4:].numpy()
            x = T.step(x, u0)
            x_traj.append(x.copy())
            us.append(u0)
            un = torch.cat([un[1:].detach(), un[-1:].detach()])
    except alib.SIMULATOR_FAILURE_EXCEPTIONS as e:
        solver_completed = False
        failure_reason = repr(e)
        failed_at_step = len(us)
    Vtraj = [float(T.Vnp(xi)) for xi in x_traj]
    final_V = Vtraj[-1] if solver_completed else float("nan")
    return dict(final_V=final_V, max_V=float(max(Vtraj)),
                success=bool(solver_completed and final_V <= success_V),
                solver_completed=solver_completed, failure_reason=failure_reason,
                failed_at_step=failed_at_step, x_traj=x_traj, us=us, Vtraj=Vtraj)
