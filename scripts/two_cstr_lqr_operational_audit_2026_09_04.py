#!/usr/bin/env python3
"""LQR operational audit (R3.4/R3.6 two-CSTR role determination, 2026-09-04).

Pre-registered contract (fixed BEFORE running, per the user's exact
specification):
  - K_LQR is the nominal gain (`two_cstr_lgrad.K_LQR`), never retuned per
    scenario.
  - Same 5 ICs (`two_cstr_lgrad.ICS`), 120 steps, input box (`BOX_LO`/
    `BOX_HI`) as every other two-CSTR result in this codebase.
  - feed-temp +5K / k0 +5% / dH +5%: reuse `two_cstr_realism.py`'s EXISTING
    plant_step objects verbatim (same `TwoCSTRSeriesSimulator(TwoCSTRParams(...))`
    construction), not re-derived.
  - Setpoint tracking: fair baseline u = clip(u_sp - K_LQR@(x-x_sp), lo, hi),
    reusing `two_cstr_tracking.py`'s X_SP/U_SP.
  - Metrics: success, final_V, failure-penalized settling (steps+1 sentinel
    on failure, never imputed from a partial trajectory), accumulated V
    (cost), input TV (normalized), saturation frequency.
  - LQR results are a DETERMINISTIC 5-IC set; the learned-MPC comparison
    numbers are a 10-seed distribution -- never conflated into one number.

No training. No learned model touched. Deterministic, real dynamics only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import two_cstr_lgrad as T                                       # noqa: E402
import two_cstr_tracking as TT                                   # noqa: E402
from two_cstr_realism import NOM, make_plant_step                # noqa: E402
from cstr.two_cstr_series import TwoCSTRSeriesSimulator, TwoCSTRParams  # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"
STEPS = 120
SUCCESS_V = 2.0


def audit_metrics(Vtraj: list, U: list, success_v: float, steps: int) -> dict:
    V = np.asarray(Vtraj)
    success = bool(V[-1] <= success_v)
    above = np.where(V[1:] > success_v)[0]  # exclude t=0
    if success:
        settling = int(above[-1] + 2) if above.size else 0
    else:
        settling = steps + 1  # fixed sentinel, never imputed from a partial trajectory
    cost = float(np.sum(V))
    U = np.asarray(U)
    urange = T.BOX_HI - T.BOX_LO
    dU = np.diff(U, axis=0)
    tv = float(np.sum(np.abs(dU) / urange)) if len(U) > 1 else 0.0
    saturation_freq = float(np.mean(np.any(np.abs(U) >= 0.99 * T.BOX_HI, axis=1)))
    return dict(success=success, final_V=float(V[-1]), settling_time=settling,
               accumulated_V=cost, input_tv=tv, saturation_frequency=saturation_freq)


def rollout_lqr_regulation(plant_step, x0: np.ndarray, steps: int = STEPS) -> dict:
    x = np.asarray(x0, float)
    Vtraj = [float(T.Vnp(x))]
    U = []
    for _ in range(steps):
        u = np.clip(-T.K_LQR @ x, T.BOX_LO, T.BOX_HI)
        U.append(u)
        x = plant_step(x, u)
        Vtraj.append(float(T.Vnp(x)))
    return audit_metrics(Vtraj, U, SUCCESS_V, steps)


def rollout_lqr_tracking(steps: int = STEPS) -> list:
    def Vsp(x):
        d = np.asarray(x, float) - TT.X_SP
        return float(d @ T.P_MAT @ d)

    nominal_step = make_plant_step(TwoCSTRSeriesSimulator(NOM))
    rows = []
    for x0 in T.ICS:
        x = np.asarray(x0, float)
        Vtraj = [Vsp(x)]
        U = []
        for _ in range(steps):
            u = np.clip(TT.U_SP - T.K_LQR @ (x - TT.X_SP), T.BOX_LO, T.BOX_HI)
            U.append(u)
            x = nominal_step(x, u)
            Vtraj.append(Vsp(x))
        rows.append(audit_metrics(Vtraj, U, SUCCESS_V, steps))
    return rows


def main() -> dict:
    scenarios = {
        "nominal": TwoCSTRSeriesSimulator(NOM),
        "feed_temp_dist(+5K)": TwoCSTRSeriesSimulator(TwoCSTRParams(T0=NOM.T0 + 5.0)),
        "mismatch_k0(+5%)": TwoCSTRSeriesSimulator(TwoCSTRParams(k0=NOM.k0 * 1.05)),
        "mismatch_dH(+5%)": TwoCSTRSeriesSimulator(TwoCSTRParams(dH=NOM.dH * 1.05)),
    }
    out = dict(config=dict(plant="two_cstr_series", ics=[ic.tolist() for ic in T.ICS],
                           steps=STEPS, success_V=SUCCESS_V,
                           k_lqr_source="two_cstr_lgrad.K_LQR (nominal, fixed, never retuned)",
                           note="regulation scenarios use +5% only (matches code, not the "
                                "manuscript's erroneous \\pm5%% wording -- see Decision Record)"),
               regulation={}, tracking=None)

    for name, psim in scenarios.items():
        plant_step = make_plant_step(psim)
        rows = [rollout_lqr_regulation(plant_step, ic) for ic in T.ICS]
        agg = dict(
            success=float(np.mean([r["success"] for r in rows])),
            final_V_mean=float(np.mean([r["final_V"] for r in rows])),
            settling_time_mean=float(np.mean([r["settling_time"] for r in rows])),
            accumulated_V_mean=float(np.mean([r["accumulated_V"] for r in rows])),
            input_tv_mean=float(np.mean([r["input_tv"] for r in rows])),
            saturation_frequency_mean=float(np.mean([r["saturation_frequency"] for r in rows])),
        )
        out["regulation"][name] = dict(per_ic=rows, aggregate=agg)

    tracking_rows = rollout_lqr_tracking()
    out["tracking"] = dict(
        per_ic=tracking_rows,
        aggregate=dict(
            success=float(np.mean([r["success"] for r in tracking_rows])),
            final_V_mean=float(np.mean([r["final_V"] for r in tracking_rows])),
            settling_time_mean=float(np.mean([r["settling_time"] for r in tracking_rows])),
            accumulated_V_mean=float(np.mean([r["accumulated_V"] for r in tracking_rows])),
            input_tv_mean=float(np.mean([r["input_tv"] for r in tracking_rows])),
            saturation_frequency_mean=float(np.mean([r["saturation_frequency"] for r in tracking_rows])),
        ),
    )

    LOG.mkdir(parents=True, exist_ok=True)
    out_path = LOG / "two_cstr_lqr_operational_audit_2026_09_04.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("wrote", out_path)
    print(f"\n{'scenario':22} {'success':>8} {'final_V':>9} {'settle':>8} {'cost':>9} {'TV':>7} {'sat':>6}")
    for name, d in out["regulation"].items():
        a = d["aggregate"]
        print(f"{name:22} {a['success']:>8.2f} {a['final_V_mean']:>9.2f} "
              f"{a['settling_time_mean']:>8.1f} {a['accumulated_V_mean']:>9.1f} "
              f"{a['input_tv_mean']:>7.3f} {a['saturation_frequency_mean']:>6.2f}")
    a = out["tracking"]["aggregate"]
    print(f"{'setpoint_tracking':22} {a['success']:>8.2f} {a['final_V_mean']:>9.2f} "
          f"{a['settling_time_mean']:>8.1f} {a['accumulated_V_mean']:>9.1f} "
          f"{a['input_tv_mean']:>7.3f} {a['saturation_frequency_mean']:>6.2f}")
    return out


if __name__ == "__main__":
    main()
