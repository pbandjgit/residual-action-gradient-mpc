"""Per-actuator input total variation for the two-CSTR closed-loop metrics (R2.m5).

This is a SEPARATE, read-only-of-originals wrapper. It does not modify
scripts/two_cstr_metrics.py or results/interim/logs/two_cstr_metrics.json.
It reuses the same training/rollout code path (two_cstr_lgrad.py) with the
same seeds, conditions, and hyperparameters as two_cstr_metrics.py, so that
its per-seed scalar outputs (success, settle, cost, aggregate input_tv) can
be checked against the existing frozen per-seed records at a PRE-SPECIFIED
tolerance before the new per-actuator breakdown is trusted for the manuscript.

Pre-specified compatibility tolerances (fixed before running, not tuned to
the result):
  - success (per seed, mean over 5 ICs): abs diff <= 0.2 (at most one IC's
    binary outcome differs)
  - settle (per seed): both NaN, or both finite with relative diff <= 5%
  - cost (per seed): relative diff <= 2%
  - aggregate input_tv (per seed, i.e. sum over actuators of the same
    quantity two_cstr_metrics.py reports): relative diff <= 2%

Only if every one of the 20 (seed, condition) records passes on all four
metrics is the new per-actuator TV breakdown reported as reproduction-checked.

Outputs: results/interim/logs/two_cstr_actuator_tv_2026_09_07.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import two_cstr_lgrad as T  # noqa: E402
from two_cstr_metrics import metrics, record_learned  # noqa: E402  (reuse, do not redefine)

LOG = ROOT / "results" / "interim" / "logs"
FROZEN = LOG / "two_cstr_metrics.json"
SEEDS = list(range(10))
RANGE = T.BOX_HI - T.BOX_LO
ACTUATOR_NAMES = ["C_A10", "Q1", "C_A20", "Q2"]

TOL_SUCCESS_ABS = 0.2
TOL_SETTLE_REL = 0.05
TOL_COST_REL = 0.02
TOL_TV_REL = 0.02


def actuator_tv(U: np.ndarray) -> np.ndarray:
    """Per-actuator total variation, same normalization as metrics()'s scalar TV."""
    dU = np.diff(np.asarray(U), axis=0)
    return np.sum(np.abs(dU) / RANGE, axis=0)  # shape (4,)


def rel_ok(new, old, tol):
    if old == 0:
        return abs(new - old) <= tol
    return abs(new - old) / abs(old) <= tol


def settle_ok(new, old, tol):
    new_nan, old_nan = np.isnan(new), np.isnan(old)
    if new_nan and old_nan:
        return True
    if new_nan != old_nan:
        return False
    return rel_ok(new, old, tol)


def main():
    frozen = json.loads(FROZEN.read_text())
    frozen_per_seed = frozen["per_seed"]

    data = T.make_data(n=12000)
    XU, xm, xs = data[0], data[2], data[3]
    norm = T.make_norm(data)
    un_lo = torch.tensor((T.BOX_LO - xm[4:]) / xs[4:], dtype=torch.float32)
    un_hi = torch.tensor((T.BOX_HI - xm[4:]) / xs[4:], dtype=torch.float32)
    print("precomputing ...", flush=True)
    Phi, Yphi = T.precompute_phi(XU)
    true_ug = T.precompute_true_ugrad(XU, xm, xs)
    hz, steps, lr, rho_u = 3, 120, 0.2, 0.01

    out = {"config": dict(plant="two_cstr_series", seeds=SEEDS, horizon=hz,
                          steps=steps, budget=20,
                          actuator_names=ACTUATOR_NAMES,
                          tolerances=dict(success_abs=TOL_SUCCESS_ABS,
                                          settle_rel=TOL_SETTLE_REL,
                                          cost_rel=TOL_COST_REL,
                                          input_tv_rel=TOL_TV_REL)),
           "per_seed": {}, "reproduction_check": {}, "controllers_actuator_tv": {}}

    all_pass = True
    for tag, lv, lg in [("value_only", 0.003, 0.0), ("value+grad", 0.003, 0.2)]:
        per = []
        checks = []
        frozen_rows = {r["seed"]: r for r in frozen_per_seed[tag]}
        for seed in SEEDS:
            model, _ = T.train(seed, data, Phi, Yphi, true_ug, lam_val=lv, lam_grad=lg, epochs=70)
            seq = T.freeze(model)
            per_ic_tv = []
            rows = []
            for ic in T.ICS:
                Vtraj, U = record_learned(seq, norm, ic, hz, 20, steps, lr, rho_u, un_lo, un_hi)
                m = metrics(Vtraj, U)
                rows.append(m)
                per_ic_tv.append(actuator_tv(U))
            success = float(np.mean([r["success"] for r in rows]))
            settle = float(np.nanmean([r["settle"] for r in rows]))
            cost = float(np.mean([r["cost"] for r in rows]))
            input_tv = float(np.mean([r["input_tv"] for r in rows]))
            tv_per_actuator = np.mean(np.stack(per_ic_tv, axis=0), axis=0)  # (4,)

            fr = frozen_rows[seed]
            seed_pass = (
                abs(success - fr["success"]) <= TOL_SUCCESS_ABS
                and settle_ok(settle, fr["settle"], TOL_SETTLE_REL)
                and rel_ok(cost, fr["cost"], TOL_COST_REL)
                and rel_ok(input_tv, fr["input_tv"], TOL_TV_REL)
            )
            all_pass = all_pass and seed_pass
            checks.append(dict(seed=seed, seed_pass=seed_pass,
                                new=dict(success=success, settle=settle, cost=cost, input_tv=input_tv),
                                frozen=dict(success=fr["success"], settle=fr["settle"],
                                            cost=fr["cost"], input_tv=fr["input_tv"])))
            per.append(dict(seed=seed, success=success, settle=settle, cost=cost,
                             input_tv=input_tv,
                             input_tv_per_actuator=tv_per_actuator.tolist()))
            print(f"{tag} seed={seed} seed_pass={seed_pass} "
                  f"success={success:.2f} cost={cost:.1f} input_tv={input_tv:.2f} "
                  f"actuator_tv={np.round(tv_per_actuator, 2).tolist()}", flush=True)

        out["per_seed"][tag] = per
        out["reproduction_check"][tag] = checks
        arr = np.stack([p["input_tv_per_actuator"] for p in per], axis=0)  # (10,4)
        out["controllers_actuator_tv"][tag] = dict(
            mean=arr.mean(axis=0).tolist(), sd=arr.std(axis=0).tolist())

    out["reproduction_all_pass"] = bool(all_pass)
    LOG.mkdir(parents=True, exist_ok=True)
    out_path = LOG / "two_cstr_actuator_tv_2026_09_07.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\nwrote", out_path, flush=True)
    print("REPRODUCTION_ALL_PASS:", all_pass, flush=True)
    for tag in ["value_only", "value+grad"]:
        m = out["controllers_actuator_tv"][tag]["mean"]
        s = out["controllers_actuator_tv"][tag]["sd"]
        print(tag, "actuator TV mean:", [f"{ACTUATOR_NAMES[i]}={m[i]:.2f}+/-{s[i]:.2f}" for i in range(4)])


if __name__ == "__main__":
    main()
