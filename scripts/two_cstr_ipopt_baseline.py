"""Two-CSTR strong-solver stress check: learned LMPC with CasADi/IPOPT.

This is a reviewer-response reserve experiment, not part of the main JPC story.
It asks whether the value-only surrogate can be rescued by a standard NLP solver
applied to the same surrogate-based MPC problem, rather than by the finite-budget
first-order/Gauss-Newton solvers used in the manuscript. Because GroupSort enters
the symbolic model through fmax/fmin, IPOPT often exits at max_iter; the script
keeps those statuses and applies the last iterate only as a stress-test signal.

The frozen GroupSort-LCNN is converted to a CasADi expression:
materialized Björck layers -> affine maps, GroupSort(2) -> fmax/fmin, output
layer -> affine map. No L4CasADi is needed.

Outputs: results/interim/logs/two_cstr_ipopt_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import two_cstr_lgrad as T  # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"


def groupsort_ca(z: ca.MX, n: int) -> ca.MX:
    out = ca.MX.zeros(n)
    for i in range(n // 2):
        a, b = z[2 * i], z[2 * i + 1]
        out[2 * i] = ca.fmax(a, b)
        out[2 * i + 1] = ca.fmin(a, b)
    return out


def build_fhat(seq: torch.nn.Sequential, norm) -> ca.Function:
    """CasADi function (x_phys, u_phys) -> x_next_phys for frozen LCNN."""
    xm, xs, ym, ys = [t.detach().cpu().numpy().astype(float) for t in norm]
    layers = []
    for module in seq:
        if isinstance(module, torch.nn.Linear):
            layers.append(
                (
                    "lin",
                    module.weight.detach().cpu().numpy().astype(float),
                    module.bias.detach().cpu().numpy().astype(float),
                )
            )
        else:
            layers.append(("gs", None, None))

    x = ca.MX.sym("x", 4)
    u = ca.MX.sym("u", 4)
    xn = (x - ca.DM(xm[:4])) / ca.DM(xs[:4])
    un = (u - ca.DM(xm[4:])) / ca.DM(xs[4:])
    z = ca.vertcat(xn, un)
    for kind, W, b in layers:
        if kind == "lin":
            z = ca.mtimes(ca.DM(W), z) + ca.DM(b)
        else:
            z = groupsort_ca(z, int(z.shape[0]))
    y = z * ca.DM(ys) + ca.DM(ym)
    return ca.Function("two_cstr_fhat", [x, u], [y])


def make_solver(fhat: ca.Function, horizon: int, rho_u: float, norm, max_iter: int):
    xm, xs, _, _ = [t.detach().cpu().numpy().astype(float) for t in norm]
    opti = ca.Opti()
    X = opti.variable(4, horizon + 1)
    U = opti.variable(4, horizon)
    x0p = opti.parameter(4)
    opti.subject_to(X[:, 0] == x0p)
    Pd = ca.DM(T.P_MAT)
    um = ca.DM(xm[4:])
    us = ca.DM(xs[4:])
    obj = 0
    for h in range(horizon):
        opti.subject_to(X[:, h + 1] == fhat(X[:, h], U[:, h]))
        for j in range(4):
            opti.subject_to(opti.bounded(float(T.BOX_LO[j]), U[j, h], float(T.BOX_HI[j])))
        un = (U[:, h] - um) / us
        obj += ca.mtimes([X[:, h + 1].T, Pd, X[:, h + 1]]) + rho_u * ca.sumsqr(un)
    opti.minimize(obj)
    opti.solver(
        "ipopt",
        {"print_time": 0, "ipopt.print_level": 0, "ipopt.sb": "yes"},
        {
            "max_iter": int(max_iter),
            "tol": 1e-6,
            "acceptable_tol": 1e-5,
            "acceptable_iter": 10,
            "mu_strategy": "adaptive",
            "warm_start_init_point": "yes",
        },
    )
    return opti, X, U, x0p


def rollout_ipopt(
    fhat: ca.Function,
    norm,
    x0: np.ndarray,
    horizon: int,
    steps: int,
    rho_u: float,
    success_v: float,
    max_iter: int,
) -> dict:
    opti, X, U, x0p = make_solver(fhat, horizon, rho_u, norm, max_iter)
    x = np.asarray(x0, dtype=float)
    vtraj = [float(T.Vnp(x))]
    us = []
    statuses = []
    solve_times = []
    xw = np.tile(x.reshape(-1, 1), (1, horizon + 1))
    uw = np.zeros((4, horizon))
    failures = 0
    for _ in range(steps):
        opti.set_value(x0p, x)
        opti.set_initial(X, xw)
        opti.set_initial(U, uw)
        t0 = time.perf_counter()
        try:
            sol = opti.solve()
            u0 = np.asarray(sol.value(U[:, 0]), dtype=float).reshape(4)
            xw = np.asarray(sol.value(X), dtype=float).reshape(4, horizon + 1)
            uw = np.asarray(sol.value(U), dtype=float).reshape(4, horizon)
            statuses.append(str(opti.stats().get("return_status", "ok")))
        except RuntimeError:
            failures += 1
            status = str(opti.stats().get("return_status", "failed"))
            statuses.append(status)
            # IPOPT often returns a useful last iterate when it hits max_iter.
            # Keep the status count, but apply the last iterate instead of
            # replacing it by an artificial zero input.
            try:
                u0 = np.asarray(opti.debug.value(U[:, 0]), dtype=float).reshape(4)
                xw = np.asarray(opti.debug.value(X), dtype=float).reshape(4, horizon + 1)
                uw = np.asarray(opti.debug.value(U), dtype=float).reshape(4, horizon)
            except Exception:
                u0 = np.zeros(4, dtype=float)
        solve_times.append(1000.0 * (time.perf_counter() - t0))
        u0 = np.clip(u0, T.BOX_LO, T.BOX_HI)
        us.append(u0.copy())
        x = T.step(x, u0)
        vtraj.append(float(T.Vnp(x)))
        xw = np.roll(xw, -1, axis=1)
        xw[:, -1] = xw[:, -2]
        uw = np.roll(uw, -1, axis=1)
        uw[:, -1] = uw[:, -2]
    uarr = np.asarray(us)
    tv = float(np.sum(np.abs(np.diff(uarr, axis=0)) / (T.BOX_HI - T.BOX_LO))) if len(uarr) > 1 else 0.0
    return {
        "success": bool(vtraj[-1] <= success_v),
        "final_V": float(vtraj[-1]),
        "max_V": float(max(vtraj)),
        "state_cost": float(np.sum(vtraj)),
        "input_tv": tv,
        "solver_failures": int(failures),
        "status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
        "mean_solve_ms": float(np.mean(solve_times)),
        "p90_solve_ms": float(np.percentile(solve_times, 90)),
    }


def aggregate(rows: list[dict]) -> dict:
    keys = ["success", "final_V", "max_V", "state_cost", "input_tv", "solver_failures", "mean_solve_ms", "p90_solve_ms"]
    out = {}
    for key in keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=float)
        out[key] = float(np.mean(vals))
        out[key + "_sd"] = float(np.std(vals))
    status_counts: dict[str, int] = {}
    for row in rows:
        for status, count in row["status_counts"].items():
            status_counts[status] = status_counts.get(status, 0) + int(count)
    out["status_counts"] = status_counts
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--n-data", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--rho-u", type=float, default=0.01)
    parser.add_argument("--success-v", type=float, default=2.0)
    parser.add_argument("--max-ics", type=int, default=len(T.ICS))
    args = parser.parse_args()

    data = T.make_data(n=args.n_data)
    XU, xm, xs = data[0], data[2], data[3]
    norm = T.make_norm(data)
    print("precomputing Phi/Yphi and action-gradient labels ...", flush=True)
    phi_all, yphi_all = T.precompute_phi(XU)
    true_ug = T.precompute_true_ugrad(XU, xm, xs)
    configs = [("value_only", 0.003, 0.0), ("value+grad", 0.003, 0.2)]
    ics = T.ICS[: args.max_ics]
    out = {
        "config": {
            "plant": "two_cstr_series",
            "solver": "CasADi/IPOPT strong-solver stress check",
            "seeds": args.seeds,
            "n_ic": len(ics),
            "n_data": args.n_data,
            "epochs": args.epochs,
            "horizon": args.horizon,
            "steps": args.steps,
            "max_iter": args.max_iter,
            "rho_u": args.rho_u,
            "success_V": args.success_v,
            "training_configs": configs,
        },
        "per_seed": {},
        "summary": {},
    }
    print(f"{'condition':12} {'seed':>4} {'succ':>5} {'finalV':>10} {'fail':>5} {'ms':>8}", flush=True)
    for seed in args.seeds:
        out["per_seed"][str(seed)] = {}
        for tag, lam_val, lam_grad in configs:
            model, mse = T.train(
                seed,
                data,
                phi_all,
                yphi_all,
                true_ug,
                lam_val=lam_val,
                lam_grad=lam_grad,
                epochs=args.epochs,
            )
            seq = T.freeze(model)
            fhat = build_fhat(seq, norm)
            rows = [
                rollout_ipopt(
                    fhat,
                    norm,
                    x0,
                    horizon=args.horizon,
                    steps=args.steps,
                    rho_u=args.rho_u,
                    success_v=args.success_v,
                    max_iter=args.max_iter,
                )
                for x0 in ics
            ]
            agg = aggregate(rows)
            agg["test_mse"] = float(mse)
            out["per_seed"][str(seed)][tag] = {"rows": rows, "aggregate": agg}
            print(
                f"{tag:12} {seed:>4} {agg['success']:>5.2f} {agg['final_V']:>10.2f} "
                f"{agg['solver_failures']:>5.1f} {agg['mean_solve_ms']:>8.1f}",
                flush=True,
            )

    for tag, _, _ in configs:
        seed_aggs = [out["per_seed"][str(seed)][tag]["aggregate"] for seed in args.seeds]
        out["summary"][tag] = aggregate(seed_aggs)
        out["summary"][tag]["test_mse"] = float(np.mean([a["test_mse"] for a in seed_aggs]))
        out["summary"][tag]["test_mse_sd"] = float(np.std([a["test_mse"] for a in seed_aggs]))

    LOG.mkdir(parents=True, exist_ok=True)
    dest = LOG / "two_cstr_ipopt_baseline.json"
    dest.write_text(json.dumps(out, indent=2))
    print("\n=== summary over seeds ===")
    for tag in out["summary"]:
        s = out["summary"][tag]
        print(
            f"{tag:12} success={s['success']:.2f}±{s['success_sd']:.2f} "
            f"finalV={s['final_V']:.2f}±{s['final_V_sd']:.2f} "
            f"solve={s['mean_solve_ms']:.1f} ms"
        )
    print("wrote", dest)


if __name__ == "__main__":
    main()
