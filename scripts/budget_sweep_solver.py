"""Paper-H P2: fixed-budget gradient MPC, budget sweep, GroupSort-LCNN vs smooth.

Tests whether the action-gradient irregularity of GroupSort-LCNN (measured in
probe_action_gradient.py) causes finite-budget gradient-solver failures, and
whether the smooth-spectral model fixes it.

Controller: short-horizon single-shooting MPC whose cost is the rolled-out
Lyapunov value of the predicted states plus a small input penalty. The decision
variables are the normalized actions; the solver takes a FIXED number of
projected-Adam steps per MPC step (the "budget"). The closed loop runs on the
TRUE plant. Sweeping the budget gives a budget-vs-success curve: the thesis
predicts smooth succeeds at small budgets where GroupSort fails, and both improve
with budget (so the gap is a phenomenon, not a contrived handicap).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cstr.models_onestep import BjorckLinear                          # noqa: E402
import probe_action_gradient as pr                                   # noqa: E402

P = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float64)
Pt = torch.tensor(P, dtype=torch.float32)
INPUT_LO = np.array([-3.5, -5.0e5]); INPUT_HI = np.array([3.5, 5.0e5])
ICS = [np.array([0.0, 20.0]), np.array([0.0, -20.0]), np.array([0.25, 10.0]),
       np.array([-0.25, -10.0]), np.array([0.45, 0.0])]
LOG = ROOT / "results" / "interim" / "logs"


def freeze(model):
    """Materialize Bjorck layers -> fixed Linear (deterministic, fast forward).
    Works for OneStepLCNN and OneStepSmoothSpectral (same body/out structure)."""
    layers = []
    for layer in model.body:
        layers.append(layer.materialized_linear() if isinstance(layer, BjorckLinear) else layer)
    seq = nn.Sequential(*layers, model.out.materialized_linear())
    seq.eval()
    for p in seq.parameters():
        p.requires_grad_(False)
    return seq


def closed_loop(seq, norm, x0, horizon, budget, steps, lr, rho_u, success_V,
                un_lo, un_hi):
    xm, xs, ym, ys = norm
    xm2, xs2 = xm[:2], xs[:2]
    x = np.asarray(x0, float)
    un = torch.zeros(horizon, 2)            # warm start = mean input
    Vtraj = [float(x @ P @ x)]
    us = []
    for _ in range(steps):
        un = un.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([un], lr=lr)
        x0n = torch.tensor((x - xm2.numpy()) / xs2.numpy(), dtype=torch.float32).view(1, 2)
        for _it in range(budget):
            opt.zero_grad()
            xt = x0n
            J = torch.zeros(())
            for h in range(horizon):
                xun = torch.cat([xt, un[h].view(1, 2)], dim=1)   # (1,4)
                y = seq(xun) * ys + ym                           # physical next (1,2)
                J = J + (y @ Pt * y).sum() + rho_u * (un[h] ** 2).sum()
                xt = (y - xm2) / xs2                             # next normalized (1,2)
            J.backward()
            opt.step()
            with torch.no_grad():
                un.clamp_(un_lo, un_hi)
        u0n = un[0].detach().numpy()
        u0 = np.clip(u0n * xs[2:].numpy() + xm[2:].numpy(), INPUT_LO, INPUT_HI)
        us.append(u0)
        x = SIM.step(x, u0)
        Vtraj.append(float(x @ P @ x))
        un = torch.cat([un[1:].detach(), un[-1:].detach()])
    us = np.asarray(us)
    tv = float(np.sum(np.abs(np.diff(us / (INPUT_HI - INPUT_LO), axis=0))))
    incr = sum(Vtraj[i + 1] > Vtraj[i] + 1e-9 for i in range(len(Vtraj) - 1))
    return dict(final_V=Vtraj[-1], max_V=float(max(Vtraj)),
                success=bool(Vtraj[-1] <= success_V), violations=int(incr), tv=tv)


SIM = pr.create_lcnn_paper_cstr(shifted=True)


def main():
    d = np.load(ROOT / "data" / "processed" / "lcnn_paper_cstr_onestep_20k.npz")
    data = (d["XU"].astype("float32"), d["Y"].astype("float32"),
            d["x_mean"].astype("float32"), d["x_std"].astype("float32"),
            d["y_mean"].astype("float32"), d["y_std"].astype("float32"),
            d["train_idx"], d["test_idx"])
    xm, xs = torch.tensor(data[2]), torch.tensor(data[3])
    ym, ys = torch.tensor(data[4]), torch.tensor(data[5])
    norm = (xm, xs, ym, ys)
    un_lo = torch.tensor((INPUT_LO - data[2][2:]) / data[3][2:], dtype=torch.float32)
    un_hi = torch.tensor((INPUT_HI - data[2][2:]) / data[3][2:], dtype=torch.float32)

    print("precomputing Phi/Yphi ...")
    Phi, Yphi = pr.precompute_phi(SIM, data[0][:, :2].astype("float64"))

    budgets = [3, 5, 10, 20, 40]
    horizon, steps, lr, rho_u, success_V = 3, 120, 0.2, 0.01, 2.0
    seeds = [0, 1]
    out = {"config": dict(budgets=budgets, horizon=horizon, steps=steps, lr=lr,
                          rho_u=rho_u, success_V=success_V, seeds=seeds, n_ic=len(ICS)),
           "results": {}}
    print(f"{'model':7} {'seed':>4} {'budget':>6} {'success':>8} {'p90 final V':>12} {'mean viol':>10} {'mean TV':>9}")
    for kind in ["lcnn", "smooth"]:
        out["results"][kind] = []
        for seed in seeds:
            model, _ = pr.train(kind, seed, data, SIM, Phi, Yphi)
            seq = freeze(model)
            for B in budgets:
                rows = [closed_loop(seq, norm, x0, horizon, B, steps, lr, rho_u,
                                    success_V, un_lo, un_hi) for x0 in ICS]
                succ = float(np.mean([r["success"] for r in rows]))
                p90V = float(np.percentile([r["final_V"] for r in rows], 90))
                viol = float(np.mean([r["violations"] for r in rows]))
                tv = float(np.mean([r["tv"] for r in rows]))
                out["results"][kind].append(dict(seed=seed, budget=B, success=succ,
                                                  p90_final_V=p90V, mean_viol=viol, mean_tv=tv))
                print(f"{kind:7} {seed:>4} {B:>6} {succ:>8.2f} {p90V:>12.2f} {viol:>10.1f} {tv:>9.1f}")
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "budget_sweep_solver.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "budget_sweep_solver.json")
    # budget-success curve (seed-averaged)
    print("\n=== budget vs success (seed-averaged) ===")
    print(f"{'budget':>6} {'LCNN':>8} {'smooth':>8}")
    for B in budgets:
        def avg(kind):
            v = [r["success"] for r in out["results"][kind] if r["budget"] == B]
            return float(np.mean(v))
        print(f"{B:>6} {avg('lcnn'):>8.2f} {avg('smooth'):>8.2f}")


if __name__ == "__main__":
    main()
