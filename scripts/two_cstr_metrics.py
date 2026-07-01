"""Two-CSTR closed-loop control-performance metrics (JPC).

Beyond binary success, reports standard control metrics for value-only,
value+grad, and the auxiliary LQR reference: settling time (closed-loop steps to
reach and stay at the target), accumulated state cost (integral of the Lyapunov
value), and input total variation (normalized). First-order MPC, budget 20.

Outputs: results/interim/logs/two_cstr_metrics.json
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

LOG = ROOT / "results" / "interim" / "logs"
SEEDS = list(range(10))
RANGE = T.BOX_HI - T.BOX_LO


def metrics(Vtraj, U, thr=2.0):
    V = np.asarray(Vtraj)
    success = bool(V[-1] <= thr)
    above = np.where(V > thr)[0]
    settle = float(above[-1] + 1) if (above.size and success) else float("nan")
    if not above.size:
        settle = 0.0
    cost = float(np.sum(V))
    dU = np.diff(np.asarray(U), axis=0)
    tv = float(np.sum(np.abs(dU) / RANGE))
    return dict(success=success, settle=settle, cost=cost, input_tv=tv)


def record_learned(seq, norm, x0, hz, budget, steps, lr, rho_u, un_lo, un_hi):
    xm, xs, ym, ys = norm; xm4, xs4 = xm[:4], xs[:4]
    x = np.asarray(x0, float); Vtraj = [float(T.Vnp(x))]; U = []
    un = torch.zeros(hz, 4)
    for _ in range(steps):
        un = un.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([un], lr=lr)
        x0n = torch.tensor((x - xm4.numpy()) / xs4.numpy(), dtype=torch.float32).view(1, 4)
        for _it in range(budget):
            opt.zero_grad(); xt = x0n; J = torch.zeros(())
            for h in range(hz):
                y = seq(torch.cat([xt, un[h].view(1, 4)], 1)) * ys + ym
                J = J + (y @ T.Pt * y).sum() + rho_u * (un[h] ** 2).sum()
                xt = (y - xm4) / xs4
            J.backward(); opt.step()
            with torch.no_grad():
                un.clamp_(un_lo, un_hi)
        u0 = np.clip(un[0].detach().numpy() * xs[4:].numpy() + xm[4:].numpy(), T.BOX_LO, T.BOX_HI)
        U.append(u0); x = T.step(x, u0); Vtraj.append(float(T.Vnp(x)))
        un = torch.cat([un[1:].detach(), un[-1:].detach()])
    return Vtraj, np.asarray(U)


def record_aux(x0, steps):
    x = np.asarray(x0, float); Vtraj = [float(T.Vnp(x))]; U = []
    for _ in range(steps):
        u = T.phi(x); U.append(u); x = T.step(x, u); Vtraj.append(float(T.Vnp(x)))
    return Vtraj, np.asarray(U)


def agg(rows):
    return dict(success=float(np.mean([r["success"] for r in rows])),
                settle=float(np.nanmean([r["settle"] for r in rows])),
                cost=float(np.mean([r["cost"] for r in rows])),
                input_tv=float(np.mean([r["input_tv"] for r in rows])))


def main():
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
                          steps=steps, budget=20), "controllers": {}, "per_seed": {}}
    # auxiliary LQR reference (deterministic)
    aux_rows = [metrics(*record_aux(ic, steps)) for ic in T.ICS]
    out["controllers"]["aux_LQR"] = agg(aux_rows)  # deterministic: no seed std
    # learned controllers: keep per-seed aggregates so we can report mean +/- s.d.
    for tag, lv, lg in [("value_only", 0.003, 0.0), ("value+grad", 0.003, 0.2)]:
        per = []
        for seed in SEEDS:
            model, _ = T.train(seed, data, Phi, Yphi, true_ug, lam_val=lv, lam_grad=lg, epochs=70)
            seq = T.freeze(model)
            rows = [metrics(*record_learned(seq, norm, ic, hz, 20, steps, lr, rho_u,
                                            un_lo, un_hi)) for ic in T.ICS]
            a = agg(rows); a["seed"] = seed; per.append(a)
        out["per_seed"][tag] = per
        keys = [k for k in per[0] if k != "seed"]
        out["controllers"][tag] = {k: float(np.nanmean([p[k] for p in per])) for k in keys}
        out["controllers"][tag].update(
            {k + "_sd": float(np.nanstd([p[k] for p in per])) for k in keys})
        print(f"{tag} done", flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "two_cstr_metrics.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "two_cstr_metrics.json", flush=True)
    print(f"\n{'controller':16} {'success':>8} {'settle':>8} {'state cost':>11} {'input TV':>9}")
    for k, v in out["controllers"].items():
        print(f"{k:16} {v['success']:>8.2f} {v['settle']:>8.1f} {v['cost']:>11.1f} {v['input_tv']:>9.2f}")


if __name__ == "__main__":
    main()
