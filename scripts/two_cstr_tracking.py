"""Two-CSTR setpoint tracking (JPC): does the L_grad benefit extend beyond origin
regulation to tracking a non-zero setpoint?

The dynamics are control-affine, f(x,u)=f0(x)+B u, so any in-region target x_sp is
held by the steady input u_sp = -B^{-1} f0(x_sp). We pick a feasible, in-box
setpoint and run the finite-budget first-order MPC with the tracking cost
  sum_h (x_h - x_sp)' P (x_h - x_sp) + rho_u || u_h - u_sp ||^2
from each initial condition, comparing value-only vs value+grad. Success is
V_sp(x_final) = (x_final - x_sp)' P (x_final - x_sp) <= 2.

Outputs: results/interim/logs/two_cstr_tracking.json
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

# Feasible in-region setpoint and its holding input (verified: |f(x_sp,u_sp)|~1e-15).
X_SP = np.array([-0.5, -25.0, -0.5, -25.0])
_, _Blin = T.SIM.analytic_jacobian(np.zeros(4), np.zeros(4))
U_SP = -np.linalg.inv(_Blin) @ T.SIM._dynamics(X_SP, np.zeros(4))


def closed_loop_tracking(seq, norm, x0, horizon, budget, steps, lr, rho_u,
                         success_Vsp, un_lo, un_hi):
    xm, xs, ym, ys = norm; xm4, xs4 = xm[:4], xs[:4]
    xsp_t = torch.tensor(X_SP, dtype=torch.float32)
    un_sp = torch.tensor((U_SP - xm[4:].numpy()) / xs[4:].numpy(), dtype=torch.float32)

    def Vsp(xx):
        d = np.asarray(xx, float) - X_SP
        return float(d @ T.P_MAT @ d)

    x = np.asarray(x0, float); Vtraj = [Vsp(x)]
    un = torch.zeros(horizon, 4)
    for _ in range(steps):
        un = un.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([un], lr=lr)
        x0n = torch.tensor((x - xm4.numpy()) / xs4.numpy(), dtype=torch.float32).view(1, 4)
        for _it in range(budget):
            opt.zero_grad(); xt = x0n; J = torch.zeros(())
            for h in range(horizon):
                xun = torch.cat([xt, un[h].view(1, 4)], 1)
                y = seq(xun) * ys + ym
                d = y.view(1, 4) - xsp_t
                J = J + (d @ T.Pt * d).sum() + rho_u * ((un[h] - un_sp) ** 2).sum()
                xt = (y - xm4) / xs4
            J.backward(); opt.step()
            with torch.no_grad():
                un.clamp_(un_lo, un_hi)
        u0 = un[0].detach().numpy() * xs[4:].numpy() + xm[4:].numpy()
        x = T.step(x, u0); Vtraj.append(Vsp(x))
        un = torch.cat([un[1:].detach(), un[-1:].detach()])
    return dict(final_Vsp=Vtraj[-1], success=bool(Vtraj[-1] <= success_Vsp))


def main():
    print(f"setpoint x_sp={X_SP.tolist()}  u_sp={np.round(U_SP,2).tolist()}  in_box="
          f"{bool(np.all(U_SP>=T.BOX_LO) and np.all(U_SP<=T.BOX_HI))}", flush=True)
    data = T.make_data(n=12000)
    XU, xm, xs = data[0], data[2], data[3]
    norm = T.make_norm(data)
    un_lo = torch.tensor((T.BOX_LO - xm[4:]) / xs[4:], dtype=torch.float32)
    un_hi = torch.tensor((T.BOX_HI - xm[4:]) / xs[4:], dtype=torch.float32)
    print("precomputing ...", flush=True)
    Phi, Yphi = T.precompute_phi(XU)
    true_ug = T.precompute_true_ugrad(XU, xm, xs)
    hz, steps, lr, rho_u, sV = 3, 120, 0.2, 0.01, 2.0
    configs = [("value_only", 0.003, 0.0), ("value+grad", 0.003, 0.2)]

    out = {"config": dict(plant="two_cstr_series", seeds=SEEDS, x_sp=X_SP.tolist(),
                          u_sp=U_SP.tolist(), horizon=hz, steps=steps, success_Vsp=sV,
                          configs=configs), "per_seed": {}}
    print(f"{'config':11} {'seed':>4} {'track_succ':>10} {'final_Vsp':>10}", flush=True)
    for seed in SEEDS:
        out["per_seed"][seed] = {}
        for tag, lv, lg in configs:
            model, _ = T.train(seed, data, Phi, Yphi, true_ug, lam_val=lv, lam_grad=lg, epochs=70)
            seq = T.freeze(model)
            rows = [closed_loop_tracking(seq, norm, x0, hz, 20, steps, lr, rho_u,
                                         sV, un_lo, un_hi) for x0 in T.ICS]
            succ = float(np.mean([r["success"] for r in rows]))
            fV = float(np.mean([r["final_Vsp"] for r in rows]))
            out["per_seed"][seed][tag] = dict(track_success=succ, final_Vsp=fV)
            print(f"{tag:11} {seed:>4} {succ:>10.2f} {fV:>10.2e}", flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "two_cstr_tracking.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "two_cstr_tracking.json", flush=True)

    def avg(tag, key):
        return float(np.mean([out["per_seed"][s][tag][key] for s in SEEDS]))
    print("\n=== setpoint tracking (seed-averaged) ===")
    for tag, _, _ in configs:
        print(f"{tag:11} track_success={avg(tag,'track_success'):.2f}  "
              f"final_Vsp={avg(tag,'final_Vsp'):.2e}")


if __name__ == "__main__":
    main()
