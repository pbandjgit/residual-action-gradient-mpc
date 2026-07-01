"""Two-CSTR state-constraint probe (JPC): reactor-temperature safety limit.

During stabilization the value+grad controller drives a thermal transient that
can exceed a desired temperature cap. This script tests whether a simple soft
temperature penalty is enough to enforce that cap in the finite-budget learned
MPC:
  J = sum_h [ V(x_h) + rho_u ||u_h||^2 + rho_c ( relu(T1_h - T_cap)^2
                                                 + relu(T2_h - T_cap)^2 ) ],
and sweeps rho_c. The observed result is a characterized limitation: this simple
penalty does not reliably enforce the limit and can be counterproductive, so
constraint handling should be treated as a separate CLBF/constraint-Jacobian
extension rather than as a solved part of the present L_grad contribution.

Outputs: results/interim/logs/two_cstr_constraint.json
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
SEEDS = [0, 1, 2]
T_CAP = 70.0
RHO_C = [0.0, 0.02, 0.1]


def closed_loop_constrained(seq, norm, x0, hz, budget, steps, lr, rho_u,
                            success_V, un_lo, un_hi, t_cap, rho_c):
    xm, xs, ym, ys = norm; xm4, xs4 = xm[:4], xs[:4]
    x = np.asarray(x0, float); Vtraj = [float(T.Vnp(x))]
    max_temp = float(max(x[1], x[3]))
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
                if rho_c > 0:
                    J = J + rho_c * (torch.relu(y[0, 1] - t_cap) ** 2
                                     + torch.relu(y[0, 3] - t_cap) ** 2)
                xt = (y - xm4) / xs4
            J.backward(); opt.step()
            with torch.no_grad():
                un.clamp_(un_lo, un_hi)
        u0 = un[0].detach().numpy() * xs[4:].numpy() + xm[4:].numpy()
        x = T.step(x, u0); Vtraj.append(float(T.Vnp(x)))
        max_temp = max(max_temp, float(x[1]), float(x[3]))
        un = torch.cat([un[1:].detach(), un[-1:].detach()])
    return dict(final_V=Vtraj[-1], success=bool(Vtraj[-1] <= success_V),
                max_temp=max_temp, violation=max(0.0, max_temp - t_cap))


def main():
    data = T.make_data(n=12000)
    XU, xm, xs = data[0], data[2], data[3]
    norm = T.make_norm(data)
    un_lo = torch.tensor((T.BOX_LO - xm[4:]) / xs[4:], dtype=torch.float32)
    un_hi = torch.tensor((T.BOX_HI - xm[4:]) / xs[4:], dtype=torch.float32)
    print("precomputing ...", flush=True)
    Phi, Yphi = T.precompute_phi(XU)
    true_ug = T.precompute_true_ugrad(XU, xm, xs)
    hz, steps, lr, rho_u, sV = 3, 120, 0.2, 0.01, 2.0

    out = {"config": dict(plant="two_cstr_series", seeds=SEEDS, T_cap=T_CAP,
                          rho_c=RHO_C, model="value+grad", horizon=hz, steps=steps,
                          success_V=sV), "per_seed": {}}
    print(f"{'rho_c':>6} {'seed':>4} {'success':>8} {'max_temp':>9} {'violation':>10}", flush=True)
    for seed in SEEDS:
        out["per_seed"][seed] = {}
        model, _ = T.train(seed, data, Phi, Yphi, true_ug, lam_val=0.003, lam_grad=0.2, epochs=70)
        seq = T.freeze(model)
        for rc in RHO_C:
            rows = [closed_loop_constrained(seq, norm, x0, hz, 20, steps, lr, rho_u,
                                            sV, un_lo, un_hi, T_CAP, rc) for x0 in T.ICS]
            rec = dict(success=float(np.mean([r["success"] for r in rows])),
                       max_temp=float(np.max([r["max_temp"] for r in rows])),
                       mean_violation=float(np.mean([r["violation"] for r in rows])))
            out["per_seed"][seed][f"{rc:g}"] = rec
            print(f"{rc:>6g} {seed:>4} {rec['success']:>8.2f} {rec['max_temp']:>9.1f} "
                  f"{rec['mean_violation']:>10.2f}", flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "two_cstr_constraint.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "two_cstr_constraint.json", flush=True)

    def avg(rc, key):
        return float(np.mean([out["per_seed"][s][f"{rc:g}"][key] for s in SEEDS]))
    print(f"\n=== state-constraint (T_cap={T_CAP}, value+grad, seed-averaged) ===")
    print(f"{'rho_c':>6} {'success':>8} {'peak_temp':>10} {'mean_viol':>10}")
    for rc in RHO_C:
        print(f"{rc:>6g} {avg(rc,'success'):>8.2f} {avg(rc,'max_temp'):>10.1f} "
              f"{avg(rc,'mean_violation'):>10.2f}")


if __name__ == "__main__":
    main()
