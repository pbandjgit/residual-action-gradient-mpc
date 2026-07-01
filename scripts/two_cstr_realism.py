"""Two-CSTR process-realism robustness (JPC): does the L_grad benefit persist
under realistic operation, not just nominal origin stabilization?

The surrogate is trained on the NOMINAL plant; closed-loop evaluation runs the
finite-budget first-order MPC against a PERTURBED true plant:
  - nominal              : reference
  - feed-temp disturbance: sustained T0 += 5 K on the true plant
  - kinetic mismatch     : true k0 x 1.05 (surrogate trained on nominal)
  - enthalpy mismatch    : true dH x 1.05

Everything is in nominal shifted coordinates (V is x'Px around the nominal steady
state); the perturbed plant is integrated in absolute coordinates with its own
parameters. We compare value-only vs value+grad on success (V<=2) and final V.

Outputs: results/interim/logs/two_cstr_realism.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import two_cstr_lgrad as T  # noqa: E402
from cstr.two_cstr_series import TwoCSTRSeriesSimulator, TwoCSTRParams  # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"
SEEDS = list(range(10))
XS_ABS = T.SIM.xs_abs
US_ABS = T.SIM.us_abs
NOM = TwoCSTRParams()


def abs_rk4(psim, x_abs, u_abs, dt=1e-3, sub=100):
    h = dt / sub; y = np.asarray(x_abs, float)
    for _ in range(sub):
        k1 = psim.rhs_absolute(y, u_abs); k2 = psim.rhs_absolute(y + 0.5 * h * k1, u_abs)
        k3 = psim.rhs_absolute(y + 0.5 * h * k2, u_abs); k4 = psim.rhs_absolute(y + h * k3, u_abs)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def make_plant_step(psim):
    def ps(x_shift, u_shift):
        u_abs = np.clip(u_shift, T.BOX_LO, T.BOX_HI) + US_ABS
        x_abs = np.asarray(x_shift, float) + XS_ABS
        return abs_rk4(psim, x_abs, u_abs) - XS_ABS
    return ps


def closed_loop_robust(seq, norm, x0, horizon, budget, steps, lr, rho_u,
                       success_V, un_lo, un_hi, plant_step):
    xm, xs, ym, ys = norm; xm4, xs4 = xm[:4], xs[:4]
    x = np.asarray(x0, float); Vtraj = [float(T.Vnp(x))]
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
                J = J + (y @ T.Pt * y).sum() + rho_u * (un[h] ** 2).sum()
                xt = (y - xm4) / xs4
            J.backward(); opt.step()
            with torch.no_grad():
                un.clamp_(un_lo, un_hi)
        u0 = un[0].detach().numpy() * xs[4:].numpy() + xm[4:].numpy()
        x = plant_step(x, u0); Vtraj.append(float(T.Vnp(x)))
        un = torch.cat([un[1:].detach(), un[-1:].detach()])
    return dict(final_V=Vtraj[-1], success=bool(Vtraj[-1] <= success_V))


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

    scenarios = {
        "nominal": TwoCSTRSeriesSimulator(NOM),
        "feed_temp_dist(+5K)": TwoCSTRSeriesSimulator(TwoCSTRParams(T0=NOM.T0 + 5.0)),
        "mismatch_k0(+5%)": TwoCSTRSeriesSimulator(TwoCSTRParams(k0=NOM.k0 * 1.05)),
        "mismatch_dH(+5%)": TwoCSTRSeriesSimulator(TwoCSTRParams(dH=NOM.dH * 1.05)),
    }
    plant_steps = {k: make_plant_step(v) for k, v in scenarios.items()}
    configs = [("value_only", 0.003, 0.0), ("value+grad", 0.003, 0.2)]

    out = {"config": dict(plant="two_cstr_series", seeds=SEEDS, scenarios=list(scenarios),
                          horizon=hz, steps=steps, success_V=sV, configs=configs),
           "per_seed": {}}
    print(f"{'scenario':22} {'config':11} {'seed':>4} {'succ':>5} {'finalV':>9}", flush=True)
    for seed in SEEDS:
        out["per_seed"][seed] = {}
        for tag, lv, lg in configs:
            model, _ = T.train(seed, data, Phi, Yphi, true_ug, lam_val=lv, lam_grad=lg, epochs=70)
            seq = T.freeze(model)
            out["per_seed"][seed][tag] = {}
            for sc, pstep in plant_steps.items():
                rows = [closed_loop_robust(seq, norm, x0, hz, 20, steps, lr, rho_u,
                                           sV, un_lo, un_hi, pstep) for x0 in T.ICS]
                succ = float(np.mean([r["success"] for r in rows]))
                fV = float(np.mean([r["final_V"] for r in rows]))
                out["per_seed"][seed][tag][sc] = dict(success=succ, final_V=fV)
                print(f"{sc:22} {tag:11} {seed:>4} {succ:>5.2f} {fV:>9.2e}", flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "two_cstr_realism.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "two_cstr_realism.json", flush=True)

    def avg(tag, sc, key):
        return float(np.mean([out["per_seed"][s][tag][sc][key] for s in SEEDS]))
    print("\n=== process-realism robustness (seed-averaged) ===")
    print(f"{'scenario':22} {'value-only succ/V':>20} {'value+grad succ/V':>20}")
    for sc in scenarios:
        print(f"{sc:22} {avg('value_only',sc,'success'):>8.2f} / {avg('value_only',sc,'final_V'):>8.2e}"
              f"   {avg('value+grad',sc,'success'):>8.2f} / {avg('value+grad',sc,'final_V'):>8.2e}")


if __name__ == "__main__":
    main()
