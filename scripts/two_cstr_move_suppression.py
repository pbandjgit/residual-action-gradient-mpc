"""Move-suppression sweep for the two-CSTR value+L_grad controller.

The main comparisons deliberately use no input-rate penalty, to isolate the
action-gradient effect. Here we show, separately, that the high closed-loop input
total variation (TV) of the finite-budget MPC is a tuning artifact of that bare
objective, not intrinsic to the method: adding the standard move-suppression term
rho_move * sum_t ||u_t - u_{t-1}||^2 (with the first horizon move tied to the last
applied input) trades TV down toward the smooth-feedback reference while keeping
closed-loop success high, over a wide interior range of rho_move.

The learned surrogate and its training are unchanged; only the online controller
objective changes. Outputs: results/interim/logs/two_cstr_move_suppression.json
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
import two_cstr_metrics as M  # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"
SEEDS = [0, 1, 2]
RANGE = T.BOX_HI - T.BOX_LO
RHO_MOVE = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]


def record_ms(seq, norm, x0, hz, budget, steps, lr, rho_u, rho_move, un_lo, un_hi):
    """Closed loop with a move-suppression penalty in the MPC objective."""
    xm, xs, ym, ys = norm; xm4, xs4 = xm[:4], xs[:4]
    x = np.asarray(x0, float); Vtraj = [float(T.Vnp(x))]; U = []
    un = torch.zeros(hz, 4)
    u_prev_n = None  # last applied input, normalized-input coords
    for _ in range(steps):
        un = un.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([un], lr=lr)
        x0n = torch.tensor((x - xm4.numpy()) / xs4.numpy(), dtype=torch.float32).view(1, 4)
        for _it in range(budget):
            opt.zero_grad(); xt = x0n; J = torch.zeros(())
            prev = (torch.tensor(u_prev_n, dtype=torch.float32)
                    if u_prev_n is not None else un[0].detach())
            for h in range(hz):
                y = seq(torch.cat([xt, un[h].view(1, 4)], 1)) * ys + ym
                J = J + (y @ T.Pt * y).sum() + rho_u * (un[h] ** 2).sum()
                J = J + rho_move * ((un[h] - prev) ** 2).sum()
                prev = un[h]
                xt = (y - xm4) / xs4
            J.backward(); opt.step()
            with torch.no_grad():
                un.clamp_(un_lo, un_hi)
        un0 = un[0].detach()
        u0 = np.clip(un0.numpy() * xs[4:].numpy() + xm[4:].numpy(), T.BOX_LO, T.BOX_HI)
        U.append(u0); u_prev_n = un0.numpy()
        x = T.step(x, u0); Vtraj.append(float(T.Vnp(x)))
        un = torch.cat([un[1:].detach(), un[-1:].detach()])
    return Vtraj, np.asarray(U)


def tv_of(U):
    return float(np.sum(np.abs(np.diff(np.asarray(U), axis=0)) / RANGE))


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

    # reference: auxiliary LQR TV
    aux_tv = float(np.mean([tv_of(M.record_aux(ic, steps)[1]) for ic in T.ICS]))

    out = {"config": dict(plant="two_cstr_series", seeds=SEEDS, rho_move=RHO_MOVE,
                          horizon=hz, budget=20, aux_tv=aux_tv), "per_seed": {}}
    print(f"aux LQR reference TV = {aux_tv:.2f}\n", flush=True)
    print(f"{'seed':>4} {'rho_move':>9} {'success':>8} {'TV':>7}", flush=True)
    for seed in SEEDS:
        model, _ = T.train(seed, data, Phi, Yphi, true_ug,
                           lam_val=0.003, lam_grad=0.2, epochs=70)
        seq = T.freeze(model)
        out["per_seed"][seed] = {}
        for rm in RHO_MOVE:
            succ, tvs = [], []
            for ic in T.ICS:
                V, U = record_ms(seq, norm, ic, hz, 20, steps, lr, rho_u, rm, un_lo, un_hi)
                succ.append(float(V[-1] <= 2.0)); tvs.append(tv_of(U))
            rec = dict(success=float(np.mean(succ)), input_tv=float(np.mean(tvs)))
            out["per_seed"][seed][f"{rm:g}"] = rec
            print(f"{seed:>4} {rm:>9g} {rec['success']:>8.2f} {rec['input_tv']:>7.1f}", flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "two_cstr_move_suppression.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "two_cstr_move_suppression.json", flush=True)

    def avg(rm, k):
        return float(np.mean([out["per_seed"][s][f"{rm:g}"][k] for s in SEEDS]))
    print(f"\n=== seed-averaged (aux TV={aux_tv:.2f}) ===")
    print(f"{'rho_move':>9} {'success':>8} {'TV':>7}")
    for rm in RHO_MOVE:
        print(f"{rm:>9g} {avg(rm,'success'):>8.2f} {avg(rm,'input_tv'):>7.1f}")


if __name__ == "__main__":
    main()
