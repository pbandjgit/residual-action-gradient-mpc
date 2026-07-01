"""Two-CSTR-in-series GGN-MPC transfer: does L_grad also help a Gauss-Newton solver?

Damped Gauss-Newton (Levenberg-Marquardt) single-shooting MPC with backtracking
line search on the rolled-out Lyapunov objective, on the fixed GroupSort-LCNN,
comparing {value_only, grad_only, value+grad}. Mirrors the single-CSTR GGN probe
(`ggn_mpc_probe.py`) for the two-CSTR process (4 states / 4 inputs).

Objective (rolled-out V):  r = [ L^T x_h ; sqrt(rho_u) u_h ],  V(x)=||L^T x||^2.

Decisive diagnostics (line search is built to be robust to bad geometry):
accepted-step fraction, predicted GN decrease, fraction of applied steps that
increase the true-plant V, and true-plant V decrease per step. L_grad should make
the learned GN linearization faithful (predicted decrease materializes).

Outputs: results/interim/logs/ggn_mpc_two_cstr_V.json
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
P = T.P_MAT
Pt = T.Pt
ICS = T.ICS
LT = torch.tensor(np.linalg.cholesky(P).T, dtype=torch.float32)   # ||LT x||^2 = V(x)


def make_residual_fn(seq, norm, x0, horizon, rho_u, objective="V"):
    xm, xs, ym, ys = norm; xm4, xs4 = xm[:4], xs[:4]
    x0n = torch.tensor((x0 - xm4.numpy()) / xs4.numpy(), dtype=torch.float32).view(1, 4)
    srho = float(np.sqrt(rho_u))

    def resfn(Uflat):
        un = Uflat.view(horizon, 4)
        xt = x0n; parts = []
        for h in range(horizon):
            if objective == "V":
                y = seq(torch.cat([xt, un[h:h + 1]], 1)) * ys + ym
                parts.append(LT @ y.view(4))
                parts.append(srho * un[h])
                xt = (y - xm4) / xs4
            else:  # exact residual violation  g_hat = V(f_hat(x,u)) - V(f_hat(x,Phi(x)))
                x_np = (xt.detach() * xs4 + xm4).numpy()[0]
                phi_h = T.phi(x_np)
                phin = torch.tensor((phi_h - xm[4:].numpy()) / xs[4:].numpy(),
                                    dtype=torch.float32).view(1, 4)
                y_u = seq(torch.cat([xt, un[h:h + 1]], 1)) * ys + ym
                y_phi = seq(torch.cat([xt, phin], 1)) * ys + ym
                g = (y_u @ Pt * y_u).sum() - (y_phi @ Pt * y_phi).sum()
                parts.append(torch.relu(g).view(1))
                parts.append(srho * un[h])
                xt = (y_u - xm4) / xs4
        return torch.cat(parts)
    return resfn


def ggn_closed_loop(seq, norm, x0, horizon, budget, steps, rho_u, success_V,
                    un_lo, un_hi, objective="V"):
    xm, xs, ym, ys = norm
    x = np.asarray(x0, float); Vtraj = [float(x @ P @ x)]
    n = horizon * 4
    U = torch.zeros(n)
    un_lo_f = un_lo.repeat(horizon); un_hi_f = un_hi.repeat(horizon)
    accepted = total_it = true_incr = 0
    pred_dec_applied, pred_dec_full, true_dec = [], [], []
    for _ in range(steps):
        resfn = make_residual_fn(seq, norm, x, horizon, rho_u, objective)
        lam = 1e-2
        for _it in range(budget):
            total_it += 1
            U = U.detach()
            r = resfn(U)
            J = torch.autograd.functional.jacobian(
                resfn, U, vectorize=True, strategy="forward-mode")
            obj0 = 0.5 * float(r @ r)
            g = J.t() @ r
            JTJ = J.t() @ J
            H = JTJ + lam * torch.eye(n)
            try:
                delta = torch.linalg.solve(H, -g)
            except RuntimeError:
                lam *= 10; continue
            pred_full = float(-(g @ delta) - 0.5 * (delta @ JTJ @ delta))
            acc = False; U_old = U; pred_applied = 0.0
            for alpha in (1.0, 0.5, 0.25, 0.125):
                Ut = torch.clamp(U_old + alpha * delta, un_lo_f, un_hi_f)
                rt = resfn(Ut)
                if 0.5 * float(rt @ rt) < obj0:
                    dU = Ut - U_old
                    pred_applied = float(-(g @ dU) - 0.5 * (dU @ JTJ @ dU))
                    U = Ut; acc = True; break
            if acc:
                accepted += 1; lam = max(lam * 0.5, 1e-4)
                pred_dec_full.append(max(pred_full, 0.0))
                pred_dec_applied.append(max(pred_applied, 0.0))
            else:
                lam = min(lam * 10, 1e4)
        u0 = np.clip(U.view(horizon, 4)[0].numpy() * xs[4:].numpy() + xm[4:].numpy(),
                     T.BOX_LO, T.BOX_HI)
        Vbefore = float(x @ P @ x)
        x = T.step(x, u0); Vafter = float(x @ P @ x)
        true_dec.append(Vbefore - Vafter)
        if Vafter > Vbefore + 1e-9:
            true_incr += 1
        Vtraj.append(Vafter)
        U = torch.cat([U.view(horizon, 4)[1:], U.view(horizon, 4)[-1:]]).reshape(-1)
    return dict(final_V=Vtraj[-1], max_V=float(max(Vtraj)),
                success=bool(Vtraj[-1] <= success_V),
                accepted_frac=accepted / max(total_it, 1),
                true_incr_frac=true_incr / max(steps, 1),
                mean_pred_dec_applied=float(np.mean(pred_dec_applied)) if pred_dec_applied else 0.0,
                mean_true_dec=float(np.mean(true_dec)))


def main(objective="V"):
    data = T.make_data(n=12000)
    XU, xm, xs = data[0], data[2], data[3]
    norm = T.make_norm(data)
    un_lo = torch.tensor((T.BOX_LO - xm[4:]) / xs[4:], dtype=torch.float32)
    un_hi = torch.tensor((T.BOX_HI - xm[4:]) / xs[4:], dtype=torch.float32)
    print(f"GGN objective={objective}; precomputing Phi/Yphi + true action-grad ...", flush=True)
    Phi, Yphi = T.precompute_phi(XU)
    true_ug = T.precompute_true_ugrad(XU, xm, xs)

    configs = [("value_only", 0.003, 0.0), ("grad_only", 0.0, 0.2),
               ("value+grad", 0.003, 0.2)]
    horizon, steps, rho_u, success_V = 3, 120, 0.01, 2.0
    budgets = [3, 10]
    out = {"config": dict(plant="two_cstr_series", solver="damped-GN + line search",
                          objective=objective, horizon=horizon, steps=steps, rho_u=rho_u,
                          success_V=success_V, seeds=SEEDS, budgets=budgets,
                          configs=configs), "per_seed": {}}
    print(f"{'config':12} {'seed':>4} {'bud':>4} {'succ':>5} {'acc':>5} "
          f"{'trueIncr':>8} {'predDec':>9} {'trueDec':>9}", flush=True)
    for seed in SEEDS:
        out["per_seed"][seed] = {}
        for tag, lv, lg in configs:
            model, _ = T.train(seed, data, Phi, Yphi, true_ug, lam_val=lv,
                               lam_grad=lg, epochs=70)
            seq = T.freeze(model)
            out["per_seed"][seed][tag] = {}
            for B in budgets:
                rows = [ggn_closed_loop(seq, norm, x0, horizon, B, steps, rho_u,
                        success_V, un_lo, un_hi, objective) for x0 in ICS]
                rec = dict(
                    success=float(np.mean([r["success"] for r in rows])),
                    accepted_frac=float(np.mean([r["accepted_frac"] for r in rows])),
                    true_incr_frac=float(np.mean([r["true_incr_frac"] for r in rows])),
                    mean_pred_dec_applied=float(np.mean([r["mean_pred_dec_applied"] for r in rows])),
                    mean_true_dec=float(np.mean([r["mean_true_dec"] for r in rows])))
                out["per_seed"][seed][tag][B] = rec
                print(f"{tag:12} {seed:>4} {B:>4} {rec['success']:>5.2f} "
                      f"{rec['accepted_frac']:>5.2f} {rec['true_incr_frac']:>8.2f} "
                      f"{rec['mean_pred_dec_applied']:>9.2e} {rec['mean_true_dec']:>9.2e}", flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    fn = LOG / f"ggn_mpc_two_cstr_{objective}.json"
    fn.write_text(json.dumps(out, indent=2))
    print("\nwrote", fn, flush=True)

    def avg(tag, B, key):
        return float(np.mean([out["per_seed"][s][tag][B][key] for s in SEEDS]))
    print(f"\n=== two-CSTR GGN (objective={objective}) seed-averaged ===")
    print(f"{'config':12} {'bud':>4} {'succ':>5} {'acc':>5} {'trueIncr':>8} {'trueDec':>9}")
    for tag, _, _ in configs:
        for B in budgets:
            print(f"{tag:12} {B:>4} {avg(tag,B,'success'):>5.2f} "
                  f"{avg(tag,B,'accepted_frac'):>5.2f} {avg(tag,B,'true_incr_frac'):>8.2f} "
                  f"{avg(tag,B,'mean_true_dec'):>9.2e}")


if __name__ == "__main__":
    obj = sys.argv[1] if len(sys.argv) > 1 else "V"
    main(obj)
