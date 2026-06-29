"""Paper-H GGN-MPC probe (CSTR): does residual-action-gradient consistency (L_grad)
improve the local linearization used by a Gauss-Newton MPC solver?

A damped Gauss-Newton (Levenberg-Marquardt) single-shooting MPC with line search runs
on a FIXED GroupSort-LCNN, comparing training objectives {value_only, grad_only,
value+grad}. Two least-squares objectives:

  A (rolled-out V):   r = [ L^T x_h ; sqrt(rho_u) u_h ],  V(x)=||L^T x||^2 (P=L L^T)
  B (exact residual): r = [ relu(g_hat(x_h,u_h)) ; sqrt(rho_u) u_h ],
                      g_hat(x,u)=V(f_hat(x,u))-V(f_hat(x,Phi(x)))

GGN step: (J^T J + lam I) delta = -J^T r, J = dr/dU by autograd; LM damping plus
monotone backtracking line search from the start (so failures are not simply
undamped-solver artifacts).

Because GGN line search is *built to be robust to bad geometry*, the decisive signal
is in the SOLVER diagnostics, not only closed-loop success:
  - accepted-step fraction,
  - predicted GN-model decrease for the accepted/clamped update,
  - fraction of applied steps that INCREASE the true-plant Lyapunov value,
  - true-plant V decrease per step.
L_grad should make the learned GN model's predicted decrease materialize on the true
plant (faithful linearization) even where projected-Adam success saturates.

Outputs: results/interim/logs/ggn_mpc_probe_{V,residual}.json
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
import budget_sweep_solver as bs                                      # noqa: E402
import probe_action_gradient as pr                                   # noqa: E402
import lgrad_experiment as L                                         # noqa: E402
from cstr.sontag import saturated_sontag_np                          # noqa: E402

P = bs.P; Pt = bs.Pt; SIM = bs.SIM; ICS = bs.ICS
INPUT_LO = bs.INPUT_LO; INPUT_HI = bs.INPUT_HI
LOG = ROOT / "results" / "interim" / "logs"
SEEDS = [0, 1, 2]
ARCH = {"model": "lcnn", "hidden": 40, "lipschitz_bound": 4.0, "n_layers": 2}
LT = torch.tensor(np.linalg.cholesky(P).T, dtype=torch.float32)   # ||LT x||^2 = V(x)


def make_residual_fn(seq, norm, x0, horizon, rho_u, objective):
    xm, xs, ym, ys = norm; xm2, xs2 = xm[:2], xs[:2]
    x0n = torch.tensor((x0 - xm2.numpy()) / xs2.numpy(), dtype=torch.float32).view(1, 2)
    srho = float(np.sqrt(rho_u))

    def resfn(Uflat):
        un = Uflat.view(horizon, 2)
        xt = x0n
        parts = []
        for h in range(horizon):
            if objective == "V":
                y = seq(torch.cat([xt, un[h:h + 1]], 1)) * ys + ym
                parts.append(LT @ y.view(2))
                parts.append(srho * un[h])
                xt = (y - xm2) / xs2
            else:  # exact residual violation
                x_np = (xt.detach() * xs2 + xm2).numpy()[0]
                phi_h = saturated_sontag_np(SIM, x_np, INPUT_LO, INPUT_HI, P)
                phin = torch.tensor((phi_h - xm[2:].numpy()) / xs[2:].numpy(),
                                    dtype=torch.float32).view(1, 2)
                y_u = seq(torch.cat([xt, un[h:h + 1]], 1)) * ys + ym
                y_phi = seq(torch.cat([xt, phin], 1)) * ys + ym
                g = (y_u @ Pt * y_u).sum() - (y_phi @ Pt * y_phi).sum()
                parts.append(torch.relu(g).view(1))
                parts.append(srho * un[h])
                xt = (y_u - xm2) / xs2
        return torch.cat(parts)
    return resfn


def ggn_closed_loop(seq, norm, x0, horizon, budget, steps, rho_u, success_V,
                    un_lo, un_hi, objective):
    xm, xs, ym, ys = norm
    x = np.asarray(x0, float); Vtraj = [float(x @ P @ x)]
    n = horizon * 2
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
                resfn, U, vectorize=True, strategy="forward-mode")      # (m, n)
            obj0 = 0.5 * float(r @ r)
            g = J.t() @ r
            JTJ = J.t() @ J
            H = JTJ + lam * torch.eye(n)
            try:
                delta = torch.linalg.solve(H, -g)
            except RuntimeError:
                lam *= 10; continue
            # predicted decrease from GN quadratic model
            pred_full = float(-(g @ delta) - 0.5 * (delta @ JTJ @ delta))
            acc = False
            U_old = U
            pred_applied = 0.0
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
        u0 = np.clip(U.view(horizon, 2)[0].numpy() * xs[2:].numpy() + xm[2:].numpy(),
                     INPUT_LO, INPUT_HI)
        Vbefore = float(x @ P @ x)
        x = SIM.step(x, u0); Vafter = float(x @ P @ x)
        true_dec.append(Vbefore - Vafter)
        if Vafter > Vbefore + 1e-9:
            true_incr += 1
        Vtraj.append(Vafter)
        U = torch.cat([U.view(horizon, 2)[1:], U.view(horizon, 2)[-1:]]).reshape(-1)
    return dict(final_V=Vtraj[-1], max_V=float(max(Vtraj)),
                success=bool(Vtraj[-1] <= success_V),
                accepted_frac=accepted / max(total_it, 1),
                true_incr_frac=true_incr / max(steps, 1),
                mean_pred_dec=float(np.mean(pred_dec_applied)) if pred_dec_applied else 0.0,
                mean_pred_dec_applied=float(np.mean(pred_dec_applied)) if pred_dec_applied else 0.0,
                mean_pred_dec_full=float(np.mean(pred_dec_full)) if pred_dec_full else 0.0,
                mean_true_dec=float(np.mean(true_dec)))


def main(objective="V"):
    d = np.load(ROOT / "data" / "processed" / "lcnn_paper_cstr_onestep_20k.npz")
    data = (d["XU"].astype("float32"), d["Y"].astype("float32"),
            d["x_mean"].astype("float32"), d["x_std"].astype("float32"),
            d["y_mean"].astype("float32"), d["y_std"].astype("float32"),
            d["train_idx"], d["test_idx"])
    norm, un_lo, un_hi = L.make_norm(data)
    print("precomputing Phi/Yphi + true action-grad ...")
    L.PHI, L.YPHI = pr.precompute_phi(SIM, data[0][:, :2].astype("float64"))
    true_ug = L.precompute_true_ugrad(SIM, data[0], data[2], data[3])

    configs = [("value_only", 0.003, 0.0), ("grad_only", 0.0, 0.05),
               ("value+grad", 0.003, 0.05)]
    horizon, steps, rho_u, success_V = 3, 120, 0.01, 2.0
    budgets = [3, 10]
    out = {"config": dict(plant="cstr", solver="damped-GN + line search",
                          objective=objective, horizon=horizon, steps=steps,
                          rho_u=rho_u, success_V=success_V, seeds=SEEDS,
                          budgets=budgets, configs=configs), "per_seed": {}}
    print(f"GGN-MPC objective={objective}")
    print(f"{'config':12} {'seed':>4} {'bud':>4} {'succ':>5} {'acc':>5} "
          f"{'trueIncr':>8} {'predDec':>9} {'trueDec':>9}")
    for seed in SEEDS:
        out["per_seed"][seed] = {}
        for tag, lv, lg in configs:
            model, _ = L.train(ARCH, seed, data, L.PHI, L.YPHI, true_ug,
                               lam_val=lv, lam_grad=lg)
            seq = bs.freeze(model)
            out["per_seed"][seed][tag] = {}
            for B in budgets:
                rows = [ggn_closed_loop(seq, norm, x0, horizon, B, steps, rho_u,
                        success_V, un_lo, un_hi, objective) for x0 in ICS]
                rec = dict(
                    success=float(np.mean([r["success"] for r in rows])),
                    accepted_frac=float(np.mean([r["accepted_frac"] for r in rows])),
                    true_incr_frac=float(np.mean([r["true_incr_frac"] for r in rows])),
                    mean_pred_dec=float(np.mean([r["mean_pred_dec"] for r in rows])),
                    mean_pred_dec_applied=float(np.mean([r["mean_pred_dec_applied"] for r in rows])),
                    mean_pred_dec_full=float(np.mean([r["mean_pred_dec_full"] for r in rows])),
                    mean_true_dec=float(np.mean([r["mean_true_dec"] for r in rows])))
                out["per_seed"][seed][tag][B] = rec
                print(f"{tag:12} {seed:>4} {B:>4} {rec['success']:>5.2f} "
                      f"{rec['accepted_frac']:>5.2f} {rec['true_incr_frac']:>8.2f} "
                      f"{rec['mean_pred_dec']:>9.2e} {rec['mean_true_dec']:>9.2e}")
    LOG.mkdir(parents=True, exist_ok=True)
    fn = LOG / f"ggn_mpc_probe_{objective}.json"
    fn.write_text(json.dumps(out, indent=2))
    print("\nwrote", fn)

    def avg(tag, B, key):
        return float(np.mean([out["per_seed"][s][tag][B][key] for s in SEEDS]))
    print(f"\n=== GGN objective={objective} seed-averaged ===")
    print(f"{'config':12} {'bud':>4} {'succ':>5} {'acc':>5} {'trueIncr':>8} {'trueDec':>9}")
    for tag, _, _ in configs:
        for B in budgets:
            print(f"{tag:12} {B:>4} {avg(tag,B,'success'):>5.2f} "
                  f"{avg(tag,B,'accepted_frac'):>5.2f} {avg(tag,B,'true_incr_frac'):>8.2f} "
                  f"{avg(tag,B,'mean_true_dec'):>9.2e}")


if __name__ == "__main__":
    obj = sys.argv[1] if len(sys.argv) > 1 else "V"
    main(obj)
