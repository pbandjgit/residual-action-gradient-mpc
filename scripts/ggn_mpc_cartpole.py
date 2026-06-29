"""Paper-H GGN-MPC, cart-pole port: does L_grad improve Gauss-Newton linearization
quality on the second plant? Mirrors ggn_mpc_probe.py (CSTR). Damped GN (LM) +
monotone backtracking line search, FIXED GroupSort-LCNN, ablation {value_only, grad_only,
value+grad}, Objective A (rolled-out V) and B (exact residual). Decisive diagnostics:
accepted-step fraction, accepted-update predicted decrease, true-V-increasing-step fraction.

Outputs: results/interim/logs/ggn_mpc_cartpole_{V,residual}.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import cartpole_lgrad as C                                            # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
ARCH_HIDDEN = 48
LT = torch.tensor(np.linalg.cholesky(C.P_MAT).T, dtype=torch.float32)  # ||LT x||^2 = V
FLIM = C.FLIM


def make_residual_fn(seq, norm, x0, horizon, rho_u, objective):
    xm, xs, ym, ys = norm; xm4, xs4 = xm[:4], xs[:4]
    x0n = torch.tensor((x0 - xm4.numpy()) / xs4.numpy(), dtype=torch.float32).view(1, 4)
    srho = float(np.sqrt(rho_u))

    def resfn(Uflat):
        un = Uflat.view(horizon, 1)
        xt = x0n; parts = []
        for h in range(horizon):
            if objective == "V":
                y = seq(torch.cat([xt, un[h:h + 1]], 1)) * ys + ym
                parts.append(LT @ y.view(4))
                parts.append(srho * un[h])
                xt = (y - xm4) / xs4
            else:
                x_np = (xt.detach() * xs4 + xm4).numpy()[0]
                phi_h = C.phi(x_np)  # (1,)
                phin = torch.tensor((phi_h - xm[4:].numpy()) / xs[4:].numpy(),
                                    dtype=torch.float32).view(1, 1)
                y_u = seq(torch.cat([xt, un[h:h + 1]], 1)) * ys + ym
                y_phi = seq(torch.cat([xt, phin], 1)) * ys + ym
                g = (y_u @ C.Pt * y_u).sum() - (y_phi @ C.Pt * y_phi).sum()
                parts.append(torch.relu(g).view(1))
                parts.append(srho * un[h])
                xt = (y_u - xm4) / xs4
        return torch.cat(parts)
    return resfn


def ggn_closed_loop(seq, norm, x0, horizon, budget, steps, rho_u, success_V,
                    un_lo, un_hi, objective):
    xm, xs, ym, ys = norm
    x = np.asarray(x0, float); Vtraj = [float(C.Vnp(x))]
    n = horizon
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
        u0 = float(np.clip(U.view(horizon, 1)[0].numpy() * xs[4:].numpy() + xm[4:].numpy(),
                           -FLIM, FLIM)[0])
        Vb = float(C.Vnp(x)); x = C.step(x, u0); Va = float(C.Vnp(x))
        true_dec.append(Vb - Va)
        if Va > Vb + 1e-9:
            true_incr += 1
        Vtraj.append(Va)
        U = torch.cat([U.view(horizon, 1)[1:], U.view(horizon, 1)[-1:]]).reshape(-1)
    return dict(final_V=Vtraj[-1], max_V=float(max(Vtraj)),
                success_V2=bool(Vtraj[-1] <= 2.0), success_V10=bool(Vtraj[-1] <= 10.0),
                accepted_frac=accepted / max(total_it, 1),
                true_incr_frac=true_incr / max(steps, 1),
                mean_pred_dec=float(np.mean(pred_dec_applied)) if pred_dec_applied else 0.0,
                mean_pred_dec_applied=float(np.mean(pred_dec_applied)) if pred_dec_applied else 0.0,
                mean_pred_dec_full=float(np.mean(pred_dec_full)) if pred_dec_full else 0.0,
                mean_true_dec=float(np.mean(true_dec)))


def main(objective="V", seed_arg=None, out_suffix="", threads=1):
    seed_text = seed_arg if seed_arg is not None else os.environ.get(
        "CP_SEEDS", ",".join(map(str, SEEDS)))
    seeds = [int(s) for s in seed_text.split(",") if s.strip()]
    out_suffix = out_suffix or os.environ.get("CP_OUT_SUFFIX", "").strip()
    torch.set_num_threads(int(threads))
    C.DEFAULT_HIDDEN = ARCH_HIDDEN
    data = C.make_data()
    XU, Y, xm, xs, ym, ys, _, _ = data
    norm = (torch.tensor(xm), torch.tensor(xs), torch.tensor(ym), torch.tensor(ys))
    un_lo = torch.tensor((-FLIM - xm[4:]) / xs[4:], dtype=torch.float32)
    un_hi = torch.tensor((FLIM - xm[4:]) / xs[4:], dtype=torch.float32)
    print("precomputing Phi/Yphi + true action-grad ...")
    Phi, Yphi = C.precompute_phi(XU)
    true_ug = C.precompute_true_ugrad(XU, xm, xs)

    configs = [("value_only", 0.003, 0.0), ("grad_only", 0.0, 0.05),
               ("value+grad", 0.003, 0.05)]
    horizon, steps, rho_u = 10, 180, 0.01
    budgets = [3, 10]
    out = {"config": dict(plant="cartpole", objective=objective, horizon=horizon,
                          steps=steps, rho_u=rho_u, seeds=SEEDS, budgets=budgets,
                          run_seeds=seeds, pred_dec="accepted_update",
                          configs=configs), "per_seed": {}}
    fn = LOG / f"ggn_mpc_cartpole_{objective}{('_' + out_suffix) if out_suffix else ''}.json"
    LOG.mkdir(parents=True, exist_ok=True)
    print(f"GGN-MPC cart-pole objective={objective}, seeds={seeds}", flush=True)
    print(f"{'config':12} {'seed':>4} {'bud':>4} {'s@2':>5} {'s@10':>5} {'acc':>5} "
          f"{'trueInc':>7} {'predDec':>9} {'trueDec':>9}", flush=True)
    for seed in seeds:
        out["per_seed"][seed] = {}
        for tag, lv, lg in configs:
            model, _ = C.train(seed, data, Phi, Yphi, true_ug, lam_val=lv, lam_grad=lg)
            seq = C.freeze(model)
            out["per_seed"][seed][tag] = {}
            for B in budgets:
                rows = [ggn_closed_loop(seq, norm, x0, horizon, B, steps, rho_u, 2.0,
                        un_lo, un_hi, objective) for x0 in C.ICS]
                rec = {k: float(np.mean([r[k] for r in rows])) for k in
                       ("success_V2", "success_V10", "accepted_frac", "true_incr_frac",
                        "mean_pred_dec", "mean_pred_dec_applied", "mean_pred_dec_full",
                        "mean_true_dec")}
                out["per_seed"][seed][tag][B] = rec
                print(f"{tag:12} {seed:>4} {B:>4} {rec['success_V2']:>5.2f} "
                      f"{rec['success_V10']:>5.2f} {rec['accepted_frac']:>5.2f} "
                      f"{rec['true_incr_frac']:>7.2f} {rec['mean_pred_dec']:>9.2e} "
                      f"{rec['mean_true_dec']:>9.2e}", flush=True)
                fn.write_text(json.dumps(out, indent=2))
    fn.write_text(json.dumps(out, indent=2))
    print("\nwrote", fn, flush=True)

    def avg(tag, B, k):
        return float(np.mean([out["per_seed"][s][tag][B][k] for s in seeds]))
    print(f"\n=== GGN cart-pole objective={objective} seed-averaged ===")
    print(f"{'config':12} {'bud':>4} {'s@2':>5} {'s@10':>6} {'acc':>5} {'trueInc':>7} {'predDec':>9}")
    for tag, _, _ in configs:
        for B in budgets:
            print(f"{tag:12} {B:>4} {avg(tag,B,'success_V2'):>5.2f} "
                  f"{avg(tag,B,'success_V10'):>6.2f} {avg(tag,B,'accepted_frac'):>5.2f} "
                  f"{avg(tag,B,'true_incr_frac'):>7.2f} {avg(tag,B,'mean_pred_dec'):>9.2e}")


def parse_cli(argv):
    objective = "V"
    seed_arg = None
    suffix = ""
    threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
    for arg in argv:
        if arg in {"V", "residual"}:
            objective = arg
        elif arg.startswith("--seeds="):
            seed_arg = arg.split("=", 1)[1]
        elif arg == "--seeds":
            raise SystemExit("Use --seeds=0,1,2")
        elif arg.startswith("--suffix="):
            suffix = arg.split("=", 1)[1]
        elif arg.startswith("--threads="):
            threads = int(arg.split("=", 1)[1])
    return objective, seed_arg, suffix, threads


if __name__ == "__main__":
    main(*parse_cli(sys.argv[1:]))
