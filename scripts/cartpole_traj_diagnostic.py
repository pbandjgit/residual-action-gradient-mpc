"""Paper-H cart-pole trajectory-distribution diagnostic.

Question: why do cart-pole seeds 3,4 fail closed-loop (b20=0) despite a low
TEST-SET wrong-sign fraction (0.03-0.07) after L_grad? Hypothesis: covariate shift —
the cheap-budget solver drives the closed loop into states the model was not trained
on (outside the sampling box), where its action-gradient and prediction degrade, even
though it is well aligned on the training/test distribution.

For each seed's value+grad model, run the closed loop (budget 20), record the visited
(x_t, u_t), and measure ALONG THE TRAJECTORY:
  - wrong-sign fraction of the action-gradient d_u V(f(x,u))  (learned vs true);
  - one-step prediction error of the surrogate;
  - fraction of visited states outside the training sampling box;
and compare to the TEST-SET values. If bad seeds show large trajectory-vs-test
gaps (high out-of-box fraction, high trajectory wrong-sign / pred error) while good
seeds stay in-distribution, the sufficiency gap is covariate shift, not a failure of
the action-gradient idea.

Outputs: results/interim/logs/cartpole_traj_diagnostic.json
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
import cartpole_lgrad as C                                            # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"
SEEDS = [0, 1, 2, 3, 4]
HORIZON, BUDGET, STEPS, LR, RHO_U, SUCCESS_V = 10, 20, 180, 0.15, 0.01, 2.0
BOX_LO = np.array([-1.8, -2.5, -0.55, -3.5]); BOX_HI = np.array([1.8, 2.5, 0.55, 3.5])


def closed_loop_record(seq, norm, x0, horizon, budget, steps, lr, rho_u, un_lo, un_hi):
    xm, xs, ym, ys = norm; xm4, xs4 = xm[:4], xs[:4]
    x = np.asarray(x0, float); Vtraj = [float(C.Vnp(x))]
    Xs, Us = [], []
    un = torch.zeros(horizon, 1)
    for _ in range(steps):
        un = un.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([un], lr=lr)
        x0n = torch.tensor((x - xm4.numpy()) / xs4.numpy(), dtype=torch.float32).view(1, 4)
        for _it in range(budget):
            opt.zero_grad(); xt = x0n; J = torch.zeros(())
            for h in range(horizon):
                y = seq(torch.cat([xt, un[h].view(1, 1)], 1)) * ys + ym
                J = J + (y @ C.Pt * y).sum() + rho_u * (un[h] ** 2).sum()
                xt = (y - xm4) / xs4
            J.backward(); opt.step()
            with torch.no_grad():
                un.clamp_(un_lo, un_hi)
        u0 = float(np.clip(un[0].detach().numpy() * xs[4:].numpy() + xm[4:].numpy(),
                           -C.FLIM, C.FLIM)[0])
        Xs.append(x.copy()); Us.append(u0)
        x = C.step(x, u0); Vtraj.append(float(C.Vnp(x)))
        un = torch.cat([un[1:].detach(), un[-1:].detach()])
    return (np.array(Xs), np.array(Us), float(Vtraj[-1]),
            bool(Vtraj[-1] <= SUCCESS_V))


def traj_metrics(seq, norm, Xs, Us):
    """wrong-sign fraction of action-grad, one-step pred MSE, out-of-box frac, over
    the recorded trajectory points."""
    xm, xs, ym, ys = [t.numpy() for t in norm]
    x_mean, x_std = xm[:4], xs[:4]; u_mean, u_std = xm[4:], xs[4:]
    ys_t = torch.tensor(ys); ym_t = torch.tensor(ym)
    # clamp absurd states to avoid overflow in finite diff on diverged trajectories
    Xs = np.clip(Xs, -1e3, 1e3)
    un = ((Us[:, None] - u_mean) / u_std)
    xn = (Xs - x_mean) / x_std
    un_t = torch.tensor(un, dtype=torch.float32, requires_grad=True)
    xn_t = torch.tensor(xn, dtype=torch.float32)
    y = seq(torch.cat([xn_t, un_t], 1)) * ys_t + ym_t
    g = C.Vt(y).sum()
    (lg,) = torch.autograd.grad(g, un_t)
    lg = lg.detach().numpy()
    eps = 1e-3
    up = un.copy(); up[:, 0] += eps; dn = un.copy(); dn[:, 0] -= eps
    uu_p = up * u_std + u_mean; uu_m = dn * u_std + u_mean
    Vp = C.Vnp(np.array([C.step(Xs[i], float(uu_p[i, 0])) for i in range(len(Xs))]))
    Vm = C.Vnp(np.array([C.step(Xs[i], float(uu_m[i, 0])) for i in range(len(Xs))]))
    tg = ((Vp - Vm) / (2 * eps)).reshape(-1, 1)
    frac_neg = float(np.mean((lg[:, 0] * tg[:, 0]) < 0))
    # one-step pred error (normalized)
    with torch.no_grad():
        yhat = (seq(torch.cat([xn_t, un_t], 1)) * ys_t + ym_t).numpy()
    ytrue = np.array([C.step(Xs[i], float(Us[i])) for i in range(len(Xs))])
    pred_mse = float(np.mean(((yhat - ytrue) / ys) ** 2))
    oob = float(np.mean(np.any((Xs < BOX_LO) | (Xs > BOX_HI), axis=1)))
    return dict(traj_frac_neg=frac_neg, traj_pred_mse=pred_mse, oob_frac=oob,
                n_pts=int(len(Xs)))


def main():
    data = C.make_data()
    XU, Y, xm, xs, ym, ys, _, _ = data
    norm = (torch.tensor(xm), torch.tensor(xs), torch.tensor(ym), torch.tensor(ys))
    un_lo = torch.tensor((-C.FLIM - xm[4:]) / xs[4:], dtype=torch.float32)
    un_hi = torch.tensor((C.FLIM - xm[4:]) / xs[4:], dtype=torch.float32)
    print("precomputing Phi/Yphi + true action-grad ...")
    Phi, Yphi = C.precompute_phi(XU)
    true_ug = C.precompute_true_ugrad(XU, xm, xs)

    out = {"config": dict(seeds=SEEDS, horizon=HORIZON, budget=BUDGET, steps=STEPS,
                          note="value+grad models; trajectory vs test-set alignment"),
           "per_seed": {}}
    print(f"{'seed':>4} {'b20':>5} {'test_fneg':>9} {'traj_fneg':>9} "
          f"{'test_mse':>9} {'traj_mse':>9} {'oob_frac':>8}")
    for seed in SEEDS:
        model, test_mse = C.train(seed, data, Phi, Yphi, true_ug, lam_val=0.003, lam_grad=0.05)
        seq = C.freeze(model)
        _, test_fneg = C.align_med(seq, data, norm, seed=seed)
        succs, allX, allU = [], [], []
        for x0 in C.ICS:
            Xs, Us, fv, sc = closed_loop_record(seq, norm, x0, HORIZON, BUDGET, STEPS,
                                                LR, RHO_U, un_lo, un_hi)
            succs.append(sc); allX.append(Xs); allU.append(Us)
        b20 = float(np.mean(succs))
        Xall = np.concatenate(allX); Uall = np.concatenate(allU)
        tm = traj_metrics(seq, norm, Xall, Uall)
        # split good vs bad ICs within the seed
        good_mask = np.array(succs, dtype=bool)
        rec = dict(b20=b20, test_fneg=test_fneg, test_mse=test_mse, **tm,
                   per_ic_success=[bool(s) for s in succs])
        out["per_seed"][seed] = rec
        print(f"{seed:>4} {b20:>5.2f} {test_fneg:>9.3f} {tm['traj_frac_neg']:>9.3f} "
              f"{test_mse:>9.2e} {tm['traj_pred_mse']:>9.2e} {tm['oob_frac']:>8.2f}")
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "cartpole_traj_diagnostic.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "cartpole_traj_diagnostic.json")

    print("\n=== good seeds (b20>=0.8) vs bad (b20<=0.2) ===")
    good = [s for s in SEEDS if out["per_seed"][s]["b20"] >= 0.8]
    bad = [s for s in SEEDS if out["per_seed"][s]["b20"] <= 0.2]
    for name, grp in [("good", good), ("bad", bad)]:
        if not grp:
            continue
        tf = np.mean([out["per_seed"][s]["traj_frac_neg"] for s in grp])
        tm = np.mean([out["per_seed"][s]["traj_pred_mse"] for s in grp])
        ob = np.mean([out["per_seed"][s]["oob_frac"] for s in grp])
        ef = np.mean([out["per_seed"][s]["test_fneg"] for s in grp])
        print(f"{name} seeds {grp}: test_fneg={ef:.3f} traj_fneg={tf:.3f} "
              f"traj_mse={tm:.2e} oob_frac={ob:.2f}")


if __name__ == "__main__":
    main()
