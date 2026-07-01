"""CSTR phase-portrait figure for the process-control paper.

fig_traj_cstr: CSTR state-space (x1-x2) closed-loop trajectories under the damped
Gauss-Newton MPC, value-only vs value+L_grad, ten seeds overlaid with transparency.
Shows stabilizing control reaching the unstable operating point versus divergence.

Output: results/figures/paper/fig_traj_cstr.{png,pdf}
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import budget_sweep_solver as bs
import probe_action_gradient as pr
import lgrad_experiment as Lx
import ggn_mpc_probe as Gc            # CSTR GGN (make_residual_fn)
OUT = ROOT / "results" / "figures" / "paper"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
CV, CVG = "#c44e52", "#55a868"


def gn_optimize(resfn, U, n, budget, un_lo_f, un_hi_f):
    lam = 1e-2
    for _ in range(budget):
        U = U.detach(); r = resfn(U)
        J = torch.autograd.functional.jacobian(resfn, U, vectorize=True, strategy="forward-mode")
        obj0 = 0.5 * float(r @ r); g = J.t() @ r; JTJ = J.t() @ J
        try:
            delta = torch.linalg.solve(JTJ + lam*torch.eye(n), -g)
        except RuntimeError:
            lam *= 10; continue
        acc = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            Ut = torch.clamp(U + alpha*delta, un_lo_f, un_hi_f)
            if 0.5*float(resfn(Ut) @ resfn(Ut)) < obj0:
                U = Ut; acc = True; break
        lam = max(lam*0.5, 1e-4) if acc else min(lam*10, 1e4)
    return U


def record_cstr(seq, norm, x0, horizon=3, budget=10, steps=120, rho_u=0.01):
    xm, xs, ym, ys = norm; xm2, xs2 = xm[:2], xs[:2]
    x = np.asarray(x0, float); V = [float(x @ bs.P @ x)]; Xs = [x.copy()]; Us = []
    n = horizon*2; U = torch.zeros(n)
    un_lo = torch.tensor((bs.INPUT_LO - xm[2:].numpy())/xs[2:].numpy(), dtype=torch.float32)
    un_hi = torch.tensor((bs.INPUT_HI - xm[2:].numpy())/xs[2:].numpy(), dtype=torch.float32)
    ll, hh = un_lo.repeat(horizon), un_hi.repeat(horizon)
    for _ in range(steps):
        resfn = Gc.make_residual_fn(seq, norm, x, horizon, rho_u, "V")
        U = gn_optimize(resfn, U, n, budget, ll, hh)
        u0 = np.clip(U.view(horizon, 2)[0].numpy()*xs[2:].numpy()+xm[2:].numpy(), bs.INPUT_LO, bs.INPUT_HI)
        x = bs.SIM.step(x, u0); V.append(float(x @ bs.P @ x)); Xs.append(x.copy()); Us.append(u0)
        U = torch.cat([U.view(horizon, 2)[1:], U.view(horizon, 2)[-1:]]).reshape(-1)
    return np.array(V), np.array(Xs), np.array(Us)


def cstr_data():
    d = np.load(ROOT/"data"/"processed"/"lcnn_paper_cstr_onestep_20k.npz")
    data = tuple(d[k].astype("float32") if d[k].dtype != np.int64 else d[k] for k in
                 ["XU", "Y", "x_mean", "x_std", "y_mean", "y_std"]) + (d["train_idx"], d["test_idx"])
    norm, _, _ = Lx.make_norm(data)
    Lx.PHI, Lx.YPHI = pr.precompute_phi(bs.SIM, data[0][:, :2].astype("float64"))
    ug = Lx.precompute_true_ugrad(bs.SIM, data[0], data[2], data[3])
    return data, norm, ug


def main():
    arch = {"model": "lcnn", "hidden": 40, "lipschitz_bound": 4.0, "n_layers": 2}
    cdata, cstr_norm, cug = cstr_data()
    # Overlay many seeds transparently so the portrait does not read as a hand-picked few.
    CSTR_SEEDS = list(range(10))
    print("training CSTR models (seeds %s) ..." % CSTR_SEEDS)
    cstr_vo, cstr_vg = [], []
    for s in CSTR_SEEDS:
        mv, _ = Lx.train(arch, s, cdata, Lx.PHI, Lx.YPHI, cug, lam_val=0.003, lam_grad=0.0)
        mg, _ = Lx.train(arch, s, cdata, Lx.PHI, Lx.YPHI, cug, lam_val=0.003, lam_grad=0.05)
        cstr_vo.append(bs.freeze(mv)); cstr_vg.append(bs.freeze(mg))

    icc = 2  # IC [0.25, 10]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for i in range(len(cstr_vo)):
        _, Xv, _ = record_cstr(cstr_vo[i], cstr_norm, bs.ICS[icc])
        _, Xg, _ = record_cstr(cstr_vg[i], cstr_norm, bs.ICS[icc])
        ax.plot(Xv[:, 0], Xv[:, 1], color=CV, lw=1.0, alpha=0.4)
        ax.plot(Xg[:, 0], Xg[:, 1], color=CVG, lw=1.3, alpha=0.5)
    ax.plot(0, 0, "k*", ms=13, label="target (unstable eq.)")
    ax.plot(bs.ICS[icc][0], bs.ICS[icc][1], "ko", ms=6, label="initial state")
    ax.plot([], [], color=CV, label="value only")
    ax.plot([], [], color=CVG, label="value + $L_{grad}$")
    ax.set_xlabel("$x_1$ (concentration dev.)"); ax.set_ylabel("$x_2$ (temperature dev.)")
    # zoom on the convergence basin; diverging value-only runs leave the frame
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-50, 50)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT/"fig_traj_cstr.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT/"fig_traj_cstr.pdf", bbox_inches="tight")
    print("wrote fig_traj_cstr")


if __name__ == "__main__":
    main()
