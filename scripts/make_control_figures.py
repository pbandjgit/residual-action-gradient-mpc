"""Control-style closed-loop figures (what a control audience expects):
  fig_lyapunov_convergence : V(t) under GGN-MPC, value_only (diverges) vs value+L_grad
                              (converges), CSTR and cart-pole, multiple ICs.
  fig_trajectories         : cart-pole pole angle theta(t) and control force u(t);
                              CSTR phase portrait x1-x2. Shows constraint-respecting,
                              stabilizing control vs divergence.
Records true-plant closed loops with the damped Gauss-Newton solver (seed 0).
Outputs under results/figures/paper/.
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
import cartpole_lgrad as C
import ggn_mpc_cartpole as Gp         # cart-pole GGN
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


def record_cp(seq, norm, x0, horizon=10, budget=10, steps=180, rho_u=0.01):
    xm, xs, ym, ys = norm; xm4, xs4 = xm[:4], xs[:4]
    x = np.asarray(x0, float); V = [float(C.Vnp(x))]; Xs = [x.copy()]; Us = []
    n = horizon; U = torch.zeros(n)
    un_lo = torch.tensor((-C.FLIM - xm[4:].numpy())/xs[4:].numpy(), dtype=torch.float32)
    un_hi = torch.tensor((C.FLIM - xm[4:].numpy())/xs[4:].numpy(), dtype=torch.float32)
    ll, hh = un_lo.repeat(horizon), un_hi.repeat(horizon)
    for _ in range(steps):
        resfn = Gp.make_residual_fn(seq, norm, x, horizon, rho_u, "V")
        U = gn_optimize(resfn, U, n, budget, ll, hh)
        u0 = float(np.clip(U.view(horizon, 1)[0].numpy()*xs[4:].numpy()+xm[4:].numpy(), -C.FLIM, C.FLIM)[0])
        x = C.step(x, u0); V.append(float(C.Vnp(x))); Xs.append(x.copy()); Us.append(u0)
        U = torch.cat([U.view(horizon, 1)[1:], U.view(horizon, 1)[-1:]]).reshape(-1)
    return np.array(V), np.array(Xs), np.array(Us)


SEEDS = [0, 1, 2]


def cstr_data():
    d = np.load(ROOT/"data"/"processed"/"lcnn_paper_cstr_onestep_20k.npz")
    data = tuple(d[k].astype("float32") if d[k].dtype != np.int64 else d[k] for k in
                 ["XU", "Y", "x_mean", "x_std", "y_mean", "y_std"]) + (d["train_idx"], d["test_idx"])
    norm, _, _ = Lx.make_norm(data)
    Lx.PHI, Lx.YPHI = pr.precompute_phi(bs.SIM, data[0][:, :2].astype("float64"))
    ug = Lx.precompute_true_ugrad(bs.SIM, data[0], data[2], data[3])
    return data, norm, ug


def cp_data():
    C.DEFAULT_HIDDEN = 48
    data = C.make_data(); XU, Y, xm, xs, ym, ys, _, _ = data
    norm = (torch.tensor(xm), torch.tensor(xs), torch.tensor(ym), torch.tensor(ys))
    Phi, Yphi = C.precompute_phi(XU); ug = C.precompute_true_ugrad(XU, xm, xs)
    return data, norm, (Phi, Yphi, ug)


def main():
    arch = {"model": "lcnn", "hidden": 40, "lipschitz_bound": 4.0, "n_layers": 2}
    print("training + recording (seeds %s) ..." % SEEDS)
    cdata, cstr_norm, cug = cstr_data()
    pdata, cp_norm, (Phi, Yphi, pug) = cp_data()

    # per-seed frozen models
    cstr_vo, cstr_vg, cp_vo, cp_vg = [], [], [], []
    for s in SEEDS:
        mv, _ = Lx.train(arch, s, cdata, Lx.PHI, Lx.YPHI, cug, lam_val=0.003, lam_grad=0.0)
        mg, _ = Lx.train(arch, s, cdata, Lx.PHI, Lx.YPHI, cug, lam_val=0.003, lam_grad=0.05)
        cstr_vo.append(bs.freeze(mv)); cstr_vg.append(bs.freeze(mg))
        pv, _ = C.train(s, pdata, Phi, Yphi, pug, lam_val=0.003, lam_grad=0.0)
        pg, _ = C.train(s, pdata, Phi, Yphi, pug, lam_val=0.003, lam_grad=0.05)
        cp_vo.append(C.freeze(pv)); cp_vg.append(C.freeze(pg))

    # =============== Fig 1: Lyapunov convergence (seed 0, multi-IC) ===============
    cstr_V_vo = [record_cstr(cstr_vo[0], cstr_norm, ic)[0] for ic in bs.ICS]
    cstr_V_vg = [record_cstr(cstr_vg[0], cstr_norm, ic)[0] for ic in bs.ICS]
    cp_V_vo = [record_cp(cp_vo[0], cp_norm, ic)[0] for ic in C.ICS]
    cp_V_vg = [record_cp(cp_vg[0], cp_norm, ic)[0] for ic in C.ICS]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, Vvo, Vvg, name, panel in [(axs[0], cstr_V_vo, cstr_V_vg, "CSTR", "(a)"),
                                      (axs[1], cp_V_vo, cp_V_vg, "cart-pole", "(b)")]:
        for V in Vvo:
            ax.semilogy(np.clip(V, 1e-3, None), color=CV, alpha=0.7, lw=1.3)
        for V in Vvg:
            ax.semilogy(np.clip(V, 1e-3, None), color=CVG, alpha=0.85, lw=1.6)
        ax.axhline(2.0, color="gray", ls=":", lw=1)
        ax.set_title(f"{panel} {name}")
        ax.set_xlabel("MPC step"); ax.set_ylabel("Lyapunov value  $V(x_t)$")
    axs[0].plot([], [], color=CV, label="value only")
    axs[0].plot([], [], color=CVG, label="value + $L_{grad}$")
    axs[0].plot([], [], color="gray", ls=":", label="success threshold")
    axs[0].legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT/"fig_lyapunov_convergence.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT/"fig_lyapunov_convergence.pdf", bbox_inches="tight")
    print("wrote fig_lyapunov_convergence")

    # =============== Fig 2: cart-pole trajectories (multi-seed overlay) ===============
    icp = 4  # IC [0,0,0.40,0] ~ 23 deg
    recs_vo = [record_cp(cp_vo[i], cp_norm, C.ICS[icp]) for i in range(len(SEEDS))]
    recs_vg = [record_cp(cp_vg[i], cp_norm, C.ICS[icp]) for i in range(len(SEEDS))]
    dt = C.CPP.dt
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    for (V, X, U) in recs_vo:
        t = np.arange(len(U))*dt
        axs[0].plot(t, np.degrees(np.unwrap(X[:-1, 2])), color=CV, lw=1.3, alpha=0.7)
        axs[1].plot(t, U, color=CV, lw=1.0, alpha=0.6)
    for (V, X, U) in recs_vg:
        t = np.arange(len(U))*dt
        axs[0].plot(t, np.degrees(np.unwrap(X[:-1, 2])), color=CVG, lw=1.7, alpha=0.85)
        axs[1].plot(t, U, color=CVG, lw=1.3, alpha=0.75)
    axs[0].axhline(0, color="gray", lw=0.8)
    axs[0].set_title("(a)")
    axs[0].set_xlabel("time [s]"); axs[0].set_ylabel("$\\theta$ [deg]"); axs[0].set_ylim(-35, 60)
    axs[0].plot([], [], color=CV, label="value only (pole falls $\\to$ off-frame)")
    axs[0].plot([], [], color=CVG, label="value + $L_{grad}$ (stabilizes)")
    axs[0].legend(fontsize=8.3, loc="upper right")
    axs[1].axhline(C.FLIM, color="r", ls="--", lw=1, alpha=0.6)
    axs[1].axhline(-C.FLIM, color="r", ls="--", lw=1, alpha=0.6)
    axs[1].text(0.1, C.FLIM*1.03, "input limit $\\pm F_{max}$", color="r", fontsize=8)
    axs[1].set_title("(b)")
    axs[1].set_xlabel("time [s]"); axs[1].set_ylabel("force [N]"); axs[1].set_ylim(-12.5, 13.5)
    fig.tight_layout()
    fig.savefig(OUT/"fig_traj_cartpole.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT/"fig_traj_cartpole.pdf", bbox_inches="tight")
    print("wrote fig_traj_cartpole")

    # =============== Fig 3: CSTR phase portrait (multi-seed overlay) ===============
    icc = 2  # IC [0.25, 10]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for i in range(len(SEEDS)):
        _, Xv, _ = record_cstr(cstr_vo[i], cstr_norm, bs.ICS[icc])
        _, Xg, _ = record_cstr(cstr_vg[i], cstr_norm, bs.ICS[icc])
        ax.plot(Xv[:, 0], Xv[:, 1], color=CV, lw=1.3, alpha=0.7)
        ax.plot(Xg[:, 0], Xg[:, 1], color=CVG, lw=1.7, alpha=0.85)
    ax.plot(0, 0, "k*", ms=13, label="target (unstable eq.)")
    ax.plot(bs.ICS[icc][0], bs.ICS[icc][1], "ko", ms=6, label="initial state")
    ax.plot([], [], color=CV, label="value only")
    ax.plot([], [], color=CVG, label="value + $L_{grad}$")
    ax.set_xlabel("$x_1$ (concentration dev.)"); ax.set_ylabel("$x_2$ (temperature dev.)")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT/"fig_traj_cstr.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT/"fig_traj_cstr.pdf", bbox_inches="tight")
    print("wrote fig_traj_cstr")


if __name__ == "__main__":
    main()
