"""Region-of-attraction / multi-IC robustness: over a grid of initial conditions,
which states does the finite-budget closed loop recover? value_only vs value+L_grad,
CSTR and cart-pole (first-order projected-Adam MPC, budget 20). Larger green region =
larger recovered set. Outputs results/figures/paper/fig_roa.*
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts"))
import budget_sweep_solver as bs
import probe_action_gradient as pr
import lgrad_experiment as Lx
import cartpole_lgrad as C
OUT = ROOT/"results"/"figures"/"paper"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10})
CVG, CV = "#55a868", "#c44e52"


def cstr_models():
    d = np.load(ROOT/"data"/"processed"/"lcnn_paper_cstr_onestep_20k.npz")
    data = tuple(d[k].astype("float32") if d[k].dtype != np.int64 else d[k] for k in
                 ["XU", "Y", "x_mean", "x_std", "y_mean", "y_std"]) + (d["train_idx"], d["test_idx"])
    norm, lo, hi = Lx.make_norm(data)
    Lx.PHI, Lx.YPHI = pr.precompute_phi(bs.SIM, data[0][:, :2].astype("float64"))
    ug = Lx.precompute_true_ugrad(bs.SIM, data[0], data[2], data[3])
    arch = {"model": "lcnn", "hidden": 40, "lipschitz_bound": 4.0, "n_layers": 2}
    mv, _ = Lx.train(arch, 0, data, Lx.PHI, Lx.YPHI, ug, lam_val=0.003, lam_grad=0.0)
    mg, _ = Lx.train(arch, 0, data, Lx.PHI, Lx.YPHI, ug, lam_val=0.003, lam_grad=0.05)
    return bs.freeze(mv), bs.freeze(mg), norm, lo, hi


def cp_models():
    C.DEFAULT_HIDDEN = 48
    data = C.make_data(); XU, Y, xm, xs, ym, ys, _, _ = data
    norm = (torch.tensor(xm), torch.tensor(xs), torch.tensor(ym), torch.tensor(ys))
    lo = torch.tensor((-C.FLIM-xm[4:])/xs[4:], dtype=torch.float32)
    hi = torch.tensor((C.FLIM-xm[4:])/xs[4:], dtype=torch.float32)
    Phi, Yphi = C.precompute_phi(XU); ug = C.precompute_true_ugrad(XU, xm, xs)
    mv, _ = C.train(0, data, Phi, Yphi, ug, lam_val=0.003, lam_grad=0.0)
    mg, _ = C.train(0, data, Phi, Yphi, ug, lam_val=0.003, lam_grad=0.05)
    return C.freeze(mv), C.freeze(mg), norm, lo, hi


def main():
    print("training ...")
    csv, csg, csn, cslo, cshi = cstr_models()
    cpv, cpg, cpn, cplo, cphi = cp_models()

    # CSTR grid over (x1, x2)
    x1s = np.linspace(-0.45, 0.45, 13); x2s = np.linspace(-22, 22, 13)
    # cart-pole grid over (theta0, thetadot0), pos=vel=0
    ths = np.linspace(-0.55, 0.55, 13); thds = np.linspace(-3.0, 3.0, 13)

    def cstr_succ(seq, x1, x2):
        r = bs.closed_loop(seq, csn, np.array([x1, x2]), 3, 20, 120, 0.2, 0.01, 2.0, cslo, cshi)
        return r["success"]

    def cp_succ(seq, th, thd):
        r = C.closed_loop(seq, cpn, np.array([0.0, 0.0, th, thd]), 10, 20, 150, 0.15, 0.01, 10.0, cplo, cphi)
        return r["success"]

    print("CSTR ROA grid ...")
    Sv = np.array([[cstr_succ(csv, a, b) for a in x1s] for b in x2s])
    Sg = np.array([[cstr_succ(csg, a, b) for a in x1s] for b in x2s])
    print("cart-pole ROA grid ...")
    Pv = np.array([[cp_succ(cpv, a, b) for a in ths] for b in thds])
    Pg = np.array([[cp_succ(cpg, a, b) for a in ths] for b in thds])

    fig, axs = plt.subplots(1, 4, figsize=(15, 3.7))
    def panel(ax, S, XX, YY, title, xlabel, ylabel):
        ax.imshow(S, origin="lower", extent=[XX[0], XX[-1], YY[0], YY[-1]],
                  aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, alpha=0.85)
        ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.text(0.5, 0.04, f"recovered: {100*S.mean():.0f}%", transform=ax.transAxes,
                ha="center", fontsize=9, weight="bold",
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    panel(axs[0], Sv, x1s, x2s, "(a) CSTR — value only", "$x_1$", "$x_2$")
    panel(axs[1], Sg, x1s, x2s, "(b) CSTR — value + $L_{grad}$", "$x_1$", "$x_2$")
    panel(axs[2], Pv, ths, thds, "(c) cart-pole — value only", "$\\theta_0$ [rad]", "$\\dot\\theta_0$ [rad/s]")
    panel(axs[3], Pg, ths, thds, "(d) cart-pole — value + $L_{grad}$", "$\\theta_0$ [rad]", "$\\dot\\theta_0$ [rad/s]")
    fig.tight_layout()
    fig.savefig(OUT/"fig_roa.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT/"fig_roa.pdf", bbox_inches="tight")
    print("wrote fig_roa  (CSTR %.0f%%->%.0f%%, cart-pole %.0f%%->%.0f%%)" %
          (100*Sv.mean(), 100*Sg.mean(), 100*Pv.mean(), 100*Pg.mean()))


if __name__ == "__main__":
    main()
