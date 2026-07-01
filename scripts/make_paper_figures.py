"""Paper-H data figures from the interim logs.
  fig_decoupling   : value/gradient decoupling (same MSE, opposite gradient & closed-loop)
  fig_dose_response: lam_grad dose-response on CSTR (alignment, success, Jx, MSE)
Outputs under results/figures/paper/.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "interim" / "logs"
OUT = ROOT / "results" / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

CV = "#c44e52"   # value_only
CG = "#4c72b0"   # grad_only
CVG = "#55a868"  # value+grad
COL = {"value_only": CV, "grad_only": CG, "value+grad": CVG}


def load(fn):
    return json.loads((LOG / fn).read_text())


def g(rec, *keys):
    for k in keys:
        if k in rec:
            return rec[k]
    return float("nan")


# ---------------- Fig 1: value/gradient decoupling (CSTR, lgrad_ablation) ----------------
def fig_decoupling():
    d = load("lgrad_ablation.json"); ps = d["per_seed"]; seeds = list(ps.keys())
    cfgs = ["value_only", "value+grad"]
    def ms(c, k, sc=1.0):
        v = np.array([ps[s][c][k] for s in seeds]) * sc
        return v.mean(), v.std(), v
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
    titles = ["(a)", "(b)", "(c)"]
    ylabels = ["one-step test MSE ($\\times10^{-3}$)", "action-gradient wrong-sign frac.",
               "closed-loop success (budget 20)"]
    keys = [("test_mse", 1e3), ("align_frac_neg", 1.0), ("grad_succ_b20", 1.0)]
    xpos = [0, 1]
    for ax, t, yl, (k, sc) in zip(axs, titles, ylabels, keys):
        means, stds, raws = [], [], []
        for c in cfgs:
            mm, sd, raw = ms(c, k, sc); means.append(mm); stds.append(sd); raws.append(raw)
        ax.bar(xpos, means, yerr=stds, capsize=5, color=[CV, CVG],
               edgecolor="black", width=0.6, error_kw=dict(lw=1.3))
        for xi, raw in zip(xpos, raws):           # individual seed points
            ax.scatter([xi]*len(raw), raw, color="black", s=14, zorder=3, alpha=0.7)
        ax.set_xticks(xpos); ax.set_xticklabels(["value\nonly", "value\n+$L_{grad}$"])
        ax.set_title(t, fontsize=11); ax.set_ylabel(yl, fontsize=9); ax.margins(y=0.22)
    fig.tight_layout()
    fig.savefig(OUT / "fig_decoupling.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_decoupling.pdf", bbox_inches="tight")
    print("wrote fig_decoupling")


# ---------------- Fig 2: lam_grad dose-response (CSTR) ----------------
def fig_dose_response():
    d = load("lgrad_ablation.json"); ps = d["per_seed"]; seeds = list(ps.keys())
    sweep = [("value_only", 0.0), ("sweep_0.01", 0.01), ("value+grad", 0.05),
             ("sweep_0.2", 0.2), ("sweep_1.0", 1.0)]
    def ms(tag, k, sc=1.0):
        v = np.array([ps[s][tag][k] for s in seeds]) * sc
        return v.mean(), v.std()
    lams = [l for _, l in sweep]
    fneg, fneg_s = zip(*[ms(t, "align_frac_neg") for t, _ in sweep])
    succ, succ_s = zip(*[ms(t, "grad_succ_b20") for t, _ in sweep])
    jx, jx_s = zip(*[ms(t, "state_jac_med") for t, _ in sweep])
    mse, mse_s = zip(*[ms(t, "test_mse", 1e3) for t, _ in sweep])
    x = np.arange(len(lams))
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    # left axis: control quality in [0,1]
    ax.errorbar(x, succ, yerr=succ_s, fmt="-o", color=CVG, lw=2.2, capsize=4, label="closed-loop success (b20)")
    ax.errorbar(x, fneg, yerr=fneg_s, fmt="-s", color=CV, lw=2.2, capsize=4, label="action-grad wrong-sign frac")
    ax.set_xticks(x); ax.set_xticklabels([f"{l:g}" for l in lams])
    ax.set_xlabel("$\\lambda_{grad}$  (action-gradient loss weight)")
    ax.set_ylabel("control quality  (success, wrong-sign frac)")
    ax.set_ylim(-0.05, 1.12)
    # right axis: degradation indicators that blow up for large lambda
    ax2 = ax.twinx()
    ax2.errorbar(x, mse, yerr=mse_s, fmt="-^", color="#8172b3", lw=2.0, capsize=4, label="test MSE ($\\times10^{-3}$)")
    ax2.errorbar(x, jx, yerr=jx_s, fmt="--D", color="#937860", lw=2.0, capsize=4, label="state-Jacobian $J_x$")
    ax2.set_ylabel("degradation: MSE ($\\times10^{-3}$),  $J_x$")
    ax2.grid(False)
    ax.axvspan(1.5, 2.5, color="#dff0d8", alpha=0.55)
    ax.text(2.0, 1.05, "operating\npoint", ha="center", va="top", fontsize=8.5)
    l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lab1+lab2, fontsize=8.0, loc="upper left")
    # panel title omitted; the figure is described in the caption
    fig.tight_layout()
    fig.savefig(OUT / "fig_dose_response.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_dose_response.pdf", bbox_inches="tight")
    print("wrote fig_dose_response")




if __name__ == "__main__":
    fig_decoupling(); fig_dose_response()
    print("all figures in", OUT)
