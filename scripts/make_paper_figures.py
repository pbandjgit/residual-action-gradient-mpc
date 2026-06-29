"""Paper-H data figures from the interim logs.
  fig_decoupling   : value/gradient decoupling (same MSE, opposite gradient & closed-loop)
  fig_dose_response: lam_grad dose-response on CSTR (alignment, success, Jx, MSE)
  fig_ggn_main     : GGN headline — predicted-decrease faithfulness + success + accepted
  fig_two_condition: cart-pole per-seed — gradient fixed for all, success limited by axis (a)
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


# ---------------- Fig 3: GGN headline (4 settings, budget 10) ----------------
def fig_ggn_main():
    settings = [("ggn_mpc_probe_V.json", "success", "CSTR\nrolled-V"),
                ("ggn_mpc_probe_residual.json", "success", "CSTR\nresidual"),
                ("ggn_mpc_cartpole_V.json", "success_V10", "cart-pole\nrolled-V"),
                ("ggn_mpc_cartpole_residual.json", "success_V10", "cart-pole\nresidual")]
    cfgs = ["value_only", "grad_only", "value+grad"]; B = 10
    succ = {c: [] for c in cfgs}; pdec = {c: [] for c in cfgs}; tinc = {c: [] for c in cfgs}
    succ_s = {c: [] for c in cfgs}; pdec_s = {c: [] for c in cfgs}; tinc_s = {c: [] for c in cfgs}
    labels = []
    for fn, sk, lab in settings:
        d = load(fn); ps = d["per_seed"]; seeds = list(ps.keys()); labels.append(lab)
        for c in cfgs:
            rs = [ps[s][c][str(B)] for s in seeds]
            sv = np.array([r[sk] for r in rs]); succ[c].append(sv.mean()); succ_s[c].append(sv.std())
            pv = np.array([g(r, "mean_pred_dec_applied", "mean_pred_dec") for r in rs])
            pdec[c].append(pv.mean()); pdec_s[c].append(pv.std())
            tv = np.array([r["true_incr_frac"] for r in rs]); tinc[c].append(tv.mean()); tinc_s[c].append(tv.std())
    x = np.arange(len(settings)); w = 0.26
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ek = dict(lw=1.1)
    for i, c in enumerate(cfgs):
        axs[0].bar(x+(i-1)*w, succ[c], w, yerr=succ_s[c], capsize=3, color=COL[c],
                   edgecolor="black", label=c.replace("_", " "), error_kw=ek)
    axs[0].set_title("(a)"); axs[0].set_ylabel("closed-loop success", fontsize=9); axs[0].set_ylim(0, 1.12)
    axs[0].set_xticks(x); axs[0].set_xticklabels(labels, fontsize=8.2); axs[0].legend(fontsize=8.2)
    for i, c in enumerate(cfgs):
        m = np.array(pdec[c]); s = np.array(pdec_s[c]); lo = np.minimum(s, m*0.999)
        axs[1].bar(x+(i-1)*w, m, w, yerr=[lo, s], capsize=3, color=COL[c], edgecolor="black", error_kw=ek)
    axs[1].set_yscale("log"); axs[1].set_title("(b)"); axs[1].set_ylabel("predicted decrease (log)", fontsize=9)
    axs[1].set_xticks(x); axs[1].set_xticklabels(labels, fontsize=8.2)
    for i, c in enumerate(cfgs):
        axs[2].bar(x+(i-1)*w, tinc[c], w, yerr=tinc_s[c], capsize=3, color=COL[c], edgecolor="black", error_kw=ek)
    axs[2].set_title("(c)"); axs[2].set_ylabel("frac. steps increasing $V$", fontsize=9); axs[2].set_ylim(0, 0.85)
    axs[2].set_xticks(x); axs[2].set_xticklabels(labels, fontsize=8.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ggn_main.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_ggn_main.pdf", bbox_inches="tight")
    print("wrote fig_ggn_main")


# ---------------- Fig 4: two-condition (cart-pole per-seed) ----------------
def fig_two_condition():
    camp = load("cartpole_campaign.json"); ps = camp["per_seed"]; seeds = sorted(ps.keys(), key=int)
    ggnA = load("ggn_mpc_cartpole_V.json")["per_seed"]
    dag = load("cartpole_dagger.json")["per_seed"]["4"]   # seed-4 DAgger fix
    fneg_vo = [ps[s]["value_only"]["frac_neg"] for s in seeds]
    fneg_vg = [ps[s]["value+grad"]["frac_neg"] for s in seeds]
    succ_vo = [ggnA[s]["value_only"]["10"]["success_V10"] for s in seeds]   # GGN, V<=10
    succ_vg = [ggnA[s]["value+grad"]["10"]["success_V10"] for s in seeds]   # GGN, V<=10
    dag_it = [r["iter"] for r in dag]; dag_s = [r["b20_V10"] for r in dag]
    dag_mse = np.array([r["traj_mse"] for r in dag], dtype=float)
    dag_mse_norm = dag_mse / dag_mse[0]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.7))
    # (a) Paired seed plot: emphasizes the within-seed shift, not the seed identity.
    for a, b in zip(fneg_vo, fneg_vg):
        axs[0].plot([0, 1], [a, b], color="#999999", lw=1.2, alpha=0.75, zorder=1)
    axs[0].scatter(np.zeros(len(seeds)), fneg_vo, color=CV, edgecolor="black", s=45,
                   label="value only", zorder=2)
    axs[0].scatter(np.ones(len(seeds)), fneg_vg, color=CVG, edgecolor="black", s=45,
                   label="value+$L_{grad}$", zorder=2)
    axs[0].set_title("(a)")
    axs[0].set_xlim(-0.35, 1.35); axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(["value\nonly", "value\n+$L_{grad}$"])
    axs[0].set_ylabel("wrong-sign fraction"); axs[0].legend(fontsize=8.5)
    axs[0].text(0.5, 0.91, f"mean {np.mean(fneg_vo):.2f} $\\rightarrow$ {np.mean(fneg_vg):.2f}",
                ha="center", transform=axs[0].transAxes, fontsize=8.3,
                bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.8))

    # (b) Aggregate recovery with seed dots: shows improvement while avoiding seed-index storytelling.
    rng = np.random.default_rng(20260629)
    groups = [np.array(succ_vo), np.array(succ_vg)]
    axs[1].bar([0, 1], [g.mean() for g in groups], yerr=[g.std() for g in groups],
               capsize=5, color=[CV, CVG], edgecolor="black", width=0.55,
               error_kw=dict(lw=1.2))
    for xi, vals in enumerate(groups):
        jitter = rng.uniform(-0.07, 0.07, size=len(vals))
        axs[1].scatter(np.full(len(vals), xi) + jitter, vals, color="black",
                       alpha=0.7, s=18, zorder=3)
    axs[1].set_title("(b)")
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["value\nonly", "value\n+$L_{grad}$"])
    axs[1].set_ylabel("GGN success (V$\\leq$10)"); axs[1].set_ylim(0, 1.12)
    axs[1].text(0.5, 0.91, f"mean {np.mean(succ_vo):.2f} $\\rightarrow$ {np.mean(succ_vg):.2f}",
                ha="center", transform=axs[1].transAxes, fontsize=8.3,
                bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.8))

    # (c) DAgger refinement: success increases as on-trajectory prediction error decreases.
    axs[2].plot(dag_it, dag_s, "-o", color="#dd8452", lw=2.3, label="seed-4 success")
    axs[2].plot(dag_it, dag_mse_norm, "--s", color="#4c72b0", lw=2.0,
                label="trajectory MSE / initial")
    axs[2].set_title("(c)")
    axs[2].set_xlabel("DAgger iteration"); axs[2].set_xticks(dag_it)
    axs[2].set_ylabel("normalized value"); axs[2].set_ylim(0, 1.12)
    axs[2].legend(fontsize=8.5)
    for it, s in zip(dag_it, dag_s):
        axs[2].text(it, s+0.03, f"{s:.1f}", ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_two_condition.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_two_condition.pdf", bbox_inches="tight")
    print("wrote fig_two_condition")


if __name__ == "__main__":
    fig_decoupling(); fig_dose_response(); fig_ggn_main(); fig_two_condition()
    print("all figures in", OUT)
