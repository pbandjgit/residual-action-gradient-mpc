"""Two-CSTR-in-series figures (process second plant) from the interim logs.

  fig_two_cstr_main : (a) action-gradient wrong-sign fraction, (b) finite-budget
                      closed-loop success (first-order + Gauss-Newton),
                      (c) lam_grad dose-response, (d) Gauss-Newton mean true
                      Lyapunov decrease (faithful vs unfaithful linearization).
  fig_two_cstr_jac  : Jacobian faithfulness (state-Jacobian magnitude vs the true
                      plant, and state-Jacobian error) for value-only vs value+grad.

Outputs under results/figures/paper/.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "interim" / "logs"
OUT = ROOT / "results" / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "axes.grid": True, "grid.alpha": 0.3})

CV = "#c44e52"   # value_only
CG = "#4c72b0"   # grad_only
CVG = "#55a868"  # value+grad


def load(fn):
    return json.loads((LOG / fn).read_text())


def seed_avg(d, getter):
    seeds = list(d["per_seed"].keys())
    return float(np.mean([getter(d["per_seed"][s]) for s in seeds]))


def seed_std(d, getter):
    seeds = list(d["per_seed"].keys())
    return float(np.std([getter(d["per_seed"][s]) for s in seeds]))


def seed_vals(d, getter):
    seeds = list(d["per_seed"].keys())
    return [getter(d["per_seed"][s]) for s in seeds]


def clip01_err(m, s):
    """Asymmetric whiskers that stay inside the [0,1] success/fraction range."""
    lo = [min(si, mi) for mi, si in zip(m, s)]
    up = [min(si, 1.0 - mi) for mi, si in zip(m, s)]
    return [lo, up]


def main():
    abl = load("two_cstr_lgrad.json")
    dose = load("two_cstr_dose.json")
    ggn = load("ggn_mpc_two_cstr_V.json")
    jac = load("two_cstr_jac_faithful.json")
    cfgs = ["value_only", "grad_only", "value+grad"]
    cols = [CV, CG, CVG]
    xlab = ["value\nonly", "grad\nonly", "value\n+$L_{grad}$"]

    fig, axs = plt.subplots(2, 2, figsize=(9.5, 7.2))

    # (a) action-gradient wrong-sign fraction (mean +/- s.d., seed dots)
    fn_m = [seed_avg(abl, lambda ps, c=c: ps[c]["frac_neg"]) for c in cfgs]
    fn_s = [seed_std(abl, lambda ps, c=c: ps[c]["frac_neg"]) for c in cfgs]
    axs[0, 0].bar([0, 1, 2], fn_m, yerr=clip01_err(fn_m, fn_s), capsize=4, color=cols,
                  edgecolor="black", width=0.6, error_kw=dict(lw=1.2))
    for j, c in enumerate(cfgs):
        vs = seed_vals(abl, lambda ps, c=c: ps[c]["frac_neg"])
        axs[0, 0].scatter([j] * len(vs), vs, color="black", s=12, zorder=3, alpha=0.7)
    axs[0, 0].set_xticks([0, 1, 2]); axs[0, 0].set_xticklabels(xlab)
    axs[0, 0].set_ylabel("action-gradient wrong-sign frac."); axs[0, 0].set_title("(a)")

    # (b) finite-budget success: first-order (b20) and Gauss-Newton (b10), mean +/- s.d.
    fo_m = [seed_avg(abl, lambda ps, c=c: ps[c]["b20"]) for c in cfgs]
    fo_s = [seed_std(abl, lambda ps, c=c: ps[c]["b20"]) for c in cfgs]
    gn_m = [seed_avg(ggn, lambda ps, c=c: ps[c]["10"]["success"]) for c in cfgs]
    gn_s = [seed_std(ggn, lambda ps, c=c: ps[c]["10"]["success"]) for c in cfgs]
    x = np.arange(3); w = 0.36
    axs[0, 1].bar(x - w / 2, fo_m, w, yerr=clip01_err(fo_m, fo_s), capsize=3, color=cols,
                  edgecolor="black", error_kw=dict(lw=1.1))
    axs[0, 1].bar(x + w / 2, gn_m, w, yerr=clip01_err(gn_m, gn_s), capsize=3, color=cols,
                  edgecolor="black", hatch="///", error_kw=dict(lw=1.1))
    axs[0, 1].set_xticks(x); axs[0, 1].set_xticklabels(xlab); axs[0, 1].set_ylim(0, 1.34)
    axs[0, 1].set_ylabel("closed-loop success")
    solver_proxies = [
        Patch(facecolor="0.8", edgecolor="black", label="first-order ($B{=}20$)"),
        Patch(facecolor="0.8", edgecolor="black", hatch="////", label="Gauss--Newton ($B{=}10$)")]
    axs[0, 1].legend(handles=solver_proxies, fontsize=7.8, loc="upper center",
                     ncol=2, framealpha=0.6, columnspacing=1.0, handlelength=1.4,
                     borderpad=0.4, handletextpad=0.5)
    axs[0, 1].set_title("(b)")

    # (c) lam_grad dose-response (mean +/- s.d.)
    lams = dose["config"]["lams"]
    succ_m = [seed_avg(dose, lambda ps, l=l: ps[f"{l:g}"]["b20"]) for l in lams]
    succ_s = [seed_std(dose, lambda ps, l=l: ps[f"{l:g}"]["b20"]) for l in lams]
    wr_m = [seed_avg(dose, lambda ps, l=l: ps[f"{l:g}"]["frac_neg"]) for l in lams]
    wr_s = [seed_std(dose, lambda ps, l=l: ps[f"{l:g}"]["frac_neg"]) for l in lams]
    jx = [seed_avg(dose, lambda ps, l=l: ps[f"{l:g}"]["state_jac_med"]) for l in lams]
    xx = np.arange(len(lams))
    axs[1, 0].errorbar(xx, succ_m, yerr=succ_s, fmt="-o", color=CVG, lw=2, capsize=3,
                       label="success (b20)")
    axs[1, 0].errorbar(xx, wr_m, yerr=wr_s, fmt="-s", color=CV, lw=2, capsize=3,
                       label="wrong-sign frac.")
    axs[1, 0].set_xticks(xx); axs[1, 0].set_xticklabels([f"{l:g}" for l in lams])
    axs[1, 0].set_xlabel("$\\lambda_{grad}$"); axs[1, 0].set_ylim(-0.05, 1.18)
    axs[1, 0].set_ylabel("success / wrong-sign frac.")
    ax2 = axs[1, 0].twinx()
    ax2.plot(xx, jx, "--D", color="#937860", lw=1.8, label="state-Jac. $J_x$")
    ax2.set_ylabel("$J_x$"); ax2.grid(False)
    h1, l1 = axs[1, 0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axs[1, 0].legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    axs[1, 0].set_title("(c)")

    # (d) Gauss-Newton mean true Lyapunov decrease (mean +/- s.d.)
    td_m = [seed_avg(ggn, lambda ps, c=c: ps[c]["10"]["mean_true_dec"]) for c in cfgs]
    td_s = [seed_std(ggn, lambda ps, c=c: ps[c]["10"]["mean_true_dec"]) for c in cfgs]
    axs[1, 1].bar([0, 1, 2], td_m, yerr=td_s, capsize=4, color=cols, edgecolor="black",
                  width=0.6, error_kw=dict(lw=1.2))
    axs[1, 1].axhline(0, color="black", lw=0.8)
    axs[1, 1].set_xticks([0, 1, 2]); axs[1, 1].set_xticklabels(xlab)
    axs[1, 1].set_ylabel("GGN mean true $V$ decrease / step"); axs[1, 1].set_title("(d)")

    fig.tight_layout()
    fig.savefig(OUT / "fig_two_cstr_main.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_two_cstr_main.pdf", bbox_inches="tight")
    print("wrote fig_two_cstr_main")

    # ----- Jacobian faithfulness -----
    jc = ["value_only", "value+grad"]
    jcol = [CV, CVG]
    jxm = [seed_avg(jac, lambda ps, c=c: ps[c]["Jx_model"]) for c in jc]
    jxm_s = [seed_std(jac, lambda ps, c=c: ps[c]["Jx_model"]) for c in jc]
    jerr = [seed_avg(jac, lambda ps, c=c: ps[c]["jac_err"]) for c in jc]
    jerr_s = [seed_std(jac, lambda ps, c=c: ps[c]["jac_err"]) for c in jc]
    jtrue = seed_avg(jac, lambda ps: ps["value_only"]["Jx_true"])
    fig2, axs2 = plt.subplots(1, 2, figsize=(8, 3.4))
    axs2[0].bar([0, 1], jxm, yerr=jxm_s, capsize=4, color=jcol, edgecolor="black",
                width=0.6, error_kw=dict(lw=1.2))
    for j, c in enumerate(jc):
        vs = seed_vals(jac, lambda ps, c=c: ps[c]["Jx_model"])
        axs2[0].scatter([j] * len(vs), vs, color="black", s=12, zorder=3, alpha=0.7)
    axs2[0].axhline(jtrue, color="black", ls="--", lw=1.5, label=f"true plant $J_x$ = {jtrue:.2f}")
    axs2[0].set_xticks([0, 1]); axs2[0].set_xticklabels(["value\nonly", "value\n+$L_{grad}$"])
    axs2[0].set_ylabel("surrogate state-Jacobian $J_x$")
    axs2[0].legend(fontsize=8.5); axs2[0].set_title("(a) Jacobian magnitude")
    axs2[1].bar([0, 1], jerr, yerr=jerr_s, capsize=4, color=jcol, edgecolor="black",
                width=0.6, error_kw=dict(lw=1.2))
    for j, c in enumerate(jc):
        vs = seed_vals(jac, lambda ps, c=c: ps[c]["jac_err"])
        axs2[1].scatter([j] * len(vs), vs, color="black", s=12, zorder=3, alpha=0.7)
    axs2[1].set_xticks([0, 1]); axs2[1].set_xticklabels(["value\nonly", "value\n+$L_{grad}$"])
    axs2[1].set_ylabel("state-Jacobian error $\\|J_{model}-J_{true}\\|$")
    axs2[1].set_title("(b) Jacobian error")
    fig2.tight_layout()
    fig2.savefig(OUT / "fig_two_cstr_jac.png", dpi=200, bbox_inches="tight")
    fig2.savefig(OUT / "fig_two_cstr_jac.pdf", bbox_inches="tight")
    print("wrote fig_two_cstr_jac")

    # ----- process-realism robustness -----
    try:
        real = load("two_cstr_realism.json")
        scs = real["config"]["scenarios"]
        short = ["nominal", "feed-temp\n+5K", "$k_0$ +5%", "$\\Delta H$ +5%"]
        vo_s = [seed_avg(real, lambda ps, s=s: ps["value_only"][s]["success"]) for s in scs]
        vg_s = [seed_avg(real, lambda ps, s=s: ps["value+grad"][s]["success"]) for s in scs]
        vo_ss = [seed_std(real, lambda ps, s=s: ps["value_only"][s]["success"]) for s in scs]
        vg_ss = [seed_std(real, lambda ps, s=s: ps["value+grad"][s]["success"]) for s in scs]
        vo_v = [seed_avg(real, lambda ps, s=s: ps["value_only"][s]["final_V"]) for s in scs]
        vg_v = [seed_avg(real, lambda ps, s=s: ps["value+grad"][s]["final_V"]) for s in scs]
        vo_vs = [seed_std(real, lambda ps, s=s: ps["value_only"][s]["final_V"]) for s in scs]
        vg_vs = [seed_std(real, lambda ps, s=s: ps["value+grad"][s]["final_V"]) for s in scs]

        def logerr(m, sd):  # keep lower whisker positive on the log axis
            lo = [min(si, mi * 0.999) for mi, si in zip(m, sd)]
            return [lo, sd]
        fig3, ax3 = plt.subplots(1, 2, figsize=(9, 3.4))
        xx = np.arange(len(scs)); w = 0.38
        ax3[0].bar(xx - w / 2, vo_s, w, yerr=vo_ss, capsize=3, color=CV, edgecolor="black",
                   label="value-only", error_kw=dict(lw=1.0))
        ax3[0].bar(xx + w / 2, vg_s, w, yerr=vg_ss, capsize=3, color=CVG, edgecolor="black",
                   label="value+$L_{grad}$", error_kw=dict(lw=1.0))
        ax3[0].set_xticks(xx); ax3[0].set_xticklabels(short, fontsize=8); ax3[0].set_ylim(0, 1.18)
        ax3[0].set_ylabel("closed-loop success"); ax3[0].legend(fontsize=8)
        ax3[0].set_title("(a) success under plant variation")
        ax3[1].bar(xx - w / 2, vo_v, w, yerr=logerr(vo_v, vo_vs), capsize=3, color=CV,
                   edgecolor="black", error_kw=dict(lw=1.0))
        ax3[1].bar(xx + w / 2, vg_v, w, yerr=logerr(vg_v, vg_vs), capsize=3, color=CVG,
                   edgecolor="black", error_kw=dict(lw=1.0))
        ax3[1].set_yscale("log"); ax3[1].set_xticks(xx); ax3[1].set_xticklabels(short, fontsize=8)
        ax3[1].set_ylabel("final Lyapunov value $V$ (log)")
        ax3[1].set_title("(b) final Lyapunov value")
        fig3.tight_layout()
        fig3.savefig(OUT / "fig_two_cstr_realism.png", dpi=200, bbox_inches="tight")
        fig3.savefig(OUT / "fig_two_cstr_realism.pdf", bbox_inches="tight")
        print("wrote fig_two_cstr_realism")
    except FileNotFoundError:
        print("(skipped fig_two_cstr_realism: two_cstr_realism.json not found)")

    # ----- LaTeX ablation table -----
    def row(c):
        return (xlab[cfgs.index(c)].replace("\n", " "),
                seed_avg(abl, lambda ps: ps[c]["test_mse"]),
                seed_avg(abl, lambda ps: ps[c]["frac_neg"]),
                seed_avg(abl, lambda ps: ps[c]["b20"]),
                seed_avg(ggn, lambda ps: ps[c]["10"]["success"]),
                seed_avg(abl, lambda ps: ps[c]["state_jac_med"]))
    lines = [
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "training & one-step MSE & wrong-sign & first-order succ. & GGN succ. & $J_x$ \\\\",
        "\\midrule"]
    for c in cfgs:
        n, mse, fn_, fo_, gn_, jx_ = row(c)
        lines.append(f"{n} & {mse:.2e} & {fn_:.2f} & {fo_:.2f} & {gn_:.2f} & {jx_:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_two_cstr.tex").write_text("\n".join(lines))
    print("wrote table_two_cstr.tex")
    print("\n".join(lines))

    # ----- GGN both-objectives table (budget 10) -----
    ggn_res = load("ggn_mpc_two_cstr_residual.json")
    glines = ["\\begin{tabular}{llccc}", "\\toprule",
              "objective & training & success & $V$-increase frac. & mean true $V$ dec. \\\\",
              "\\midrule"]
    for obj, d in [("rolled-out $V$", ggn), ("exact residual", ggn_res)]:
        for c in cfgs:
            su = seed_avg(d, lambda ps, c=c: ps[c]["10"]["success"])
            su_s = seed_std(d, lambda ps, c=c: ps[c]["10"]["success"])
            ti = seed_avg(d, lambda ps, c=c: ps[c]["10"]["true_incr_frac"])
            ti_s = seed_std(d, lambda ps, c=c: ps[c]["10"]["true_incr_frac"])
            td = seed_avg(d, lambda ps, c=c: ps[c]["10"]["mean_true_dec"])
            td_s = seed_std(d, lambda ps, c=c: ps[c]["10"]["mean_true_dec"])
            nm = xlab[cfgs.index(c)].replace("\n", " ")
            glines.append(f"{obj} & {nm} & ${su:.2f}\\pm{su_s:.2f}$ & "
                          f"${ti:.2f}\\pm{ti_s:.2f}$ & ${td:.2e}\\pm{td_s:.0e}$ \\\\")
        glines.append("\\midrule")
    glines[-1] = "\\bottomrule"
    glines.append("\\end{tabular}")
    (OUT / "table_two_cstr_ggn.tex").write_text("\n".join(glines))
    print("\nwrote table_two_cstr_ggn.tex")
    print("\n".join(glines))

    # ----- process-operation summary table (success; value-only vs value+grad) -----
    real = load("two_cstr_realism.json")
    track = load("two_cstr_tracking.json")
    nice = {"nominal": "nominal",
            "feed_temp_dist(+5K)": "feed-temp disturbance (+5\\,K)",
            "mismatch_k0(+5%)": "kinetic mismatch ($k_0$\\,+5\\%)",
            "mismatch_dH(+5%)": "enthalpy mismatch ($\\Delta H$\\,+5\\%)"}
    olines = ["\\begin{tabular}{lcc}", "\\toprule",
              "operating scenario & value-only & value+$L_{grad}$ \\\\", "\\midrule"]
    for s in real["config"]["scenarios"]:
        vo = seed_avg(real, lambda ps, s=s: ps["value_only"][s]["success"])
        vo_s = seed_std(real, lambda ps, s=s: ps["value_only"][s]["success"])
        vg = seed_avg(real, lambda ps, s=s: ps["value+grad"][s]["success"])
        vg_s = seed_std(real, lambda ps, s=s: ps["value+grad"][s]["success"])
        olines.append(f"{nice.get(s, s)} & ${vo:.2f}\\pm{vo_s:.2f}$ & ${vg:.2f}\\pm{vg_s:.2f}$ \\\\")
    vot = seed_avg(track, lambda ps: ps["value_only"]["track_success"])
    vot_s = seed_std(track, lambda ps: ps["value_only"]["track_success"])
    vgt = seed_avg(track, lambda ps: ps["value+grad"]["track_success"])
    vgt_s = seed_std(track, lambda ps: ps["value+grad"]["track_success"])
    olines.append(f"setpoint tracking & ${vot:.2f}\\pm{vot_s:.2f}$ & ${vgt:.2f}\\pm{vgt_s:.2f}$ \\\\")
    olines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_two_cstr_operation.tex").write_text("\n".join(olines))
    print("\nwrote table_two_cstr_operation.tex")
    print("\n".join(olines))


if __name__ == "__main__":
    main()
