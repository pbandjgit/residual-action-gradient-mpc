"""Combined Jacobian-faithfulness figure: CSTR (top row) and two-CSTR (bottom row).

Reads cstr_jac_faithful.json and two_cstr_jac_faithful.json and draws, for each
plant, (left) surrogate state-Jacobian magnitude vs the true plant and (right) the
state-Jacobian error, for value-only vs value+grad. Error bars are seed s.d.,
with individual-seed dots.

Output: results/figures/paper/fig_jac.{png,pdf}
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
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

CV = "#c44e52"    # value_only
CVG = "#55a868"   # value+grad
JC = [CV, CVG]


def load(fn):
    return json.loads((LOG / fn).read_text())


def sa(d, get):
    s = list(d["per_seed"].keys()); return float(np.mean([get(d["per_seed"][k]) for k in s]))


def ss(d, get):
    s = list(d["per_seed"].keys()); return float(np.std([get(d["per_seed"][k]) for k in s]))


def sv(d, get):
    s = list(d["per_seed"].keys()); return [get(d["per_seed"][k]) for k in s]


def main():
    cstr = load("cstr_jac_faithful.json")
    two = load("two_cstr_jac_faithful.json")
    jc = ["value_only", "value+grad"]
    xlab = ["value\nonly", "value\n+$L_{grad}$"]

    fig, axs = plt.subplots(2, 2, figsize=(8.2, 6.8))
    rows = [("CSTR", cstr, "(a)", "(b)"), ("two CSTRs in series", two, "(c)", "(d)")]
    for r, (name, d, pa, pb) in enumerate(rows):
        jxm = [sa(d, lambda ps, c=c: ps[c]["Jx_model"]) for c in jc]
        jxs = [ss(d, lambda ps, c=c: ps[c]["Jx_model"]) for c in jc]
        jer = [sa(d, lambda ps, c=c: ps[c]["jac_err"]) for c in jc]
        jes = [ss(d, lambda ps, c=c: ps[c]["jac_err"]) for c in jc]
        jtrue = sa(d, lambda ps: ps["value_only"]["Jx_true"])
        # (left) magnitude vs true
        ax = axs[r, 0]
        ax.bar([0, 1], jxm, yerr=jxs, capsize=4, color=JC, edgecolor="black",
               width=0.6, error_kw=dict(lw=1.2))
        for j, c in enumerate(jc):
            vs = sv(d, lambda ps, c=c: ps[c]["Jx_model"])
            ax.scatter([j] * len(vs), vs, color="black", s=12, zorder=3, alpha=0.7)
        ax.axhline(jtrue, color="black", ls="--", lw=1.5,
                   label=f"true plant $J_x$ = {jtrue:.2f}")
        ax.set_xticks([0, 1]); ax.set_xticklabels(xlab)
        ax.set_ylabel("surrogate state-Jacobian $J_x$")
        ax.legend(fontsize=8.5, loc="upper right")
        ax.set_title(f"{pa} {name}: magnitude")
        # (right) error
        ax = axs[r, 1]
        ax.bar([0, 1], jer, yerr=jes, capsize=4, color=JC, edgecolor="black",
               width=0.6, error_kw=dict(lw=1.2))
        for j, c in enumerate(jc):
            vs = sv(d, lambda ps, c=c: ps[c]["jac_err"])
            ax.scatter([j] * len(vs), vs, color="black", s=12, zorder=3, alpha=0.7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(xlab)
        ax.set_ylabel("state-Jac. error $\\|J_{\\mathrm{model}}-J_{\\mathrm{true}}\\|$")
        ax.set_title(f"{pb} {name}: error")

    fig.tight_layout()
    fig.savefig(OUT / "fig_jac.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_jac.pdf", bbox_inches="tight")
    print("wrote fig_jac", flush=True)


if __name__ == "__main__":
    main()
