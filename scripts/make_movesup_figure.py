"""Move-suppression trade-off figure from two_cstr_move_suppression.json.

Closed-loop success and input total variation (TV) versus the move-suppression
weight rho_move for the two-CSTR value+L_grad controller, with the auxiliary LQR
reference TV drawn as a horizontal line. Shows that a standard input-rate term
brings TV down toward the smooth-feedback reference over a wide interior range
while success stays high.

Output: results/figures/paper/fig_move_suppression.{png,pdf}
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
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3})

CVG = "#55a868"   # success
CB = "#4c72b0"    # TV


def main():
    d = json.loads((LOG / "two_cstr_move_suppression.json").read_text())
    rms = d["config"]["rho_move"]
    seeds = list(d["per_seed"].keys())
    aux_tv = d["config"]["aux_tv"]

    def ms(rm, k):
        v = [d["per_seed"][s][f"{rm:g}"][k] for s in seeds]
        return np.mean(v), np.std(v)
    succ = np.array([ms(rm, "success") for rm in rms])
    tv = np.array([ms(rm, "input_tv") for rm in rms])
    x = np.arange(len(rms))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    l1 = ax.errorbar(x, succ[:, 0], yerr=succ[:, 1], fmt="-o", color=CVG, lw=2,
                     capsize=3, label="closed-loop success")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("move-suppression weight  $\\rho_{\\Delta}$")
    ax.set_ylabel("closed-loop success", color=CVG)
    ax.tick_params(axis="y", labelcolor=CVG)
    ax.set_xticks(x); ax.set_xticklabels([f"{rm:g}" for rm in rms])

    ax2 = ax.twinx(); ax2.grid(False)
    l2 = ax2.errorbar(x, tv[:, 0], yerr=tv[:, 1], fmt="-s", color=CB, lw=2,
                      capsize=3, label="input total variation")
    l3 = ax2.axhline(aux_tv, color="black", ls="--", lw=1.5,
                     label=f"LQR reference TV = {aux_tv:.1f}")
    ax2.set_ylabel("input total variation", color=CB)
    ax2.tick_params(axis="y", labelcolor=CB)
    ax2.set_ylim(0, max(tv[:, 0]) * 1.15)

    lines = [l1, l2, l3]
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_move_suppression.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_move_suppression.pdf", bbox_inches="tight")
    print("wrote fig_move_suppression", flush=True)


if __name__ == "__main__":
    main()
