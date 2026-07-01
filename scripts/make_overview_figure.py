"""Study-overview schematic for the residual action-gradient paper.

The figure is intentionally sparse: it should tell the reader, at first glance,
where the proposed training target enters and why it matters for the online MPC
solver. Details belong in the caption and body text.

Output: results/figures/paper/fig_overview.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

CV = "#c44e52"    # value-only / wrong
CVG = "#55a868"   # value+grad / right
CBLUE = "#4c72b0"
INK = "#222222"
HL = "#f2c14e"    # highlight (L_grad)


def box(ax, x, y, w, h, fc, ec=INK, lw=1.4, rad=0.025, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={rad}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, mutation_aspect=0.6)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color=INK, lw=2.0, style="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16, lw=lw,
                        color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    return a


def main():
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Main boxes.
    box(ax, 0.035, 0.34, 0.20, 0.40, "#eef2f7")
    ax.text(0.135, 0.63, "Offline process\nsource", ha="center", va="center",
            fontsize=12, weight="bold", color=INK)
    ax.text(0.135, 0.48, "transition data\n+ local input perturbations",
            ha="center", va="center", fontsize=9.2, color="#444444")

    box(ax, 0.305, 0.25, 0.32, 0.58, "#f7f7f7")
    ax.text(0.465, 0.75, "Train fixed surrogate $\\hat f$", ha="center",
            va="center", fontsize=12, weight="bold", color=INK)
    ax.text(0.465, 0.64, "data fit  +  residual value $E_+$",
            ha="center", va="center", fontsize=10.2, color=INK)
    box(ax, 0.35, 0.42, 0.23, 0.13, HL, ec="#b8860b", lw=1.8)
    ax.text(0.465, 0.485, "$L_{\\mathrm{grad}}$: match $\\nabla_{\\!u}g$",
            ha="center", va="center", fontsize=11.0, weight="bold", color="#4b3a00")
    ax.text(0.465, 0.33, "same architecture; only the loss changes",
            ha="center", va="center", fontsize=9.0, color="#555555")

    box(ax, 0.71, 0.49, 0.23, 0.28, "#eaf3ee", ec=CVG, lw=1.8)
    ax.text(0.825, 0.69, "Finite-budget MPC", ha="center", va="center",
            fontsize=12, weight="bold", color=INK)
    ax.text(0.825, 0.58, "updates $u$ using $\\nabla_{\\!u}\\hat g$",
            ha="center", va="center", fontsize=10.5, weight="bold", color=CVG)

    box(ax, 0.71, 0.18, 0.23, 0.20, "#eef2f7")
    ax.text(0.825, 0.30, "Nonlinear process", ha="center", va="center",
            fontsize=11.5, weight="bold", color=INK)
    ax.text(0.825, 0.225, "CSTR and two-CSTRs", ha="center", va="center",
            fontsize=9.0, color="#444444")

    # Bottom design principle.
    box(ax, 0.095, 0.045, 0.77, 0.105, "#fff8e8", ec="#d7a62f", lw=1.5)
    ax.text(0.480, 0.098,
            "Design principle: match the derivative the online optimizer follows, not only the residual value.",
            ha="center", va="center", fontsize=10.7, weight="bold", color=INK)

    # Arrows and closed loop.
    arrow(ax, (0.238, 0.54), (0.303, 0.54))
    arrow(ax, (0.628, 0.625), (0.708, 0.625), color=CVG)
    arrow(ax, (0.805, 0.485), (0.805, 0.385), color=INK)
    ax.text(0.778, 0.435, "$u$", ha="center", va="center", fontsize=10, color=INK)
    arrow(ax, (0.845, 0.385), (0.845, 0.485), color=INK)
    ax.text(0.872, 0.435, "$x$", ha="center", va="center", fontsize=10, color=INK)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_overview.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig_overview.pdf", bbox_inches="tight")
    print("wrote fig_overview", flush=True)


if __name__ == "__main__":
    main()
