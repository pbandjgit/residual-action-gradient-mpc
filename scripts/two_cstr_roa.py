"""Two-CSTR region-of-attraction estimate (empirical stabilizable set).

Grids the initial reactor-temperature deviations $(T_1,T_2)$ (with concentrations
at the steady state) and records whether the finite-budget first-order MPC recovers
the process, for value-only vs value+grad. This is the empirical counterpart of the
stabilizable set in Proposition 1: it is much larger with L_grad.

Outputs: results/figures/paper/fig_two_cstr_roa.{png,pdf}, log two_cstr_roa.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import two_cstr_lgrad as T  # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"
OUT = ROOT / "results" / "figures" / "paper"
GRID = np.linspace(-60.0, 60.0, 9)   # T1_0, T2_0 deviations


def main():
    data = T.make_data(n=12000)
    XU, xm, xs = data[0], data[2], data[3]
    norm = T.make_norm(data)
    un_lo = torch.tensor((T.BOX_LO - xm[4:]) / xs[4:], dtype=torch.float32)
    un_hi = torch.tensor((T.BOX_HI - xm[4:]) / xs[4:], dtype=torch.float32)
    print("precomputing ...", flush=True)
    Phi, Yphi = T.precompute_phi(XU)
    true_ug = T.precompute_true_ugrad(XU, xm, xs)
    hz, steps, lr, rho_u, sV, B = 3, 120, 0.2, 0.01, 2.0, 20

    def roa(seq):
        S = np.zeros((len(GRID), len(GRID)))
        for i, t2 in enumerate(GRID):           # rows: T2 (y-axis)
            for j, t1 in enumerate(GRID):       # cols: T1 (x-axis)
                x0 = np.array([0.0, t1, 0.0, t2])
                S[i, j] = T.closed_loop(seq, norm, x0, hz, B, steps, lr, rho_u,
                                        sV, un_lo, un_hi)["success"]
        return S

    maps = {}
    for tag, lv, lg in [("value_only", 0.003, 0.0), ("value+grad", 0.003, 0.2)]:
        model, _ = T.train(0, data, Phi, Yphi, true_ug, lam_val=lv, lam_grad=lg, epochs=70)
        seq = T.freeze(model)
        S = roa(seq); maps[tag] = S
        print(f"{tag}: recovered {100*S.mean():.0f}% of the grid", flush=True)

    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "two_cstr_roa.json").write_text(json.dumps(
        {"grid": GRID.tolist(), "budget": B,
         "value_only": maps["value_only"].tolist(),
         "value+grad": maps["value+grad"].tolist()}, indent=2))

    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, tag, lab in [(axs[0], "value_only", "(a) value-only"),
                         (axs[1], "value+grad", "(b) value+$L_{grad}$")]:
        S = maps[tag]
        ax.imshow(S, origin="lower", extent=[GRID[0], GRID[-1], GRID[0], GRID[-1]],
                  aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, alpha=0.85)
        ax.text(0.5, 0.04, f"recovered: {100*S.mean():.0f}%", transform=ax.transAxes,
                ha="center", fontsize=9, weight="bold",
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        ax.set_xlabel("$T_1$ deviation [K]"); ax.set_ylabel("$T_2$ deviation [K]")
        ax.set_title(lab)
    fig.tight_layout()
    fig.savefig(OUT / "fig_two_cstr_roa.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_two_cstr_roa.pdf", bbox_inches="tight")
    print("wrote fig_two_cstr_roa", flush=True)


if __name__ == "__main__":
    main()
