"""Combined region-of-attraction figure: single CSTR (top row) and two CSTRs in
series (bottom row), value-only vs value+grad, finite-budget first-order MPC.

The single-CSTR row is computed here on a full 2-state grid (concentration and
temperature deviation); the two-CSTR row reuses results/interim/logs/two_cstr_roa.json.
This is the empirical counterpart of the stabilizable set of Proposition 1 on both
processes.

Outputs: results/figures/paper/fig_roa.{png,pdf}, log cstr_roa.json
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
import budget_sweep_solver as bs  # noqa: E402
import probe_action_gradient as pr  # noqa: E402
import lgrad_experiment as Lx  # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"
OUT = ROOT / "results" / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": False})

X1 = np.linspace(-1.5, 1.5, 13)      # concentration deviation
X2 = np.linspace(-160.0, 160.0, 13)  # temperature deviation


def cstr_setup():
    d = np.load(ROOT / "data" / "processed" / "lcnn_paper_cstr_onestep_20k.npz")
    data = tuple(d[k].astype("float32") if d[k].dtype != np.int64 else d[k] for k in
                 ["XU", "Y", "x_mean", "x_std", "y_mean", "y_std"]) + (d["train_idx"], d["test_idx"])
    norm, un_lo, un_hi = Lx.make_norm(data)
    Lx.PHI, Lx.YPHI = pr.precompute_phi(bs.SIM, data[0][:, :2].astype("float64"))
    ug = Lx.precompute_true_ugrad(bs.SIM, data[0], data[2], data[3])
    return data, norm, un_lo, un_hi, ug


def cstr_roa(seq, norm, un_lo, un_hi):
    S = np.zeros((len(X2), len(X1)))
    for i, x2 in enumerate(X2):           # rows: temperature deviation (y-axis)
        for j, x1 in enumerate(X1):       # cols: concentration deviation (x-axis)
            x0 = np.array([x1, x2])
            S[i, j] = bs.closed_loop(seq, norm, x0, 3, 20, 120, 0.1, 0.01,
                                     2.0, un_lo, un_hi)["success"]
    return S


def main():
    arch = {"model": "lcnn", "hidden": 40, "lipschitz_bound": 4.0, "n_layers": 2}
    data, norm, un_lo, un_hi, ug = cstr_setup()
    cmaps = {}
    for tag, lv, lg in [("value_only", 0.003, 0.0), ("value+grad", 0.003, 0.05)]:
        m, _ = Lx.train(arch, 0, data, Lx.PHI, Lx.YPHI, ug, lam_val=lv, lam_grad=lg)
        S = cstr_roa(bs.freeze(m), norm, un_lo, un_hi)
        cmaps[tag] = S
        print(f"CSTR {tag}: recovered {100 * S.mean():.0f}% of the grid", flush=True)
    (LOG / "cstr_roa.json").write_text(json.dumps(
        {"x1": X1.tolist(), "x2": X2.tolist(), "budget": 20,
         "value_only": cmaps["value_only"].tolist(),
         "value+grad": cmaps["value+grad"].tolist()}, indent=2))

    two = json.loads((LOG / "two_cstr_roa.json").read_text())
    G = np.array(two["grid"])
    rows = [("CSTR", cmaps, [X1[0], X1[-1], X2[0], X2[-1]],
             "$x_1$ (concentration dev.)", "$x_2$ (temperature dev.)"),
            ("two CSTRs in series", {k: np.array(two[k]) for k in ("value_only", "value+grad")},
             [G[0], G[-1], G[0], G[-1]], "$T_1$ deviation [K]", "$T_2$ deviation [K]")]
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 7.0))
    for r, (name, mp, ext, xl, yl) in enumerate(rows):
        for c, (tag, lab) in enumerate([("value_only", "value-only"),
                                        ("value+grad", "value+$L_{grad}$")]):
            ax = axs[r, c]; S = mp[tag]
            ax.imshow(S, origin="lower", extent=ext, aspect="auto", cmap="RdYlGn",
                      vmin=0, vmax=1, alpha=0.9)
            ax.text(0.5, 0.05, f"recovered: {100 * S.mean():.0f}%", transform=ax.transAxes,
                    ha="center", fontsize=9, weight="bold",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_title(f"({'abcd'[r * 2 + c]}) {name}: {lab}")
    fig.tight_layout()
    fig.savefig(OUT / "fig_roa.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_roa.pdf", bbox_inches="tight")
    print("wrote fig_roa", flush=True)


if __name__ == "__main__":
    main()
