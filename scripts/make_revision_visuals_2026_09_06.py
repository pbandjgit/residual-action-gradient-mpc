"""Read-only evidence extraction and reproducible manuscript plots; no solves."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cstr.ablation_io import read_and_verify_json_artifact

OUT = ROOT / "manuscript_jpc/figures/revision_2026_09_06"
AUD = ROOT / "results/solver_audit_execution_2026_09_01"
ABL = ROOT / "results/ablation_execution_2026_08_31"
ORA = ROOT / "results/oracle_nlp_execution_2026_09_03_corrective"
SOURCES = {}
COLORS = {"B": "#0072B2", "E": "#D55E00"}
LABELS = {"B": "Value-only training", "E": "Value + residual action-gradient training"}
STYLES = {"B": "-", "E": "--"}
plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.labelsize": 9, "axes.titlesize": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path)
    _, data = read_and_verify_json_artifact(path)
    SOURCES[str(path)] = sha(path)
    return data


def companion(path, expected):
    path = Path(path)
    assert sha(path) == expected, path
    SOURCES[str(path)] = expected
    return np.load(path, allow_pickle=False)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def overview_plot():
    fig, ax = plt.subplots(figsize=(7.05, 2.05))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    def box(x, y, w, h, label, color=".35", fill=".97"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                   edgecolor=color, facecolor=fill, lw=1))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=8.5)
    def arrow(a, b, label=None):
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": 1, "color": ".3"})
        if label:
            ax.text((a[0]+b[0])/2+.018, (a[1]+b[1])/2, label, fontsize=8,
                    ha="left", va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": .4})
    box(.02, .35, .19, .34, "Offline process source\n\nShared transition and\nperturbation queries")
    box(.30, .60, .29, .23, "Value-only training\nPerturbations as transitions", COLORS["B"])
    box(.30, .19, .29, .23, "Value + residual action-gradient\ntraining", COLORS["E"])
    arrow((.22, .60), (.285, .71))
    arrow((.22, .43), (.285, .31))
    box(.72, .60, .25, .23, "Fixed-budget learned MPC\nSurrogate rollout objective")
    box(.72, .12, .25, .20, "True process\nApplied input / observed state")
    arrow((.605, .71), (.705, .71))
    arrow((.605, .31), (.705, .63))
    arrow((.80, .58), (.80, .34), "$u_0$")
    arrow((.91, .34), (.91, .58), "$x$")
    ax.text(.45, .99, "Primary comparison: equal query information, architecture and online budget",
            ha="center", va="top", fontsize=8.5)
    ax.text(.44, .035, "Offline training", ha="center", fontsize=8, color=".35")
    ax.text(.85, .035, "Online feedback", ha="center", fontsize=8, color=".35")
    fig.subplots_adjust(left=.01, right=.99, top=.97, bottom=.03)
    save(fig, "method_overview")


def paired_plot():
    records = [record(ABL / f"ablation_training_manifest_seed{s}_2026_08_31.json")
               for s in range(10)]
    specs = [("test MSE", lambda r, c: r["test_mse"][c] * 1000,
              r"Test MSE ($\times10^{-3}$)"),
             ("offline gradient", lambda r, c: r["oracle_summary"][c]["align_cos"]["frac_neg"],
              "Negative-cosine fraction"),
             ("recovery", lambda r, c: r["closed_loop_success"][c], "Recovery fraction")]
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.35))
    plotdata = {}
    for i, (ax, (name, extract, ylabel)) in enumerate(zip(axes, specs)):
        values = np.array([[extract(r, c) for c in ("B", "E")] for r in records])
        plotdata[name] = values.tolist()
        offsets = np.linspace(-0.06, 0.06, 10)
        for seed, row in enumerate(values):
            xx = np.array([0., 1.]) + offsets[seed]
            ax.plot(xx, row, color="0.65", lw=.65, alpha=.7, zorder=1)
            for j, c in enumerate(("B", "E")):
                ax.scatter(xx[j], row[j], s=16, color=COLORS[c],
                           marker="o" if c == "B" else "s", clip_on=False, zorder=3)
        for j, c in enumerate(("B", "E")):
            ax.plot([j-.13, j+.13], [values[:, j].mean()]*2, color="black", lw=2, zorder=4)
        ax.set(xlim=(-.3, 1.3), xticks=[0, 1],
               xticklabels=["Value-only\ntraining", "Value + residual\naction-gradient\ntraining"], ylabel=ylabel)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylim(0, 1 if i == 2 else values.max()*1.12)
        ax.set_title(f"({chr(97+i)}) {name}", loc="left")
        ax.grid(axis="y", alpha=.18)
    fig.subplots_adjust(left=.085, right=.99, bottom=.29, top=.86, wspace=.58)
    save(fig, "paired_evidence")
    return plotdata


def recovery_plot():
    traces = {c: np.empty((10, 5, 121)) for c in ("B", "E")}
    settling = {c: np.empty((10, 5), dtype=int) for c in ("B", "E")}
    tv = {c: [] for c in ("B", "E")}
    for c in traces:
        for seed in range(10):
            r = record(AUD / f"solver_audit_battery2_{c}_seed{seed}_2026_09_01.json")
            assert r["seed"] == seed and r["condition"] == c
            with companion(r["iteration_arrays_path"], r["iteration_arrays_sha256"]) as z:
                per_tv = []
                for ic, entry in enumerate(r["per_ic_summary"]):
                    assert entry["initial_condition_index"] == ic
                    summ = entry["summary"]
                    assert summ["solver_completed"] and summ["n_steps_completed"] == 120
                    before = z[f"ic{ic}_perstep_V_before"]
                    after = z[f"ic{ic}_perstep_V_after_realized"]
                    assert before.shape == after.shape == (120,)
                    assert np.allclose(before[1:], after[:-1], rtol=1e-10, atol=1e-10)
                    v = np.r_[before[0], after]
                    assert np.isfinite(v).all() and (v > 0).all()
                    assert np.isclose(v[-1], summ["final_V"])
                    assert np.isclose(v.max(), summ["max_V"])
                    t = next((i for i in range(121) if np.all(v[i:] <= 2)), 121)
                    assert t == summ["settling_time"], (c, seed, ic, t)
                    traces[c][seed, ic] = v
                    settling[c][seed, ic] = t
                    per_tv.append(summ["input_tv_per_actuator_normalized"])
                tv[c].append(np.mean(per_tv, axis=0).tolist())
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.0))
    low = min(t.min() for t in traces.values()) * .8
    high = max(t.max() for t in traces.values()) * 1.3
    for ic, ax in enumerate(axes.flat):
        if ic == 5:
            for c in traces:
                curves = np.mean(settling[c][:, :, None] <= np.arange(121), axis=1)
                for curve in curves:
                    ax.step(np.arange(121), curve, where="post", color=COLORS[c], alpha=.16, lw=.6)
                ax.step(np.arange(121), curves.mean(axis=0), where="post",
                        color=COLORS[c], ls=STYLES[c], lw=1.7)
            ax.set(ylim=(0, 1.04), ylabel="Fraction recovered and staying")
            ax.set_title("(f) All five initial conditions", loc="left")
        else:
            for c in traces:
                for v in traces[c][:, ic]:
                    ax.plot(v, color=COLORS[c], ls=STYLES[c], alpha=.22, lw=.55)
                ax.plot(np.median(traces[c][:, ic], axis=0), color=COLORS[c], ls=STYLES[c], lw=1.6)
            ax.set(yscale="log", ylim=(low, high), ylabel="True $V(x_t)$")
            ax.axhline(2, color="0.3", ls=":", lw=.8)
            ax.set_title(f"({chr(97+ic)}) Initial condition {ic+1}", loc="left")
        ax.set(xlim=(0, 120), xticks=[0, 60, 120], xlabel="MPC step")
        ax.grid(alpha=.12)
    fig.legend(handles=[Line2D([], [], color=COLORS[c], ls=STYLES[c], label=LABELS[c]) for c in traces]
               + [Line2D([], [], color=".3", ls=":", label="$V=2$")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.015))
    fig.subplots_adjust(left=.09, right=.99, bottom=.1, top=.87, wspace=.55, hspace=.6)
    save(fig, "recovery_dynamics")
    return {"V": {c: a.tolist() for c, a in traces.items()},
            "settling": {c: a.tolist() for c, a in settling.items()}, "normalized_tv_per_seed": tv}


def oracle_plot(recovery):
    r = record(ORA / "oracle_nlp_summary_manifest_2026_09_03.json")
    assert r["decomposition_identity_passed"] and r["outcome"] == "REPORTED"
    with companion(r["resample_arrays_path"], r["resample_arrays_sha256"]):
        pass
    oracle, nlp = [], []
    for ic in range(5):
        for kind, target in [("oracle20", oracle), ("certnlp", nlp)]:
            rec = record(ORA / f"oracle_nlp_{kind}_ic{ic}_2026_09_03.json")
            assert rec["summary"]["reference_valid"]
            target.append(rec["summary"]["final_V"])
    optimizer = np.array(oracle) - np.array(nlp)
    assert np.isclose(optimizer.mean(), r["optimizer_gap"]["point_estimate"])
    for c in ("B", "E"):
        actual = np.array(recovery["V"][c])[:, :, -1].mean(axis=1) - np.mean(oracle)
        assert np.allclose(actual, r[f"gap_{c}"]["per_seed"], rtol=1e-5, atol=1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.1), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    for i, (ax, conditions) in enumerate(zip(axes[:2], [("B", "E"), ("E",)])):
        for y, c in enumerate(conditions):
            gap = r[f"gap_{c}"]
            raw = np.array(gap["per_seed"])
            assert np.isclose(raw.mean(), gap["point_estimate"])
            ax.scatter(raw, y+np.linspace(-.12, .12, 10), s=12, color=COLORS[c], alpha=.6)
            ax.plot([gap["ci_lower"], gap["ci_upper"]], [y, y], color="black", lw=1.3)
            ax.plot(gap["point_estimate"], y, "D", color=COLORS[c], ms=5)
        ax.set(yticks=[], ylim=(-.45, len(conditions)-.55))
        ax.invert_yaxis()
        ax.set_xlabel("Final-$V$ gap to TM-20")
        ax.set_xlim(-.03 if i else -12, .55 if i else 260)
        ax.axvline(0, color=".6", lw=.7)
        ax.grid(axis="x", alpha=.15)
    axes[0].set_title("(a) Full scale", loc="left")
    axes[1].set_title("(b) Value + residual-gradient: zoom", loc="left", fontsize=8)
    ax = axes[2]
    ax.scatter(optimizer * 1000, np.arange(1, 6), color=".3", s=18)
    ax.axvline(optimizer.mean() * 1000, color=".3", ls="--", lw=1)
    ax.set(yticks=range(1, 6), ylabel="Initial condition",
           xlabel="TM-20 minus NLP\n" + r"(final $V$, $\times10^{-3}$)", xlim=(-.03, 1.2))
    ax.set_title("(c) Fixed-IC reference gap", loc="left")
    ax.grid(axis="x", alpha=.15)
    fig.legend(handles=[Line2D([], [], color=COLORS[c], marker="o", ls="", label=LABELS[c])
                        for c in ("B", "E")], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, 1.08), fontsize=8)
    fig.subplots_adjust(left=.035, right=.99, bottom=.32, top=.76, wspace=.48)
    save(fig, "oracle_reference")
    return {"gap_B": r["gap_B"], "gap_E": r["gap_E"], "optimizer_per_ic": optimizer.tolist()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    overview_plot()
    data = {"paired": paired_plot(), "recovery": recovery_plot()}
    data["oracle"] = oracle_plot(data["recovery"])
    assert all(sha(p) == h for p, h in SOURCES.items())
    table = OUT / "plot_data.json"
    table.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    manifest = {"purpose": "read-only figure production, no new inference",
                "source_file_sha256": SOURCES, "script_sha256": sha(__file__),
                "plot_data_sha256": sha(table),
                "figure_sha256": {p.name: sha(p) for p in OUT.glob("*.pdf")},
                "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__}
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"source_count": len(SOURCES), "normalized_tv_per_seed": data["recovery"]["normalized_tv_per_seed"]}, indent=2))


if __name__ == "__main__":
    main()
