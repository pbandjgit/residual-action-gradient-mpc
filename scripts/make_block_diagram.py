"""Paper-H overview block diagram (minimal): method pipeline + research logic.
Short labels only; details belong in the body text. Outputs results/figures/paper/fig_overview.*
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

PROB, METH, EVID = "#fbe5d6", "#dbe6d3", "#dde7f3"


def box(ax, cx, cy, w, h, text, fc, fs=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.04",
                 linewidth=1.2, edgecolor="#555555", facecolor=fc))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, weight=weight)


def arrow(ax, x1, cy1, x2, cy2):
    ax.add_patch(FancyArrowPatch((x1, cy1), (x2, cy2), arrowstyle="-|>",
                 mutation_scale=16, lw=1.6, color="#444444", shrinkA=1, shrinkB=1))


fig, (axa, axb) = plt.subplots(2, 1, figsize=(12.5, 6.0))
for ax in (axa, axb):
    ax.set_xlim(0, 100); ax.set_ylim(0, 24); ax.axis("off")

# ---- (a) method pipeline : 4 uniform boxes ----
axa.text(1, 22, "(a)  Method", fontsize=12, weight="bold")
ya, h, w = 12.5, 9.0, 20.5
cx = [12.5, 37, 62.5, 88]
labels_a = ["Plant data\n(+ heavy-tailed\nnoise)",
            "GroupSort-LCNN\nsurrogate $\\hat f$\nloss: data + $E_+$ + $L_{grad}$",
            "Finite-budget MPC\nfirst-order  |\nGauss–Newton",
            "True-plant\nclosed loop\n$V(x_t)\\rightarrow 0$"]
fcs_a = [PROB, METH, EVID, PROB]
for c, t, fc in zip(cx, labels_a, fcs_a):
    box(axa, c, ya, w, h, t, fc, fs=9.5)
for i in range(3):
    arrow(axa, cx[i]+w/2, ya, cx[i+1]-w/2, ya)
axa.text(50, 3.0, "$L_{grad}=\\;\\|\\,\\nabla_u V(\\hat f(x,u))-\\nabla_u V(f(x,u))\\,\\|^2$",
         ha="center", fontsize=10.5, style="italic",
         bbox=dict(boxstyle="round", fc="#fffbe6", ec="#cccccc"))

# ---- (b) research logic : 5 uniform boxes ----
axb.text(1, 22, "(b)  Logic", fontsize=12, weight="bold")
yb, hb, wb = 12.0, 9.5, 17.0
cxb = [10.5, 31, 51.5, 72, 90.5]
labels_b = ["Problem\nfinite-budget\nlearned LMPC",
            "Gap\nvalue consistency\n$\\neq$ gradient\nconsistency",
            "Method\n$L_{grad}$\n(fixed\narchitecture)",
            "Evidence\n2 solvers ×\n2 plants ×\n2 objectives",
            "Result\n(a) prediction +\n(b) gradient\nboth necessary"]
fcs_b = [PROB, EVID, METH, EVID, PROB]
for c, t, fc in zip(cxb, labels_b, fcs_b):
    box(axb, c, yb, wb, hb, t, fc, fs=9.0)
for i in range(4):
    arrow(axb, cxb[i]+wb/2, yb, cxb[i+1]-wb/2, yb)
axb.text(50, 3.2, "value_only fails everywhere  →  +$L_{grad}$ rescues (not via contractivity);   "
         "limitation: full sufficiency also needs prediction accuracy (axis a)",
         ha="center", fontsize=8.8, style="italic")

fig.tight_layout(h_pad=1.5)
fig.savefig(OUT / "fig_overview.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "fig_overview.pdf", bbox_inches="tight")
print("wrote", OUT / "fig_overview.png")
