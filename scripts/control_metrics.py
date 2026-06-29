"""Control-venue quantitative evaluation:
  - baselines: auxiliary controller alone (LQR / saturated-Sontag), value_only-GGN,
    value+L_grad-GGN, and value+L_grad-GGN with an input-rate (Delta-u) penalty;
  - metrics: success rate, settling time, closed-loop state cost, input total
    variation (TV), input-constraint-active fraction;
  - figure fig_input_rate: the Delta-u penalty removes the budget-GGN force chatter.
CSTR and cart-pole, seed 0, GGN budget 10. Outputs:
  results/interim/logs/control_metrics.json  and  results/figures/paper/fig_input_rate.*
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts"))
import budget_sweep_solver as bs
import probe_action_gradient as pr
import lgrad_experiment as Lx
import cartpole_lgrad as C
LOG = ROOT/"results"/"interim"/"logs"; OUT = ROOT/"results"/"figures"/"paper"
OUT.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
CV, CVG, CB = "#c44e52", "#55a868", "#dd8452"
LT_CP = torch.tensor(np.linalg.cholesky(C.P_MAT).T, dtype=torch.float32)
LT_CS = torch.tensor(np.linalg.cholesky(bs.P).T, dtype=torch.float32)


def gn_opt(resfn, U, n, budget, lo, hi):
    lam = 1e-2
    for _ in range(budget):
        U = U.detach(); r = resfn(U)
        J = torch.autograd.functional.jacobian(resfn, U, vectorize=True, strategy="forward-mode")
        obj0 = 0.5*float(r @ r); g = J.t() @ r
        try:
            d = torch.linalg.solve(J.t() @ J + lam*torch.eye(n), -g)
        except RuntimeError:
            lam *= 10; continue
        acc = False
        for a in (1.0, 0.5, 0.25, 0.125):
            Ut = torch.clamp(U + a*d, lo, hi)
            if 0.5*float(resfn(Ut) @ resfn(Ut)) < obj0:
                U = Ut; acc = True; break
        lam = max(lam*0.5, 1e-4) if acc else min(lam*10, 1e4)
    return U


def metrics(Vtraj, Us, urange, lo_hi, thr, dt):
    V = np.asarray(Vtraj); U = np.asarray(Us)
    success = bool(V[-1] <= thr)
    below = np.where(V <= thr)[0]
    settle = float("nan")
    if below.size:                              # last time it is ABOVE thr, +1
        above = np.where(V > thr)[0]
        settle = (above[-1]+1)*dt if above.size else 0.0
        if V[-1] > thr:
            settle = float("nan")
    cost = float(np.sum(V)*dt)
    dU = np.diff(U, axis=0)
    tv = float(np.sum(np.abs(dU)/urange)) if U.ndim == 1 else float(np.sum(np.abs(dU)/urange))
    act = float(np.mean(np.any(np.abs(U) >= 0.99*np.asarray(lo_hi), axis=-1))) if U.ndim > 1 \
        else float(np.mean(np.abs(U) >= 0.99*lo_hi))
    return dict(success=success, settle_time=settle, cost=cost, input_tv=tv, constr_active=act)


# ----------------- cart-pole rollouts -----------------
def cp_models(seed=0):
    C.DEFAULT_HIDDEN = 48
    data = C.make_data(); XU, Y, xm, xs, ym, ys, _, _ = data
    norm = (torch.tensor(xm), torch.tensor(xs), torch.tensor(ym), torch.tensor(ys))
    Phi, Yphi = C.precompute_phi(XU); ug = C.precompute_true_ugrad(XU, xm, xs)
    mv, _ = C.train(seed, data, Phi, Yphi, ug, lam_val=0.003, lam_grad=0.0)
    mg, _ = C.train(seed, data, Phi, Yphi, ug, lam_val=0.003, lam_grad=0.05)
    return C.freeze(mv), C.freeze(mg), norm


def cp_rollout(seq, norm, x0, mode="ggn", H=10, budget=10, steps=180, rho_u=0.01, rho_du=0.0):
    xm, xs, ym, ys = norm; xm4, xs4 = xm[:4], xs[:4]
    x = np.asarray(x0, float); V = [float(C.Vnp(x))]; Us = []
    n = H; U = torch.zeros(n); srho, srdu = np.sqrt(rho_u), np.sqrt(rho_du)
    lo = torch.tensor((-C.FLIM-xm[4:].numpy())/xs[4:].numpy(), dtype=torch.float32).repeat(H)
    hi = torch.tensor((C.FLIM-xm[4:].numpy())/xs[4:].numpy(), dtype=torch.float32).repeat(H)
    uprev_n = float((0.0 - xm[4].item())/xs[4].item())
    for _ in range(steps):
        if mode == "aux":
            u0 = float(C.phi(x)[0])
        else:
            x0n = torch.tensor((x-xm4.numpy())/xs4.numpy(), dtype=torch.float32).view(1, 4)
            up = torch.tensor([uprev_n], dtype=torch.float32)
            def resfn(Uf):
                un = Uf.view(H, 1); xt = x0n; prev = up; parts = []
                for h in range(H):
                    y = seq(torch.cat([xt, un[h:h+1]], 1))*ys+ym
                    parts += [LT_CP @ y.view(4), srho*un[h]]
                    if srdu > 0: parts.append(srdu*(un[h]-prev))
                    prev = un[h]; xt = (y-xm4)/xs4
                return torch.cat(parts)
            U = gn_opt(resfn, U, n, budget, lo, hi)
            u0 = float(np.clip(U.view(H, 1)[0].numpy()*xs[4:].numpy()+xm[4:].numpy(), -C.FLIM, C.FLIM)[0])
            U = torch.cat([U.view(H, 1)[1:], U.view(H, 1)[-1:]]).reshape(-1)
        uprev_n = float((u0-xm[4].item())/xs[4].item())
        x = C.step(x, u0); V.append(float(C.Vnp(x))); Us.append(u0)
    return np.array(V), np.array(Us)


# ----------------- CSTR rollouts -----------------
def cstr_models(seed=0):
    d = np.load(ROOT/"data"/"processed"/"lcnn_paper_cstr_onestep_20k.npz")
    data = tuple(d[k].astype("float32") if d[k].dtype != np.int64 else d[k] for k in
                 ["XU", "Y", "x_mean", "x_std", "y_mean", "y_std"]) + (d["train_idx"], d["test_idx"])
    norm, _, _ = Lx.make_norm(data)
    Lx.PHI, Lx.YPHI = pr.precompute_phi(bs.SIM, data[0][:, :2].astype("float64"))
    ug = Lx.precompute_true_ugrad(bs.SIM, data[0], data[2], data[3])
    arch = {"model": "lcnn", "hidden": 40, "lipschitz_bound": 4.0, "n_layers": 2}
    mv, _ = Lx.train(arch, seed, data, Lx.PHI, Lx.YPHI, ug, lam_val=0.003, lam_grad=0.0)
    mg, _ = Lx.train(arch, seed, data, Lx.PHI, Lx.YPHI, ug, lam_val=0.003, lam_grad=0.05)
    return bs.freeze(mv), bs.freeze(mg), norm


def cstr_rollout(seq, norm, x0, mode="ggn", H=3, budget=10, steps=120, rho_u=0.01, rho_du=0.0):
    from cstr.sontag import saturated_sontag_np
    xm, xs, ym, ys = norm; xm2, xs2 = xm[:2], xs[:2]
    x = np.asarray(x0, float); V = [float(x @ bs.P @ x)]; Us = []
    n = H*2; U = torch.zeros(n); srho, srdu = np.sqrt(rho_u), np.sqrt(rho_du)
    lo = torch.tensor((bs.INPUT_LO-xm[2:].numpy())/xs[2:].numpy(), dtype=torch.float32).repeat(H)
    hi = torch.tensor((bs.INPUT_HI-xm[2:].numpy())/xs[2:].numpy(), dtype=torch.float32).repeat(H)
    uprev_n = (np.zeros(2)-xm[2:].numpy())/xs[2:].numpy()
    for _ in range(steps):
        if mode == "aux":
            u0 = saturated_sontag_np(bs.SIM, x, bs.INPUT_LO, bs.INPUT_HI, bs.P)
        else:
            x0n = torch.tensor((x-xm2.numpy())/xs2.numpy(), dtype=torch.float32).view(1, 2)
            up = torch.tensor(uprev_n, dtype=torch.float32)
            def resfn(Uf):
                un = Uf.view(H, 2); xt = x0n; prev = up; parts = []
                for h in range(H):
                    y = seq(torch.cat([xt, un[h:h+1]], 1))*ys+ym
                    parts += [LT_CS @ y.view(2), srho*un[h]]
                    if srdu > 0: parts.append(srdu*(un[h]-prev))
                    prev = un[h]; xt = (y-xm2)/xs2
                return torch.cat(parts)
            U = gn_opt(resfn, U, n, budget, lo, hi)
            u0 = np.clip(U.view(H, 2)[0].numpy()*xs[2:].numpy()+xm[2:].numpy(), bs.INPUT_LO, bs.INPUT_HI)
            U = torch.cat([U.view(H, 2)[1:], U.view(H, 2)[-1:]]).reshape(-1)
        uprev_n = (u0-xm[2:].numpy())/xs[2:].numpy()
        x = bs.SIM.step(x, u0); V.append(float(x @ bs.P @ x)); Us.append(u0)
    return np.array(V), np.array(Us)


def main():
    out = {}
    print("CSTR models ..."); cv, cg, cn = cstr_models()
    print("cart-pole models ..."); pv, pg, pn = cp_models()
    cstr_methods = [("aux (Sontag)", cv, "aux", 0.0), ("value_only GGN", cv, "ggn", 0.0),
                    ("value+Lgrad GGN", cg, "ggn", 0.0), ("value+Lgrad GGN +Δu", cg, "ggn", 0.05)]
    cp_methods = [("aux (LQR)", pv, "aux", 0.0), ("value_only GGN", pv, "ggn", 0.0),
                  ("value+Lgrad GGN", pg, "ggn", 0.0), ("value+Lgrad GGN +Δu", pg, "ggn", 0.05)]

    def run(plant, methods, roll, norm, ICs, urange, lohi, thr, dt):
        rows = {}
        for name, seq, mode, rdu in methods:
            ms = []
            for ic in ICs:
                V, U = roll(seq, norm, ic, mode=mode, rho_du=rdu)
                ms.append(metrics(V, U, urange, lohi, thr, dt))
            agg = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0] if k != "success"}
            agg["success"] = float(np.mean([m["success"] for m in ms]))
            rows[name] = agg
            print(f"  [{plant}] {name:22} succ={agg['success']:.2f} settle={agg['settle_time']:.2f} "
                  f"cost={agg['cost']:.1f} TV={agg['input_tv']:.2f} constr={agg['constr_active']:.2f}")
        return rows

    print("CSTR metrics ...")
    out["cstr"] = run("CSTR", cstr_methods, cstr_rollout, cn, bs.ICS,
                      bs.INPUT_HI-bs.INPUT_LO, bs.INPUT_HI, 2.0, 1.0)
    print("cart-pole metrics ...")
    out["cartpole"] = run("cart-pole", cp_methods, cp_rollout, pn, C.ICS,
                          2*C.FLIM, C.FLIM, 10.0, C.CPP.dt)
    (LOG/"control_metrics.json").write_text(json.dumps(out, indent=2))
    print("wrote", LOG/"control_metrics.json")

    # markdown table
    def md(plant, rows):
        lines = [f"\n### {plant}",
                 "| method | success | settling [s] | state cost | input TV | constr. active |",
                 "|---|---:|---:|---:|---:|---:|"]
        for n, a in rows.items():
            st = "—" if np.isnan(a["settle_time"]) else f"{a['settle_time']:.2f}"
            lines.append(f"| {n} | {a['success']:.2f} | {st} | {a['cost']:.1f} | {a['input_tv']:.2f} | {a['constr_active']:.2f} |")
        return "\n".join(lines)
    table = md("CSTR", out["cstr"]) + "\n" + md("cart-pole", out["cartpole"])
    (ROOT/"docs"/"CONTROL_METRICS_TABLE_2026_06_26.md").write_text(
        "# Control metrics (GGN budget 10, seed 0, 5 ICs)\n" + table + "\n")
    print(table)

    # ---- fig_input_rate: cart-pole force with vs without Delta-u penalty ----
    ic = C.ICS[4]
    Vn, Un = cp_rollout(pg, pn, ic, mode="ggn", rho_du=0.0)
    Vr, Ur = cp_rollout(pg, pn, ic, mode="ggn", rho_du=0.05)
    t = np.arange(len(Un))*C.CPP.dt
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(t, Un, color="#bbbbbb", lw=1.0, label="value+$L_{grad}$ (no $\\Delta u$ penalty)")
    ax.plot(t, Ur, color=CVG, lw=1.7, label="value+$L_{grad}$ (+ $\\Delta u$ penalty)")
    ax.axhline(C.FLIM, color="r", ls="--", lw=1, alpha=0.6); ax.axhline(-C.FLIM, color="r", ls="--", lw=1, alpha=0.6)
    tv0 = metrics(Vn, Un, 2*C.FLIM, C.FLIM, 10, C.CPP.dt)["input_tv"]
    tv1 = metrics(Vr, Ur, 2*C.FLIM, C.FLIM, 10, C.CPP.dt)["input_tv"]
    ax.set_title(f"cart-pole control force: input-rate penalty cuts chatter\ninput TV {tv0:.1f} $\\to$ {tv1:.1f}")
    ax.set_xlabel("time [s]"); ax.set_ylabel("force [N]"); ax.set_ylim(-12.5, 13.5); ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT/"fig_input_rate.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT/"fig_input_rate.pdf", bbox_inches="tight")
    print("wrote fig_input_rate")


if __name__ == "__main__":
    main()
