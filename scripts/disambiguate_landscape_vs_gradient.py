"""Paper-H disambiguation: WHY does cheap-budget MPC fail on GroupSort-LCNN but
succeed on the smooth-spectral model? Reproducible script for the checks that were
previously run inline (Caveat 3), plus the contractivity control (Caveat 4) and
multi-seed aggregation (Caveat 5).

Candidate mechanisms separated here:
  (M1) model exploitation / spurious value-surface minima (Paper G):
       near-oracle optimizer would also be captured; large model-vs-plant V gap.
  (M2) action-gradient-specific failure (sharp Paper H thesis):
       derivative-free (CEM) solver would rescue LCNN.
  (M3) optimization-landscape regularity, decoupled from value accuracy:
       LCNN value-usable (near-oracle recovers) but hard to optimize cheaply;
       smooth easy at every budget.

Contractivity control (M3 vs trivial artifact): tanh is gradient-shrinking, so the
smooth model could win merely by predicting more contractive dynamics, not by
better landscape geometry. We compare input/state Jacobian-norm distributions, the
one-step residual value error / E+, and the model's own self-rollout contraction;
and we re-run the key budget comparison with a DIFFERENT smooth activation
(softsign) to check the advantage is not tanh-specific shrinkage.

Outputs: results/interim/logs/disambiguation_landscape_vs_gradient.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import budget_sweep_solver as bs                                      # noqa: E402
import probe_action_gradient as pr                                   # noqa: E402
from cstr.models_onestep import build_onestep                        # noqa: E402

P = bs.P; Pt = bs.Pt; SIM = bs.SIM
INPUT_LO = bs.INPUT_LO; INPUT_HI = bs.INPUT_HI; ICS = bs.ICS
LOG = ROOT / "results" / "interim" / "logs"
SEEDS = [0, 1, 2]


def make_norm(data):
    xm, xs, ym, ys = [torch.tensor(data[i]) for i in (2, 3, 4, 5)]
    un_lo = torch.tensor((INPUT_LO - data[2][2:]) / data[3][2:], dtype=torch.float32)
    un_hi = torch.tensor((INPUT_HI - data[2][2:]) / data[3][2:], dtype=torch.float32)
    return (xm, xs, ym, ys), un_lo, un_hi


def train_kind(kind, seed, data, Phi, Yphi):
    """kind in {lcnn, smooth(tanh), smooth_softsign}. Returns frozen seq."""
    if kind == "smooth_softsign":
        # build the smooth-spectral model directly with softsign activation
        torch.manual_seed(seed)
        cfg = {"model": "smooth", "hidden": 40, "lipschitz_bound": 4.0,
               "n_layers": 2, "activation": "softsign"}
        model = _train_with_cfg(cfg, seed, data, Phi, Yphi)
    else:
        model, _ = pr.train(kind, seed, data, SIM, Phi, Yphi)
    return bs.freeze(model)


def _train_with_cfg(cfg, seed, data, Phi_all, Yphi_all,
                    epochs=50, lam=0.003, warmup=30, noise_scale=0.2, noise_clip=5.0):
    """Same training loop as probe_action_gradient.train but for an explicit cfg."""
    XU, Y, xm, xs, ym, ys, train_idx, _ = data
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 1000)
    xm_t, xs_t = torch.tensor(xm), torch.tensor(xs)
    ym_t, ys_t = torch.tensor(ym), torch.tensor(ys)
    Xn = torch.tensor((XU - xm) / xs)
    Yn = ((Y - ym) / ys).copy()
    n = noise_scale * rng.standard_cauchy(size=Yn[train_idx].shape)
    Yn[train_idx] += np.clip(n, -noise_clip, noise_clip).astype(np.float32)
    Yn_t = torch.tensor(Yn)
    Phi_t = torch.tensor(Phi_all)
    g_true_all = pr.Vt(torch.tensor(Y)) - pr.Vt(torch.tensor(Yphi_all))
    g_scale = torch.clamp(g_true_all[train_idx].std(), min=torch.tensor(1.0))
    disp = torch.tensor(np.median(np.abs(Yn[train_idx] - np.median(Yn[train_idx], 0)), 0)
                        * 1.4826, dtype=torch.float32)
    model = build_onestep(cfg)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    tr = np.asarray(train_idx)
    for ep in range(1, epochs + 1):
        model.train(); order = rng.permutation(len(tr))
        for k in range(0, len(tr), 256):
            b = tr[order[k:k + 256]]
            opt.zero_grad()
            pred = model(Xn[b])
            loss = pr.cauchy_nll(pred, Yn_t[b], disp)
            if lam > 0 and ep > warmup:
                y_hat = pred * ys_t + ym_t
                xphi = torch.cat([torch.tensor(XU[b, :2]), Phi_t[b]], 1)
                yphi_hat = model((xphi - xm_t) / xs_t) * ys_t + ym_t
                g_hat = pr.Vt(y_hat) - pr.Vt(yphi_hat)
                loss = loss + lam * torch.relu((g_true_all[b] - g_hat) / g_scale).pow(2).mean()
            loss.backward(); opt.step()
            if hasattr(model, "apply_constraints"):
                model.apply_constraints()
    model.eval()
    return model


def cem(seq, norm, un_lo, un_hi, x0, horizon=3, n=96, iters=6, elite=12,
        steps=120, rho_u=0.01, report_exploit=False):
    xm, xs, ym, ys = norm; xm2, xs2 = xm[:2], xs[:2]
    un_lo = un_lo.numpy(); un_hi = un_hi.numpy()
    x = np.asarray(x0, float); Vtraj = [float(x @ P @ x)]; expl = []
    with torch.no_grad():
        for _ in range(steps):
            mu = np.zeros((horizon, 2), "float32"); sig = np.ones((horizon, 2), "float32") * 0.8
            x0n = torch.tensor((x - xm2.numpy()) / xs2.numpy(), dtype=torch.float32).view(1, 2)
            for _it in range(iters):
                samp = np.clip(mu[None] + sig[None] * np.random.randn(n, horizon, 2).astype("float32"),
                               un_lo, un_hi)
                U = torch.tensor(samp); xt = x0n.repeat(n, 1); J = torch.zeros(n)
                for h in range(horizon):
                    xun = torch.cat([xt, U[:, h]], 1); y = seq(xun) * ys + ym
                    J = J + (y @ Pt * y).sum(1) + rho_u * (U[:, h] ** 2).sum(1); xt = (y - xm2) / xs2
                order = torch.argsort(J)[:elite].numpy(); el = samp[order]
                mu = el.mean(0); sig = el.std(0) + 1e-3
            u0 = np.clip(mu[0] * xs[2:].numpy() + xm[2:].numpy(), INPUT_LO, INPUT_HI)
            xun = torch.cat([x0n, torch.tensor(mu[0][None], dtype=torch.float32)], 1)
            ypred = (seq(xun) * ys + ym).numpy()[0]; Vpred = float(ypred @ P @ ypred)
            xnext = SIM.step(x, u0); Vtrue = float(xnext @ P @ xnext)
            expl.append((Vpred, Vtrue)); x = xnext; Vtraj.append(Vtrue)
    out = dict(final_V=Vtraj[-1], success=bool(Vtraj[-1] <= 2.0))
    if report_exploit:
        e = np.array(expl); out["mean_Vpred"] = float(e[:, 0].mean()); out["mean_Vtrue"] = float(e[:, 1].mean())
    return out


def succ_rate(fn):
    rows = [fn(x0) for x0 in ICS]
    return float(np.mean([r["success"] for r in rows])), \
        float(np.percentile([r["final_V"] for r in rows], 90)), rows


def jacobian_stats(seq, data, norm, n_pts=400, seed=0):
    """Distribution of ||d f_hat/d x|| and ||d f_hat/d u|| (physical) over test pts.
    Contractivity control: a much smaller state-Jacobian for smooth would mean it
    predicts more contractive dynamics (could explain easier optimization trivially)."""
    XU, Y, xm, xs, ym, ys, _, test_idx = data
    rng = np.random.default_rng(seed)
    idx = rng.choice(test_idx, size=min(n_pts, len(test_idx)), replace=False)
    xm_t, xs_t = torch.tensor(xm), torch.tensor(xs)
    ym_t, ys_t = torch.tensor(ym), torch.tensor(ys)
    jx, ju = [], []
    for i in idx:
        xun = torch.tensor((XU[i] - xm) / xs, dtype=torch.float32, requires_grad=True)
        y = seq(xun.view(1, 4)).view(2) * ys_t + ym_t   # physical next state
        Jrows = []
        for k in range(2):
            g, = torch.autograd.grad(y[k], xun, retain_graph=(k == 0), create_graph=False)
            Jrows.append(g.detach())
        Jfull = torch.stack(Jrows)            # d y_phys / d (xn,un)
        # convert columns to physical input scaling: d/d x_phys = (d/d xn)/xs
        Jphys = Jfull / xs_t                  # (2,4)
        jx.append(float(torch.linalg.matrix_norm(Jphys[:, :2], 2)))
        ju.append(float(torch.linalg.matrix_norm(Jphys[:, 2:], 2)))
    q = lambda a: dict(median=float(np.median(a)), p90=float(np.percentile(a, 90)))
    return {"state_jac_spec": q(jx), "input_jac_spec": q(ju)}


def selfrollout_contraction(seq, data, norm, n_pts=200, seed=0, H=8):
    """Model's own closed-form rollout under Phi: does smooth predict faster decay?
    Reports mean V(model rollout step H)/V(0) ratio. Smaller = more contractive."""
    XU, Y, xm, xs, ym, ys, _, test_idx = data
    rng = np.random.default_rng(seed + 7)
    idx = rng.choice(test_idx, size=min(n_pts, len(test_idx)), replace=False)
    xm2, xs2 = torch.tensor(xm[:2]), torch.tensor(xs[:2])
    ym_t, ys_t = torch.tensor(ym), torch.tensor(ys)
    X0 = XU[idx, :2].astype(np.float64)
    ratios = []
    with torch.no_grad():
        for x0 in X0:
            x = x0.copy(); V0 = float(x @ P @ x)
            for _ in range(H):
                phi = np.zeros(2)  # decay under zero shifted-input (regulation target)
                xn = torch.tensor((x - xm2.numpy()) / xs2.numpy(), dtype=torch.float32)
                un = torch.tensor((phi - xm[2:]) / xs[2:], dtype=torch.float32)
                y = (seq(torch.cat([xn, un]).view(1, 4)).view(2) * ys_t + ym_t).numpy()
                x = y
            ratios.append(float(x @ P @ x) / max(V0, 1e-9))
    return dict(median_VH_over_V0=float(np.median(ratios)),
                p90_VH_over_V0=float(np.percentile(ratios, 90)))


def main():
    d = np.load(ROOT / "data" / "processed" / "lcnn_paper_cstr_onestep_20k.npz")
    data = (d["XU"].astype("float32"), d["Y"].astype("float32"),
            d["x_mean"].astype("float32"), d["x_std"].astype("float32"),
            d["y_mean"].astype("float32"), d["y_std"].astype("float32"),
            d["train_idx"], d["test_idx"])
    norm, un_lo, un_hi = make_norm(data)
    print("precomputing Phi/Yphi ...")
    Phi, Yphi = pr.precompute_phi(SIM, data[0][:, :2].astype("float64"))

    out = {"config": dict(seeds=SEEDS, n_ic=len(ICS), horizon=3, steps=120,
                          success_V=2.0, note="reproducible disambiguation + controls"),
           "per_seed": {}}

    for seed in SEEDS:
        print(f"\n===== seed {seed} =====")
        rec = {}
        for kind in ["lcnn", "smooth", "smooth_softsign"]:
            seq = train_kind(kind, seed, data, Phi, Yphi)
            r = {}
            # (1) gradient budget sweep (cheap + high)
            r["grad_lr0.1"] = {B: succ_rate(lambda x0, B=B: bs.closed_loop(
                seq, norm, x0, 3, B, 120, 0.1, 0.01, 2.0, un_lo, un_hi))[0]
                for B in [3, 20, 400]}
            # (2) lr robustness at budget 20
            r["lr_robust_b20"] = {lr: succ_rate(lambda x0, lr=lr: bs.closed_loop(
                seq, norm, x0, 3, 20, 120, lr, 0.01, 2.0, un_lo, un_hi))[0]
                for lr in [0.05, 0.2, 0.8]}
            # (3) cheap CEM + (4) near-oracle CEM with exploitation gap (seed 0 only for oracle cost)
            np.random.seed(seed)
            s_cheap, p_cheap, _ = succ_rate(lambda x0: cem(seq, norm, un_lo, un_hi, x0,
                                                           n=96, iters=6, elite=12))
            r["cem_cheap"] = dict(success=s_cheap, p90_final_V=p_cheap)
            if seed == 0:
                np.random.seed(100)
                rows = [cem(seq, norm, un_lo, un_hi, x0, n=2500, iters=14, elite=120,
                            report_exploit=True) for x0 in ICS]
                r["cem_oracle"] = dict(
                    success=float(np.mean([x["success"] for x in rows])),
                    p90_final_V=float(np.percentile([x["final_V"] for x in rows], 90)),
                    mean_Vpred=float(np.mean([x["mean_Vpred"] for x in rows])),
                    mean_Vtrue=float(np.mean([x["mean_Vtrue"] for x in rows])))
            # contractivity controls
            r["jacobian"] = jacobian_stats(seq, data, norm, seed=seed)
            r["selfrollout"] = selfrollout_contraction(seq, data, norm, seed=seed)
            rec[kind] = r
            print(f"  {kind:15} grad(b3/b20/b400)="
                  f"{r['grad_lr0.1'][3]:.2f}/{r['grad_lr0.1'][20]:.2f}/{r['grad_lr0.1'][400]:.2f}"
                  f"  cem_cheap={s_cheap:.2f}"
                  f"  Jx_med={r['jacobian']['state_jac_spec']['median']:.3f}"
                  f"  Ju_med={r['jacobian']['input_jac_spec']['median']:.2e}"
                  f"  rollVH/V0={r['selfrollout']['median_VH_over_V0']:.3f}")
        out["per_seed"][seed] = rec

    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "disambiguation_landscape_vs_gradient.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "disambiguation_landscape_vs_gradient.json")


if __name__ == "__main__":
    main()
