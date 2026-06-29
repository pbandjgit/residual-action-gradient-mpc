"""Paper-H probe: action-gradient consistency of the Lyapunov residual.

Thesis: in finite-budget learned LMPC, residual-VALUE consistency (Paper G) is not
sufficient; the solver also needs action-GRADIENT consistency of the Lyapunov
residual it optimizes. GroupSort-LCNN is 1-Lipschitz but piecewise linear, so its
action-gradient can be value-accurate yet irregular and poorly aligned with the
true residual gradient; a smooth 1-Lipschitz spectral model should be better.

Trains two models under the SAME loss (Cauchy + residual-value consistency) and
same data/seed, then measures in normalized action space:
  (1) residual VALUE error          |g_hat - g_true|
  (2) action-gradient ALIGNMENT     cos(d_un V(f_hat), d_un V(f_true))
  (3) action-gradient IRREGULARITY  ||d_un V(f_hat)(un+delta) - d_un V(f_hat)(un)||
The relevant action-gradient is d_u V(f(x,u)); the residual's Phi term is
u-independent and cancels. d is taken in normalized action coordinates un.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cstr.lcnn_paper_simulator import create_lcnn_paper_cstr          # noqa: E402
from cstr.sontag import saturated_sontag_np                           # noqa: E402
from cstr.models_onestep import build_onestep, cauchy_nll            # noqa: E402

P = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float64)
Pt = torch.tensor(P, dtype=torch.float32)
INPUT_LO = np.array([-3.5, -5.0e5]); INPUT_HI = np.array([3.5, 5.0e5])
LOG = ROOT / "results" / "interim" / "logs"


def Vt(y):
    return (y @ Pt * y).sum(-1)


def Vnp(y):
    return np.einsum("ni,ij,nj->n", y, P, y)


def precompute_phi(sim, X):
    """Phi(x) and next state under Phi for every sample (global-indexed)."""
    Phi = np.array([saturated_sontag_np(sim, X[i], INPUT_LO, INPUT_HI, P)
                    for i in range(len(X))], dtype=np.float32)
    Yphi = np.array([sim.step(X[i], Phi[i]) for i in range(len(X))], dtype=np.float32)
    return Phi, Yphi


def train(kind, seed, data, sim, Phi_all, Yphi_all,
          epochs=50, lam=0.003, warmup=30, noise_scale=0.2, noise_clip=5.0):
    XU, Y, xm, xs, ym, ys, train_idx, _ = data
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 1000)
    xm_t, xs_t = torch.tensor(xm), torch.tensor(xs)
    ym_t, ys_t = torch.tensor(ym), torch.tensor(ys)
    Xn = torch.tensor((XU - xm) / xs)
    Yn = ((Y - ym) / ys).copy()
    n = noise_scale * rng.standard_cauchy(size=Yn[train_idx].shape)
    Yn[train_idx] += np.clip(n, -noise_clip, noise_clip).astype(np.float32)
    Yn_t = torch.tensor(Yn)
    rawY = torch.tensor(Y)
    Phi_t = torch.tensor(Phi_all)
    g_true_all = Vt(rawY) - Vt(torch.tensor(Yphi_all))      # (N,) global-indexed
    g_scale = torch.clamp(g_true_all[train_idx].std(), min=torch.tensor(1.0))
    disp = torch.tensor(np.median(np.abs(Yn[train_idx] - np.median(Yn[train_idx], 0)), 0)
                        * 1.4826, dtype=torch.float32)

    model = build_onestep({"model": kind, "hidden": 40, "lipschitz_bound": 4.0, "n_layers": 2})
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    tr = np.asarray(train_idx)
    for ep in range(1, epochs + 1):
        model.train(); order = rng.permutation(len(tr))
        for k in range(0, len(tr), 256):
            b = tr[order[k:k + 256]]
            opt.zero_grad()
            pred = model(Xn[b])
            loss = cauchy_nll(pred, Yn_t[b], disp)
            if lam > 0 and ep > warmup:
                y_hat = pred * ys_t + ym_t
                xphi = torch.cat([torch.tensor(XU[b, :2]), Phi_t[b]], 1)
                yphi_hat = model((xphi - xm_t) / xs_t) * ys_t + ym_t
                g_hat = Vt(y_hat) - Vt(yphi_hat)
                loss = loss + lam * torch.relu((g_true_all[b] - g_hat) / g_scale).pow(2).mean()
            loss.backward(); opt.step()
            if hasattr(model, "apply_constraints"):
                model.apply_constraints()
    model.eval()
    # clean test MSE (sanity)
    _, _, _, _, _, _, _, test_idx = data
    with torch.no_grad():
        pv = (model(Xn[test_idx]) * ys_t + ym_t).numpy()
    mse = float(np.mean(((pv - Y[test_idx]) / ys) ** 2))
    return model, mse


def diagnose(model, data, sim, Phi_all, Yphi_all, n_pts=800, seed=0, delta=0.05):
    XU, Y, xm, xs, ym, ys, _, test_idx = data
    rng = np.random.default_rng(seed)
    idx = rng.choice(test_idx, size=min(n_pts, len(test_idx)), replace=False)
    x = XU[idx, :2].astype(np.float64); u = XU[idx, 2:].astype(np.float64)
    xm_t, xs_t = torch.tensor(xm), torch.tensor(xs)
    ym_t, ys_t = torch.tensor(ym), torch.tensor(ys)
    u_mean, u_std = xm[2:].astype(np.float64), xs[2:].astype(np.float64)
    x_mean, x_std = xm[:2].astype(np.float64), xs[:2].astype(np.float64)

    def learned_u_grad(un_np):
        """d_un V(f_hat(x,u)) for a batch, via autograd (M,2)."""
        un = torch.tensor(un_np, dtype=torch.float32, requires_grad=True)
        xn = torch.tensor((x - x_mean) / x_std, dtype=torch.float32)
        xun = torch.cat([xn, un], 1)
        y = model(xun) * ys_t + ym_t
        g = Vt(y).sum()
        (grad,) = torch.autograd.grad(g, un)
        return grad.detach().numpy()

    un = (u - u_mean) / u_std
    g_hat_grad = learned_u_grad(un)

    # true d_un V(f(x,u)) via central finite difference in un space
    eps = 1e-3
    true_grad = np.zeros_like(un)
    for j in range(2):
        up = un.copy(); up[:, j] += eps
        dn = un.copy(); dn[:, j] -= eps
        uu_p = up * u_std + u_mean; uu_m = dn * u_std + u_mean
        Vp = Vnp(np.array([sim.step(x[i], uu_p[i]) for i in range(len(x))]))
        Vm = Vnp(np.array([sim.step(x[i], uu_m[i]) for i in range(len(x))]))
        true_grad[:, j] = (Vp - Vm) / (2 * eps)

    def cos(a, b):
        na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
        ok = (na > 1e-9) & (nb > 1e-9)
        c = np.full(len(a), np.nan)
        c[ok] = (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])
        return c

    align = cos(true_grad, g_hat_grad)

    # gradient irregularity: variation of learned grad over a small un step
    dvec = rng.standard_normal(un.shape); dvec /= np.linalg.norm(dvec, axis=1, keepdims=True)
    g2 = learned_u_grad(un + delta * dvec)
    irr = np.linalg.norm(g2 - g_hat_grad, axis=1) / delta

    # residual value error (full residual, with Phi)
    Phi = Phi_all[idx]
    yhat = (model(torch.tensor((np.concatenate([x, u], 1) - xm) / xs, dtype=torch.float32))
            * ys_t + ym_t).detach().numpy()
    yphi_hat = (model(torch.tensor((np.concatenate([x, Phi], 1).astype(np.float64) - xm) / xs,
                dtype=torch.float32)) * ys_t + ym_t).detach().numpy()
    g_hat = Vnp(yhat) - Vnp(yphi_hat)
    g_true = Vnp(Y[idx].astype(np.float64)) - Vnp(Yphi_all[idx].astype(np.float64))
    val_err = np.abs(g_hat - g_true)

    def q(a):
        a = a[~np.isnan(a)]
        return dict(median=float(np.median(a)), p90=float(np.percentile(a, 90)),
                    mean=float(np.mean(a)))
    return {
        "align_cos": {**q(align), "frac_neg": float(np.mean(align[~np.isnan(align)] < 0))},
        "grad_irregularity": q(irr),
        "value_error": q(val_err),
    }


def main():
    sim = create_lcnn_paper_cstr(shifted=True)
    d = np.load(ROOT / "data" / "processed" / "lcnn_paper_cstr_onestep_20k.npz")
    data = (d["XU"].astype(np.float32), d["Y"].astype(np.float32),
            d["x_mean"].astype(np.float32), d["x_std"].astype(np.float32),
            d["y_mean"].astype(np.float32), d["y_std"].astype(np.float32),
            d["train_idx"], d["test_idx"])
    print("precomputing Phi / Yphi for all samples ...")
    Phi_all, Yphi_all = precompute_phi(sim, data[0][:, :2].astype(np.float64))

    results = {}
    for kind in ["lcnn", "smooth"]:
        print(f"training {kind} ...")
        model, mse = train(kind, seed=0, data=data, sim=sim, Phi_all=Phi_all, Yphi_all=Yphi_all)
        diag = diagnose(model, data, sim, Phi_all, Yphi_all)
        diag["test_mse"] = mse
        results[kind] = diag
        print(f"  {kind}: test_mse={mse:.2e}  "
              f"value_err(med)={diag['value_error']['median']:.3g}  "
              f"align_cos(med)={diag['align_cos']['median']:.3f} "
              f"frac_neg={diag['align_cos']['frac_neg']:.2f}  "
              f"grad_irreg(p90)={diag['grad_irregularity']['p90']:.3g}")
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "probe_action_gradient.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", LOG / "probe_action_gradient.json")
    print("\n=== thesis read ===")
    l, s = results["lcnn"], results["smooth"]
    print(f"value error (med):     LCNN {l['value_error']['median']:.3g}  vs  smooth {s['value_error']['median']:.3g}")
    print(f"grad alignment (med):  LCNN {l['align_cos']['median']:.3f}    vs  smooth {s['align_cos']['median']:.3f}")
    print(f"grad align frac<0:     LCNN {l['align_cos']['frac_neg']:.2f}     vs  smooth {s['align_cos']['frac_neg']:.2f}")
    print(f"grad irregularity(p90):LCNN {l['grad_irregularity']['p90']:.3g}  vs  smooth {s['grad_irregularity']['p90']:.3g}")


if __name__ == "__main__":
    main()
