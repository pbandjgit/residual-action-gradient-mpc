"""Paper-H second-plant replication: does the L_grad finding transfer to cart-pole?

Self-contained (cart-pole dynamics + LQR auxiliary controller copied from the
paper_g cross-plant pilot; this script writes ONLY under paper_h). Same controlled
test as on the CSTR: a FIXED GroupSort-LCNN architecture, ablation over the training
objective {value-only (Cauchy + value-residual E+), grad-only (Cauchy + L_grad),
value+grad}, with the contractivity (state-Jacobian) control. The thesis predicts:
value-only fails cheap-budget closed-loop; adding L_grad rescues it, without
inflating the Jacobian.

Plant: cart-pole (4 states, 1 input), RK4, force_limit 10, dt 0.02, unstable upright.
V(x)=x'Px with P from the discrete LQR; auxiliary controller Phi(x)=clip(-Kx).

Outputs: results/interim/logs/cartpole_lgrad.json
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_discrete_are

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cstr.models_onestep import OneStepLCNN, BjorckLinear              # noqa: E402
import probe_action_gradient as pr                                    # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"
SEEDS = [0, 1, 2]


@dataclass(frozen=True)
class CP:
    g: float = 9.81; mc: float = 1.0; mp: float = 0.1
    l: float = 0.5; dt: float = 0.02; flim: float = 10.0


CPP = CP()


def dyn(x, u, p=CPP):
    pos, vel, th, thd = x
    f = float(np.clip(u, -p.flim, p.flim))
    tm = p.mc + p.mp; pml = p.mp * p.l
    s = math.sin(th); c = math.cos(th)
    temp = (f + pml * thd ** 2 * s) / tm
    tha = (p.g * s - c * temp) / (p.l * (4.0 / 3.0 - p.mp * c ** 2 / tm))
    xa = temp - pml * tha * c / tm
    return np.array([vel, xa, thd, tha], dtype=np.float64)


def step(x, u, p=CPP):
    dt = p.dt
    k1 = dyn(x, u, p); k2 = dyn(x + 0.5 * dt * k1, u, p)
    k3 = dyn(x + 0.5 * dt * k2, u, p); k4 = dyn(x + dt * k3, u, p)
    y = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    y[2] = ((y[2] + np.pi) % (2 * np.pi)) - np.pi
    return y.astype(np.float64)


def design_lqr(p=CPP, eps=1e-5):
    x0 = np.zeros(4)
    A = np.zeros((4, 4)); B = np.zeros((4, 1))
    for i in range(4):
        dx = np.zeros(4); dx[i] = eps
        A[:, i] = (step(x0 + dx, 0.0, p) - step(x0 - dx, 0.0, p)) / (2 * eps)
    B[:, 0] = (step(x0, eps, p) - step(x0, -eps, p)) / (2 * eps)
    Q = np.diag([1.0, 0.15, 12.0, 0.4]); R = np.array([[0.08]])
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
    return K, P


K_LQR, P_MAT = design_lqr()
Pt = torch.tensor(P_MAT, dtype=torch.float32)
FLIM = CPP.flim
ICS = [np.array([0.0, 0.0, 0.30, 0.0]), np.array([0.0, 0.0, -0.30, 0.0]),
       np.array([0.4, 0.0, 0.20, 0.0]), np.array([-0.4, 0.0, -0.20, 0.0]),
       np.array([0.0, 0.0, 0.40, 0.0])]


def phi(x):
    u = -np.asarray(x, float) @ K_LQR.T
    return np.clip(u.reshape(-1), -FLIM, FLIM)


def Vnp(x):
    x = np.asarray(x, float)
    return np.einsum("...i,ij,...j->...", x, P_MAT, x)


def Vt(y):
    return (y @ Pt * y).sum(-1)


def sample_dataset(n, seed):
    rng = np.random.default_rng(seed)
    lows = np.array([-1.8, -2.5, -0.55, -3.5]); highs = np.array([1.8, 2.5, 0.55, 3.5])
    X = rng.uniform(lows, highs, size=(n, 4)); U = rng.uniform(-FLIM, FLIM, size=(n, 1))
    Y = np.array([step(x, float(u[0])) for x, u in zip(X, U)])
    return np.concatenate([X, U], 1).astype(np.float32), Y.astype(np.float32)


def make_data(n=12000, seed=7):
    XU, Y = sample_dataset(n, seed)
    idx = np.arange(n); rng = np.random.default_rng(123); rng.shuffle(idx)
    ntr = int(0.85 * n); train_idx, test_idx = idx[:ntr], idx[ntr:]
    xm = XU.mean(0); xs = XU.std(0) + 1e-6
    ym = Y.mean(0); ys = Y.std(0) + 1e-6
    return (XU, Y, xm.astype(np.float32), xs.astype(np.float32),
            ym.astype(np.float32), ys.astype(np.float32), train_idx, test_idx)


def precompute_phi(XU):
    X = XU[:, :4].astype(np.float64)
    Phi = np.array([phi(X[i]) for i in range(len(X))], dtype=np.float32)   # (N,1)
    Yphi = np.array([step(X[i], float(Phi[i, 0])) for i in range(len(X))], dtype=np.float32)
    return Phi, Yphi


def precompute_true_ugrad(XU, xm, xs, eps=1e-3):
    x = XU[:, :4].astype(np.float64); u = XU[:, 4:].astype(np.float64)
    u_mean, u_std = xm[4:].astype(np.float64), xs[4:].astype(np.float64)
    un = (u - u_mean) / u_std
    N = len(XU); g = np.zeros((N, 1), np.float32)
    up = un.copy(); up[:, 0] += eps; dn = un.copy(); dn[:, 0] -= eps
    uu_p = up * u_std + u_mean; uu_m = dn * u_std + u_mean
    Vp = Vnp(np.array([step(x[i], float(uu_p[i, 0])) for i in range(N)]))
    Vm = Vnp(np.array([step(x[i], float(uu_m[i, 0])) for i in range(N)]))
    g[:, 0] = ((Vp - Vm) / (2 * eps)).astype(np.float32)
    return g


DEFAULT_HIDDEN = 48  # overridable by callers (e.g. capacity-balanced DAgger)


def build_lcnn(seed, hidden=None, lip=6.0, n_layers=2):
    torch.manual_seed(seed)
    return OneStepLCNN(state_dim=4, input_dim=1,
                       hidden=DEFAULT_HIDDEN if hidden is None else hidden,
                       lipschitz_bound=lip, n_layers=n_layers)


def train(seed, data, Phi_all, Yphi_all, true_ug, lam_val, lam_grad,
          epochs=60, warmup=35, noise_scale=0.2, noise_clip=5.0):
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
    g_true_all = Vt(torch.tensor(Y)) - Vt(torch.tensor(Yphi_all))
    g_scale = torch.clamp(g_true_all[train_idx].std(), min=torch.tensor(1.0))
    disp = torch.tensor(np.median(np.abs(Yn[train_idx] - np.median(Yn[train_idx], 0)), 0)
                        * 1.4826, dtype=torch.float32)
    true_ug_t = torch.tensor(true_ug)
    ug_scale = torch.clamp(true_ug_t[train_idx].std(), min=torch.tensor(1.0))

    model = build_lcnn(seed)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    tr = np.asarray(train_idx)
    for ep in range(1, epochs + 1):
        model.train(); order = rng.permutation(len(tr))
        for k in range(0, len(tr), 256):
            b = tr[order[k:k + 256]]
            opt.zero_grad()
            xb = Xn[b]
            pred = model(xb)
            loss = pr.cauchy_nll(pred, Yn_t[b], disp)
            if lam_val > 0 and ep > warmup:
                y_hat = pred * ys_t + ym_t
                xphi = torch.cat([torch.tensor(XU[b, :4]), Phi_t[b]], 1)
                yphi_hat = model((xphi - xm_t) / xs_t) * ys_t + ym_t
                g_hat = Vt(y_hat) - Vt(yphi_hat)
                loss = loss + lam_val * torch.relu((g_true_all[b] - g_hat) / g_scale).pow(2).mean()
            if lam_grad > 0 and ep > warmup:
                un_b = xb[:, 4:].detach().requires_grad_(True)
                xun = torch.cat([xb[:, :4], un_b], 1)
                y = model(xun) * ys_t + ym_t
                Vsum = Vt(y).sum()
                (g_un,) = torch.autograd.grad(Vsum, un_b, create_graph=True)
                loss = loss + lam_grad * (((g_un - true_ug_t[b]) / ug_scale) ** 2).mean()
            loss.backward(); opt.step()
            if hasattr(model, "apply_constraints"):
                model.apply_constraints()
    model.eval()
    _, _, _, _, _, _, _, test_idx = data
    with torch.no_grad():
        pv = (model(Xn[test_idx]) * ys_t + ym_t).numpy()
    mse = float(np.mean(((pv - Y[test_idx]) / ys) ** 2))
    return model, mse


def freeze(model):
    layers = []
    for layer in model.body:
        layers.append(layer.materialized_linear() if isinstance(layer, BjorckLinear) else layer)
    seq = nn.Sequential(*layers, model.out.materialized_linear())
    seq.eval()
    for p in seq.parameters():
        p.requires_grad_(False)
    return seq


def closed_loop(seq, norm, x0, horizon, budget, steps, lr, rho_u, success_V, un_lo, un_hi):
    xm, xs, ym, ys = norm; xm4, xs4 = xm[:4], xs[:4]
    x = np.asarray(x0, float); Vtraj = [float(Vnp(x))]
    un = torch.zeros(horizon, 1)
    for _ in range(steps):
        un = un.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([un], lr=lr)
        x0n = torch.tensor((x - xm4.numpy()) / xs4.numpy(), dtype=torch.float32).view(1, 4)
        for _it in range(budget):
            opt.zero_grad(); xt = x0n; J = torch.zeros(())
            for h in range(horizon):
                xun = torch.cat([xt, un[h].view(1, 1)], 1)
                y = seq(xun) * ys + ym
                J = J + (y @ Pt * y).sum() + rho_u * (un[h] ** 2).sum()
                xt = (y - xm4) / xs4
            J.backward(); opt.step()
            with torch.no_grad():
                un.clamp_(un_lo, un_hi)
        u0 = float(np.clip(un[0].detach().numpy() * xs[4:].numpy() + xm[4:].numpy(), -FLIM, FLIM)[0])
        x = step(x, u0); Vtraj.append(float(Vnp(x)))
        un = torch.cat([un[1:].detach(), un[-1:].detach()])
    return dict(final_V=Vtraj[-1], max_V=float(max(Vtraj)),
                success=bool(Vtraj[-1] <= success_V))


def jac_state_med(seq, data, norm, n_pts=300, seed=0):
    XU, Y, xm, xs, ym, ys, _, test_idx = data
    rng = np.random.default_rng(seed)
    idx = rng.choice(test_idx, size=min(n_pts, len(test_idx)), replace=False)
    xs_t = torch.tensor(xs); ys_t = torch.tensor(ys); ym_t = torch.tensor(ym)
    jx = []
    for i in idx:
        xun = torch.tensor((XU[i] - xm) / xs, dtype=torch.float32, requires_grad=True)
        y = seq(xun.view(1, 5)).view(4) * ys_t + ym_t  # physical next state
        rows = []
        for k in range(4):
            g, = torch.autograd.grad(y[k], xun, retain_graph=(k < 3))
            rows.append(g.detach())
        Jfull = torch.stack(rows) / xs_t          # d y_phys / d (x_phys, u_phys)
        jx.append(float(torch.linalg.matrix_norm(Jfull[:, :4], 2)))
    return float(np.median(jx))


def align_med(seq, data, norm, n_pts=500, seed=0):
    """action-grad alignment median + frac_neg, in normalized 1-D action space."""
    XU, Y, xm, xs, ym, ys, _, test_idx = data
    rng = np.random.default_rng(seed)
    idx = rng.choice(test_idx, size=min(n_pts, len(test_idx)), replace=False)
    x = XU[idx, :4].astype(np.float64); u = XU[idx, 4:].astype(np.float64)
    u_mean, u_std = xm[4:].astype(np.float64), xs[4:].astype(np.float64)
    x_mean, x_std = xm[:4].astype(np.float64), xs[:4].astype(np.float64)
    ys_t = torch.tensor(ys); ym_t = torch.tensor(ym)
    un = (u - u_mean) / u_std
    un_t = torch.tensor(un, dtype=torch.float32, requires_grad=True)
    xn = torch.tensor((x - x_mean) / x_std, dtype=torch.float32)
    y = seq(torch.cat([xn, un_t], 1)) * ys_t + ym_t
    g = Vt(y).sum()
    (lg,) = torch.autograd.grad(g, un_t)
    lg = lg.detach().numpy()
    eps = 1e-3
    up = un.copy(); up[:, 0] += eps; dn = un.copy(); dn[:, 0] -= eps
    uu_p = up * u_std + u_mean; uu_m = dn * u_std + u_mean
    Vp = Vnp(np.array([step(x[i], float(uu_p[i, 0])) for i in range(len(x))]))
    Vm = Vnp(np.array([step(x[i], float(uu_m[i, 0])) for i in range(len(x))]))
    tg = ((Vp - Vm) / (2 * eps)).reshape(-1, 1)
    cs = (lg * tg).sum(1) / (np.linalg.norm(lg, axis=1) * np.linalg.norm(tg, axis=1) + 1e-12)
    return float(np.median(cs)), float(np.mean(cs < 0))


def main():
    data = make_data()
    XU, Y, xm, xs, ym, ys, _, _ = data
    norm = (torch.tensor(xm), torch.tensor(xs), torch.tensor(ym), torch.tensor(ys))
    un_lo = torch.tensor((-FLIM - xm[4:]) / xs[4:], dtype=torch.float32)
    un_hi = torch.tensor((FLIM - xm[4:]) / xs[4:], dtype=torch.float32)
    print("V at upright-ish [0,0,0.05,0]:", float(Vnp(np.array([0, 0, 0.05, 0]))),
          " ICs V0:", [round(float(Vnp(ic)), 1) for ic in ICS])
    print("precomputing Phi/Yphi + true action-grad ...")
    Phi, Yphi = precompute_phi(XU)
    true_ug = precompute_true_ugrad(XU, xm, xs)

    horizon, steps, lr, rho_u = 10, 150, 0.15, 0.01
    # success_V=2.0 corresponds to theta~0.04 rad (~2.6 deg) residual, clearly
    # recovered/upright (LQR-true oracle reaches ~0.26); value-only diverges to ~1e5.
    success_V = 2.0
    configs = [("value_only", 0.003, 0.0), ("grad_only", 0.0, 0.05),
               ("value+grad", 0.003, 0.05)]
    out = {"config": dict(plant="cartpole", seeds=SEEDS, horizon=horizon, steps=steps,
                          lr=lr, rho_u=rho_u, success_V=success_V, n_ic=len(ICS),
                          configs=configs), "per_seed": {}}
    print(f"{'config':12} {'seed':>4} {'mse':>9} {'align':>6} {'fneg':>5} {'b3':>4} {'b20':>4} {'Jx':>6}")
    for seed in SEEDS:
        out["per_seed"][seed] = {}
        for tag, lv, lg in configs:
            model, mse = train(seed, data, Phi, Yphi, true_ug, lam_val=lv, lam_grad=lg)
            seq = freeze(model)
            al, fn = align_med(seq, data, norm, seed=seed)
            b3 = float(np.mean([closed_loop(seq, norm, x0, horizon, 3, steps, lr, rho_u,
                                            success_V, un_lo, un_hi)["success"] for x0 in ICS]))
            b20 = float(np.mean([closed_loop(seq, norm, x0, horizon, 20, steps, lr, rho_u,
                                             success_V, un_lo, un_hi)["success"] for x0 in ICS]))
            jx = jac_state_med(seq, data, norm, seed=seed)
            out["per_seed"][seed][tag] = dict(test_mse=mse, align=al, frac_neg=fn,
                                              b3=b3, b20=b20, state_jac_med=jx,
                                              lam_val=lv, lam_grad=lg)
            print(f"{tag:12} {seed:>4} {mse:>9.2e} {al:>6.2f} {fn:>5.2f} {b3:>4.1f} {b20:>4.1f} {jx:>6.2f}")
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "cartpole_lgrad.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "cartpole_lgrad.json")

    def avg(tag, key):
        return float(np.mean([out["per_seed"][s][tag][key] for s in SEEDS]))
    print("\n=== cart-pole ablation (seed-averaged) ===")
    print(f"{'config':12} {'align':>6} {'fneg':>5} {'b3':>5} {'b20':>5} {'Jx':>6} {'mse':>9}")
    for tag, _, _ in configs:
        print(f"{tag:12} {avg(tag,'align'):>6.2f} {avg(tag,'frac_neg'):>5.2f} "
              f"{avg(tag,'b3'):>5.2f} {avg(tag,'b20'):>5.2f} {avg(tag,'state_jac_med'):>6.2f} "
              f"{avg(tag,'test_mse'):>9.2e}")


if __name__ == "__main__":
    main()
