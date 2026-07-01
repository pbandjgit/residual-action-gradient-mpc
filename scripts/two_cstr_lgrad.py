"""Paper-H second process plant: two CSTRs in series (process-control benchmark).

Self-contained replication of the CSTR L_grad study on the standard
Christofides/Wu "two CSTRs in series" process (4 states, 4 inputs), using the
SAME fixed GroupSort-LCNN architecture and the SAME training-objective ablation
{value-only (Cauchy + value-residual E+), grad-only (Cauchy + L_grad),
value+grad}. The auxiliary controller is saturated LQR. The single-CSTR benchmark
uses saturated Sontag feedback; on this four-input process, saturated LQR is used
because it stabilizes the tested region cleanly under the same input box.

Plant: src/cstr/two_cstr_series.py (second-order kinetics, open-loop unstable).
V(x)=x'Px with P from the continuous LQR at the unstable steady state.

Outputs: results/interim/logs/two_cstr_lgrad.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_continuous_are

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cstr.models_onestep import OneStepLCNN, BjorckLinear              # noqa: E402
from cstr.two_cstr_series import TwoCSTRSeriesSimulator               # noqa: E402
import probe_action_gradient as pr                                    # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"
SEEDS = list(range(10))

# ----------------------------------------------------------------------------- #
# Plant, Lyapunov function, input box
# ----------------------------------------------------------------------------- #
SIM = TwoCSTRSeriesSimulator()           # shifted coordinates, 4 states / 4 inputs
A_LIN, B_LIN = SIM.analytic_jacobian(np.zeros(4), np.zeros(4))
# Input-range-scaled LQR weights (heat inputs have a huge admissible range).
Q_LQR = np.diag([1.0, 0.02, 1.0, 0.02])
R_LQR = np.diag([1.0, 1e-9, 1.0, 1e-9])
P_MAT = solve_continuous_are(A_LIN, B_LIN, Q_LQR, R_LQR)
K_LQR = np.linalg.solve(R_LQR, B_LIN.T @ P_MAT)   # continuous LQR gain
Pt = torch.tensor(P_MAT, dtype=torch.float32)

# Per-reactor input box (shifted), same magnitudes as the single-CSTR benchmark.
BOX_HI = np.array([3.5, 5e5, 3.5, 5e5], dtype=float)
BOX_LO = -BOX_HI


def step(x, u):
    return SIM.step(np.asarray(x, float), np.clip(np.asarray(u, float), BOX_LO, BOX_HI))


def Vnp(x):
    x = np.asarray(x, float)
    return np.einsum("...i,ij,...j->...", x, P_MAT, x)


def Vt(y):
    return (y @ Pt * y).sum(-1)


def phi(x):
    """Saturated LQR feedback for the quadratic CLF V=x'Px (auxiliary controller).

    The single-CSTR benchmark uses a saturated Sontag controller; on the
    two-CSTR process the Sontag formula under input saturation/scaling did not
    fully recover the origin, so we use the saturated LQR auxiliary (the same
    choice as the cart-pole benchmark), which is a valid CLF-based controller for
    the quadratic V designed by LQR.
    """
    x = np.asarray(x, float)
    return np.clip(-K_LQR @ x, BOX_LO, BOX_HI)


# Initial conditions (shifted deviations the controller must drive to the SS).
ICS = [np.array([1.0, 50.0, 1.0, 50.0]),
       np.array([-1.0, -50.0, -1.0, -50.0]),
       np.array([1.5, 30.0, -1.0, -30.0]),
       np.array([-1.5, -30.0, 1.0, 30.0]),
       np.array([0.8, 60.0, 0.8, 60.0])]


# ----------------------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------------------- #
def sample_dataset(n, seed):
    rng = np.random.default_rng(seed)
    lows = np.array([-1.8, -70.0, -1.8, -70.0])
    highs = np.array([1.8, 70.0, 1.8, 70.0])
    X = rng.uniform(lows, highs, size=(n, 4))
    U = rng.uniform(BOX_LO, BOX_HI, size=(n, 4))
    Y = np.array([step(X[i], U[i]) for i in range(n)])
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
    Phi = np.array([phi(X[i]) for i in range(len(X))], dtype=np.float32)   # (N,4)
    Yphi = np.array([step(X[i], Phi[i]) for i in range(len(X))], dtype=np.float32)
    return Phi, Yphi


def precompute_true_ugrad(XU, xm, xs, eps=1e-3):
    """True action-gradient grad_u V(f(x,u)) in normalized control coords (N,4)."""
    x = XU[:, :4].astype(np.float64); u = XU[:, 4:].astype(np.float64)
    u_mean, u_std = xm[4:].astype(np.float64), xs[4:].astype(np.float64)
    un = (u - u_mean) / u_std
    N = len(XU); g = np.zeros((N, 4), np.float32)
    for j in range(4):
        up = un.copy(); up[:, j] += eps
        dn = un.copy(); dn[:, j] -= eps
        uu_p = up * u_std + u_mean; uu_m = dn * u_std + u_mean
        Vp = Vnp(np.array([step(x[i], uu_p[i]) for i in range(N)]))
        Vm = Vnp(np.array([step(x[i], uu_m[i]) for i in range(N)]))
        g[:, j] = ((Vp - Vm) / (2 * eps)).astype(np.float32)
    return g


# ----------------------------------------------------------------------------- #
# Model + training (same machinery as the CSTR / cart-pole modules)
# ----------------------------------------------------------------------------- #
DEFAULT_HIDDEN = 48


def build_lcnn(seed, hidden=None, lip=6.0, n_layers=2):
    torch.manual_seed(seed)
    return OneStepLCNN(state_dim=4, input_dim=4,
                       hidden=DEFAULT_HIDDEN if hidden is None else hidden,
                       lipschitz_bound=lip, n_layers=n_layers)


def train(seed, data, Phi_all, Yphi_all, true_ug, lam_val, lam_grad,
          epochs=50, warmup=30, noise_scale=0.2, noise_clip=5.0):
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
                xphi = torch.cat([torch.tensor(XU[b, :4]), Phi_t[b]], 1)
                yphi_hat = model((xphi - xm_t) / xs_t) * ys_t + ym_t
                y_hat = pred * ys_t + ym_t
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
    un = torch.zeros(horizon, 4)
    for _ in range(steps):
        un = un.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([un], lr=lr)
        x0n = torch.tensor((x - xm4.numpy()) / xs4.numpy(), dtype=torch.float32).view(1, 4)
        for _it in range(budget):
            opt.zero_grad(); xt = x0n; J = torch.zeros(())
            for h in range(horizon):
                xun = torch.cat([xt, un[h].view(1, 4)], 1)
                y = seq(xun) * ys + ym
                J = J + (y @ Pt * y).sum() + rho_u * (un[h] ** 2).sum()
                xt = (y - xm4) / xs4
            J.backward(); opt.step()
            with torch.no_grad():
                un.clamp_(un_lo, un_hi)
        u0 = un[0].detach().numpy() * xs[4:].numpy() + xm[4:].numpy()
        x = step(x, u0); Vtraj.append(float(Vnp(x)))
        un = torch.cat([un[1:].detach(), un[-1:].detach()])
    return dict(final_V=Vtraj[-1], max_V=float(max(Vtraj)),
                success=bool(Vtraj[-1] <= success_V))


def align_med(seq, data, norm, n_pts=500, seed=0):
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
    eps = 1e-3; tg = np.zeros_like(lg)
    for j in range(4):
        up = un.copy(); up[:, j] += eps; dn = un.copy(); dn[:, j] -= eps
        uu_p = up * u_std + u_mean; uu_m = dn * u_std + u_mean
        Vp = Vnp(np.array([step(x[i], uu_p[i]) for i in range(len(x))]))
        Vm = Vnp(np.array([step(x[i], uu_m[i]) for i in range(len(x))]))
        tg[:, j] = (Vp - Vm) / (2 * eps)
    cs = (lg * tg).sum(1) / (np.linalg.norm(lg, axis=1) * np.linalg.norm(tg, axis=1) + 1e-12)
    return float(np.median(cs)), float(np.mean(cs < 0))


def make_norm(data):
    _, _, xm, xs, ym, ys, _, _ = data
    return (torch.tensor(xm), torch.tensor(xs), torch.tensor(ym), torch.tensor(ys))


def jac_state_med(seq, data, norm, n_pts=300, seed=0):
    XU, Y, xm, xs, ym, ys, _, test_idx = data
    rng = np.random.default_rng(seed)
    idx = rng.choice(test_idx, size=min(n_pts, len(test_idx)), replace=False)
    xs_t = torch.tensor(xs); ys_t = torch.tensor(ys); ym_t = torch.tensor(ym)
    jx = []
    for i in idx:
        xun = torch.tensor((XU[i] - xm) / xs, dtype=torch.float32, requires_grad=True)
        y = seq(xun.view(1, 8)).view(4) * ys_t + ym_t
        rows = []
        for k in range(4):
            g, = torch.autograd.grad(y[k], xun, retain_graph=(k < 3))
            rows.append(g.detach())
        Jfull = torch.stack(rows) / xs_t
        jx.append(float(torch.linalg.matrix_norm(Jfull[:, :4], 2)))
    return float(np.median(jx))


def main():
    data = make_data(n=12000)
    XU, Y, xm, xs, ym, ys, _, _ = data
    norm = make_norm(data)
    un_lo = torch.tensor((BOX_LO - xm[4:]) / xs[4:], dtype=torch.float32)
    un_hi = torch.tensor((BOX_HI - xm[4:]) / xs[4:], dtype=torch.float32)
    print("V at ICs:", [round(float(Vnp(ic)), 1) for ic in ICS], flush=True)
    print("precomputing Phi/Yphi + true action-grad ...", flush=True)
    Phi, Yphi = precompute_phi(XU)
    true_ug = precompute_true_ugrad(XU, xm, xs)
    horizon, steps, lr, rho_u, success_V = 3, 120, 0.2, 0.01, 2.0
    configs = [("value_only", 0.003, 0.0), ("grad_only", 0.0, 0.2), ("value+grad", 0.003, 0.2)]
    out = {"config": dict(plant="two_cstr_series", seeds=SEEDS, horizon=horizon, steps=steps,
                          lr=lr, rho_u=rho_u, success_V=success_V, n_ic=len(ICS),
                          n_data=12000, epochs=70, configs=configs), "per_seed": {}}
    print(f"{'config':12} {'seed':>4} {'mse':>9} {'cos':>6} {'fneg':>5} {'b3':>4} {'b20':>4} {'Jx':>7}", flush=True)
    for seed in SEEDS:
        out["per_seed"][seed] = {}
        for tag, lv, lg in configs:
            model, mse = train(seed, data, Phi, Yphi, true_ug, lam_val=lv, lam_grad=lg, epochs=70)
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
            print(f"{tag:12} {seed:>4} {mse:>9.2e} {al:>6.2f} {fn:>5.2f} {b3:>4.1f} {b20:>4.1f} {jx:>7.2f}", flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "two_cstr_lgrad.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "two_cstr_lgrad.json", flush=True)

    def avg(tag, key):
        return float(np.mean([out["per_seed"][s][tag][key] for s in SEEDS]))
    print("\n=== two-CSTR-series ablation (seed-averaged) ===")
    print(f"{'config':12} {'cos':>6} {'fneg':>5} {'b3':>5} {'b20':>5} {'Jx':>7} {'mse':>9}")
    for tag, _, _ in configs:
        print(f"{tag:12} {avg(tag,'align'):>6.2f} {avg(tag,'frac_neg'):>5.2f} "
              f"{avg(tag,'b3'):>5.2f} {avg(tag,'b20'):>5.2f} {avg(tag,'state_jac_med'):>7.2f} "
              f"{avg(tag,'test_mse'):>9.2e}")


if __name__ == "__main__":
    main()
