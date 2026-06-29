"""Paper-H Experiment C: explicit residual-gradient-consistency training (L_grad).

Decisive constructive test that CONTROLS the architecture/contractivity confound:
within a FIXED architecture, compare value-only training against value + L_grad,
changing only the training objective. If L_grad improves cheap-budget closed-loop
on a fixed architecture, the gain is attributable to the training target (not to
the activation's smoothness or predicted contractivity, which are held fixed).

L_grad = mean || d_un V(f_hat(x,u)) - d_un V(f_true(x,u)) ||^2  (normalized action
space; the Phi term of the Lyapunov residual g is u-independent and cancels in
d_un g). d_un g_true is precomputed by central finite difference on the simulator;
d_un g_hat is autograd through the model (double-backward during training).

Architectures tested (each value-only vs value+L_grad):
  - lcnn            : Bjorck + GroupSort (nonsmooth; value-robust per Paper G)
  - smooth_softsign : Bjorck + softsign (smooth but NOT contractive; fails value-only)

Outputs: results/interim/logs/lgrad_experiment.json
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
from disambiguate_landscape_vs_gradient import cem, make_norm, jacobian_stats  # noqa: E402
from cstr.models_onestep import build_onestep                        # noqa: E402

P = bs.P; SIM = bs.SIM; ICS = bs.ICS
INPUT_LO = bs.INPUT_LO; INPUT_HI = bs.INPUT_HI
LOG = ROOT / "results" / "interim" / "logs"
SEEDS = [0, 1, 2]


def precompute_true_ugrad(sim, XU, xm, xs, eps=1e-3):
    """d_un V(f(x,u)) for every sample, central finite diff on the simulator,
    in NORMALIZED action coordinates un. Architecture-independent."""
    x = XU[:, :2].astype(np.float64); u = XU[:, 2:].astype(np.float64)
    u_mean, u_std = xm[2:].astype(np.float64), xs[2:].astype(np.float64)
    un = (u - u_mean) / u_std
    N = len(XU); g = np.zeros((N, 2), np.float32)
    for j in range(2):
        up = un.copy(); up[:, j] += eps
        dn = un.copy(); dn[:, j] -= eps
        uu_p = up * u_std + u_mean; uu_m = dn * u_std + u_mean
        Vp = pr.Vnp(np.array([sim.step(x[i], uu_p[i]) for i in range(N)]))
        Vm = pr.Vnp(np.array([sim.step(x[i], uu_m[i]) for i in range(N)]))
        g[:, j] = ((Vp - Vm) / (2 * eps)).astype(np.float32)
    return g


def train(cfg, seed, data, Phi_all, Yphi_all, true_ug,
          epochs=50, lam_val=0.003, lam_grad=0.0, warmup=30,
          noise_scale=0.2, noise_clip=5.0):
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
    true_ug_t = torch.tensor(true_ug)
    ug_scale = torch.clamp(true_ug_t[train_idx].std(), min=torch.tensor(1.0))

    model = build_onestep(cfg)
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
                xphi = torch.cat([torch.tensor(XU[b, :2]), Phi_t[b]], 1)
                yphi_hat = model((xphi - xm_t) / xs_t) * ys_t + ym_t
                g_hat = pr.Vt(y_hat) - pr.Vt(yphi_hat)
                loss = loss + lam_val * torch.relu((g_true_all[b] - g_hat) / g_scale).pow(2).mean()
            if lam_grad > 0 and ep > warmup:
                un_b = xb[:, 2:].detach().requires_grad_(True)
                xun = torch.cat([xb[:, :2], un_b], 1)
                y = model(xun) * ys_t + ym_t
                Vsum = pr.Vt(y).sum()
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


def eval_model(model, data, norm, un_lo, un_hi, seed):
    """test MSE already had; here: action-grad alignment + cheap closed-loop."""
    diag = pr.diagnose(model, data, SIM, PHI, YPHI, seed=seed)
    seq = bs.freeze(model)
    grad_succ = {B: float(np.mean([bs.closed_loop(seq, norm, x0, 3, B, 120, 0.1, 0.01,
                                                  2.0, un_lo, un_hi)["success"] for x0 in ICS]))
                 for B in [3, 20]}
    np.random.seed(seed)
    cem_succ = float(np.mean([cem(seq, norm, un_lo, un_hi, x0, n=96, iters=6,
                                  elite=12)["success"] for x0 in ICS]))
    jac = jacobian_stats(seq, data, norm, seed=seed)
    return dict(align_cos_med=diag["align_cos"]["median"],
                align_frac_neg=diag["align_cos"]["frac_neg"],
                grad_irreg_p90=diag["grad_irregularity"]["p90"],
                grad_succ_b3=grad_succ[3], grad_succ_b20=grad_succ[20],
                cem_cheap_succ=cem_succ,
                state_jac_med=jac["state_jac_spec"]["median"],
                input_jac_med=jac["input_jac_spec"]["median"])


PHI = YPHI = None


def main():
    global PHI, YPHI
    d = np.load(ROOT / "data" / "processed" / "lcnn_paper_cstr_onestep_20k.npz")
    data = (d["XU"].astype("float32"), d["Y"].astype("float32"),
            d["x_mean"].astype("float32"), d["x_std"].astype("float32"),
            d["y_mean"].astype("float32"), d["y_std"].astype("float32"),
            d["train_idx"], d["test_idx"])
    norm, un_lo, un_hi = make_norm(data)
    print("precomputing Phi/Yphi + true action-grad ...")
    PHI, YPHI = pr.precompute_phi(SIM, data[0][:, :2].astype("float64"))
    true_ug = precompute_true_ugrad(SIM, data[0], data[2], data[3])

    archs = {
        "lcnn": {"model": "lcnn", "hidden": 40, "lipschitz_bound": 4.0, "n_layers": 2},
        "smooth_softsign": {"model": "smooth", "hidden": 40, "lipschitz_bound": 4.0,
                            "n_layers": 2, "activation": "softsign"},
    }
    variants = {"value_only": dict(lam_grad=0.0), "value+Lgrad": dict(lam_grad=0.05)}

    out = {"config": dict(seeds=SEEDS, n_ic=len(ICS), lam_grad=0.05,
                          note="Experiment C: L_grad within fixed architecture"),
           "per_seed": {}}
    hdr = f"{'arch':16} {'variant':12} {'seed':>4} {'mse':>9} {'align':>6} {'fneg':>5} {'irr_p90':>8} {'b3':>4} {'b20':>4} {'cem':>4}"
    print(hdr)
    for seed in SEEDS:
        out["per_seed"][seed] = {}
        for aname, cfg in archs.items():
            out["per_seed"][seed][aname] = {}
            for vname, vkw in variants.items():
                model, mse = train(cfg, seed, data, PHI, YPHI, true_ug, **vkw)
                ev = eval_model(model, data, norm, un_lo, un_hi, seed)
                ev["test_mse"] = mse
                out["per_seed"][seed][aname][vname] = ev
                print(f"{aname:16} {vname:12} {seed:>4} {mse:>9.2e} "
                      f"{ev['align_cos_med']:>6.2f} {ev['align_frac_neg']:>5.2f} "
                      f"{ev['grad_irreg_p90']:>8.1f} {ev['grad_succ_b3']:>4.1f} "
                      f"{ev['grad_succ_b20']:>4.1f} {ev['cem_cheap_succ']:>4.1f}")

    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "lgrad_experiment.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "lgrad_experiment.json")
    # seed-averaged summary
    print("\n=== seed-averaged (closed-loop success + contractivity) ===")
    print(f"{'arch':16} {'variant':12} {'align':>6} {'b3':>5} {'b20':>5} {'cem':>5} {'Jx':>5}")
    for aname in archs:
        for vname in variants:
            rows = [out["per_seed"][s][aname][vname] for s in SEEDS]
            al = np.mean([r["align_cos_med"] for r in rows])
            b3 = np.mean([r["grad_succ_b3"] for r in rows])
            b20 = np.mean([r["grad_succ_b20"] for r in rows])
            cm = np.mean([r["cem_cheap_succ"] for r in rows])
            jx = np.mean([r["state_jac_med"] for r in rows])
            print(f"{aname:16} {vname:12} {al:>6.2f} {b3:>5.2f} {b20:>5.2f} {cm:>5.2f} {jx:>5.2f}")


if __name__ == "__main__":
    main()
