"""Reviewer-facing CSTR supplement for Paper H.

This script addresses three likely EAAI reviewer questions in one controlled
experiment:

1. Does L_grad require analytic access to the true model?
   No. We train with action-gradient labels obtained from offline local input
   perturbation queries. The simulator is used only to emulate those transition
   queries; the online controller uses the learned surrogate.

2. Is the gain just due to using a smoother surrogate?
   We compare fixed GroupSort-LCNN and smooth-spectral Softsign models, each
   with value-only and value+L_grad training.

3. Is the finite-budget setting meaningful?
   We log model-evaluation counts and indicative closed-loop planning time.

Outputs:
  results/interim/logs/reviewer_supplement_cstr.json
  docs/REVIEWER_SUPPLEMENT_CSTR_2026_06_29.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import budget_sweep_solver as bs                                      # noqa: E402
import probe_action_gradient as pr                                    # noqa: E402
from disambiguate_landscape_vs_gradient import make_norm              # noqa: E402
from cstr.models_onestep import build_onestep                         # noqa: E402

P = bs.P
SIM = bs.SIM
ICS = bs.ICS
LOG = ROOT / "results" / "interim" / "logs"
DOC = ROOT / "docs" / "REVIEWER_SUPPLEMENT_CSTR_2026_06_29.md"


def load_data():
    d = np.load(ROOT / "data" / "processed" / "lcnn_paper_cstr_onestep_20k.npz")
    return (d["XU"].astype("float32"), d["Y"].astype("float32"),
            d["x_mean"].astype("float32"), d["x_std"].astype("float32"),
            d["y_mean"].astype("float32"), d["y_std"].astype("float32"),
            d["train_idx"], d["test_idx"])


def make_subset_data(data, seed: int = 29, n_train: int = 6000, n_test: int = 3000):
    XU, Y, xm, xs, ym, ys, train_idx, test_idx = data
    rng = np.random.default_rng(seed)
    train_sub = rng.choice(train_idx, size=min(n_train, len(train_idx)), replace=False)
    test_sub = rng.choice(test_idx, size=min(n_test, len(test_idx)), replace=False)
    return XU, Y, xm, xs, ym, ys, np.sort(train_sub), np.sort(test_sub)


def offline_perturbation_label(data, eps_un: float, y_noise_scale: float,
                               seed: int) -> np.ndarray:
    """Estimate d_un V(f(x,u)) from local transition queries.

    The input perturbation is applied in normalized action coordinates. Optional
    output noise emulates measurement/simulator noise in the queried next states.
    """
    XU, _, xm, xs, _, ys, *_ = data
    x = XU[:, :2].astype(np.float64)
    u = XU[:, 2:].astype(np.float64)
    u_mean, u_std = xm[2:].astype(np.float64), xs[2:].astype(np.float64)
    un = (u - u_mean) / u_std
    rng = np.random.default_rng(seed + 4242)
    train_idx = data[6]
    labels = np.zeros((len(XU), 2), dtype=np.float32)
    for j in range(2):
        up = un[train_idx].copy()
        dn = un[train_idx].copy()
        up[:, j] += eps_un
        dn[:, j] -= eps_un
        uu_p = up * u_std + u_mean
        uu_m = dn * u_std + u_mean
        xp = x[train_idx]
        yp = np.array([SIM.step(xp[i], uu_p[i]) for i in range(len(xp))], dtype=np.float64)
        ym = np.array([SIM.step(xp[i], uu_m[i]) for i in range(len(xp))], dtype=np.float64)
        if y_noise_scale > 0:
            # Noise is specified in normalized-output units, then mapped back to
            # physical state units before V is evaluated.
            yp = yp + rng.normal(0.0, y_noise_scale, size=yp.shape) * ys
            ym = ym + rng.normal(0.0, y_noise_scale, size=ym.shape) * ys
        labels[train_idx, j] = ((pr.Vnp(yp) - pr.Vnp(ym)) / (2.0 * eps_un)).astype(np.float32)
    return labels


def label_quality(label, reference, train_idx):
    err = label[train_idx] - reference[train_idx]
    rel = np.linalg.norm(err, axis=1) / (np.linalg.norm(reference[train_idx], axis=1) + 1e-9)
    cos_num = np.sum(label[train_idx] * reference[train_idx], axis=1)
    cos_den = (np.linalg.norm(label[train_idx], axis=1)
               * np.linalg.norm(reference[train_idx], axis=1) + 1e-9)
    cos = cos_num / cos_den
    return {
        "rel_err_median": float(np.median(rel)),
        "rel_err_p90": float(np.percentile(rel, 90)),
        "cos_median": float(np.median(cos)),
        "wrong_sign_frac": float(np.mean(cos < 0.0)),
    }


def timed_closed_loop(seq, norm, un_lo, un_hi, budget: int):
    tic = time.perf_counter()
    rows = [bs.closed_loop(seq, norm, x0, horizon=3, budget=budget, steps=120,
                           lr=0.1, rho_u=0.01, success_V=2.0,
                           un_lo=un_lo, un_hi=un_hi)
            for x0 in ICS]
    elapsed = time.perf_counter() - tic
    n_control = len(ICS) * 120
    return {
        "success": float(np.mean([r["success"] for r in rows])),
        "p90_final_V": float(np.percentile([r["final_V"] for r in rows], 90)),
        "mean_tv": float(np.mean([r["tv"] for r in rows])),
        "ms_per_control": float(1000.0 * elapsed / n_control),
        "model_evals_per_control": int(budget * 3),
    }


def train_model(cfg, seed, data, Phi_all, Yphi_all, true_ug,
                epochs=42, lam_val=0.003, lam_grad=0.0, warmup=24,
                noise_scale=0.2, noise_clip=5.0):
    XU, Y, xm, xs, ym, ys, train_idx, test_idx = data
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)
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
        model.train()
        order = rng.permutation(len(tr))
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
            loss.backward()
            opt.step()
            if hasattr(model, "apply_constraints"):
                model.apply_constraints()
    model.eval()
    with torch.no_grad():
        pv = (model(Xn[test_idx]) * ys_t + ym_t).numpy()
    mse = float(np.mean(((pv - Y[test_idx]) / ys) ** 2))
    return model, mse


def diagnose_model(model, data, sim, Phi, Yphi, seed):
    diag = pr.diagnose(model, data, sim, Phi, Yphi, seed=seed)
    return {
        "align_cos_med": diag["align_cos"]["median"],
        "align_frac_neg": diag["align_cos"]["frac_neg"],
        "grad_irreg_p90": diag["grad_irregularity"]["p90"],
    }


def summarize(out):
    lines = [
        "# Reviewer supplement: offline perturbation labels, smooth baseline, timing",
        "",
        "## Purpose",
        "This supplement addresses the reviewer concern that L_grad appears to require",
        "analytic access to the true model. The experiment trains from offline local",
        "input-perturbation labels. The simulator is used only to emulate transition",
        "queries during identification; the online controller uses only the surrogate.",
        "",
        "It also checks whether smoothness alone is sufficient by comparing a",
        "GroupSort-LCNN with a smooth-spectral Softsign surrogate under the same",
        "training objectives.",
        "",
        "## Label sources",
    ]
    for name, q in out["label_quality"].items():
        lines.append(
            f"- `{name}`: median relative error {q['rel_err_median']:.3g}, "
            f"p90 relative error {q['rel_err_p90']:.3g}, median cosine "
            f"{q['cos_median']:.3f}, wrong-sign {q['wrong_sign_frac']:.3f}."
        )
    lines += [
        "",
        "## Closed-loop and solver-facing diagnostics",
        "| architecture | condition | seed | MSE | frac wrong grad | b20 success | ms/control | model evals/control |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, seed_rows in out["per_seed"].items():
        for arch, arch_rows in seed_rows.items():
            for condition, r in arch_rows.items():
                lines.append(
                    f"| {arch} | {condition} | {seed} | {r['test_mse']:.2e} | "
                    f"{r['align_frac_neg']:.3f} | {r['b20']['success']:.2f} | "
                    f"{r['b20']['ms_per_control']:.2f} | {r['b20']['model_evals_per_control']} |"
                )
    lines += [
        "",
        "## Seed-averaged reading",
        "| architecture | condition | MSE | frac wrong grad | b20 success | ms/control |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arch, arch_rows in out["summary"].items():
        for condition, r in arch_rows.items():
            lines.append(
                f"| {arch} | {condition} | {r['test_mse']:.2e} | "
                f"{r['align_frac_neg']:.3f} | {r['b20_success']:.2f} | "
                f"{r['b20_ms_per_control']:.2f} |"
            )
    lines += [
        "",
        "## Manuscript implication",
        "- L_grad need not be described as requiring analytic true-model gradients.",
        "  It can be trained from local offline perturbation labels.",
        "- Smoothness alone is not the same as residual action-gradient consistency;",
        "  the smooth value-only baseline is the relevant check.",
        "- Timing is indicative Python/autograd timing, not a speed claim. The primary",
        "  finite-budget quantity is the model-evaluation count per control step.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


def main():
    data = make_subset_data(load_data())
    norm, un_lo, un_hi = make_norm(data)
    print("precomputing Phi/Yphi ...", flush=True)
    Phi, Yphi = pr.precompute_phi(SIM, data[0][:, :2].astype("float64"))

    print("precomputing reference and offline perturbation labels ...", flush=True)
    reference = offline_perturbation_label(data, eps_un=1e-3, y_noise_scale=0.0, seed=0)
    fd_clean = offline_perturbation_label(data, eps_un=2e-2, y_noise_scale=0.0, seed=0)
    fd_noisy = offline_perturbation_label(data, eps_un=2e-2, y_noise_scale=0.001, seed=0)
    label_sources = {
        "fd_clean": fd_clean,
        "fd_noisy": fd_noisy,
    }

    archs = {
        "lcnn": {"model": "lcnn", "hidden": 40, "lipschitz_bound": 4.0, "n_layers": 2},
        "smooth_softsign": {"model": "smooth", "hidden": 40, "lipschitz_bound": 4.0,
                            "n_layers": 2, "activation": "softsign"},
    }
    common_configs = [
        ("value_only", None, 0.0),
        ("Lgrad_fd_clean", "fd_clean", 0.05),
    ]
    configs_by_arch = {
        "lcnn": common_configs + [("Lgrad_fd_noisy", "fd_noisy", 0.05)],
        "smooth_softsign": common_configs,
    }
    seeds = [0, 1, 2]
    out = {
        "config": {
            "plant": "CSTR",
            "seeds": seeds,
            "train_subset_size": int(len(data[6])),
            "test_subset_size": int(len(data[7])),
            "horizon": 3,
            "budget_reported": 20,
            "offline_query_count_per_sample": 4,
            "fd_clean_eps_un": 2e-2,
            "fd_noisy_eps_un": 2e-2,
            "fd_noisy_y_noise_scale_normalized": 0.001,
        },
        "label_quality": {
            name: label_quality(lbl, reference, data[6])
            for name, lbl in label_sources.items()
        },
        "per_seed": {},
    }

    print(f"{'arch':16} {'condition':15} {'seed':>4} {'mse':>9} {'fneg':>6} {'b20':>5} {'ms':>7}", flush=True)
    for seed in seeds:
        out["per_seed"][str(seed)] = {}
        for arch, cfg in archs.items():
            out["per_seed"][str(seed)][arch] = {}
            for condition, label_name, lam_grad in configs_by_arch[arch]:
                label = reference if label_name is None else label_sources[label_name]
                model, mse = train_model(cfg, seed, data, Phi, Yphi, label, lam_grad=lam_grad)
                ev = diagnose_model(model, data, SIM, Phi, Yphi, seed)
                seq = bs.freeze(model)
                b20 = timed_closed_loop(seq, norm, un_lo, un_hi, budget=20)
                rec = {
                    "test_mse": mse,
                    "align_cos_med": ev["align_cos_med"],
                    "align_frac_neg": ev["align_frac_neg"],
                    "grad_irreg_p90": ev["grad_irreg_p90"],
                    "b20": b20,
                }
                out["per_seed"][str(seed)][arch][condition] = rec
                print(f"{arch:16} {condition:15} {seed:>4} {mse:>9.2e} "
                      f"{rec['align_frac_neg']:>6.3f} {b20['success']:>5.2f} "
                      f"{b20['ms_per_control']:>7.2f}", flush=True)

    out["summary"] = {}
    for arch in archs:
        out["summary"][arch] = {}
        for condition, _, _ in configs_by_arch[arch]:
            rows = [out["per_seed"][str(seed)][arch][condition] for seed in seeds]
            out["summary"][arch][condition] = {
                "test_mse": float(np.mean([r["test_mse"] for r in rows])),
                "align_frac_neg": float(np.mean([r["align_frac_neg"] for r in rows])),
                "align_cos_med": float(np.mean([r["align_cos_med"] for r in rows])),
                "grad_irreg_p90": float(np.mean([r["grad_irreg_p90"] for r in rows])),
                "b20_success": float(np.mean([r["b20"]["success"] for r in rows])),
                "b20_p90_final_V": float(np.mean([r["b20"]["p90_final_V"] for r in rows])),
                "b20_ms_per_control": float(np.mean([r["b20"]["ms_per_control"] for r in rows])),
            }

    LOG.mkdir(parents=True, exist_ok=True)
    out_path = LOG / "reviewer_supplement_cstr.json"
    out_path.write_text(json.dumps(out, indent=2))
    summarize(out)
    print("\nwrote", out_path, flush=True)
    print("wrote", DOC, flush=True)


if __name__ == "__main__":
    main()
