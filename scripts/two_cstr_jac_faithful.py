"""Two-CSTR Jacobian-faithfulness analysis (resolves the Jx attribution caveat).

The concern: does L_grad stabilize by making the surrogate artificially more
contractive (smaller state-Jacobian Jx), rather than by aligning derivatives?

This measures, at matched test points, the surrogate's physical one-step state
Jacobian against the TRUE plant's, for value-only vs value+grad:
  - Jx_model  : spectral norm of d f_hat / d x   (state Jacobian magnitude)
  - Jx_true   : spectral norm of d f / d x        (true plant)
  - jac_err   : spectral norm of (d f_hat/dx - d f/dx)  (state-Jacobian error)

Hypothesis: value-only has a spuriously inflated, less accurate state Jacobian;
L_grad de-inflates it TOWARD the true value (jac_err down), and the model stays
MORE expansive than the true plant (Jx_model > Jx_true), so the gain is improved
derivative faithfulness, not artificial contractivity.

Outputs: results/interim/logs/two_cstr_jac_faithful.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import two_cstr_lgrad as T  # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"
SEEDS = [0, 1, 2]
EPS = np.array([1e-5, 1e-3, 1e-5, 1e-3])


def true_state_jac(x_phys, u_phys):
    J = np.zeros((4, 4))
    for j in range(4):
        xp = x_phys.copy(); xmn = x_phys.copy(); xp[j] += EPS[j]; xmn[j] -= EPS[j]
        J[:, j] = (T.step(xp, u_phys) - T.step(xmn, u_phys)) / (2 * EPS[j])
    return J


def model_state_jac(seq, xun_norm, xs_t, ys_t, ym_t):
    xun = xun_norm.clone().requires_grad_(True)
    y = seq(xun.view(1, 8)).view(4) * ys_t + ym_t
    rows = []
    for k in range(4):
        g, = torch.autograd.grad(y[k], xun, retain_graph=(k < 3))
        rows.append(g.detach())
    Jfull = torch.stack(rows) / xs_t        # d y_phys / d (x_phys, u_phys)
    return Jfull[:, :4].numpy()


def main():
    data = T.make_data(n=12000)
    XU, Y, xm, xs, ym, ys, _, test_idx = data
    xs_t = torch.tensor(xs); ys_t = torch.tensor(ys); ym_t = torch.tensor(ym)
    print("precomputing ...", flush=True)
    Phi, Yphi = T.precompute_phi(XU)
    true_ug = T.precompute_true_ugrad(XU, xm, xs)
    configs = [("value_only", 0.003, 0.0), ("value+grad", 0.003, 0.2)]
    out = {"config": dict(plant="two_cstr_series", seeds=SEEDS, n_pts=200,
                          configs=configs), "per_seed": {}}
    print(f"{'config':12} {'seed':>4} {'Jx_model':>9} {'Jx_true':>8} {'jac_err':>8} {'rel_err':>8}", flush=True)
    for seed in SEEDS:
        out["per_seed"][seed] = {}
        rng = np.random.default_rng(seed)
        idx = rng.choice(test_idx, size=200, replace=False)
        Jt = [true_state_jac(XU[i, :4].astype(float), XU[i, 4:].astype(float)) for i in idx]
        Jt_norm = np.array([np.linalg.norm(J, 2) for J in Jt])
        for tag, lv, lg in configs:
            model, _ = T.train(seed, data, Phi, Yphi, true_ug, lam_val=lv,
                               lam_grad=lg, epochs=70)
            seq = T.freeze(model)
            Jm_norm, err = [], []
            for c, i in enumerate(idx):
                xun = torch.tensor((XU[i] - xm) / xs, dtype=torch.float32)
                Jm = model_state_jac(seq, xun, xs_t, ys_t, ym_t)
                Jm_norm.append(np.linalg.norm(Jm, 2))
                err.append(np.linalg.norm(Jm - Jt[c], 2))
            Jm_norm = np.array(Jm_norm); err = np.array(err)
            rel = float(np.median(err / (Jt_norm + 1e-9)))
            rec = dict(Jx_model=float(np.median(Jm_norm)),
                       Jx_true=float(np.median(Jt_norm)),
                       jac_err=float(np.median(err)), rel_err=rel)
            out["per_seed"][seed][tag] = rec
            print(f"{tag:12} {seed:>4} {rec['Jx_model']:>9.3f} {rec['Jx_true']:>8.3f} "
                  f"{rec['jac_err']:>8.3f} {rel:>8.3f}", flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "two_cstr_jac_faithful.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "two_cstr_jac_faithful.json", flush=True)

    def avg(tag, key):
        return float(np.mean([out["per_seed"][s][tag][key] for s in SEEDS]))
    print("\n=== Jacobian faithfulness (seed-averaged) ===")
    print(f"{'config':12} {'Jx_model':>9} {'Jx_true':>8} {'jac_err':>8} {'rel_err':>8}")
    for tag, _, _ in configs:
        print(f"{tag:12} {avg(tag,'Jx_model'):>9.3f} {avg(tag,'Jx_true'):>8.3f} "
              f"{avg(tag,'jac_err'):>8.3f} {avg(tag,'rel_err'):>8.3f}")


if __name__ == "__main__":
    main()
