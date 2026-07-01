"""Two-CSTR-in-series lam_grad dose-response (contractivity / Jx attribution).

Sweeps the action-gradient weight lam_grad (value-residual fixed at 0.003) and
records the action-gradient wrong-sign fraction, finite-budget closed-loop
success (b20), the state-Jacobian Jx (contractivity control), and one-step MSE.
Goal: show that closed-loop success rises with the wrong-sign correction at an
operating point where Jx is still close to the value-only model, so the gain is
attributable to gradient alignment rather than added contractivity.

Outputs: results/interim/logs/two_cstr_dose.json
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
LAMS = [0.0, 0.05, 0.1, 0.2, 0.5]
SEEDS = [0, 1, 2]


def main():
    data = T.make_data(n=12000)
    XU = data[0]; xm, xs = data[2], data[3]
    norm = T.make_norm(data)
    un_lo = torch.tensor((T.BOX_LO - xm[4:]) / xs[4:], dtype=torch.float32)
    un_hi = torch.tensor((T.BOX_HI - xm[4:]) / xs[4:], dtype=torch.float32)
    print("precomputing ...", flush=True)
    Phi, Yphi = T.precompute_phi(XU)
    true_ug = T.precompute_true_ugrad(XU, xm, xs)
    hz, steps, lr, rho_u, sV = 3, 120, 0.2, 0.01, 2.0

    out = {"config": dict(plant="two_cstr_series", seeds=SEEDS, lams=LAMS,
                          lam_val=0.003, n_data=12000, epochs=70, budget=20,
                          success_V=sV), "per_seed": {}}
    print(f"{'lam':>6} {'seed':>4} {'mse':>9} {'fneg':>5} {'b20':>5} {'Jx':>7}", flush=True)
    for seed in SEEDS:
        out["per_seed"][seed] = {}
        for lam in LAMS:
            model, mse = T.train(seed, data, Phi, Yphi, true_ug,
                                 lam_val=0.003, lam_grad=lam, epochs=70)
            seq = T.freeze(model)
            _, fn = T.align_med(seq, data, norm, seed=seed)
            b20 = float(np.mean([T.closed_loop(seq, norm, x0, hz, 20, steps, lr, rho_u,
                                               sV, un_lo, un_hi)["success"] for x0 in T.ICS]))
            jx = T.jac_state_med(seq, data, norm, seed=seed)
            out["per_seed"][seed][f"{lam:g}"] = dict(test_mse=mse, frac_neg=fn, b20=b20,
                                                     state_jac_med=jx)
            print(f"{lam:>6g} {seed:>4} {mse:>9.2e} {fn:>5.2f} {b20:>5.2f} {jx:>7.2f}", flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "two_cstr_dose.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "two_cstr_dose.json", flush=True)

    def avg(lam, key):
        return float(np.mean([out["per_seed"][s][f"{lam:g}"][key] for s in SEEDS]))
    print("\n=== two-CSTR dose-response (seed-averaged) ===")
    print(f"{'lam_grad':>8} {'fneg':>5} {'b20':>5} {'Jx':>7} {'mse':>9}")
    for lam in LAMS:
        print(f"{lam:>8g} {avg(lam,'frac_neg'):>5.2f} {avg(lam,'b20'):>5.2f} "
              f"{avg(lam,'state_jac_med'):>7.2f} {avg(lam,'test_mse'):>9.2e}")


if __name__ == "__main__":
    main()
