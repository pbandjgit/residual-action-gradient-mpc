"""Paper-H cart-pole robustness campaign: 5-seed ablation + lam_grad sweep.

Reduces the "lucky 3-seed" concern and tests whether lam_grad has a dose-response on
cart-pole (as on the CSTR). Fixed GroupSort-LCNN, fixed solver config (horizon 10,
budget 20, 180 steps, lr 0.15). The earlier tuning sweep showed b20<1.0 is partly
seed-structural (one seed's model caps ~0.6 regardless of horizon/budget/lr), so the
campaign reports the value-only vs +L_grad separation across 5 seeds and the
lam_grad dose-response, not just a single tuned number.

Outputs: results/interim/logs/cartpole_campaign.json
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
import cartpole_lgrad as C                                            # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
HORIZON, BUDGET, STEPS, LR, RHO_U, SUCCESS_V = 10, 20, 180, 0.15, 0.01, 2.0

# (tag, lam_val, lam_grad)
CONFIGS = [
    ("value_only", 0.003, 0.0),     # ablation; sweep lam_grad=0
    ("grad_only",  0.0,   0.05),    # ablation
    ("value+grad", 0.003, 0.05),    # ablation; sweep lam_grad=0.05
    ("sweep_0.01", 0.003, 0.01),    # sweep
    ("sweep_0.2",  0.003, 0.2),     # sweep
]


def main():
    data = C.make_data()
    XU, Y, xm, xs, ym, ys, _, _ = data
    norm = (torch.tensor(xm), torch.tensor(xs), torch.tensor(ym), torch.tensor(ys))
    un_lo = torch.tensor((-C.FLIM - xm[4:]) / xs[4:], dtype=torch.float32)
    un_hi = torch.tensor((C.FLIM - xm[4:]) / xs[4:], dtype=torch.float32)
    print("precomputing Phi/Yphi + true action-grad ...")
    Phi, Yphi = C.precompute_phi(XU)
    true_ug = C.precompute_true_ugrad(XU, xm, xs)

    out = {"config": dict(plant="cartpole", seeds=SEEDS, horizon=HORIZON, budget=BUDGET,
                          steps=STEPS, lr=LR, success_V=SUCCESS_V, configs=CONFIGS),
           "per_seed": {}}
    print(f"{'config':12} {'seed':>4} {'mse':>9} {'fneg':>5} {'b20':>5} {'Jx':>6}")
    for seed in SEEDS:
        out["per_seed"][seed] = {}
        for tag, lv, lg in CONFIGS:
            model, mse = C.train(seed, data, Phi, Yphi, true_ug, lam_val=lv, lam_grad=lg)
            seq = C.freeze(model)
            _, fn = C.align_med(seq, data, norm, seed=seed)
            b20 = float(np.mean([C.closed_loop(seq, norm, x0, HORIZON, BUDGET, STEPS, LR,
                                 RHO_U, SUCCESS_V, un_lo, un_hi)["success"] for x0 in C.ICS]))
            jx = C.jac_state_med(seq, data, norm, seed=seed)
            out["per_seed"][seed][tag] = dict(test_mse=mse, frac_neg=fn, b20=b20,
                                              state_jac_med=jx, lam_val=lv, lam_grad=lg)
            print(f"{tag:12} {seed:>4} {mse:>9.2e} {fn:>5.2f} {b20:>5.2f} {jx:>6.2f}")
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "cartpole_campaign.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "cartpole_campaign.json")

    def agg(tag, key):
        v = [out["per_seed"][s][tag][key] for s in SEEDS]
        return float(np.mean(v)), float(np.std(v))

    print("\n=== ablation (mean +/- std) ===")
    print(f"{'config':12} {'fneg':>12} {'b20':>12} {'Jx':>12} {'mse':>10}")
    for tag in ["value_only", "grad_only", "value+grad"]:
        fm, fs = agg(tag, "frac_neg"); bm, bsd = agg(tag, "b20")
        jm, js = agg(tag, "state_jac_med"); mm, _ = agg(tag, "test_mse")
        print(f"{tag:12} {fm:>6.2f}+/-{fs:<4.2f} {bm:>6.2f}+/-{bsd:<4.2f} "
              f"{jm:>6.2f}+/-{js:<4.2f} {mm:>10.2e}")

    print("\n=== lam_grad sweep (value on, seed mean) ===")
    sweep = [("0.0", "value_only"), ("0.01", "sweep_0.01"), ("0.05", "value+grad"),
             ("0.2", "sweep_0.2")]
    print(f"{'lam_grad':>9} {'fneg':>6} {'b20':>6} {'Jx':>6} {'mse':>10}")
    for lab, tag in sweep:
        print(f"{lab:>9} {agg(tag,'frac_neg')[0]:>6.2f} {agg(tag,'b20')[0]:>6.2f} "
              f"{agg(tag,'state_jac_med')[0]:>6.2f} {agg(tag,'test_mse')[0]:>10.2e}")


if __name__ == "__main__":
    main()
