"""Two-CSTR budget sweep: is value-only failure a finite-budget (too-few-iterations)
artifact, or a wrong-gradient effect that more iterations cannot fix?

Runs the first-order projected-gradient MPC at B in {3,10,20,40,100} for value-only
and value+grad (3 seeds). If value-only fails across all budgets while value+grad
succeeds even at small B, the failure is the misaligned action-gradient, not
insufficient optimization -- the correct control for the "strong-solver baseline"
question (a high-accuracy gradient/Hessian solver is misled by the same wrong
derivatives, so IPOPT would not rescue value-only either).

Outputs: results/interim/logs/two_cstr_budget.json
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
BUDGETS = [3, 10, 20, 40, 100]


def main():
    data = T.make_data(n=12000)
    XU, xm, xs = data[0], data[2], data[3]
    norm = T.make_norm(data)
    un_lo = torch.tensor((T.BOX_LO - xm[4:]) / xs[4:], dtype=torch.float32)
    un_hi = torch.tensor((T.BOX_HI - xm[4:]) / xs[4:], dtype=torch.float32)
    print("precomputing ...", flush=True)
    Phi, Yphi = T.precompute_phi(XU)
    true_ug = T.precompute_true_ugrad(XU, xm, xs)
    hz, steps, lr, rho_u, sV = 3, 120, 0.2, 0.01, 2.0
    configs = [("value_only", 0.003, 0.0), ("value+grad", 0.003, 0.2)]

    out = {"config": dict(plant="two_cstr_series", seeds=SEEDS, budgets=BUDGETS,
                          configs=configs), "per_seed": {}}
    print(f"{'config':11} {'seed':>4} " + " ".join(f"b{b:>3}" for b in BUDGETS), flush=True)
    for seed in SEEDS:
        out["per_seed"][seed] = {}
        for tag, lv, lg in configs:
            model, _ = T.train(seed, data, Phi, Yphi, true_ug, lam_val=lv, lam_grad=lg, epochs=70)
            seq = T.freeze(model)
            row = {}
            for B in BUDGETS:
                s = float(np.mean([T.closed_loop(seq, norm, x0, hz, B, steps, lr, rho_u,
                                                 sV, un_lo, un_hi)["success"] for x0 in T.ICS]))
                row[str(B)] = s
            out["per_seed"][seed][tag] = row
            print(f"{tag:11} {seed:>4} " + " ".join(f"{row[str(b)]:>4.1f}" for b in BUDGETS), flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "two_cstr_budget.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "two_cstr_budget.json", flush=True)

    def avg(tag, B):
        return float(np.mean([out["per_seed"][s][tag][str(B)] for s in SEEDS]))
    print("\n=== budget sweep (seed-averaged success) ===")
    print(f"{'config':11} " + " ".join(f"b{b:>3}" for b in BUDGETS))
    for tag, _, _ in configs:
        print(f"{tag:11} " + " ".join(f"{avg(tag,b):>4.2f}" for b in BUDGETS))


if __name__ == "__main__":
    main()
