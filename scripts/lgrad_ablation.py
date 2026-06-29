"""Paper-H Experiment C ablation: separate the value and gradient training terms,
and a lam_grad dose-response sweep, on a FIXED GroupSort-LCNN architecture.

Questions:
  (1) Is the gradient term necessary? (value-only should keep failing.)
  (2) Is the value term still necessary, or does grad-only suffice?
  (3) Dose-response: how much lam_grad is needed; is the rescue robust / monotone?

All runs: lcnn (Bjorck + GroupSort), hidden 40, Lipschitz 4, 2 layers, 50 epochs,
3 seeds. Only the loss weights (lam_val, lam_grad) change. Metrics: test MSE,
action-grad alignment, irregularity, cheap closed-loop (gradient b3/b20, cheap CEM),
and state-Jacobian (contractivity control).

Outputs: results/interim/logs/lgrad_ablation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import budget_sweep_solver as bs                                      # noqa: E402
import probe_action_gradient as pr                                   # noqa: E402
import lgrad_experiment as L                                         # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"
SEEDS = [0, 1, 2]
ARCH = {"model": "lcnn", "hidden": 40, "lipschitz_bound": 4.0, "n_layers": 2}

# (tag, lam_val, lam_grad). Covers the 3-way ablation and the lam_grad sweep
# (value term fixed at 0.003 for the sweep; lam_grad in {0,.01,.05,.2,1.0}).
CONFIGS = [
    ("value_only",   0.003, 0.0),    # ablation + sweep lam_grad=0
    ("grad_only",    0.0,   0.05),   # ablation
    ("value+grad",   0.003, 0.05),   # ablation + sweep lam_grad=0.05
    ("sweep_0.01",   0.003, 0.01),   # sweep
    ("sweep_0.2",    0.003, 0.2),    # sweep
    ("sweep_1.0",    0.003, 1.0),    # sweep
]


def main():
    d = np.load(ROOT / "data" / "processed" / "lcnn_paper_cstr_onestep_20k.npz")
    data = (d["XU"].astype("float32"), d["Y"].astype("float32"),
            d["x_mean"].astype("float32"), d["x_std"].astype("float32"),
            d["y_mean"].astype("float32"), d["y_std"].astype("float32"),
            d["train_idx"], d["test_idx"])
    norm, un_lo, un_hi = L.make_norm(data)
    print("precomputing Phi/Yphi + true action-grad ...")
    L.PHI, L.YPHI = pr.precompute_phi(bs.SIM, data[0][:, :2].astype("float64"))
    true_ug = L.precompute_true_ugrad(bs.SIM, data[0], data[2], data[3])

    out = {"config": dict(arch="lcnn_groupsort", seeds=SEEDS, configs=CONFIGS,
                          note="ablation + lam_grad sweep, fixed architecture"),
           "per_seed": {}}
    print(f"{'config':12} {'lv':>6} {'lg':>5} {'seed':>4} {'mse':>9} {'align':>6} "
          f"{'fneg':>5} {'irr':>7} {'b3':>4} {'b20':>4} {'cem':>4} {'Jx':>5}")
    for seed in SEEDS:
        out["per_seed"][seed] = {}
        for tag, lv, lg in CONFIGS:
            model, mse = L.train(ARCH, seed, data, L.PHI, L.YPHI, true_ug,
                                 lam_val=lv, lam_grad=lg)
            ev = L.eval_model(model, data, norm, un_lo, un_hi, seed)
            ev.update(test_mse=mse, lam_val=lv, lam_grad=lg)
            out["per_seed"][seed][tag] = ev
            print(f"{tag:12} {lv:>6.3f} {lg:>5.2f} {seed:>4} {mse:>9.2e} "
                  f"{ev['align_cos_med']:>6.2f} {ev['align_frac_neg']:>5.2f} "
                  f"{ev['grad_irreg_p90']:>7.1f} {ev['grad_succ_b3']:>4.1f} "
                  f"{ev['grad_succ_b20']:>4.1f} {ev['cem_cheap_succ']:>4.1f} "
                  f"{ev['state_jac_med']:>5.2f}")

    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "lgrad_ablation.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / "lgrad_ablation.json")

    def avg(tag, key):
        return float(np.mean([out["per_seed"][s][tag][key] for s in SEEDS]))

    print("\n=== ablation (seed-averaged) ===")
    print(f"{'config':12} {'align':>6} {'irr':>7} {'b3':>5} {'b20':>5} {'cem':>5} {'Jx':>5} {'mse':>9}")
    for tag in ["value_only", "grad_only", "value+grad"]:
        print(f"{tag:12} {avg(tag,'align_cos_med'):>6.2f} {avg(tag,'grad_irreg_p90'):>7.1f} "
              f"{avg(tag,'grad_succ_b3'):>5.2f} {avg(tag,'grad_succ_b20'):>5.2f} "
              f"{avg(tag,'cem_cheap_succ'):>5.2f} {avg(tag,'state_jac_med'):>5.2f} "
              f"{avg(tag,'test_mse'):>9.2e}")

    print("\n=== lam_grad sweep (value term on, seed-averaged) ===")
    sweep = [("0.0", "value_only"), ("0.01", "sweep_0.01"), ("0.05", "value+grad"),
             ("0.2", "sweep_0.2"), ("1.0", "sweep_1.0")]
    print(f"{'lam_grad':>9} {'align':>6} {'irr':>7} {'b3':>5} {'b20':>5} {'cem':>5} {'Jx':>5} {'mse':>9}")
    for lab, tag in sweep:
        print(f"{lab:>9} {avg(tag,'align_cos_med'):>6.2f} {avg(tag,'grad_irreg_p90'):>7.1f} "
              f"{avg(tag,'grad_succ_b3'):>5.2f} {avg(tag,'grad_succ_b20'):>5.2f} "
              f"{avg(tag,'cem_cheap_succ'):>5.2f} {avg(tag,'state_jac_med'):>5.2f} "
              f"{avg(tag,'test_mse'):>9.2e}")


if __name__ == "__main__":
    main()
