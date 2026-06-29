"""Paper-H cart-pole on-policy (DAgger-style) data augmentation.

Closes the trajectory-vs-test gap diagnosed in cartpole_traj_diagnostic: the cheap
solver drives the closed loop onto a near-upright manifold under-represented by
uniform box sampling, so the surrogate's action-gradient (and, for seed 4, its
one-step prediction) degrade exactly where they are used. This is covariate shift; the
standard fix is on-policy data (Ross et al. 2011, DAgger).

Loop (per seed): train value+grad on the current dataset -> roll out the cheap-budget
MPC from a set of collection ICs -> add (x_visited, u, step(x_visited,u)) pairs (with
true action-gradient labels for L_grad) to the dataset -> retrain -> repeat. Evaluation
stays on the ORIGINAL 5 ICs for a fair before/after comparison.

Tests two predictions: (i) the trajectory wrong-sign fraction and trajectory
prediction error drop as on-policy data is added; (ii) the closed-loop success of the
failing seeds (esp. seed 4) improves -> the sufficiency gap is addressable.

Outputs: results/interim/logs/cartpole_dagger.json
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
from cartpole_traj_diagnostic import closed_loop_record, traj_metrics, BOX_LO, BOX_HI  # noqa: E402

LOG = ROOT / "results" / "interim" / "logs"
SEEDS = [0, 1, 2, 3, 4]
HORIZON, BUDGET, STEPS, LR, RHO_U = 10, 20, 180, 0.15, 0.01
N_DAGGER = 2
# capacity-balanced refinement: bigger model + augmentation kept well below the base
# size (so the large-angle basin is not swamped by near-equilibrium points).
HIDDEN = int(__import__("os").environ.get("CP_HIDDEN", "48"))
AUG_VISITED_CAP = int(__import__("os").environ.get("CP_AUGCAP", "2200"))
AUG_N_ACT = int(__import__("os").environ.get("CP_NACT", "2"))
# collection ICs: a denser grid than the 5 eval ICs, to cover approach corridors
COLLECT_ICS = [np.array([px, 0.0, th, 0.0])
               for px in (-0.4, 0.0, 0.4)
               for th in (-0.45, -0.3, -0.15, 0.15, 0.3, 0.45)]


def augment_pairs(seq, norm, un_lo, un_hi, collect_ics, n_act, rng):
    """Roll out the cheap MPC from collect_ics, sample actions at visited states,
    return (XU_aug, Y_aug) plus the visited states for diagnostics."""
    Xvis = []
    for x0 in collect_ics:
        Xs, _, _, _ = closed_loop_record(seq, norm, x0, HORIZON, BUDGET, STEPS,
                                         LR, RHO_U, un_lo, un_hi)
        Xvis.append(np.clip(Xs, BOX_LO - 0.5, BOX_HI + 0.5))  # keep in a sane range
    Xvis = np.concatenate(Xvis)
    # subsample to bound growth, keep near-equilibrium density
    if len(Xvis) > AUG_VISITED_CAP:
        idx = rng.choice(len(Xvis), size=AUG_VISITED_CAP, replace=False)
        Xvis = Xvis[idx]
    XU, Y = [], []
    for x in Xvis:
        us = list(rng.uniform(-C.FLIM, C.FLIM, size=n_act)) + [float(C.phi(x)[0])]
        for u in us:
            XU.append(np.concatenate([x, [u]])); Y.append(C.step(x, u))
    return (np.array(XU, np.float32), np.array(Y, np.float32), Xvis)


def eval_seq(seq, norm, data, un_lo, un_hi, seed):
    _, test_fneg = C.align_med(seq, data, norm, seed=seed)
    allX, fvs = [], []
    for x0 in C.ICS:
        Xs, Us, fv, _ = closed_loop_record(seq, norm, x0, HORIZON, BUDGET, STEPS,
                                           LR, RHO_U, un_lo, un_hi)
        allX.append((Xs, Us)); fvs.append(fv)
    Xall = np.concatenate([a for a, _ in allX]); Uall = np.concatenate([b for _, b in allX])
    tm = traj_metrics(seq, norm, Xall, Uall)
    return dict(b20_V2=float(np.mean([f <= 2.0 for f in fvs])),
                b20_V10=float(np.mean([f <= 10.0 for f in fvs])),
                final_Vs=[round(float(f), 2) for f in fvs],
                test_fneg=test_fneg, traj_fneg=tm["traj_frac_neg"],
                traj_mse=tm["traj_pred_mse"], oob=tm["oob_frac"])


def run_seed(seed, base_XU, base_Y, norm, un_lo, un_hi, n_test):
    rng = np.random.default_rng(seed + 7777)
    xm, xs, ym, ys = [t.numpy() for t in norm]
    XU = base_XU.copy(); Y = base_Y.copy()
    N0 = len(XU)
    test_idx = np.arange(N0 - n_test, N0)   # fixed uniform test block
    # precompute labels for base once
    Phi, Yphi = C.precompute_phi(XU)
    ug = C.precompute_true_ugrad(XU, xm, xs)
    rows = []
    for it in range(N_DAGGER + 1):
        train_idx = np.array([i for i in range(len(XU)) if i not in set(test_idx.tolist())])
        data = (XU, Y, xm, xs, ym, ys, train_idx, test_idx)
        model, mse = C.train(seed, data, Phi, Yphi, ug, lam_val=0.003, lam_grad=0.05)
        seq = C.freeze(model)
        ev = eval_seq(seq, norm, data, un_lo, un_hi, seed)
        ev["iter"] = it; ev["n_data"] = int(len(XU)); ev["test_mse"] = mse
        rows.append(ev)
        print(f"  seed {seed} iter {it}: n={len(XU):6d} b20@2={ev['b20_V2']:.2f} "
              f"b20@10={ev['b20_V10']:.2f} traj_fneg={ev['traj_fneg']:.3f} "
              f"traj_mse={ev['traj_mse']:.2e} finalVs={ev['final_Vs']}")
        if it < N_DAGGER:
            XUa, Ya, _ = augment_pairs(seq, norm, un_lo, un_hi, COLLECT_ICS, AUG_N_ACT, rng)
            Phia, Yphia = C.precompute_phi(XUa)
            uga = C.precompute_true_ugrad(XUa, xm, xs)
            XU = np.concatenate([XU, XUa]); Y = np.concatenate([Y, Ya])
            Phi = np.concatenate([Phi, Phia]); Yphi = np.concatenate([Yphi, Yphia])
            ug = np.concatenate([ug, uga])
    return rows


def main():
    data0 = C.make_data()
    base_XU, base_Y, xm, xs, ym, ys, train_idx, test_idx = data0
    norm = (torch.tensor(xm), torch.tensor(xs), torch.tensor(ym), torch.tensor(ys))
    un_lo = torch.tensor((-C.FLIM - xm[4:]) / xs[4:], dtype=torch.float32)
    un_hi = torch.tensor((C.FLIM - xm[4:]) / xs[4:], dtype=torch.float32)
    n_test = len(test_idx)
    # reorder so the test block is contiguous at the end (run_seed slices it)
    order = np.concatenate([train_idx, test_idx])
    base_XU = base_XU[order]; base_Y = base_Y[order]

    C.DEFAULT_HIDDEN = HIDDEN
    tag = "" if (HIDDEN == 48 and AUG_VISITED_CAP == 2200 and AUG_N_ACT == 2) else "_balanced"
    print(f"DAgger: {N_DAGGER} iters, hidden={HIDDEN}, aug_cap={AUG_VISITED_CAP}, "
          f"n_act={AUG_N_ACT}, collect ICs={len(COLLECT_ICS)}, eval ICs={len(C.ICS)}")
    out = {"config": dict(seeds=SEEDS, n_dagger=N_DAGGER, horizon=HORIZON, hidden=HIDDEN,
                          aug_visited_cap=AUG_VISITED_CAP, aug_n_act=AUG_N_ACT,
                          budget=BUDGET, steps=STEPS, n_collect_ics=len(COLLECT_ICS)),
           "per_seed": {}}
    for seed in SEEDS:
        out["per_seed"][seed] = run_seed(seed, base_XU, base_Y, norm, un_lo, un_hi, n_test)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / f"cartpole_dagger{tag}.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", LOG / f"cartpole_dagger{tag}.json")

    print("\n=== DAgger effect (seed-averaged by iteration) ===")
    print(f"{'iter':>4} {'b20@2':>6} {'b20@10':>7} {'traj_fneg':>9} {'traj_mse':>9}")
    for it in range(N_DAGGER + 1):
        rs = [out["per_seed"][s][it] for s in SEEDS]
        print(f"{it:>4} {np.mean([r['b20_V2'] for r in rs]):>6.2f} "
              f"{np.mean([r['b20_V10'] for r in rs]):>7.2f} "
              f"{np.mean([r['traj_fneg'] for r in rs]):>9.3f} "
              f"{np.mean([r['traj_mse'] for r in rs]):>9.2e}")
    print("\nseed 4 (the genuine failure) by iter:",
          [out["per_seed"][4][it]["b20_V10"] for it in range(N_DAGGER + 1)])


if __name__ == "__main__":
    main()
