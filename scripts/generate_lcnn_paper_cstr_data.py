"""Generate one-step data for the LCNN paper CSTR benchmark.

The generated supervised task matches the LCNN paper formulation:

    input  : [x0, u] = [CA-CAs, T-Ts, CA0-CA0s, Q-Qs]
    target : F_tilde(x0, u), the shifted state after Delta = 1e-3 hr

The paper specifies P, rho, Delta, and the train/val/test split, but does not
give the numerical input-constraint set U in the LCNN paper itself. Therefore,
U is configurable here and recorded in metadata.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cstr.lcnn_paper_simulator import create_lcnn_paper_cstr


P_PAPER = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=float)


def summarize(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return {
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.quantile(arr, 0.90)),
            "p99": float(np.quantile(arr, 0.99)),
            "max": float(np.max(arr)),
        }
    return [summarize(arr[:, i]) for i in range(arr.shape[1])]


def sample_ellipsoid(rng, n, P, rho):
    """Uniform samples from {x: x.T P x <= rho} in 2D."""
    L = np.linalg.cholesky(P)  # P = L L.T
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=n)) * np.sqrt(rho)
    y = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    return np.linalg.solve(L.T, y.T).T


def finite_difference_step_jacobian(sim, row, eps=(1e-5, 1e-3, 1e-5, 10.0)):
    J = np.zeros((2, 4), dtype=float)
    for i, h in enumerate(eps):
        rp = row.copy()
        rm = row.copy()
        rp[i] += h
        rm[i] -= h
        yp = sim.step(rp[:2], rp[2:])
        ym = sim.step(rm[:2], rm[2:])
        J[:, i] = (yp - ym) / (2.0 * h)
    return J


def sensitivity_diagnostics(sim, XU, Y, train_idx, n=256, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(XU), size=min(n, len(XU)), replace=False)
    xum = XU[train_idx].mean(axis=0)
    xus = np.maximum(XU[train_idx].std(axis=0), 1e-12)
    ym = Y[train_idx].mean(axis=0)
    ys = np.maximum(Y[train_idx].std(axis=0), 1e-12)
    phys_specs = []
    norm_specs = []
    for i in idx:
        J = finite_difference_step_jacobian(sim, XU[i].astype(float))
        Jn = np.diag(1.0 / ys) @ J @ np.diag(xus)
        phys_specs.append(float(np.linalg.svd(J, compute_uv=False)[0]))
        norm_specs.append(float(np.linalg.svd(Jn, compute_uv=False)[0]))
    return {
        "n": int(len(idx)),
        "physical_step_jacobian_spectral": summarize(np.asarray(phys_specs)),
        "normalized_step_jacobian_spectral": summarize(np.asarray(norm_specs)),
    }


def integrate_one_step(sim, x, u, method: str) -> np.ndarray:
    if method == "rk4":
        return sim.step(x, u)
    if method == "euler":
        y = np.asarray(x, dtype=float).copy()
        h = sim.dt_hr / sim.integration_substeps
        for _ in range(sim.integration_substeps):
            y = y + h * sim._dynamics(y, u)
        return y.astype(np.float64)
    raise ValueError(f"Unsupported integration method: {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--rho", type=float, default=372.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt-hr", type=float, default=1e-3)
    ap.add_argument("--integration-substeps", type=int, default=100)
    ap.add_argument("--integrator", choices=["rk4", "euler"], default="rk4")
    ap.add_argument("--ca0-dev-min", type=float, default=-3.5)
    ap.add_argument("--ca0-dev-max", type=float, default=3.5)
    ap.add_argument("--q-dev-min", type=float, default=-5.0e5)
    ap.add_argument("--q-dev-max", type=float, default=5.0e5)
    ap.add_argument("--sensitivity-samples", type=int, default=256)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "processed" / "lcnn_paper_cstr_onestep_20k.npz"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sim = create_lcnn_paper_cstr(
        dt_hr=args.dt_hr,
        integration_substeps=args.integration_substeps,
        shifted=True,
        seed=args.seed,
    )

    X = sample_ellipsoid(rng, args.n_samples, P_PAPER, args.rho)
    U = np.column_stack([
        rng.uniform(args.ca0_dev_min, args.ca0_dev_max, size=args.n_samples),
        rng.uniform(args.q_dev_min, args.q_dev_max, size=args.n_samples),
    ])
    XU = np.column_stack([X, U]).astype(np.float32)
    Y = np.array(
        [integrate_one_step(sim, XU[i, :2], XU[i, 2:], args.integrator) for i in range(args.n_samples)],
        dtype=np.float32,
    )

    perm = rng.permutation(args.n_samples)
    n_train = int(round(0.525 * args.n_samples))
    n_val = int(round(0.175 * args.n_samples))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    x_mean = XU[train_idx].mean(axis=0)
    x_std = np.maximum(XU[train_idx].std(axis=0), 1e-12)
    y_mean = Y[train_idx].mean(axis=0)
    y_std = np.maximum(Y[train_idx].std(axis=0), 1e-12)

    V = np.einsum("ni,ij,nj->n", X, P_PAPER, X)
    X_abs = X + sim.xs_abs
    U_abs = U + sim.us_abs
    sensitivity = sensitivity_diagnostics(
        sim, XU, Y, train_idx, n=args.sensitivity_samples, seed=args.seed + 101
    )

    out = Path(args.out)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        XU=XU,
        Y=Y,
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
        P=P_PAPER.astype(np.float32),
    )

    metadata = {
        "dataset": str(out),
        "n_samples": int(args.n_samples),
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "train_fraction": float(len(train_idx) / args.n_samples),
            "val_fraction": float(len(val_idx) / args.n_samples),
            "test_fraction": float(len(test_idx) / args.n_samples),
        },
        "plant": {
            "coordinate": "shifted",
            "state": ["CA-CAs", "T-Ts"],
            "input": ["CA0-CA0s", "Q-Qs"],
            "dt_hr": float(args.dt_hr),
            "integration_substeps": int(args.integration_substeps),
            "integrator": args.integrator,
            "xs_abs": sim.xs_abs.astype(float).tolist(),
            "us_abs": sim.us_abs.astype(float).tolist(),
        },
        "sampling": {
            "rho": float(args.rho),
            "P": P_PAPER.astype(float).tolist(),
            "input_box_shifted": {
                "CA0_dev": [float(args.ca0_dev_min), float(args.ca0_dev_max)],
                "Q_dev": [float(args.q_dev_min), float(args.q_dev_max)],
            },
            "V_summary": summarize(V),
            "X_shifted_summary": summarize(X),
            "X_abs_summary": summarize(X_abs),
            "U_shifted_summary": summarize(U),
            "U_abs_summary": summarize(U_abs),
            "Y_shifted_summary": summarize(Y),
            "Y_abs_summary": summarize(Y + sim.xs_abs),
        },
        "scaler": {
            "x_mean": x_mean.astype(float).tolist(),
            "x_std": x_std.astype(float).tolist(),
            "y_mean": y_mean.astype(float).tolist(),
            "y_std": y_std.astype(float).tolist(),
        },
        "sensitivity": sensitivity,
        "caveat": "The LCNN paper omits the exact input-constraint set U; this dataset records the configurable U box used here.",
    }
    meta_path = out.with_suffix(".json")
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps({
        "dataset": str(out),
        "metadata": str(meta_path),
        "split": metadata["split"],
        "V_max": metadata["sampling"]["V_summary"]["max"],
        "X_abs_summary": metadata["sampling"]["X_abs_summary"],
        "U_abs_summary": metadata["sampling"]["U_abs_summary"],
        "Y_abs_summary": metadata["sampling"]["Y_abs_summary"],
        "normalized_jacobian": sensitivity["normalized_step_jacobian_spectral"],
    }, indent=2))


if __name__ == "__main__":
    main()
