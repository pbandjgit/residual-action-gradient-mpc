#!/usr/bin/env python3
"""D's `lambda_jac` grid extension -- follow-up to the frozen go/no-go
ablation protocol (`docs/JPC_LAMJAC_GRID_EXTENSION_PROTOCOL_2026_09_03.md`).

Extends `alib.LAM_JAC_GRID` with `{1.0, 2.0, 5.0}` (30 new models: 10
seeds x 3 lambda), selects the final `lambda_jac` over the COMBINED
9-point grid using validation MSE only (never test MSE / closed-loop
success / oracle metrics), and reports `BOUNDARY_PERSISTS` if `5.0` is
selected again. Never modifies `results/ablation_execution_2026_08_31/`
or any of its frozen manifests -- all new artifacts live under
`results/ablation_lamjac_extension_2026_09_03/` and a new Freeze Manifest
that references (does not replace) the original one.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cstr.ablation_io as aio      # noqa: E402
import cstr.ablation_lib as alib    # noqa: E402
import ablation_driver_2026_08_31 as orig  # noqa: E402

RESULTS_DIR = ROOT / "results"
NAMESPACE = RESULTS_DIR / "ablation_lamjac_extension_2026_09_03"
SMOKE_RESULTS_PATH = RESULTS_DIR / "ablation_lamjac_extension_smoke_results_2026_09_03.json"
FREEZE_MANIFEST_PATH = ROOT / "docs" / "JPC_LAMJAC_GRID_EXTENSION_FREEZE_MANIFEST_2026_09_03.json"
PROTOCOL_DOC = ROOT / "docs" / "JPC_LAMJAC_GRID_EXTENSION_PROTOCOL_2026_09_03.md"
THIS_FILE = ROOT / "scripts" / "ablation_lamjac_extension_2026_09_03.py"

EXTENSION_GRID = [1.0, 2.0, 5.0]
ORIGINAL_FREEZE_MANIFEST_PATH = orig.FREEZE_MANIFEST_PATH
ORIGINAL_NAMESPACE = orig.NAMESPACE

REQUIRED_FREEZE_FILES = [
    PROTOCOL_DOC,
    THIS_FILE,
    SMOKE_RESULTS_PATH,
]

EXPECTED_FREEZE_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "file_sha256", "smoke_n_pass", "smoke_n_total",
    "predecessor_original_freeze_sha256",
}
EXPECTED_SEED_RECORD_KEYS = {
    "seed", "old_trace", "new_trace", "merged_selected_lambda_jac",
    "boundary_outcome", "init_state_hash_consistent_with_original",
    "new_candidate_closed_loop_success", "new_candidate_oracle",
    "selected_candidate_closed_loop_success", "selected_candidate_test_mse",
}
EXPECTED_RESULTS_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_freeze_sha256",
    "predecessor_original_freeze_sha256", "predecessor_original_query_manifest_sha256",
    "extension_grid", "per_seed", "overall_boundary_outcome",
    "d_extended_vs_e_contrast", "d_extended_vs_e_permutation",
}


def verify_freeze_manifest(path: Path = FREEZE_MANIFEST_PATH) -> str:
    self_hash, payload = aio.read_and_verify_json_artifact(
        path, expected_kind="lamjac_extension_freeze_manifest", expected_keys=EXPECTED_FREEZE_PAYLOAD_KEYS)
    expected_keys = {str(p.relative_to(ROOT)) for p in REQUIRED_FREEZE_FILES}
    actual_keys = set(payload["file_sha256"].keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"FREEZE_KEYSET_MISMATCH: manifest covers {sorted(actual_keys)}, expected {sorted(expected_keys)}"
        )
    for rel_path, expected_hash in payload["file_sha256"].items():
        actual = aio.file_sha256(ROOT / rel_path)
        if actual != expected_hash:
            raise RuntimeError(f"FREEZE_DRIFT: {rel_path} hash changed since freeze ({expected_hash} -> {actual})")
    current_original = orig.verify_freeze_manifest(ORIGINAL_FREEZE_MANIFEST_PATH)
    if current_original != payload["predecessor_original_freeze_sha256"]:
        raise RuntimeError(
            "ORIGINAL_FREEZE_CHAIN_STALE: the original 2026-08-31 Freeze Manifest no longer "
            "matches what this extension's own Freeze Manifest recorded as its predecessor"
        )
    return self_hash


def build_freeze_manifest(dest_path: Path = FREEZE_MANIFEST_PATH) -> Path:
    if not SMOKE_RESULTS_PATH.exists():
        raise RuntimeError(f"cannot freeze: {SMOKE_RESULTS_PATH} does not exist -- run the smoke suite first")
    smoke = json.loads(SMOKE_RESULTS_PATH.read_text())
    if not smoke.get("all_passed"):
        raise RuntimeError(f"cannot freeze: smoke suite reports all_passed=False ({smoke.get('n_pass')}/{smoke.get('n_total')})")
    checks = smoke.get("checks")
    n_pass_actual = sum(1 for c in checks if c.get("passed") is True)
    if n_pass_actual != smoke.get("n_pass") or len(checks) != smoke.get("n_total"):
        raise RuntimeError("cannot freeze: smoke results file is internally inconsistent")
    original_freeze_hash = orig.verify_freeze_manifest(ORIGINAL_FREEZE_MANIFEST_PATH)
    file_hashes = {str(p.relative_to(ROOT)): aio.file_sha256(p) for p in REQUIRED_FREEZE_FILES}
    for p in REQUIRED_FREEZE_FILES:
        if not p.exists():
            raise RuntimeError(f"cannot freeze: required file missing: {p}")
    payload = dict(
        kind="lamjac_extension_freeze_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        file_sha256=file_hashes,
        smoke_n_pass=smoke["n_pass"], smoke_n_total=smoke["n_total"],
        predecessor_original_freeze_sha256=original_freeze_hash,
    )
    aio.write_with_self_verification(dest_path, payload)
    return dest_path


def load_original_frozen_inputs(freeze_manifest_path: Path = ORIGINAL_FREEZE_MANIFEST_PATH):
    """Reconstructs pool/clean/scale/sim EXACTLY as `run_real_protocol`
    does, and verifies the reconstruction against the original frozen M1
    -- proving byte-identical inputs to the original 60-model run."""
    orig.verify_freeze_manifest(freeze_manifest_path)
    import budget_sweep_solver as bs  # noqa: PLC0415
    import probe_action_gradient as pr  # noqa: PLC0415

    sim = bs.SIM
    pool = alib.load_pool()
    clean = alib.build_clean_query_table(pool, sim, pr.precompute_phi)
    scale = alib.shared_scale_constants(pool, clean)

    m1_path = ORIGINAL_NAMESPACE / "ablation_query_manifest_2026_08_31.json"
    m1_verified = orig.verify_query_manifest(m1_path, pool=pool, clean=clean,
                                              freeze_manifest_path=freeze_manifest_path)
    return dict(pool=pool, clean=clean, scale=scale, sim=sim, m1_self_hash=m1_verified["self_hash"])


def load_original_seed_trace(seed: int, m1_self_hash: str, pool: dict, clean: dict, scale: dict) -> dict:
    m2_path = ORIGINAL_NAMESPACE / f"ablation_training_manifest_seed{seed}_2026_08_31.json"
    m2_verified = orig.verify_training_manifest(m2_path, m1_self_hash, pool, clean, scale, seed)
    payload = m2_verified["payload"]
    return dict(
        old_trace=payload["lambda_jac_grid_trace"],
        old_selected=payload["lambda_jac_selected"],
        init_state_hash=payload["init_state_hashes"]["D"],
    )


def _select_over_merged_grid(old_trace: list, new_trace: list) -> float:
    """Same rule as `alib.select_lambda_jac`: minimum validation MSE,
    deterministic tie-break by smallest lambda. Never looks at test MSE,
    closed-loop success, or oracle metrics."""
    merged = list(old_trace) + list(new_trace)
    min_mse = min(t["val_mse"] for t in merged)
    candidates = sorted(t["lam_jac"] for t in merged if t["val_mse"] == min_mse)
    return candidates[0]


def _closed_loop_success_rate(seq, norm, un_lo, un_hi) -> float:
    import budget_sweep_solver as bs  # noqa: PLC0415
    successes = [
        alib.safe_closed_loop_success(bs.closed_loop, seq, norm, x0, 3, 20, 120, 0.1, 0.01, 2.0, un_lo, un_hi)
        for x0 in bs.ICS
    ]
    return float(np.mean(successes))


def run_seed_extension(seed: int, pool: dict, clean: dict, scale: dict, sim, m1_self_hash: str,
                        checkpoint_dir: Path, oracle_idx: np.ndarray, grid: list = None) -> dict:
    from disambiguate_landscape_vs_gradient import make_norm  # noqa: PLC0415
    import budget_sweep_solver as bs  # noqa: PLC0415

    if sim is not bs.SIM:
        raise RuntimeError("SIMULATOR_IDENTITY_MISMATCH: sim passed to run_seed_extension is not budget_sweep_solver.SIM")

    grid = grid if grid is not None else EXTENSION_GRID
    original = load_original_seed_trace(seed, m1_self_hash, pool, clean, scale)

    data_tuple = (pool["XU"], pool["Y"], pool["xm"], pool["xs"], pool["ym"], pool["ys"],
                  pool["train_idx"], pool["test_idx"])
    norm, un_lo, un_hi = make_norm(data_tuple)

    new_trace = []
    new_models = {}
    init_hash_consistent = True
    for lam in grid:
        r = alib.train_base_condition("D", seed, pool, clean, scale, lam_jac=lam,
                                       checkpoint_dir=checkpoint_dir, checkpoint_prefix=f"D_lamjac{lam}_ext")
        if r["init_state_hash"] != original["init_state_hash"]:
            init_hash_consistent = False
        new_trace.append(dict(lam_jac=lam, val_mse=r["val_mse"], test_mse=r["test_mse"],
                              checkpoints=r["checkpoints"]))
        new_models[lam] = r["model"]

    selected_lambda = _select_over_merged_grid(original["old_trace"], new_trace)
    boundary_outcome = "BOUNDARY_PERSISTS" if selected_lambda == max(grid) else "INTERIOR_OPTIMUM_FOUND"

    new_candidate_closed_loop_success = {}
    new_candidate_oracle = {}
    new_candidate_test_mse = {}
    for lam, model in new_models.items():
        seq = alib.freeze_any(model)
        new_candidate_closed_loop_success[lam] = _closed_loop_success_rate(seq, norm, un_lo, un_hi)
        new_candidate_oracle[lam] = alib.diagnose_fixed_idx(model, pool, sim, oracle_idx)
        new_candidate_test_mse[lam] = next(t["test_mse"] for t in new_trace if t["lam_jac"] == lam)

    if selected_lambda in new_models:
        selected_closed_loop = new_candidate_closed_loop_success[selected_lambda]
        selected_test_mse = new_candidate_test_mse[selected_lambda]
    else:
        # Selected lambda is one of the original 6 -- the extension did
        # not change the outcome; original per-seed values already exist
        # in the frozen M2 (`lambda_jac_full_grid_closed_loop`/`test_mse`).
        m2_path = ORIGINAL_NAMESPACE / f"ablation_training_manifest_seed{seed}_2026_08_31.json"
        m2_payload = json.loads(m2_path.read_text())
        selected_closed_loop = m2_payload["lambda_jac_full_grid_closed_loop"][str(selected_lambda)]
        selected_test_mse = None
        for t in original["old_trace"]:
            if t["lam_jac"] == selected_lambda:
                selected_test_mse = t["test_mse"]

    return dict(
        seed=seed,
        old_trace=original["old_trace"],
        new_trace=new_trace,
        merged_selected_lambda_jac=selected_lambda,
        boundary_outcome=boundary_outcome,
        init_state_hash_consistent_with_original=init_hash_consistent,
        new_candidate_closed_loop_success=new_candidate_closed_loop_success,
        new_candidate_oracle=new_candidate_oracle,
        selected_candidate_closed_loop_success=selected_closed_loop,
        selected_candidate_test_mse=selected_test_mse,
    )


def build_results_manifest(dest_path: Path, per_seed: dict, e_closed_loop_success: dict,
                           freeze_self_hash: str, original_freeze_hash: str, m1_self_hash: str) -> dict:
    seeds = sorted(per_seed.keys())
    outcomes = {per_seed[s]["boundary_outcome"] for s in seeds}
    overall_outcome = "BOUNDARY_PERSISTS" if outcomes == {"BOUNDARY_PERSISTS"} else (
        "MIXED" if len(outcomes) > 1 else "INTERIOR_OPTIMUM_FOUND")

    d_ext = np.array([per_seed[s]["selected_candidate_closed_loop_success"] for s in seeds])
    e_vals = np.array([e_closed_loop_success[s] for s in seeds])
    diffs = e_vals - d_ext
    ci = alib.bootstrap_ci(diffs)
    perm = alib.exact_permutation_pvalue(diffs)

    resample_idx = ci.pop("resample_idx")
    resample_path = dest_path.parent / "ablation_lamjac_extension_resample_2026_09_03.npz"
    aio.write_npz_verify_if_exists(resample_path, dict(resample_idx=resample_idx, diffs=diffs))

    per_seed_jsonable = {}
    for s in seeds:
        rec = dict(per_seed[s])
        rec["new_candidate_closed_loop_success"] = {str(k): v for k, v in rec["new_candidate_closed_loop_success"].items()}
        rec["new_candidate_oracle"] = {str(k): v for k, v in rec["new_candidate_oracle"].items()}
        per_seed_jsonable[str(s)] = rec

    payload = dict(
        kind="lamjac_extension_results_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_freeze_sha256=freeze_self_hash,
        predecessor_original_freeze_sha256=original_freeze_hash,
        predecessor_original_query_manifest_sha256=m1_self_hash,
        extension_grid=EXTENSION_GRID,
        per_seed=per_seed_jsonable,
        overall_boundary_outcome=overall_outcome,
        d_extended_vs_e_contrast=ci,
        d_extended_vs_e_permutation=perm,
    )
    self_hash = aio.write_with_self_verification(dest_path, payload)
    return dict(path=dest_path, self_hash=self_hash, payload=payload)


def run_real_extension(out_dir: Path = NAMESPACE, freeze_manifest_path: Path = FREEZE_MANIFEST_PATH) -> dict:
    freeze_self_hash = verify_freeze_manifest(freeze_manifest_path)
    original_freeze_hash = orig.verify_freeze_manifest(ORIGINAL_FREEZE_MANIFEST_PATH)

    loaded = load_original_frozen_inputs(ORIGINAL_FREEZE_MANIFEST_PATH)
    pool, clean, scale, sim, m1_self_hash = (loaded["pool"], loaded["clean"], loaded["scale"],
                                              loaded["sim"], loaded["m1_self_hash"])
    oracle_idx = alib.fixed_oracle_idx(pool["test_idx"])

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"

    per_seed = {}
    e_closed_loop_success = {}
    for seed in alib.SEEDS:
        t0 = time.monotonic()
        per_seed[seed] = run_seed_extension(seed, pool, clean, scale, sim, m1_self_hash,
                                             checkpoint_dir, oracle_idx)
        m2_path = ORIGINAL_NAMESPACE / f"ablation_training_manifest_seed{seed}_2026_08_31.json"
        m2_payload = json.loads(m2_path.read_text())
        e_closed_loop_success[seed] = m2_payload["closed_loop_success"]["E"]
        print(f"seed {seed}: selected_lambda={per_seed[seed]['merged_selected_lambda_jac']} "
              f"outcome={per_seed[seed]['boundary_outcome']} "
              f"({time.monotonic()-t0:.1f}s)", flush=True)

    dest_path = out_dir / "ablation_lamjac_extension_results_2026_09_03.json"
    result = build_results_manifest(dest_path, per_seed, e_closed_loop_success,
                                     freeze_self_hash, original_freeze_hash, m1_self_hash)
    return result


if __name__ == "__main__":
    r = run_real_extension()
    print(json.dumps(dict(overall_boundary_outcome=r["payload"]["overall_boundary_outcome"],
                          d_extended_vs_e_contrast=r["payload"]["d_extended_vs_e_contrast"],
                          d_extended_vs_e_permutation=r["payload"]["d_extended_vs_e_permutation"]),
                     indent=2, default=str))
