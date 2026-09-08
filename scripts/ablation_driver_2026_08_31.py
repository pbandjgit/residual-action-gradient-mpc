#!/usr/bin/env python3
"""Driver / manifest-chain orchestration for the JPC go/no-go ablation
protocol (`docs/JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_2026_08_31.md`,
v7, Section 10). Implements the 4-stage manifest chain (Freeze -> Query
Manifest M1 -> Training Manifest M2 -> Results Manifest M3) AND the real
per-seed/full-protocol orchestration that produces them, including
resume-or-verify (Section 10, M2's own text): a seed whose M2 already
exists and verifies is never retrained -- its persisted M2 payload is the
sole source of truth for aggregation, never in-memory state from a prior
process.

Nothing in this module is called against real data by importing it --
`build_freeze_manifest()` and `run_full_protocol()` are the only
functions with real side effects, and both are gated (the former on a
passing smoke suite, the latter on a verified freeze). No entry point
here has been invoked for real; this file exists to be reviewed alongside
the library it orchestrates, per Section 0's design-review -> implement
-> smoke -> freeze -> real-execution sequence.
"""
from __future__ import annotations

import hashlib
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

RESULTS_DIR = ROOT / "results"
NAMESPACE = RESULTS_DIR / "ablation_execution_2026_08_31"
SMOKE_RESULTS_PATH = RESULTS_DIR / "ablation_synthetic_smoke_results_2026_08_31.json"
FREEZE_MANIFEST_PATH = ROOT / "docs" / "JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_FREEZE_MANIFEST_2026_08_31.json"

REQUIRED_FREEZE_FILES = [
    ROOT / "docs" / "JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_2026_08_31.md",
    ROOT / "src" / "cstr" / "ablation_lib.py",
    ROOT / "src" / "cstr" / "ablation_io.py",
    ROOT / "src" / "cstr" / "models_onestep.py",
    ROOT / "src" / "cstr" / "lcnn_paper_simulator.py",
    ROOT / "src" / "cstr" / "sontag.py",
    ROOT / "scripts" / "budget_sweep_solver.py",
    ROOT / "scripts" / "probe_action_gradient.py",
    ROOT / "scripts" / "disambiguate_landscape_vs_gradient.py",
    ROOT / "scripts" / "ablation_synthetic_smoke_2026_08_31.py",
    ROOT / "scripts" / "ablation_driver_2026_08_31.py",
    SMOKE_RESULTS_PATH,
]

FNN_CONDITIONS = ("B-FNN", "E-FNN")
LCNN_CONDITIONS = ("A", "B", "C", "D", "E", "F")
SENSITIVITY_CONDITIONS = ("B", "C")

# Exact payload keysets for each manifest stage (Section 10's "each stage
# hash-locks its own exact keyset" promise) -- kept in one place, next to
# the `dict(...)` construction each corresponds to, so the two stay in
# sync by inspection whenever either is edited.
EXPECTED_FREEZE_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "file_sha256", "smoke_n_pass", "smoke_n_total",
}
EXPECTED_QUERY_MANIFEST_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_freeze_sha256", "pool_content_sha256",
    "clean_query_table_path", "clean_query_table_sha256", "clean_query_table_content_sha256",
    "clean_query_table_fields", "n_train", "n_val", "n_test",
    "valid_pair_counts", "query_counts_per_condition",
}
EXPECTED_TRAINING_MANIFEST_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_query_manifest_sha256", "seed",
    "rng_stream_mapping", "scale_constants", "noise_realized_table_path",
    "noise_realized_table_sha256", "init_state_hashes", "lcnn_init_hash_consistent",
    "checkpoint_sha256", "val_mse", "test_mse", "lambda_jac_grid_trace",
    "lambda_jac_selected", "lambda_jac_full_grid_closed_loop", "mse_match_outcome",
    "mse_match_selected_test_mse", "mse_match_attempt_log", "closed_loop_success",
    "oracle_summary", "sensitivity_arm", "wall_clock_seconds", "optimizer_steps",
}
EXPECTED_RESULTS_MANIFEST_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_training_manifest_sha256",
    "oracle_fixed_idx_path", "oracle_fixed_idx_sha256", "raw_metrics_path",
    "raw_metrics_sha256", "raw_resample_arrays_path", "raw_resample_arrays_sha256",
    "primary_contrast", "secondary_family", "lambda_jac_full_grid_contrasts",
    "mse_attribution_detail", "outcome",
}


# ---------------------------------------------------------------------------
# Stage 1: Freeze Manifest
# ---------------------------------------------------------------------------

def _validate_smoke_results_internal_consistency(smoke: dict) -> None:
    checks = smoke.get("checks")
    if not isinstance(checks, list) or not checks:
        raise RuntimeError("cannot freeze: smoke results file has no 'checks' list")
    n_pass_actual = sum(1 for c in checks if c.get("passed") is True)
    if n_pass_actual != smoke.get("n_pass"):
        raise RuntimeError(
            f"cannot freeze: smoke results file is internally inconsistent -- "
            f"recorded n_pass={smoke.get('n_pass')} but {n_pass_actual} checks actually have passed=true"
        )
    if len(checks) != smoke.get("n_total"):
        raise RuntimeError(
            f"cannot freeze: smoke results file is internally inconsistent -- "
            f"recorded n_total={smoke.get('n_total')} but checks list has {len(checks)} entries"
        )
    if smoke.get("all_passed") != (n_pass_actual == len(checks)):
        raise RuntimeError("cannot freeze: smoke results file's all_passed flag contradicts its own checks list")


def build_freeze_manifest(dest_path: Path = FREEZE_MANIFEST_PATH) -> Path:
    if not SMOKE_RESULTS_PATH.exists():
        raise RuntimeError(f"cannot freeze: {SMOKE_RESULTS_PATH} does not exist -- run the smoke suite first")
    smoke = json.loads(SMOKE_RESULTS_PATH.read_text())
    if not smoke.get("all_passed"):
        raise RuntimeError(
            f"cannot freeze: synthetic smoke suite reports all_passed=False "
            f"({smoke.get('n_pass')}/{smoke.get('n_total')})"
        )
    _validate_smoke_results_internal_consistency(smoke)
    file_hashes = {}
    for p in REQUIRED_FREEZE_FILES:
        if not p.exists():
            raise RuntimeError(f"cannot freeze: required file missing: {p}")
        file_hashes[str(p.relative_to(ROOT))] = aio.file_sha256(p)
    payload = dict(
        kind="ablation_freeze_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        file_sha256=file_hashes,
        smoke_n_pass=smoke["n_pass"], smoke_n_total=smoke["n_total"],
    )
    aio.write_with_self_verification(dest_path, payload)
    return dest_path


def verify_freeze_manifest(path: Path = FREEZE_MANIFEST_PATH) -> str:
    """Returns the manifest's own self_hash on success. Rejects a
    manifest whose `file_sha256` keyset does not EXACTLY match the
    CURRENT `REQUIRED_FREEZE_FILES` set."""
    self_hash, payload = aio.read_and_verify_json_artifact(
        path, expected_kind="ablation_freeze_manifest", expected_keys=EXPECTED_FREEZE_PAYLOAD_KEYS)
    expected_keys = {str(p.relative_to(ROOT)) for p in REQUIRED_FREEZE_FILES}
    actual_keys = set(payload["file_sha256"].keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"FREEZE_KEYSET_MISMATCH: manifest covers {sorted(actual_keys)} but the current "
            f"REQUIRED_FREEZE_FILES set is {sorted(expected_keys)} "
            f"(missing={sorted(expected_keys - actual_keys)}, extra={sorted(actual_keys - expected_keys)})"
        )
    for rel_path, expected_hash in payload["file_sha256"].items():
        actual = aio.file_sha256(ROOT / rel_path)
        if actual != expected_hash:
            raise RuntimeError(f"FREEZE_DRIFT: {rel_path} hash changed since freeze ({expected_hash} -> {actual})")
    return self_hash


# ---------------------------------------------------------------------------
# Stage 2: Query Manifest (M1)
# ---------------------------------------------------------------------------

def build_query_manifest(pool: dict, clean: dict, out_dir: Path,
                          freeze_manifest_path: Path = FREEZE_MANIFEST_PATH) -> dict:
    """The predecessor freeze hash is obtained by INTERNALLY calling
    `verify_freeze_manifest` (never accepted as a caller-supplied string).
    Hash-locks the ACTUAL `CLEAN_QUERY_TABLE` array content (a companion
    `.npz`) AND the actual `pool` array content passed in (via `aio.
    pool_content_sha256`, NOT the on-disk pool file's hash, which could
    silently diverge from what was actually used for training)."""
    freeze_hash = verify_freeze_manifest(freeze_manifest_path)
    counts = alib.query_counts_per_condition(pool, clean)
    pair_counts = alib.valid_pair_counts(clean)

    array_fields = {k: v for k, v in clean.items() if isinstance(v, np.ndarray)}
    npz_path = out_dir / "ablation_clean_query_table_2026_08_31.npz"
    # verify-if-exists: a crash between this write and the JSON below
    # (which is what actually marks M1 "complete", per run_full_protocol's
    # own m1_path.exists() check) must not make a retry crash on a stale
    # npz from the interrupted attempt.
    clean_query_table_sha256 = aio.write_npz_verify_if_exists(npz_path, array_fields)
    clean_query_table_content_sha256 = aio.dict_arrays_sha256(clean)
    pool_content_sha256 = aio.pool_content_sha256(pool)

    payload = dict(
        kind="ablation_query_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_freeze_sha256=freeze_hash,
        pool_content_sha256=pool_content_sha256,
        clean_query_table_path=str(npz_path.relative_to(ROOT)),
        clean_query_table_sha256=clean_query_table_sha256,
        clean_query_table_content_sha256=clean_query_table_content_sha256,
        clean_query_table_fields=sorted(array_fields.keys()),
        n_train=int(len(pool["train_idx"])), n_val=int(len(pool["val_idx"])), n_test=int(len(pool["test_idx"])),
        valid_pair_counts=pair_counts,
        query_counts_per_condition=counts,
    )
    path = out_dir / "ablation_query_manifest_2026_08_31.json"
    return dict(path=path, self_hash=aio.write_with_self_verification(path, payload))


def verify_query_manifest(m1_path: Path, pool: dict = None, clean: dict = None,
                          freeze_manifest_path: Path = None) -> dict:
    """Corrected (P0): also verifies (a) the freshly-passed `clean` dict's
    CONTENT matches what M1 recorded (not just the on-disk npz file's
    byte-hash, which only catches file tampering, not "a different `clean`
    object was passed on resume"), and (b) the freeze chain is STILL
    currently valid by calling `verify_freeze_manifest` live and
    confirming its hash matches M1's recorded predecessor -- a stale or
    since-tampered freeze manifest must not be silently trusted just
    because M1 once recorded its hash."""
    self_hash, payload = aio.read_and_verify_json_artifact(
        m1_path, expected_kind="ablation_query_manifest", expected_keys=EXPECTED_QUERY_MANIFEST_PAYLOAD_KEYS)
    npz_path = ROOT / payload["clean_query_table_path"]
    actual = aio.file_sha256(npz_path)
    if actual != payload["clean_query_table_sha256"]:
        raise RuntimeError(f"QUERY_TABLE_DRIFT: {payload['clean_query_table_path']} hash changed since M1 was built")
    with np.load(npz_path, allow_pickle=True) as npz:
        actual_fields = sorted(npz.files)
    if actual_fields != payload["clean_query_table_fields"]:
        raise RuntimeError(
            f"QUERY_TABLE_KEYSET_MISMATCH: {payload['clean_query_table_path']} has fields {actual_fields}, "
            f"but M1 recorded {payload['clean_query_table_fields']}"
        )
    if pool is not None:
        actual_pool_hash = aio.pool_content_sha256(pool)
        if actual_pool_hash != payload["pool_content_sha256"]:
            raise RuntimeError("POOL_CONTENT_MISMATCH: the pool passed to this call differs from the one M1 recorded")
    if clean is not None:
        actual_clean_hash = aio.dict_arrays_sha256(clean)
        if actual_clean_hash != payload["clean_query_table_content_sha256"]:
            raise RuntimeError(
                "CLEAN_TABLE_CONTENT_MISMATCH: the clean-query-table object passed to this call "
                "differs from the one M1 recorded, even though the on-disk npz file is unchanged"
            )
    if freeze_manifest_path is not None:
        current_freeze_hash = verify_freeze_manifest(freeze_manifest_path)
        if current_freeze_hash != payload["predecessor_freeze_sha256"]:
            raise RuntimeError(
                "FREEZE_CHAIN_STALE: the current Freeze Manifest's hash no longer matches what "
                "M1 recorded as its predecessor -- the freeze may have been rebuilt or replaced since M1 was built"
            )
    return dict(self_hash=self_hash, payload=payload)


# ---------------------------------------------------------------------------
# Stage 3: real per-seed training/evaluation + Training Manifest (M2)
# ---------------------------------------------------------------------------

def _closed_loop_success_rate(seq, norm, un_lo, un_hi) -> float:
    import budget_sweep_solver as bs  # noqa: PLC0415
    successes = [
        alib.safe_closed_loop_success(bs.closed_loop, seq, norm, x0, 3, 20, 120, 0.1, 0.01, 2.0, un_lo, un_hi)
        for x0 in bs.ICS
    ]
    return float(np.mean(successes))


def run_seed(seed: int, pool: dict, clean: dict, scale: dict, checkpoint_dir: Path,
             oracle_idx: np.ndarray, sim) -> dict:
    """Real per-seed training for every condition (A-F, B-FNN, E-FNN),
    the pre-registered non-gating noisy-augmented SENSITIVITY ARM for B/C
    (Section 2.5), the lambda_Jac full grid (Section 5.3), MSE-matching
    (Section 8), the fixed-oracle diagnostics (Section 6), and REAL
    closed-loop success rates at the primary `horizon=3, budget=20`
    endpoint (Section 1) via `budget_sweep_solver.closed_loop` +
    `freeze_any`. Records wall-clock and optimizer-step counts per
    condition for Section 8.6's disclosure requirement."""
    import budget_sweep_solver as bs  # noqa: PLC0415
    from disambiguate_landscape_vs_gradient import make_norm  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if sim is not bs.SIM:
        raise RuntimeError(
            "SIMULATOR_IDENTITY_MISMATCH: the `sim` passed to run_seed (used for training/oracle) is "
            "not the SAME object as `budget_sweep_solver.SIM` (used internally by the real closed_loop() "
            "call, which cannot accept a `sim` parameter). Training a model on one plant while evaluating "
            "its closed-loop success on a different plant would silently produce meaningless results. "
            "Use `run_real_protocol()` for production runs (fixes this by construction), or explicitly "
            "set `budget_sweep_solver.SIM = sim` before calling run_seed/run_full_protocol directly."
        )

    data_tuple = (pool["XU"], pool["Y"], pool["xm"], pool["xs"], pool["ym"], pool["ys"],
                  pool["train_idx"], pool["test_idx"])
    norm, un_lo, un_hi = make_norm(data_tuple)

    results = {}
    wall_clock = {}
    optimizer_steps = {}
    n_tr = len(pool["train_idx"])
    steps_per_epoch_base = -(-n_tr // alib.BATCH)
    n_aug = alib.query_counts_per_condition(pool, clean)["B"] - 2 * n_tr
    steps_per_epoch_aug = -(-(n_tr + n_aug) // alib.BATCH)

    def _train_base(condition, **kw):
        t0 = time.monotonic()
        r = alib.train_base_condition(condition, seed, pool, clean, scale, checkpoint_dir=checkpoint_dir, **kw)
        wall_clock[condition] = time.monotonic() - t0
        return r

    def _train_aug(condition, **kw):
        t0 = time.monotonic()
        r = alib.train_augmented_condition(condition, seed, pool, clean, scale, checkpoint_dir=checkpoint_dir, **kw)
        wall_clock[condition] = time.monotonic() - t0
        return r

    results["A"] = _train_base("A"); optimizer_steps["A"] = alib.EPOCHS * steps_per_epoch_base
    results["E"] = _train_base("E"); optimizer_steps["E"] = alib.EPOCHS * steps_per_epoch_base
    results["F"] = _train_base("F"); optimizer_steps["F"] = alib.EPOCHS * steps_per_epoch_base

    t0 = time.monotonic()
    lam_jac_result = alib.select_lambda_jac(seed, pool, clean, scale, checkpoint_dir=checkpoint_dir)
    results["D"] = lam_jac_result["result"]
    wall_clock["D"] = time.monotonic() - t0
    optimizer_steps["D"] = alib.EPOCHS * steps_per_epoch_base * len(alib.LAM_JAC_GRID)

    results["B"] = _train_aug("B"); optimizer_steps["B"] = alib.EPOCHS * steps_per_epoch_aug
    results["C"] = _train_aug("C"); optimizer_steps["C"] = alib.EPOCHS * steps_per_epoch_aug
    results["B-FNN"] = _train_aug("B-FNN", use_fnn=True, checkpoint_prefix="BFNN")
    optimizer_steps["B-FNN"] = alib.EPOCHS * steps_per_epoch_aug
    results["E-FNN"] = _train_base("E-FNN", use_fnn=True)
    optimizer_steps["E-FNN"] = alib.EPOCHS * steps_per_epoch_base

    # Section 2.5 sensitivity arm: non-primary, non-gating, reported only.
    sensitivity_results = {}
    sensitivity_wall_clock = {}
    sensitivity_optimizer_steps = {}
    for condition in SENSITIVITY_CONDITIONS:
        t0 = time.monotonic()
        sensitivity_results[condition] = alib.train_augmented_condition(
            condition, seed, pool, clean, scale, sensitivity=True,
            checkpoint_dir=checkpoint_dir, checkpoint_prefix=f"{condition}_sensitivity")
        sensitivity_wall_clock[condition] = time.monotonic() - t0
        sensitivity_optimizer_steps[condition] = alib.EPOCHS * steps_per_epoch_aug

    mse_match = alib.mse_match_select(seed, pool, clean, scale, results["E"]["val_mse"], results["B"],
                                      checkpoint_dir=checkpoint_dir)

    closed_loop_success = {}
    for condition, r in results.items():
        seq = alib.freeze_any(r["model"])
        closed_loop_success[condition] = _closed_loop_success_rate(seq, norm, un_lo, un_hi)

    mse_match_selected_test_mse = None
    if mse_match["outcome"] == "MATCHED":
        matched_state = torch.load(mse_match["selected"]["path"], map_location="cpu")
        matched_model = alib.build_model("B", use_fnn=False)
        matched_model.load_state_dict(matched_state)
        matched_seq = alib.freeze_any(matched_model)
        closed_loop_success["B_matched"] = _closed_loop_success_rate(matched_seq, norm, un_lo, un_hi)
        mse_match_selected_test_mse = mse_match["selected"]["test_mse"]

    sensitivity_closed_loop_success = {}
    for condition, r in sensitivity_results.items():
        seq = alib.freeze_any(r["model"])
        sensitivity_closed_loop_success[condition] = _closed_loop_success_rate(seq, norm, un_lo, un_hi)

    oracle = {}
    for condition, r in results.items():
        oracle[condition] = alib.diagnose_fixed_idx(r["model"], pool, sim, oracle_idx)

    lam_jac_full_grid_closed_loop = {}
    for lam, r in lam_jac_result["all_results"].items():
        seq = alib.freeze_any(r["model"])
        lam_jac_full_grid_closed_loop[lam] = _closed_loop_success_rate(seq, norm, un_lo, un_hi)

    return dict(
        results=results, sensitivity_results=sensitivity_results, mse_match=mse_match,
        mse_match_selected_test_mse=mse_match_selected_test_mse,
        closed_loop_success=closed_loop_success,
        sensitivity_closed_loop_success=sensitivity_closed_loop_success, oracle=oracle,
        lambda_jac_grid_trace=lam_jac_result["grid_trace"],
        lambda_jac_selected=lam_jac_result["selected_lambda_jac"],
        lambda_jac_full_grid_closed_loop=lam_jac_full_grid_closed_loop,
        wall_clock_seconds=wall_clock, sensitivity_wall_clock_seconds=sensitivity_wall_clock,
        optimizer_steps=optimizer_steps, sensitivity_optimizer_steps=sensitivity_optimizer_steps,
    )


def build_training_manifest(seed: int, out_dir: Path, m1_path: Path, seed_run: dict, scale: dict,
                            pool: dict, clean: dict) -> dict:
    """Includes the 7-item shared scaling keyset (incl. `disp`), the
    per-seed NOISE_REALIZED_TABLE's hash, the FULL `lambda_Jac` grid trace,
    wall-clock/optimizer-step disclosure, the MSE-match fallback
    checkpoint(s) and full attempt log (hash-locked, not just the primary/
    D-grid checkpoints), the sensitivity-arm results, and full oracle
    summary statistics per condition -- everything `aggregate_and_
    evaluate`/`build_results_manifest` need, so a resumed run can
    reconstruct a seed's contribution from THIS file alone, never from
    in-memory state a prior process held."""
    m1_verified = verify_query_manifest(m1_path)
    results = seed_run["results"]

    noise = alib.build_noise_realized_table(pool, clean, seed, sensitivity=False)
    sensitivity_noise = alib.build_noise_realized_table(pool, clean, seed, sensitivity=True)
    noise_npz_path = out_dir / f"ablation_noise_table_seed{seed}_2026_08_31.npz"
    noise_npz_arrays = {f"primary_{k}": v for k, v in noise.items() if isinstance(v, np.ndarray)}
    noise_npz_arrays.update({f"sensitivity_{k}": v for k, v in sensitivity_noise.items() if isinstance(v, np.ndarray)})
    # verify-if-exists: a crash between this write and M2's own JSON
    # (written at the end of this function) must not make a retried
    # run_seed()+build_training_manifest() crash on a stale noise npz.
    noise_table_sha256 = aio.write_npz_verify_if_exists(noise_npz_path, noise_npz_arrays)

    checkpoint_hashes = {}
    for condition, r in results.items():
        for ckpt in r.get("checkpoints", []):
            checkpoint_hashes[ckpt["path"]] = aio.file_sha256(Path(ckpt["path"]))
    for condition, r in seed_run["sensitivity_results"].items():
        for ckpt in r.get("checkpoints", []):
            checkpoint_hashes[ckpt["path"]] = aio.file_sha256(Path(ckpt["path"]))
    for entry in seed_run["lambda_jac_grid_trace"]:
        for ckpt in entry.get("checkpoints", []):
            checkpoint_hashes[ckpt["path"]] = aio.file_sha256(Path(ckpt["path"]))
    for attempt in seed_run["mse_match"]["attempt_log"]:
        if attempt.get("path"):
            checkpoint_hashes[attempt["path"]] = aio.file_sha256(Path(attempt["path"]))

    init_hashes = {c: r["init_state_hash"] for c, r in results.items()}
    lcnn_hash_set = {init_hashes[c] for c in LCNN_CONDITIONS if c in init_hashes}

    oracle_summary = {c: {k: v for k, v in r.items() if k != "n_points"} for c, r in seed_run["oracle"].items()}

    payload = dict(
        kind="ablation_training_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_query_manifest_sha256=m1_verified["self_hash"],
        seed=seed,
        rng_stream_mapping=dict(base_noise_seed=alib.base_noise_seed(seed),
                                sensitivity_noise_seed=alib.sensitivity_noise_seed(seed),
                                shuffle_seed=alib.shuffle_seed(seed)),
        scale_constants=dict(
            xm=scale["xm"].tolist(), xs=scale["xs"].tolist(), ym=scale["ym"].tolist(), ys=scale["ys"].tolist(),
            g_scale=scale["g_scale"], ug_scale=scale["ug_scale"], disp=noise["disp"].tolist(),
        ),
        noise_realized_table_path=str(noise_npz_path.relative_to(ROOT)),
        noise_realized_table_sha256=noise_table_sha256,
        init_state_hashes=init_hashes,
        lcnn_init_hash_consistent=(len(lcnn_hash_set) <= 1),
        checkpoint_sha256=checkpoint_hashes,
        val_mse={c: r["val_mse"] for c, r in results.items()},
        test_mse={c: r["test_mse"] for c, r in results.items()},
        lambda_jac_grid_trace=[{k: v for k, v in entry.items() if k != "checkpoints"}
                               for entry in seed_run["lambda_jac_grid_trace"]],
        lambda_jac_selected=seed_run["lambda_jac_selected"],
        lambda_jac_full_grid_closed_loop={str(lam): v for lam, v in
                                          seed_run["lambda_jac_full_grid_closed_loop"].items()},
        mse_match_outcome=seed_run["mse_match"]["outcome"],
        mse_match_selected_test_mse=seed_run["mse_match_selected_test_mse"],
        mse_match_attempt_log=seed_run["mse_match"]["attempt_log"],
        closed_loop_success=seed_run["closed_loop_success"],
        oracle_summary=oracle_summary,
        sensitivity_arm=dict(
            val_mse={c: r["val_mse"] for c, r in seed_run["sensitivity_results"].items()},
            test_mse={c: r["test_mse"] for c, r in seed_run["sensitivity_results"].items()},
            closed_loop_success=seed_run["sensitivity_closed_loop_success"],
            wall_clock_seconds=seed_run["sensitivity_wall_clock_seconds"],
            optimizer_steps=seed_run["sensitivity_optimizer_steps"],
        ),
        wall_clock_seconds=seed_run["wall_clock_seconds"],
        optimizer_steps=seed_run["optimizer_steps"],
    )
    if not payload["lcnn_init_hash_consistent"]:
        raise RuntimeError(f"INIT_HASH_MISMATCH: seed {seed}'s LCNN conditions do not share an initial state_dict")
    path = out_dir / f"ablation_training_manifest_seed{seed}_2026_08_31.json"
    return dict(path=path, self_hash=aio.write_with_self_verification(path, payload))


def verify_training_manifest(m2_path: Path, expected_m1_hash: str, pool: dict, clean: dict,
                             scale: dict, seed: int) -> dict:
    """Corrected (P0): a resumed seed's M2 is not trusted on its own
    self-hash and predecessor-M1-hash alone. Re-verifies EVERY checkpoint
    file referenced in `checkpoint_sha256` (existence + hash), the
    noise-realized-table companion npz (existence + hash), and the
    recorded `scale_constants` against the CURRENTLY computed `scale`
    (including `disp`, recomputed fresh from `pool`/`clean`/`seed` -- a
    cheap noise-generation call, not a retrain) -- catching a deleted or
    modified checkpoint/noise-table, or a scale/seed drift, that the JSON
    manifest's own self-hash cannot detect since only the JSON's own
    bytes are covered by that hash."""
    self_hash, payload = aio.read_and_verify_json_artifact(
        m2_path, expected_kind="ablation_training_manifest", expected_keys=EXPECTED_TRAINING_MANIFEST_PAYLOAD_KEYS)
    if payload["predecessor_query_manifest_sha256"] != expected_m1_hash:
        raise RuntimeError(
            f"RESUME_PROVENANCE_MISMATCH: seed {seed}'s existing M2 was built against a "
            f"different Query Manifest than the current M1 -- refusing to silently mix provenance"
        )
    if payload["seed"] != seed:
        raise RuntimeError(f"SEED_MISMATCH: M2 at {m2_path} records seed={payload['seed']}, expected {seed}")
    for path_str, expected_hash in payload["checkpoint_sha256"].items():
        p = Path(path_str)
        if not p.exists():
            raise RuntimeError(f"CHECKPOINT_MISSING: {path_str} referenced by seed {seed}'s M2 no longer exists")
        if aio.file_sha256(p) != expected_hash:
            raise RuntimeError(f"CHECKPOINT_DRIFT: {path_str} hash changed since seed {seed}'s M2 was built")
    noise_p = ROOT / payload["noise_realized_table_path"]
    if not noise_p.exists():
        raise RuntimeError(f"NOISE_TABLE_MISSING: {payload['noise_realized_table_path']} no longer exists")
    if aio.file_sha256(noise_p) != payload["noise_realized_table_sha256"]:
        raise RuntimeError(f"NOISE_TABLE_DRIFT: {payload['noise_realized_table_path']} hash changed since M2 was built")
    # Exact (not allclose) comparison throughout: these are all
    # deterministic recomputations from the same pool/clean/seed inputs,
    # so any REAL divergence (a different scale-computation methodology,
    # a silently different pool/sim) should produce a bit-different
    # value, not merely a value outside some tolerance band -- allclose's
    # default rtol/atol would mask exactly the kind of small-but-real
    # drift this check exists to catch.
    sc = payload["scale_constants"]
    if sc["g_scale"] != scale["g_scale"] or sc["ug_scale"] != scale["ug_scale"]:
        raise RuntimeError(f"SCALE_CONSTANTS_MISMATCH: seed {seed}'s recorded g_scale/ug_scale differ from current")
    for k in ("xm", "xs", "ym", "ys"):
        if not np.array_equal(np.asarray(sc[k], dtype=scale[k].dtype), scale[k]):
            raise RuntimeError(f"SCALE_CONSTANTS_MISMATCH: seed {seed}'s recorded {k} differs from current")
    fresh_noise = alib.build_noise_realized_table(pool, clean, seed, sensitivity=False)
    if not np.array_equal(np.asarray(sc["disp"], dtype=fresh_noise["disp"].dtype), fresh_noise["disp"]):
        raise RuntimeError(f"SCALE_CONSTANTS_MISMATCH: seed {seed}'s recorded disp differs from a fresh recomputation")
    return dict(self_hash=self_hash, payload=payload)


def _seed_summary_from_m2_payload(payload: dict) -> dict:
    """The sole reconstruction path for a RESUMED seed: everything
    `aggregate_and_evaluate`/`build_results_manifest` need, read back from
    a verified M2 payload alone -- never from in-memory state a prior
    process might have held (which would not survive a restart)."""
    return dict(
        closed_loop_success=payload["closed_loop_success"],
        mse_match_outcome=payload["mse_match_outcome"],
        mse_match_selected_test_mse=payload["mse_match_selected_test_mse"],
        test_mse=payload["test_mse"],
        oracle_summary=payload["oracle_summary"],
        lambda_jac_full_grid_closed_loop={float(lam): v for lam, v in
                                          payload["lambda_jac_full_grid_closed_loop"].items()},
        sensitivity_arm=payload["sensitivity_arm"],
    )


# ---------------------------------------------------------------------------
# Stage 4: aggregation + Results Manifest (M3)
# ---------------------------------------------------------------------------

def aggregate_and_evaluate(all_seed_summaries: dict, seeds: list) -> dict:
    """`all_seed_summaries`: {seed: summary dict, either freshly built or
    reconstructed from a verified M2 via `_seed_summary_from_m2_payload`}.
    Builds per-condition success-rate dicts, the primary contrast, the
    7-member secondary family (Section 7.4/7.5), the D full-grid's
    descriptive per-candidate E-vs-lambda contrasts (Section 5.3
    addendum), and the full outcome map (Section 9)."""
    conditions = list(LCNN_CONDITIONS) + list(FNN_CONDITIONS)
    success_rates = {c: {s: all_seed_summaries[s]["closed_loop_success"][c] for s in seeds} for c in conditions}

    primary = alib.evaluate_contrast(alib.paired_success_rate_diff(success_rates["E"], success_rates["B"], seeds))

    mse_match_seed_status = {s: all_seed_summaries[s]["mse_match_outcome"] for s in seeds}
    feasibility = alib.mse_attribution_seed_status(
        {s: dict(outcome=mse_match_seed_status[s]) for s in seeds})
    if feasibility == "MSE_MATCH_INFEASIBLE":
        mse_match_diffs = None
        test_mse_fail_count = 0
    else:
        matched_seeds = [s for s in seeds if mse_match_seed_status[s] == "MATCHED"]
        mse_match_diffs = np.array([
            all_seed_summaries[s]["closed_loop_success"]["E"] - all_seed_summaries[s]["closed_loop_success"]["B_matched"]
            for s in matched_seeds
        ])
        test_mse_fail_count = sum(
            1 for s in matched_seeds
            if alib._relative_diff(all_seed_summaries[s]["mse_match_selected_test_mse"],
                                    all_seed_summaries[s]["test_mse"]["E"]) > alib.MSE_MATCH_TOL
        )

    secondary = alib.compute_secondary_family(success_rates, seeds, mse_match_diffs=mse_match_diffs)
    mse_attribution = alib.classify_mse_attribution(feasibility, secondary["MSE_match"], test_mse_fail_count)
    outcome = alib.evaluate_full_outcome(primary, secondary, mse_attribution)

    # Section 5.3 addendum: descriptive-only per-lambda-candidate E-vs-
    # candidate contrast, never Holm-corrected, never gating.
    lambda_grid = sorted(all_seed_summaries[seeds[0]]["lambda_jac_full_grid_closed_loop"].keys())
    lambda_jac_contrasts = {}
    for lam in lambda_grid:
        cand_rates = {s: all_seed_summaries[s]["lambda_jac_full_grid_closed_loop"][lam] for s in seeds}
        lambda_jac_contrasts[lam] = alib.evaluate_contrast(
            alib.paired_success_rate_diff(success_rates["E"], cand_rates, seeds))

    return dict(success_rates=success_rates, primary_contrast=primary, secondary_family=secondary,
                mse_attribution_detail=mse_attribution, outcome=outcome,
                lambda_jac_full_grid_contrasts=lambda_jac_contrasts)


def _compute_m2_hashes(m2_paths: dict, seeds: list) -> dict:
    m2_hashes = {}
    for seed in seeds:
        self_hash, _ = aio.read_and_verify_json_artifact(m2_paths[seed])
        m2_hashes[str(seed)] = self_hash
    return m2_hashes


def _build_raw_metrics(all_seed_summaries: dict, seeds: list) -> dict:
    """Shared by `build_results_manifest` (writes it) and
    `_assert_m3_companions_match_fresh_evidence` (recomputes it fresh on
    resume and compares) -- kept as ONE function so the two can never
    silently drift apart."""
    conditions = list(LCNN_CONDITIONS) + list(FNN_CONDITIONS)
    return dict(
        closed_loop_success={c: {str(s): all_seed_summaries[s]["closed_loop_success"].get(c) for s in seeds}
                             for c in conditions},
        oracle_align_cos={c: {str(s): all_seed_summaries[s]["oracle_summary"][c]["align_cos"] for s in seeds}
                          for c in conditions},
        oracle_rel_grad_error={c: {str(s): all_seed_summaries[s]["oracle_summary"][c]["rel_grad_error"] for s in seeds}
                              for c in conditions},
        oracle_excluded_fraction={c: {str(s): all_seed_summaries[s]["oracle_summary"][c]["excluded_fraction"]
                                      for s in seeds} for c in conditions},
        sensitivity_arm_closed_loop_success={
            c: {str(s): all_seed_summaries[s]["sensitivity_arm"]["closed_loop_success"].get(c) for s in seeds}
            for c in SENSITIVITY_CONDITIONS},
        lambda_jac_full_grid_closed_loop={
            str(s): {str(lam): v for lam, v in all_seed_summaries[s]["lambda_jac_full_grid_closed_loop"].items()}
            for s in seeds},
    )


def _build_resample_arrays(aggregated: dict) -> tuple:
    """Shared by `build_results_manifest` (writes the companion npz) and
    `_assert_m3_companions_match_fresh_evidence` (recomputes it fresh on
    resume and compares). Returns `(resample_arrays, primary_clean,
    secondary_clean, lambda_jac_contrasts_clean)` -- the stripped contrast
    dicts are needed by `build_results_manifest` for the JSON payload, the
    raw arrays by both callers."""
    resample_arrays = {}

    def _strip_resample(name, c):
        c = dict(c)
        idx = c.pop("resample_idx", None)
        if idx is not None:
            resample_arrays[f"{name}_resample_idx"] = idx
        return c

    primary_clean = _strip_resample("primary", aggregated["primary_contrast"])
    secondary_clean = {name: _strip_resample(name, c) for name, c in aggregated["secondary_family"].items()}
    lambda_jac_contrasts_clean = {
        str(lam): _strip_resample(f"lambda_jac_{lam}", c)
        for lam, c in aggregated["lambda_jac_full_grid_contrasts"].items()
    }
    return resample_arrays, primary_clean, secondary_clean, lambda_jac_contrasts_clean


def build_results_manifest(m3_path: Path, m2_paths: dict, aggregated: dict, oracle_idx: np.ndarray,
                            all_seed_summaries: dict, seeds: list) -> dict:
    """Includes raw per-seed per-condition closed-loop success and FULL
    oracle metrics (median/p90/mean/frac_neg/excluded_fraction, not just
    median), the ACTUAL bootstrap resample-index arrays for every
    contrast (a companion `.npz`), the ACTUAL fixed oracle index array
    (a companion `.npz`, not merely its hash), the sensitivity-arm
    results, and the D full-grid's per-candidate contrasts. `m3_path`
    (including its seed-set tag) determines every companion file's name,
    so a different seed set never collides with a prior M3's files."""
    out_dir = m3_path.parent
    tag = m3_path.stem.replace("ablation_results_manifest_", "")
    m2_hashes = _compute_m2_hashes(m2_paths, seeds)

    resample_arrays, primary_clean, secondary_clean, lambda_jac_contrasts_clean = _build_resample_arrays(aggregated)

    # verify-if-exists throughout: a crash between any of these companion
    # writes and M3's own JSON (written last, below) must not make a
    # retry crash on stale companions from the interrupted attempt.
    raw_npz_path = out_dir / f"ablation_results_raw_{tag}.npz"
    raw_npz_sha256 = aio.write_npz_verify_if_exists(raw_npz_path, resample_arrays)

    oracle_idx_npz_path = out_dir / f"ablation_oracle_fixed_idx_{tag}.npz"
    oracle_idx_npz_sha256 = aio.write_npz_verify_if_exists(oracle_idx_npz_path, {"oracle_idx": oracle_idx})

    raw_metrics = _build_raw_metrics(all_seed_summaries, seeds)
    raw_json_path = out_dir / f"ablation_results_raw_metrics_{tag}.json"
    raw_json_sha256 = aio.write_json_verify_if_exists(raw_json_path, raw_metrics)

    payload = dict(
        kind="ablation_results_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_training_manifest_sha256=m2_hashes,
        oracle_fixed_idx_path=str(oracle_idx_npz_path.relative_to(ROOT)),
        oracle_fixed_idx_sha256=oracle_idx_npz_sha256,
        raw_metrics_path=str(raw_json_path.relative_to(ROOT)), raw_metrics_sha256=raw_json_sha256,
        raw_resample_arrays_path=str(raw_npz_path.relative_to(ROOT)), raw_resample_arrays_sha256=raw_npz_sha256,
        primary_contrast=primary_clean,
        secondary_family=secondary_clean,
        lambda_jac_full_grid_contrasts=lambda_jac_contrasts_clean,
        mse_attribution_detail=aggregated["mse_attribution_detail"],
        outcome=aggregated["outcome"],
    )
    return dict(path=m3_path, self_hash=aio.write_with_self_verification(m3_path, payload))


def verify_results_manifest(m3_path: Path) -> dict:
    """Corrected (P0): a reused M3 is not trusted on its own self-hash and
    predecessor-M2-hashes alone. Re-verifies the three companion files
    (raw-metrics JSON, raw-resample-arrays npz, oracle-fixed-idx npz) both
    exist and match their recorded hashes -- a deleted or modified
    companion file would otherwise go undetected since only M3's own JSON
    bytes are covered by its self-hash."""
    self_hash, payload = aio.read_and_verify_json_artifact(
        m3_path, expected_kind="ablation_results_manifest", expected_keys=EXPECTED_RESULTS_MANIFEST_PAYLOAD_KEYS)
    for path_key, hash_key in (
        ("raw_metrics_path", "raw_metrics_sha256"),
        ("raw_resample_arrays_path", "raw_resample_arrays_sha256"),
        ("oracle_fixed_idx_path", "oracle_fixed_idx_sha256"),
    ):
        p = ROOT / payload[path_key]
        if not p.exists():
            raise RuntimeError(f"M3_COMPANION_MISSING: {payload[path_key]} referenced by {m3_path} no longer exists")
        if aio.file_sha256(p) != payload[hash_key]:
            raise RuntimeError(f"M3_COMPANION_DRIFT: {payload[path_key]} hash changed since M3 was built")
    return dict(self_hash=self_hash, payload=payload)


def _assert_aggregated_matches_m3_payload(aggregated: dict, payload: dict) -> None:
    """Corrected (P0): `verify_results_manifest` only proves the M3 file
    and its companions are UNMODIFIED since M3 was built -- it does not
    prove the numbers inside M3 were ever CORRECT. A self-consistent but
    wrong M3 (e.g. written by a buggy `build_results_manifest` call, then
    never touched again) would otherwise pass resume silently. This
    re-derives `primary_contrast`/`secondary_family`/`outcome`/
    `mse_attribution_detail` FRESH from the resumed seeds' M2 data (via
    `aggregate_and_evaluate`, already computed by the caller) and requires
    an EXACT match (after NaN->None normalization, since NaN != NaN) to
    what M3 actually recorded."""
    def strip(c):
        return aio._sanitize_nonfinite({k: v for k, v in c.items() if k != "resample_idx"})

    fresh_primary = strip(aggregated["primary_contrast"])
    fresh_secondary = {n: strip(c) for n, c in aggregated["secondary_family"].items()}
    fresh_lambda = {str(lam): strip(c) for lam, c in aggregated["lambda_jac_full_grid_contrasts"].items()}
    if fresh_primary != payload["primary_contrast"]:
        raise RuntimeError("M3_CONTENT_MISMATCH: recomputed primary_contrast differs from what M3 recorded")
    if fresh_secondary != payload["secondary_family"]:
        raise RuntimeError("M3_CONTENT_MISMATCH: recomputed secondary_family differs from what M3 recorded")
    if fresh_lambda != payload["lambda_jac_full_grid_contrasts"]:
        raise RuntimeError("M3_CONTENT_MISMATCH: recomputed lambda_jac_full_grid_contrasts differ from M3")
    if aggregated["mse_attribution_detail"] != payload["mse_attribution_detail"]:
        raise RuntimeError("M3_CONTENT_MISMATCH: recomputed mse_attribution_detail differs from what M3 recorded")
    if aggregated["outcome"] != payload["outcome"]:
        raise RuntimeError("M3_CONTENT_MISMATCH: recomputed outcome differs from what M3 recorded")


def _assert_m3_companions_match_fresh_evidence(payload: dict, all_seed_summaries: dict, seeds: list,
                                               oracle_idx: np.ndarray, aggregated: dict) -> None:
    """Corrected (P1, 6th review round): `verify_results_manifest` only
    proved the three M3 companion files (raw-metrics JSON, oracle-fixed-
    idx npz, raw-resample-arrays npz) were byte-for-byte UNMODIFIED since
    M3 was built -- exactly like `verify_freeze_manifest`/`verify_query_
    manifest`/`verify_training_manifest` before this same fix was applied
    to THEM, it never proved their CONTENT was ever correct to begin with.
    A companion file written once by a buggy `build_results_manifest` call
    and never touched again would pass silently forever. This re-derives
    all three companions' expected content FRESH -- `raw_metrics` via the
    same `_build_raw_metrics` helper `build_results_manifest` itself calls
    (so the two can never silently drift apart), `oracle_fixed_idx` via a
    fresh `alib.fixed_oracle_idx` call (passed in by the caller, which
    already computes it once per `run_full_protocol` call), and the
    resample-index arrays via the SAME deterministic-reseed contract
    `aggregate_and_evaluate`/`bootstrap_ci` already rely on (fixed literal
    seed 31415, fresh generator per contrast) -- and requires an EXACT
    match to what the companion files actually contain."""
    fresh_raw_metrics = aio._sanitize_nonfinite(_build_raw_metrics(all_seed_summaries, seeds))
    stored_raw_metrics = json.loads((ROOT / payload["raw_metrics_path"]).read_text())
    if stored_raw_metrics != fresh_raw_metrics:
        raise RuntimeError(
            "M3_COMPANION_CONTENT_MISMATCH: raw_metrics companion's CONTENT does not match a fresh "
            "recomputation from the resumed seeds' M2 data, even though its file hash is unchanged"
        )

    with np.load(ROOT / payload["oracle_fixed_idx_path"]) as npz:
        stored_oracle_idx = npz["oracle_idx"]
    if not np.array_equal(stored_oracle_idx, oracle_idx):
        raise RuntimeError(
            "M3_COMPANION_CONTENT_MISMATCH: oracle_fixed_idx companion's CONTENT does not match a fresh "
            "recomputation of the fixed oracle index, even though its file hash is unchanged"
        )

    fresh_resample_arrays, _, _, _ = _build_resample_arrays(aggregated)
    with np.load(ROOT / payload["raw_resample_arrays_path"]) as npz:
        stored_resample_arrays = {k: npz[k] for k in npz.files}
    if set(stored_resample_arrays.keys()) != set(fresh_resample_arrays.keys()):
        raise RuntimeError(
            "M3_COMPANION_CONTENT_MISMATCH: raw_resample_arrays companion's keyset does not match a "
            "fresh recomputation, even though its file hash is unchanged"
        )
    for key, value in fresh_resample_arrays.items():
        if not np.array_equal(stored_resample_arrays[key], value):
            raise RuntimeError(
                f"M3_COMPANION_CONTENT_MISMATCH: raw_resample_arrays companion's {key!r} content does not "
                "match a fresh recomputation, even though its file hash is unchanged"
            )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_full_protocol(pool: dict, clean: dict, scale: dict, sim, out_dir: Path = NAMESPACE,
                       seeds: list = None, freeze_manifest_path: Path = None) -> dict:
    """The real, end-to-end entry point: verify freeze -> build-or-resume
    M1 -> per-seed training/evaluation-or-resume + M2 -> aggregate + M3.
    Resume-or-verify (Section 10, M2): if `out_dir` already has a verified
    M1/M2 for a given seed, it is reused as-is (never rebuilt, never
    retrained) and its persisted payload -- not any in-memory state -- is
    the source of truth for that seed's contribution to aggregation.
    `require_seed_completeness` (Section 7.7c) is enforced before Section
    9's outcome computation runs."""
    seeds = seeds if seeds is not None else alib.SEEDS
    freeze_manifest_path = freeze_manifest_path if freeze_manifest_path is not None else FREEZE_MANIFEST_PATH
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"

    m1_path = out_dir / "ablation_query_manifest_2026_08_31.json"
    if m1_path.exists():
        m1_verified = verify_query_manifest(m1_path, pool=pool, clean=clean, freeze_manifest_path=freeze_manifest_path)
        m1 = dict(path=m1_path, self_hash=m1_verified["self_hash"])
    else:
        m1 = build_query_manifest(pool, clean, out_dir, freeze_manifest_path=freeze_manifest_path)

    oracle_idx = alib.fixed_oracle_idx(pool["test_idx"])

    all_seed_summaries = {}
    m2_paths = {}
    for seed in seeds:
        m2_path = out_dir / f"ablation_training_manifest_seed{seed}_2026_08_31.json"
        if m2_path.exists():
            m2_verified = verify_training_manifest(m2_path, m1["self_hash"], pool, clean, scale, seed)
            summary = _seed_summary_from_m2_payload(m2_verified["payload"])
        else:
            # Partial-publication crash recovery: if a prior run_seed()
            # attempt for this seed crashed mid-training (before M2 was
            # built, so M2 doesn't exist and we're here again), a naive
            # retry would reuse the SAME deterministic checkpoint
            # filenames and hit FileExistsError on the very first
            # checkpoint write. Each attempt gets its OWN fresh,
            # never-before-used subdirectory instead -- the crashed
            # attempt's partial checkpoints are left on disk untouched,
            # as historical residue, never deleted or overwritten.
            attempt_n = 1
            seed_checkpoint_dir = checkpoint_dir / f"seed{seed}" / f"attempt{attempt_n}"
            while seed_checkpoint_dir.exists():
                attempt_n += 1
                seed_checkpoint_dir = checkpoint_dir / f"seed{seed}" / f"attempt{attempt_n}"
            seed_run = run_seed(seed, pool, clean, scale, seed_checkpoint_dir, oracle_idx, sim)
            m2 = build_training_manifest(seed, out_dir, m1["path"], seed_run, scale, pool, clean)
            m2_path = m2["path"]
            summary = _seed_summary_from_m2_payload(json.loads(m2_path.read_text()))
        m2_paths[seed] = m2_path
        all_seed_summaries[seed] = summary

    conditions = list(LCNN_CONDITIONS) + list(FNN_CONDITIONS)
    alib.require_seed_completeness({c: list(all_seed_summaries.keys()) for c in conditions}, seeds=seeds)

    # M3's filename is scoped to the EXACT seed set it covers: re-running
    # with the same seed set reuses the same M3 (resume-or-verify);
    # extending to a different seed set (e.g. more seeds) naturally
    # writes a NEW M3 rather than requiring an overwrite of the old one,
    # which the atomic no-overwrite writers refuse by design. The prior
    # M3 for a smaller seed set is left on disk untouched, as a historical
    # record, exactly like Paper M's dated-namespace convention.
    seed_set_tag = hashlib.sha256(repr(sorted(seeds)).encode()).hexdigest()[:12]
    m3_path = out_dir / f"ablation_results_manifest_{seed_set_tag}_2026_08_31.json"
    expected_m2_hashes = _compute_m2_hashes(m2_paths, seeds)
    if m3_path.exists():
        m3_verified = verify_results_manifest(m3_path)
        if m3_verified["payload"]["predecessor_training_manifest_sha256"] != expected_m2_hashes:
            raise RuntimeError(
                f"RESUME_PROVENANCE_MISMATCH: existing Results Manifest {m3_path} does not match "
                "the current seed set's M2 hashes despite matching its seed-set tag -- refusing to reuse it"
            )
        aggregated = aggregate_and_evaluate(all_seed_summaries, seeds)
        _assert_aggregated_matches_m3_payload(aggregated, m3_verified["payload"])
        _assert_m3_companions_match_fresh_evidence(m3_verified["payload"], all_seed_summaries, seeds,
                                                   oracle_idx, aggregated)
        m3 = dict(path=m3_path, self_hash=m3_verified["self_hash"])
        return dict(m1=m1, m2_paths=m2_paths, m3=m3, aggregated=aggregated)

    aggregated = aggregate_and_evaluate(all_seed_summaries, seeds)
    m3 = build_results_manifest(m3_path, m2_paths, aggregated, oracle_idx, all_seed_summaries, seeds)
    return dict(m1=m1, m2_paths=m2_paths, m3=m3, aggregated=aggregated)


def run_real_protocol(out_dir: Path = NAMESPACE, seeds: list = None,
                      freeze_manifest_path: Path = None) -> dict:
    """The CANONICAL production entry point (new): loads the real pool,
    builds the real CLEAN_QUERY_TABLE/scale constants against the REAL
    `budget_sweep_solver.SIM`, and calls `run_full_protocol` with that
    SAME `sim` object -- fixing the simulator-identity consistency
    `run_seed` now enforces by CONSTRUCTION rather than by caller
    discipline. This is the only entry point that should be used for a
    real (non-smoke) run; `run_full_protocol`/`run_seed` remain directly
    callable for testing with a fabricated `sim`, provided the caller
    also points `budget_sweep_solver.SIM` at that same fabricated object
    first (as the synthetic smoke suite does).

    Corrected (P0, 6th review round): the freeze gate is verified FIRST,
    before anything else in this function runs -- previously the real
    simulator was imported and the real pool/CLEAN_QUERY_TABLE were
    computed BEFORE `run_full_protocol` got around to checking the
    freeze, meaning a missing or corrupted freeze manifest was only
    discovered after real simulator queries had already happened. Also
    enforces that `seeds`, if given, is EXACTLY the pre-registered
    10-seed set (`alib.SEEDS`) -- an arbitrary subset is a smoke/test-only
    concept and must not be reachable through the canonical real entry
    point, which is the only one that should ever touch the real pool."""
    resolved_freeze_path = freeze_manifest_path if freeze_manifest_path is not None else FREEZE_MANIFEST_PATH
    verify_freeze_manifest(resolved_freeze_path)

    resolved_seeds = seeds if seeds is not None else alib.SEEDS
    if list(resolved_seeds) != list(alib.SEEDS):
        raise ValueError(
            f"run_real_protocol requires the full pre-registered seed set {list(alib.SEEDS)}, "
            f"not an arbitrary subset ({list(resolved_seeds)}) -- partial or custom seed sets "
            "are for the generic run_full_protocol smoke/test entry point only."
        )

    import budget_sweep_solver as bs  # noqa: PLC0415
    import probe_action_gradient as pr  # noqa: PLC0415

    sim = bs.SIM
    pool = alib.load_pool()
    clean = alib.build_clean_query_table(pool, sim, pr.precompute_phi)
    scale = alib.shared_scale_constants(pool, clean)
    return run_full_protocol(pool, clean, scale, sim, out_dir=out_dir, seeds=resolved_seeds,
                             freeze_manifest_path=resolved_freeze_path)
