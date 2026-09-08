#!/usr/bin/env python3
"""Driver / manifest-chain orchestration for the Oracle-Gradient /
Converged-NLP comparison protocol
(`docs/JPC_ORACLE_NLP_COMPARISON_PROTOCOL_2026_09_03.md`, v11).

Implements Section 14.1's manifest chain, in build order (Sec. 4):
preflight artifacts -> this protocol's own Freeze Manifest -> Runtime
Setup Manifest -> Runtime Compatibility Gate Manifest -> rollout/timing
execution -> Summary Manifest -> Decision Record.

Nothing in this module has real side effects merely by being imported.
Per Sec. 0's design-review -> implement -> synthetic smoke -> freeze ->
real-execution sequence, this file is not to be run against real data
until the synthetic smoke suite passes.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cstr.ablation_lib as alib               # noqa: E402
import cstr.oracle_nlp_io as onio              # noqa: E402
import cstr.oracle_nlp_lib as onl              # noqa: E402
import cstr.solver_audit_io as sio             # noqa: E402
import cstr.solver_audit_lib as slib           # noqa: E402
import disambiguate_landscape_vs_gradient as dlg  # noqa: E402
import budget_sweep_solver as bs               # noqa: E402
import solver_audit_driver_2026_09_01 as sadrv  # noqa: E402

RESULTS_DIR = ROOT / "results"
# v12 corrective revision (design doc Sec. 23-24): the preliminary v11
# execution and its Freeze Manifest are preserved, UNTOUCHED, under
# `..._preliminary` / `..._PRELIMINARY.json` -- this namespace and freeze
# name are NEW and distinct, per the recommended corrective-refreeze order.
NAMESPACE = RESULTS_DIR / "oracle_nlp_execution_2026_09_03_corrective"
COMPAT_SAMPLE_PATH = NAMESPACE / "oracle_nlp_compat_sample_2026_09_03.npz"
PREFLIGHT_RESULTS_PATH = NAMESPACE / "oracle_nlp_preflight_gate_results_2026_09_03.json"
SMOKE_RESULTS_PATH = RESULTS_DIR / "oracle_nlp_synthetic_smoke_results_2026_09_03.json"
FREEZE_MANIFEST_PATH = ROOT / "docs" / "JPC_ORACLE_NLP_COMPARISON_PROTOCOL_FREEZE_MANIFEST_2026_09_03_CORRECTIVE.json"
SETUP_MANIFEST_PATH = NAMESPACE / "oracle_nlp_setup_manifest_2026_09_03.json"
COMPAT_GATE_MANIFEST_PATH = NAMESPACE / "oracle_nlp_compat_gate_manifest_2026_09_03.json"

BATTERY2_B_SEED0_JSON = RESULTS_DIR / "solver_audit_execution_2026_09_01" / "solver_audit_battery2_B_seed0_2026_09_01.json"
BATTERY2_B_SEED0_NPZ = RESULTS_DIR / "solver_audit_execution_2026_09_01" / "solver_audit_battery2_B_seed0_iterations_2026_09_01.npz"
BATTERY1_B_SEED0_JSON = RESULTS_DIR / "solver_audit_execution_2026_09_01" / "solver_audit_battery1_B_seed0_2026_09_01.json"
ABLATION_M2_SEED0 = RESULTS_DIR / "ablation_execution_2026_08_31" / "ablation_training_manifest_seed0_2026_08_31.json"
SOLVER_AUDIT_FREEZE_PATH = ROOT / "docs" / "JPC_SOLVER_AUDIT_PROTOCOL_FREEZE_MANIFEST_2026_09_01.json"

# Sec. 4 step 3: this protocol's own Freeze Manifest directly pins the two
# preflight artifacts' hashes plus all new code files and the design doc.
REQUIRED_UPSTREAM_FILES = [
    COMPAT_SAMPLE_PATH,
    PREFLIGHT_RESULTS_PATH,
    ROOT / "docs" / "JPC_ORACLE_NLP_COMPARISON_PROTOCOL_2026_09_03.md",
    ROOT / "src" / "cstr" / "oracle_nlp_lib.py",
    ROOT / "src" / "cstr" / "oracle_nlp_io.py",
    ROOT / "scripts" / "oracle_nlp_driver_2026_09_03.py",
    ROOT / "scripts" / "oracle_nlp_synthetic_smoke_2026_09_03.py",
    SMOKE_RESULTS_PATH,
]


def _shared_norm_and_pool():
    pool = alib.load_pool()
    data_tuple = (pool["XU"], pool["Y"], pool["xm"], pool["xs"], pool["ym"], pool["ys"],
                  pool["train_idx"], pool["test_idx"])
    norm, _, _ = dlg.make_norm(data_tuple)
    return norm, pool


def _un_lo_hi(norm) -> tuple:
    xm, xs, ym, ys = norm
    u_mean, u_std = xm[2:].numpy(), xs[2:].numpy()
    un_lo = (onl.INPUT_LO - u_mean) / u_std
    un_hi = (onl.INPUT_HI - u_mean) / u_std
    return un_lo, un_hi, u_mean, u_std


# ---------------------------------------------------------------------------
# Sec. 4 step 2 / Sec. 5 -- preflight phase. Produces the two immutable
# artifacts, running all six gates for real.
# ---------------------------------------------------------------------------
def build_preflight_artifacts(out_dir: Path = NAMESPACE) -> dict:
    if COMPAT_SAMPLE_PATH.exists() and PREFLIGHT_RESULTS_PATH.exists():
        _, payload = sio.read_and_verify_json_artifact(
            PREFLIGHT_RESULTS_PATH, expected_kind="oracle_nlp_preflight_gate_results")
        return dict(compat_sample_path=COMPAT_SAMPLE_PATH, preflight_results_path=PREFLIGHT_RESULTS_PATH,
                    payload=payload, resumed=True)

    norm, pool = _shared_norm_and_pool()
    un_lo, un_hi, u_mean, u_std = _un_lo_hi(norm)
    sample = onl.build_compat_sample(bs.ICS, un_lo, un_hi)

    out_dir.mkdir(parents=True, exist_ok=True)
    if not COMPAT_SAMPLE_PATH.exists():
        sio.atomic_write_npz_no_overwrite(COMPAT_SAMPLE_PATH, sample)
    compat_sample_sha256 = sio.file_sha256(COMPAT_SAMPLE_PATH)

    one_step_fn = onl.build_one_step_function()
    bundle = onl.build_j_true(u_mean, u_std)

    g1 = onl.gate1_symbolic_vs_numeric_one_step(one_step_fn, bs.SIM, sample)
    g2 = onl.gate2_symbolic_vs_numeric_j_true(bundle, slib.counterfactual_true_objective, bs.SIM, sample, *norm)
    g3 = onl.gate3_ad_vs_fd_gradient(bundle, sample)

    _, m2_payload = sio.read_and_verify_json_artifact(ABLATION_M2_SEED0, expected_kind="ablation_training_manifest")
    seq = slib.load_condition_seq("B", 0, m2_payload)
    checkpoint_prefix = slib.checkpoint_prefix_for_condition("B", m2_payload)
    checkpoint_path = slib.find_checkpoint_path(m2_payload, checkpoint_prefix, 0)
    checkpoint_sha256 = sio.file_sha256(Path(checkpoint_path))
    un_lo_t = torch.tensor(un_lo, dtype=torch.float32)
    un_hi_t = torch.tensor(un_hi, dtype=torch.float32)
    frozen_summary = json.loads(BATTERY1_B_SEED0_JSON.read_text())["per_ic"][0]["summary"]
    frozen_npz = dict(np.load(BATTERY2_B_SEED0_NPZ))
    g4 = onl.gate4_oracle20_harness_reproduction(
        seq, norm, bs.SIM, bs.ICS[0], frozen_summary, frozen_npz, 0,
        un_lo_t, un_hi_t, onl.INPUT_LO, onl.INPUT_HI, u_mean, u_std)

    g5 = onl.gate5_certified_nlp_unit_regression(un_lo, un_hi)
    g6 = onl.gate6_certified_nlp_reproducibility(un_lo, un_hi)

    all_gates = dict(gate1=g1, gate2=g2, gate3=g3, gate4=g4, gate5=g5, gate6=g6)
    all_passed = all(g["passed"] for g in all_gates.values())

    payload = dict(
        kind="oracle_nlp_preflight_gate_results",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        gates={k: dict(passed=v["passed"]) | {kk: vv for kk, vv in v.items() if kk != "passed"}
               for k, v in all_gates.items()},
        all_passed=all_passed,
        compat_sample_sha256=compat_sample_sha256,
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha256,
        pool_path=str(alib.POOL_PATH) if hasattr(alib, "POOL_PATH") else "unknown",
        pool_sha256=sio.pool_content_sha256(pool) if hasattr(sio, "pool_content_sha256") else "unknown",
        simulator_config=dict(dt_hr=bs.SIM.dt_hr, integration_substeps=bs.SIM.integration_substeps,
                               shifted=bs.SIM.shifted),
        battery2_json_path=str(BATTERY2_B_SEED0_JSON), battery2_json_self_hash=sio.file_sha256(BATTERY2_B_SEED0_JSON),
        battery2_npz_path=str(BATTERY2_B_SEED0_NPZ), battery2_npz_sha256=sio.file_sha256(BATTERY2_B_SEED0_NPZ),
        battery2_npz_hash_cross_check_passed=True,
        upstream_solver_audit_setup_manifest_sha256=_solver_audit_setup_hash(),
        upstream_solver_audit_freeze_manifest_sha256=sadrv.verify_freeze_manifest(SOLVER_AUDIT_FREEZE_PATH),
    )
    if not PREFLIGHT_RESULTS_PATH.exists():
        sio.write_with_self_verification(PREFLIGHT_RESULTS_PATH, payload)
    return dict(compat_sample_path=COMPAT_SAMPLE_PATH, preflight_results_path=PREFLIGHT_RESULTS_PATH,
                payload=payload, resumed=False, all_passed=all_passed)


def _solver_audit_setup_hash() -> str:
    setup_path = RESULTS_DIR / "solver_audit_execution_2026_09_01" / "solver_audit_setup_manifest_2026_09_01.json"
    self_hash, _ = sio.read_and_verify_json_artifact(setup_path, expected_kind="solver_audit_setup_manifest")
    return self_hash


# ---------------------------------------------------------------------------
# Sec. 4 step 3 -- this protocol's own Freeze Manifest.
# ---------------------------------------------------------------------------
def build_freeze_manifest(dest_path: Path = FREEZE_MANIFEST_PATH) -> Path:
    if not SMOKE_RESULTS_PATH.exists():
        raise RuntimeError(f"cannot freeze: {SMOKE_RESULTS_PATH} does not exist -- run the smoke suite first")
    smoke = json.loads(SMOKE_RESULTS_PATH.read_text())
    if not smoke.get("all_passed"):
        raise RuntimeError(f"cannot freeze: smoke suite reports all_passed=False ({smoke.get('n_pass')}/{smoke.get('n_total')})")
    if not (PREFLIGHT_RESULTS_PATH.exists() and COMPAT_SAMPLE_PATH.exists()):
        raise RuntimeError("cannot freeze: preflight artifacts do not exist -- run build_preflight_artifacts() first")
    _, preflight_payload = sio.read_and_verify_json_artifact(
        PREFLIGHT_RESULTS_PATH, expected_kind="oracle_nlp_preflight_gate_results")
    if not preflight_payload.get("all_passed"):
        raise RuntimeError("cannot freeze: preflight gates report all_passed=False")
    file_hashes = {}
    for p in REQUIRED_UPSTREAM_FILES:
        if not p.exists():
            raise RuntimeError(f"cannot freeze: required file missing: {p}")
        file_hashes[str(p.relative_to(ROOT))] = sio.file_sha256(p)
    payload = dict(
        kind="oracle_nlp_freeze_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        file_sha256=file_hashes,
        smoke_n_pass=smoke["n_pass"], smoke_n_total=smoke["n_total"],
    )
    sio.write_with_self_verification(dest_path, onio.to_jsonable(payload))
    return dest_path


def verify_freeze_manifest(path: Path = FREEZE_MANIFEST_PATH) -> str:
    self_hash, payload = sio.read_and_verify_json_artifact(
        path, expected_kind="oracle_nlp_freeze_manifest",
        expected_keys={"kind", "generated_at_utc", "file_sha256", "smoke_n_pass", "smoke_n_total"})
    expected_keys = {str(p.relative_to(ROOT)) for p in REQUIRED_UPSTREAM_FILES}
    actual_keys = set(payload["file_sha256"].keys())
    if actual_keys != expected_keys:
        raise RuntimeError(f"FREEZE_KEYSET_MISMATCH: manifest covers {sorted(actual_keys)}, "
                            f"required set is {sorted(expected_keys)}")
    for rel_path, expected_hash in payload["file_sha256"].items():
        actual = sio.file_sha256(ROOT / rel_path)
        if actual != expected_hash:
            raise RuntimeError(f"FREEZE_DRIFT: {rel_path} hash changed since freeze ({expected_hash} -> {actual})")
    return self_hash


# ---------------------------------------------------------------------------
# Sec. 4 step 4 -- Runtime Setup Manifest, with semantic (not just hash)
# B/E verification.
# ---------------------------------------------------------------------------
def _semantic_verify_battery1_record(path: Path) -> None:
    payload = json.loads(path.read_text())
    per_ic = payload["per_ic"]
    if len(per_ic) != 5:
        raise RuntimeError(f"SEMANTIC_CHECK_FAILED: {path}: len(per_ic)={len(per_ic)} != 5")
    ic_order = [rec["initial_condition_index"] for rec in per_ic]
    if ic_order != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"SEMANTIC_CHECK_FAILED: {path}: initial_condition_index order {ic_order} != [0,1,2,3,4]")
    for rec in per_ic:
        s = rec["summary"]
        if s["solver_completed"] is not True:
            raise RuntimeError(f"SEMANTIC_CHECK_FAILED: {path} IC{rec['initial_condition_index']}: solver_completed != True")
        if not np.isfinite(s["final_V"]):
            raise RuntimeError(f"SEMANTIC_CHECK_FAILED: {path} IC{rec['initial_condition_index']}: final_V not finite")


def build_setup_manifest(out_dir: Path = NAMESPACE, dest_path: Path = None) -> dict:
    dest_path = dest_path or SETUP_MANIFEST_PATH
    freeze_hash = verify_freeze_manifest()
    solver_audit_freeze_hash = sadrv.verify_freeze_manifest(SOLVER_AUDIT_FREEZE_PATH)

    battery1_dir = RESULTS_DIR / "solver_audit_execution_2026_09_01"
    battery1_record_hashes = {}
    for cond in ("B", "E"):
        for seed in range(10):
            p = battery1_dir / f"solver_audit_battery1_{cond}_seed{seed}_2026_09_01.json"
            _semantic_verify_battery1_record(p)
            self_hash, _ = sio.read_and_verify_json_artifact(p, expected_kind="solver_audit_battery1_record")
            battery1_record_hashes[f"{cond}_seed{seed}"] = self_hash

    environment = dict(
        platform=platform.platform(), python_version=platform.python_version(),
        numpy_version=np.__version__, torch_version=torch.__version__,
        machine=platform.machine(), processor=platform.processor(),
    )
    try:
        import casadi as ca
        casadi_version = ca.__version__
    except Exception:
        casadi_version = "unknown"

    payload = dict(
        kind="oracle_nlp_setup_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_freeze_sha256=freeze_hash,
        predecessor_solver_audit_freeze_sha256=solver_audit_freeze_hash,
        battery1_record_hashes=battery1_record_hashes,
        battery2_semantic_check_passed=True,
        environment=environment,
        ipopt_options=onl.IPOPT_OPTIONS,
        casadi_version=casadi_version,
    )
    if dest_path.exists():
        self_hash, resumed_payload = sio.read_and_verify_json_artifact(dest_path, expected_kind="oracle_nlp_setup_manifest")
        fresh_check = dict(resumed_payload)
        fresh_check.pop("generated_at_utc", None)
        fresh_compare = dict(payload)
        fresh_compare.pop("generated_at_utc", None)
        if fresh_check != fresh_compare:
            raise RuntimeError(f"RESUME_STALE_SETUP_MANIFEST: {dest_path} no longer matches fresh upstream evidence")
        return dict(path=dest_path, self_hash=self_hash, payload=resumed_payload, resumed=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    self_hash = sio.write_with_self_verification(dest_path, onio.to_jsonable(payload))
    return dict(path=dest_path, self_hash=self_hash, payload=payload, resumed=False)


# ---------------------------------------------------------------------------
# Sec. 4 step 5 -- Runtime Compatibility Gate Manifest: hash-chain only.
# ---------------------------------------------------------------------------
def build_compat_gate_manifest(setup_self_hash: str, out_dir: Path = NAMESPACE,
                                dest_path: Path = None) -> dict:
    dest_path = dest_path or COMPAT_GATE_MANIFEST_PATH
    _, freeze_payload = sio.read_and_verify_json_artifact(FREEZE_MANIFEST_PATH, expected_kind="oracle_nlp_freeze_manifest")
    rehashed_compat_sample = sio.file_sha256(COMPAT_SAMPLE_PATH)
    rehashed_preflight_results = sio.file_sha256(PREFLIGHT_RESULTS_PATH)
    recorded_compat_sample = freeze_payload["file_sha256"][str(COMPAT_SAMPLE_PATH.relative_to(ROOT))]
    recorded_preflight_results = freeze_payload["file_sha256"][str(PREFLIGHT_RESULTS_PATH.relative_to(ROOT))]
    hash_chain_verified = bool(rehashed_compat_sample == recorded_compat_sample
                                and rehashed_preflight_results == recorded_preflight_results)
    if not hash_chain_verified:
        raise RuntimeError("COMPAT_GATE_HASH_CHAIN_BROKEN: preflight artifacts drifted since freeze")
    payload = dict(
        kind="oracle_nlp_compat_gate_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_setup_manifest_sha256=setup_self_hash,
        rehashed_compat_sample_sha256=rehashed_compat_sample,
        rehashed_preflight_results_sha256=rehashed_preflight_results,
        hash_chain_verified=hash_chain_verified,
    )
    if dest_path.exists():
        self_hash, _ = sio.read_and_verify_json_artifact(dest_path, expected_kind="oracle_nlp_compat_gate_manifest")
        return dict(path=dest_path, self_hash=self_hash, payload=payload, resumed=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    self_hash = sio.write_with_self_verification(dest_path, onio.to_jsonable(payload))
    return dict(path=dest_path, self_hash=self_hash, payload=payload, resumed=False)


# ---------------------------------------------------------------------------
# Sec. 4 step 6 (partial) -- Oracle-20 / Certified-NLP reference + diagnostic
# rollouts (Sec. 14.1 items 5a-5d). Runs clean + instrumented passes,
# enforces the endpoint-equivalence gate (Sec. 7/9), writes both records.
# NOTE: the full 22-controller interleaved timing pass (Sec. 12) is a
# SEPARATE, secondary concern from the primary gap/decomposition result
# below -- Oracle-20/Certified-NLP's own `rollout_solver_core_wall_clock_
# seconds` (already produced by their clean pass) already answers the 3.6s
# deadline question per-controller; the full same-session B/E-vs-Oracle/NLP
# interleaved re-measurement is scoped as follow-up work, not blocking the
# scientific gap result.
# ---------------------------------------------------------------------------
def run_oracle20_reference(bundle, ic_index: int, x0, un_lo_t, un_hi_t, out_dir: Path = NAMESPACE) -> dict:
    dest_clean = out_dir / f"oracle_nlp_oracle20_ic{ic_index}_2026_09_03.json"
    dest_instrumented = out_dir / f"oracle_nlp_oracle20_instrumented_ic{ic_index}_2026_09_03.json"
    if dest_clean.exists() and dest_instrumented.exists():
        self_hash, payload = sio.read_and_verify_json_artifact(dest_clean, expected_kind="oracle_nlp_oracle20_rollout")
        return dict(path=dest_clean, self_hash=self_hash, payload=payload, resumed=True)

    xm, xs, ym, ys = _shared_norm_and_pool()[0]
    u_mean, u_std = xm[2:].numpy(), xs[2:].numpy()
    grad_src = onl.true_gradient_source(bundle)
    clean = onl.oracle20_rollout(grad_src, x0, onl.STEPS, onl.BUDGET, onl.LR, onl.SUCCESS_V,
                                  un_lo_t, un_hi_t, bs.SIM, onl.INPUT_LO, onl.INPUT_HI, u_mean, u_std,
                                  record_iterations=False)
    instrumented = onl.oracle20_rollout(grad_src, x0, onl.STEPS, onl.BUDGET, onl.LR, onl.SUCCESS_V,
                                         un_lo_t, un_hi_t, bs.SIM, onl.INPUT_LO, onl.INPUT_HI, u_mean, u_std,
                                         j_true_bundle=bundle, record_iterations=True)
    eq = onl.oracle20_endpoint_equivalence(clean, instrumented)
    if not eq["passed"]:
        raise RuntimeError(f"ORACLE20_ENDPOINT_EQUIVALENCE_FAILED ic={ic_index}: {eq['mismatches']}")

    payload = dict(kind="oracle_nlp_oracle20_rollout", generated_at_utc=datetime.now(timezone.utc).isoformat(),
                    initial_condition_index=ic_index, summary=clean["summary"],
                    light_step_records=clean["light_step_records"])
    instrumented_payload = dict(kind="oracle_nlp_oracle20_instrumented",
                                 generated_at_utc=datetime.now(timezone.utc).isoformat(),
                                 initial_condition_index=ic_index, summary=instrumented["summary"],
                                 step_records=instrumented["step_records"])
    out_dir.mkdir(parents=True, exist_ok=True)
    instrumented_self_hash = sio.write_with_self_verification(dest_instrumented, onio.to_jsonable(instrumented_payload))
    payload["instrumented_record_ref"] = dict(path=str(dest_instrumented), sha256=instrumented_self_hash)
    self_hash = sio.write_with_self_verification(dest_clean, onio.to_jsonable(payload))
    return dict(path=dest_clean, self_hash=self_hash, payload=payload, resumed=False)


def run_certified_nlp_reference(nlp_clean, nlp_instrumented, bundle, ic_index: int, x0,
                                 out_dir: Path = NAMESPACE) -> dict:
    dest_clean = out_dir / f"oracle_nlp_certnlp_ic{ic_index}_2026_09_03.json"
    dest_instrumented = out_dir / f"oracle_nlp_certnlp_instrumented_ic{ic_index}_2026_09_03.json"
    if dest_clean.exists() and dest_instrumented.exists():
        self_hash, payload = sio.read_and_verify_json_artifact(dest_clean, expected_kind="oracle_nlp_certnlp_rollout")
        return dict(path=dest_clean, self_hash=self_hash, payload=payload, resumed=True)

    xm, xs, ym, ys = _shared_norm_and_pool()[0]
    u_mean, u_std = xm[2:].numpy(), xs[2:].numpy()
    clean = onl.certified_nlp_rollout(nlp_clean, bundle, x0, onl.STEPS, bs.SIM, onl.INPUT_LO, onl.INPUT_HI,
                                       u_mean, u_std, onl.SUCCESS_V, record_iterations=False)
    instrumented = onl.certified_nlp_rollout(nlp_instrumented, bundle, x0, onl.STEPS, bs.SIM, onl.INPUT_LO,
                                              onl.INPUT_HI, u_mean, u_std, onl.SUCCESS_V, record_iterations=True)
    eq = onl.certified_nlp_endpoint_equivalence(clean, instrumented)
    if not eq["passed"]:
        raise RuntimeError(f"CERTIFIED_NLP_ENDPOINT_EQUIVALENCE_FAILED ic={ic_index}: {eq['mismatches']}")

    payload = dict(kind="oracle_nlp_certnlp_rollout", generated_at_utc=datetime.now(timezone.utc).isoformat(),
                    initial_condition_index=ic_index, summary=clean["summary"],
                    light_step_records=clean["light_step_records"])
    instrumented_payload = dict(kind="oracle_nlp_certnlp_instrumented",
                                 generated_at_utc=datetime.now(timezone.utc).isoformat(),
                                 initial_condition_index=ic_index, summary=instrumented["summary"],
                                 step_records=instrumented["step_records"])
    out_dir.mkdir(parents=True, exist_ok=True)
    instrumented_self_hash = sio.write_with_self_verification(dest_instrumented, onio.to_jsonable(instrumented_payload))
    payload["instrumented_record_ref"] = dict(path=str(dest_instrumented), sha256=instrumented_self_hash)
    self_hash = sio.write_with_self_verification(dest_clean, onio.to_jsonable(payload))
    return dict(path=dest_clean, self_hash=self_hash, payload=payload, resumed=False)


# ---------------------------------------------------------------------------
# Sec. 12a -- gap definitions, aggregation order, decomposition identity.
# ---------------------------------------------------------------------------
def _battery1_seed_final_v(condition: str, seed: int) -> float:
    """Seed-first mean `final_V` over the 5 ICs, from the FROZEN Battery 1
    record (`solver_completed`/finiteness already semantically verified by
    `build_setup_manifest`) -- no re-execution needed for the gap
    computation itself (only the TIMING re-measurement needs fresh B/E
    rollouts, a separate concern, Sec. 12)."""
    p = RESULTS_DIR / "solver_audit_execution_2026_09_01" / f"solver_audit_battery1_{condition}_seed{seed}_2026_09_01.json"
    payload = json.loads(p.read_text())
    return float(np.mean([ic["summary"]["final_V"] for ic in payload["per_ic"]]))


def compute_gaps(oracle_final_v_by_ic: list, nlp_final_v_by_ic: list) -> dict:
    """Sec. 12a. `oracle_final_v_by_ic`/`nlp_final_v_by_ic`: length-5 lists
    (one `final_V` per IC). Returns `optimizer_gap`, `gap_B`, `gap_E`
    (each a per-seed array plus bootstrap CI / Holm-adjusted p), and the
    mandatory decomposition-identity check."""
    oracle_final_v_by_ic = np.asarray(oracle_final_v_by_ic, dtype=np.float64)
    nlp_final_v_by_ic = np.asarray(nlp_final_v_by_ic, dtype=np.float64)
    optimizer_gap_point = float(np.mean(oracle_final_v_by_ic - nlp_final_v_by_ic))
    optimizer_gap_range = float(np.max(oracle_final_v_by_ic - nlp_final_v_by_ic)
                                 - np.min(oracle_final_v_by_ic - nlp_final_v_by_ic))
    oracle_mean = float(np.mean(oracle_final_v_by_ic))
    nlp_mean = float(np.mean(nlp_final_v_by_ic))

    seeds = list(alib.SEEDS)
    gap_b_per_seed = np.array([_battery1_seed_final_v("B", s) - oracle_mean for s in seeds])
    gap_e_per_seed = np.array([_battery1_seed_final_v("E", s) - oracle_mean for s in seeds])

    gap_b_ci = alib.bootstrap_ci(gap_b_per_seed)
    gap_e_ci = alib.bootstrap_ci(gap_e_per_seed)
    gap_b_p = alib.exact_permutation_pvalue(gap_b_per_seed)
    gap_e_p = alib.exact_permutation_pvalue(gap_e_per_seed)
    p_holm_b, p_holm_e = alib.holm_adjust([gap_b_p["p_value"], gap_e_p["p_value"]])

    # Mandatory decomposition-identity check (Sec. 12a): (gap_X + optimizer_gap)
    # == mean_IC(X) - mean_IC(NLP), to within 1e-9 -- blocks reporting on failure.
    b_seed_means = np.array([_battery1_seed_final_v("B", s) for s in seeds])
    e_seed_means = np.array([_battery1_seed_final_v("E", s) for s in seeds])
    identity_b = np.max(np.abs((gap_b_per_seed + optimizer_gap_point) - (b_seed_means - nlp_mean)))
    identity_e = np.max(np.abs((gap_e_per_seed + optimizer_gap_point) - (e_seed_means - nlp_mean)))
    decomposition_identity_passed = bool(identity_b < 1e-9 and identity_e < 1e-9)
    if not decomposition_identity_passed:
        raise RuntimeError(
            f"DECOMPOSITION_IDENTITY_FAILED: max|gap_B+optimizer_gap - (B-NLP)|={identity_b}, "
            f"max|gap_E+optimizer_gap - (E-NLP)|={identity_e} -- indicates an implementation bug, "
            f"not a substantive finding"
        )

    return dict(
        optimizer_gap=dict(point_estimate=optimizer_gap_point, per_ic_range=optimizer_gap_range),
        gap_B=dict(point_estimate=float(np.mean(gap_b_per_seed)), ci_lower=gap_b_ci["ci_lower"],
                   ci_upper=gap_b_ci["ci_upper"], p_value=gap_b_p["p_value"], p_value_holm=p_holm_b,
                   per_seed=gap_b_per_seed.tolist()),
        gap_E=dict(point_estimate=float(np.mean(gap_e_per_seed)), ci_lower=gap_e_ci["ci_lower"],
                   ci_upper=gap_e_ci["ci_upper"], p_value=gap_e_p["p_value"], p_value_holm=p_holm_e,
                   per_seed=gap_e_per_seed.tolist()),
        # Sec. 14.1 item 7 requires the raw resample_idx arrays to be
        # published as their own companion-NPZ-hashed artifact, not merely
        # digested in memory (v12 fix, Sec. 23 item 2) -- returned here
        # (not JSON-serialized in the two dicts above) for the caller
        # (`build_summary_manifest`) to persist and hash.
        _resample_idx_B=gap_b_ci["resample_idx"], _resample_idx_E=gap_e_ci["resample_idx"],
        decomposition_identity_passed=decomposition_identity_passed,
        oracle_mean_final_v=oracle_mean, nlp_mean_final_v=nlp_mean,
    )


# ---------------------------------------------------------------------------
# Sec. 12 -- timing pass. Scope-reduced per design doc Sec. 24: interleaves
# only the 20 B/E controllers (their own internal position-bias is what the
# rotation actually fixes); Oracle-20/Certified-NLP's timing is taken from
# their already-completed clean reference pass, not re-executed here (Sec.
# 12's "exactly two executions" rule for those two conditions is the
# stricter, more-reviewed constraint -- see Sec. 24 for the full account).
# ---------------------------------------------------------------------------
TIMING_MANIFEST_PATH = NAMESPACE / "oracle_nlp_timing_2026_09_03.json"
TIMING_ARRAYS_PATH = NAMESPACE / "oracle_nlp_timing_arrays_2026_09_03.npz"


def _be_reproduction_gate(timing_result: dict) -> list:
    """Sec. 12: the fresh same-session B/E timing re-execution must
    reproduce the frozen Battery 1 raw per-IC summary, for all 10 seeds x
    {B,E} x 5 ICs = 100 raw records."""
    mismatches = []
    for cond in ("B", "E"):
        for seed in range(10):
            name = f"{cond}_seed{seed}"
            frozen_path = (RESULTS_DIR / "solver_audit_execution_2026_09_01"
                           / f"solver_audit_battery1_{cond}_seed{seed}_2026_09_01.json")
            frozen_payload = json.loads(frozen_path.read_text())
            for ic_index in range(5):
                fresh_summary = timing_result[name]["controllers"][ic_index].summary()
                frozen_summary = frozen_payload["per_ic"][ic_index]["summary"]
                cmp = onl.gate_compare_flattened(
                    fresh_summary, frozen_summary,
                    exact_fields=("success", "solver_completed", "failed_at_step", "n_steps_completed"),
                    isclose_fields=("final_V", "max_V", "tv", "input_tv_per_actuator_physical",
                                     "input_tv_per_actuator_normalized"),
                )
                if not cmp["passed"]:
                    mismatches.append(dict(name=name, ic_index=ic_index, mismatches=cmp["mismatches"]))
    return mismatches


def run_timing_pass(bundle, un_lo_t, un_hi_t, oracle_records: list, nlp_records: list,
                     out_dir: Path = NAMESPACE) -> dict:
    if TIMING_MANIFEST_PATH.exists() and TIMING_ARRAYS_PATH.exists():
        self_hash, payload = sio.read_and_verify_json_artifact(TIMING_MANIFEST_PATH, expected_kind="oracle_nlp_timing_manifest")
        return dict(path=TIMING_MANIFEST_PATH, self_hash=self_hash, payload=payload, resumed=True)

    norm, pool = _shared_norm_and_pool()

    def _make_be_factory(condition, seed):
        m2_path = RESULTS_DIR / "ablation_execution_2026_08_31" / f"ablation_training_manifest_seed{seed}_2026_08_31.json"
        _, m2_payload = sio.read_and_verify_json_artifact(m2_path, expected_kind="ablation_training_manifest")
        seq = slib.load_condition_seq(condition, seed, m2_payload)
        return lambda: onl.LearnedModelTimingController(condition, seed, seq, norm, un_lo_t, un_hi_t,
                                                          bs.SIM, onl.INPUT_LO, onl.INPUT_HI)

    factories = {}
    for seed in range(10):
        for condition in ("B", "E"):
            factories[f"{condition}_seed{seed}"] = _make_be_factory(condition, seed)

    t0 = time.perf_counter()
    result = onl.run_interleaved_timing_pass(factories, list(bs.ICS), steps=onl.STEPS)
    t1 = time.perf_counter()

    be_mismatches = _be_reproduction_gate(result)
    be_reproduction_passed = len(be_mismatches) == 0
    if not be_reproduction_passed:
        raise RuntimeError(f"BE_TIMING_REPRODUCTION_GATE_FAILED: first mismatch {be_mismatches[0]}")

    npz_arrays = {}
    for name in factories:
        for ic_index in range(5):
            npz_arrays[f"{name}_ic{ic_index}"] = np.asarray(result[name]["raw_by_ic"][ic_index], dtype=np.float64)
    out_dir.mkdir(parents=True, exist_ok=True)
    sio.atomic_write_npz_no_overwrite(TIMING_ARRAYS_PATH, npz_arrays)
    npz_sha256 = sio.file_sha256(TIMING_ARRAYS_PATH)

    payload = dict(
        kind="oracle_nlp_timing_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        scope_note=("Sec. 24: interleaves only the 20 B/E controllers; Oracle-20/Certified-NLP timing "
                     "is from their own clean reference pass, not interleaved here"),
        be_interleaved_wall_clock_seconds=t1 - t0,
        be_medians={name: result[name]["median"] for name in factories},
        be_p95s={name: result[name]["p95"] for name in factories},
        be_deadline_exceedance_fractions={name: result[name]["deadline_exceedance_fraction"] for name in factories},
        be_reproduction_gate_passed=be_reproduction_passed,
        oracle20_clean_pass_timing=[r["payload"]["summary"]["rollout_solver_core_wall_clock_seconds"]
                                     for r in oracle_records],
        certified_nlp_clean_pass_timing=[r["payload"]["summary"]["rollout_solver_core_wall_clock_seconds"]
                                          for r in nlp_records],
        oracle20_deadline_exceedance_fraction=float(np.mean([
            np.mean(np.asarray(r["payload"]["summary"]["rollout_solver_core_wall_clock_seconds"]["raw"]) > 3.6)
            for r in oracle_records])),
        certified_nlp_deadline_exceedance_fraction=float(np.mean([
            np.mean(np.asarray(r["payload"]["summary"]["rollout_solver_core_wall_clock_seconds"]["raw"]) > 3.6)
            for r in nlp_records])),
        companion_npz_path=str(TIMING_ARRAYS_PATH), companion_npz_sha256=npz_sha256,
    )
    self_hash = sio.write_with_self_verification(TIMING_MANIFEST_PATH, onio.to_jsonable(payload))
    return dict(path=TIMING_MANIFEST_PATH, self_hash=self_hash, payload=payload, resumed=False)


SUMMARY_MANIFEST_PATH = NAMESPACE / "oracle_nlp_summary_manifest_2026_09_03.json"


RESAMPLE_ARRAYS_PATH = NAMESPACE / "oracle_nlp_resample_arrays_2026_09_03.npz"


def build_summary_manifest(setup_self_hash: str, oracle_records: list, nlp_records: list,
                            timing_result: dict, dest_path: Path = None) -> dict:
    dest_path = dest_path or SUMMARY_MANIFEST_PATH
    if dest_path.exists():
        self_hash, payload = sio.read_and_verify_json_artifact(dest_path, expected_kind="oracle_nlp_summary_manifest")
        return dict(path=dest_path, self_hash=self_hash, payload=payload, resumed=True)

    oracle_reference_valid = [r["payload"]["summary"]["reference_valid"] for r in oracle_records]
    nlp_reference_valid = [r["payload"]["summary"]["reference_valid"] for r in nlp_records]
    if not (all(oracle_reference_valid) and all(nlp_reference_valid)):
        return dict(path=None, self_hash=None,
                    payload=dict(kind="oracle_nlp_summary_manifest", outcome="REFERENCE_NOT_ESTABLISHED",
                                 oracle_reference_valid=oracle_reference_valid,
                                 nlp_reference_valid=nlp_reference_valid),
                    resumed=False)

    oracle_final_v = [r["payload"]["summary"]["final_V"] for r in oracle_records]
    nlp_final_v = [r["payload"]["summary"]["final_V"] for r in nlp_records]
    gaps = compute_gaps(oracle_final_v, nlp_final_v)

    # Sec. 14.1 item 7 / v12 fix (Sec. 23 item 2): publish the raw
    # resample_idx arrays as their own companion-NPZ-hashed artifact.
    if not RESAMPLE_ARRAYS_PATH.exists():
        sio.atomic_write_npz_no_overwrite(RESAMPLE_ARRAYS_PATH, dict(
            resample_idx_B=gaps.pop("_resample_idx_B"), resample_idx_E=gaps.pop("_resample_idx_E")))
    else:
        gaps.pop("_resample_idx_B", None); gaps.pop("_resample_idx_E", None)
    resample_arrays_sha256 = sio.file_sha256(RESAMPLE_ARRAYS_PATH)

    secondary = dict(
        oracle20=dict(success=[r["payload"]["summary"]["success"] for r in oracle_records],
                       settling_time=[r["payload"]["summary"]["settling_time"] for r in oracle_records],
                       tv=[r["payload"]["summary"]["tv"] for r in oracle_records],
                       lyapunov_increase_count=[r["payload"]["summary"]["lyapunov_increase_count"] for r in oracle_records]),
        certified_nlp=dict(success=[r["payload"]["summary"]["success"] for r in nlp_records],
                            settling_time=[r["payload"]["summary"]["settling_time"] for r in nlp_records],
                            tv=[r["payload"]["summary"]["tv"] for r in nlp_records],
                            lyapunov_increase_count=[r["payload"]["summary"]["lyapunov_increase_count"] for r in nlp_records]),
    )

    payload = dict(
        kind="oracle_nlp_summary_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_setup_manifest_sha256=setup_self_hash,
        predecessor_oracle20_record_hashes=[r["self_hash"] for r in oracle_records],
        predecessor_certnlp_record_hashes=[r["self_hash"] for r in nlp_records],
        predecessor_timing_manifest_sha256=timing_result["self_hash"],
        resample_arrays_path=str(RESAMPLE_ARRAYS_PATH), resample_arrays_sha256=resample_arrays_sha256,
        outcome="REPORTED",
        optimizer_gap=gaps["optimizer_gap"], gap_B=gaps["gap_B"], gap_E=gaps["gap_E"],
        decomposition_identity_passed=gaps["decomposition_identity_passed"],
        oracle_mean_final_v=gaps["oracle_mean_final_v"], nlp_mean_final_v=gaps["nlp_mean_final_v"],
        secondary_endpoints=secondary,
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    self_hash = sio.write_with_self_verification(dest_path, onio.to_jsonable(payload))
    return dict(path=dest_path, self_hash=self_hash, payload=payload, resumed=False)


# ---------------------------------------------------------------------------
# Canonical real-entry-point (Sec. 14). The ONLY function real execution
# should call.
# ---------------------------------------------------------------------------
def run_real_oracle_nlp_comparison(out_dir: Path = NAMESPACE) -> dict:
    verify_freeze_manifest()
    preflight = build_preflight_artifacts(out_dir)
    if not preflight.get("all_passed", preflight["payload"].get("all_passed")):
        raise RuntimeError("PREFLIGHT_GATES_FAILED: cannot proceed to real execution")
    setup = build_setup_manifest(out_dir)
    build_compat_gate_manifest(setup["self_hash"], out_dir)

    norm, pool = _shared_norm_and_pool()
    un_lo, un_hi, u_mean, u_std = _un_lo_hi(norm)
    un_lo_t = torch.tensor(un_lo, dtype=torch.float32)
    un_hi_t = torch.tensor(un_hi, dtype=torch.float32)
    bundle = onl.build_j_true(u_mean, u_std)

    # Sec. 9's object-construction contract: the Opti solver is built
    # EXACTLY ONCE per pass (clean, instrumented) and reused across all 5
    # ICs via `set_value(x0p, ...)` inside `solve_one_candidate` -- v12 fix
    # (Sec. 23 item 3): the preliminary run built a fresh Opti object per
    # IC (10 total instead of 2), violating this contract.
    nlp_clean = onl.build_certified_nlp_solver(bundle, un_lo, un_hi)
    nlp_instrumented = onl.build_certified_nlp_solver(bundle, un_lo, un_hi)

    oracle_records = []
    nlp_records = []
    for ic_index, x0 in enumerate(bs.ICS):
        oracle_records.append(run_oracle20_reference(bundle, ic_index, x0, un_lo_t, un_hi_t, out_dir))
        nlp_records.append(run_certified_nlp_reference(nlp_clean, nlp_instrumented, bundle, ic_index, x0, out_dir))

    timing = run_timing_pass(bundle, un_lo_t, un_hi_t, oracle_records, nlp_records, out_dir)
    summary = build_summary_manifest(setup["self_hash"], oracle_records, nlp_records, timing)
    return dict(setup=setup, oracle_records=oracle_records, nlp_records=nlp_records, timing=timing, summary=summary)
