#!/usr/bin/env python3
"""Driver / manifest-chain orchestration for the single-CSTR query-matched
solver audit + two-plant state-constraint minimum audit
(`docs/JPC_SOLVER_AUDIT_PROTOCOL_2026_09_01.md`, v7).

Builds this protocol's OWN manifest chain (Setup Manifest -> per-battery
Rollout records -> per-battery Summary records), separate from the real
go/no-go ablation's M1/M2/M3 (Section 4: "this protocol creates its own
manifest chain rather than resuming or extending the frozen ablation's
M1/M2/M3"). Depends on (never modifies) the real ablation's Freeze
Manifest and its per-seed Training/Results Manifests (M2/M3) for
checkpoint identity/hashes and the `closed_loop_success` values Gate 2
compares against.

Nothing in this module has real side effects merely by being imported --
`run_real_solver_audit()`/`run_real_two_cstr_minimum_audit()` are the
only functions with real side effects on the real pool/checkpoints, and
neither is called anywhere in this file or by anything that imports it.
Per Section 0's design-review -> implement -> synthetic smoke -> freeze
-> real-execution sequence, this file is not called against real data
until reviewed alongside `src/cstr/solver_audit_lib.py` and the
synthetic smoke suite passes.

Every per-battery/per-setup record writer below implements resume-or-
verify (mirroring `ablation_driver_2026_08_31.py`'s own discipline): if
the destination file already exists, it is re-verified (self-hash, kind,
exact keyset) and reused as-is -- never blindly trusted, never silently
rebuilt, never overwritten.
"""
from __future__ import annotations

import hashlib
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

import cstr.ablation_lib as alib              # noqa: E402
import cstr.solver_audit_io as sio            # noqa: E402
import cstr.solver_audit_lib as slib          # noqa: E402
import ablation_driver_2026_08_31 as adrv     # noqa: E402

RESULTS_DIR = ROOT / "results"
NAMESPACE = RESULTS_DIR / "solver_audit_execution_2026_09_01"
SMOKE_RESULTS_PATH = RESULTS_DIR / "solver_audit_synthetic_smoke_results_2026_09_01.json"
FREEZE_MANIFEST_PATH = ROOT / "docs" / "JPC_SOLVER_AUDIT_PROTOCOL_FREEZE_MANIFEST_2026_09_01.json"

# Primary MPC parameters, matching `budget_sweep_solver.main()`'s real
# closed-loop endpoint (horizon=3, budget=20) -- the SAME endpoint the
# real ablation's own `closed_loop_success` was measured at, so Gate 2's
# comparison is apples-to-apples.
HORIZON, STEPS, LR, RHO_U, SUCCESS_V, BUDGET = 3, 120, 0.1, 0.01, 2.0, 20
LCNN_CONDITIONS = ("A", "B", "C", "D", "E", "F")
FNN_CONDITIONS = ("B-FNN", "E-FNN")
ALL_CONDITIONS = LCNN_CONDITIONS + FNN_CONDITIONS
DETAILED_CONDITIONS = ("B", "E")  # Battery 2/3 scope (Section 2 items 2-3)
# Battery 1's OWN independently-executed conditions (Section 2's dedup
# contract, coverage-vs-duplication note): B/E rollout-summary/state-
# constraint data comes SOLELY from Battery 2, never re-run here.
BATTERY1_OWN_CONDITIONS = tuple(c for c in ALL_CONDITIONS if c not in DETAILED_CONDITIONS)

# GGN objective choice (Section 2 item 3 does not pin one; "V" matches
# `ggn_mpc_probe.py`'s own default/`main()` behavior) -- an explicit,
# recorded design decision, flagged here for the full re-review round.
GGN_OBJECTIVE = "V"

REQUIRED_UPSTREAM_FILES = [
    ROOT / "docs" / "JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_FREEZE_MANIFEST_2026_08_31.json",
    ROOT / "docs" / "JPC_SOLVER_AUDIT_PROTOCOL_2026_09_01.md",
    ROOT / "src" / "cstr" / "solver_audit_lib.py",
    ROOT / "src" / "cstr" / "solver_audit_io.py",
    ROOT / "scripts" / "solver_audit_driver_2026_09_01.py",
    # Gate 3/Gate 4 depend on these as READ-ONLY reference implementations
    # (the "original" side of each equivalence check) -- not part of the
    # ablation's own 12-file freeze, but drift here would silently change
    # what "the original" means without this freeze noticing.
    ROOT / "scripts" / "ggn_mpc_probe.py",
    ROOT / "src" / "cstr" / "two_cstr_series.py",
    ROOT / "scripts" / "two_cstr_lgrad.py",
    ROOT / "results" / "interim" / "logs" / "two_cstr_lgrad.json",
    ROOT / "scripts" / "solver_audit_synthetic_smoke_2026_09_01.py",
    SMOKE_RESULTS_PATH,
]

EXPECTED_SETUP_MANIFEST_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_ablation_freeze_sha256",
    "predecessor_solver_audit_freeze_sha256",
    "predecessor_ablation_m1_sha256", "predecessor_ablation_m2_sha256",
    "predecessor_ablation_m3_sha256", "pool_content_sha256_matches_m1",
    "resolved_checkpoints", "schema_version", "simulator_file_sha256",
    "ggn_mechanics_file_sha256", "ggn_objective", "torch_num_threads",
    "timing_warmup_steps", "isclose_rtol", "isclose_atol", "survivorship_threshold",
    "compute_device", "environment", "coverage", "horizon", "steps", "lr", "rho_u",
    "success_v", "budget",
}
EXPECTED_TWO_CSTR_SETUP_MANIFEST_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_solver_audit_freeze_sha256",
    "two_cstr_series_file_sha256", "two_cstr_lgrad_file_sha256",
    "legacy_log_sha256", "legacy_log_path", "seeds", "configs", "schema_version",
    "compute_device", "environment", "torch_num_threads",
}
EXPECTED_BATTERY1_RECORD_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_setup_manifest_sha256", "condition", "seed",
    "checkpoint_path", "checkpoint_sha256", "per_ic", "source",
    "seed_level_success_rate", "gate2_result",
}
EXPECTED_BATTERY2_RECORD_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_setup_manifest_sha256", "condition", "seed",
    "checkpoint_path", "checkpoint_sha256", "per_ic_summary", "gate1_results",
    "gate2_result", "iteration_arrays_path", "iteration_arrays_sha256",
}
EXPECTED_BATTERY3_RECORD_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_setup_manifest_sha256", "condition", "seed",
    "checkpoint_path", "checkpoint_sha256", "per_ic", "gate3_results", "objective",
}
EXPECTED_BATTERY5_RECORD_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_replica_manifest_sha256", "config_name", "seed",
    "replica_checkpoint_path", "replica_checkpoint_sha256", "reconstructed_metrics",
    "legacy_metrics", "gate4_result", "reload_equivalence_gate_results", "state_constraint_audit",
}
EXPECTED_BATTERY1_SUMMARY_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_setup_manifest_sha256", "seeds", "contrasts",
    "descriptive_d", "completion_rates", "settling_time_ecdf_path", "settling_time_ecdf_sha256",
    "resample_arrays_path", "resample_arrays_sha256", "input_record_hashes",
}
EXPECTED_BATTERY2_SUMMARY_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_setup_manifest_sha256", "seeds", "contrasts",
    "resample_arrays_path", "resample_arrays_sha256", "input_record_hashes",
}
EXPECTED_BATTERY3_SUMMARY_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_setup_manifest_sha256", "seeds", "descriptive",
    "input_record_hashes",
}
EXPECTED_BATTERY5_SUMMARY_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_two_cstr_setup_manifest_sha256", "seeds", "descriptive",
    "worst_case", "input_record_hashes",
}


def _resume_or_verify(path: Path, expected_kind: str, expected_keys: set, predecessor_check=None):
    """Resume-or-verify (Section 10-equivalent discipline, mirroring
    `ablation_driver_2026_08_31.py`): if `path` already exists, its
    self-hash/kind/exact-keyset are re-verified (never trusted blindly),
    then -- IMPORTANT, this is what distinguishes real evidence-aware
    resume from a self-hash-only check -- `predecessor_check(payload)`
    (when given) re-derives/re-checks whatever CURRENT evidence this
    artifact's own correctness actually depends on (an upstream hash
    chain, a checkpoint file's current content, a companion NPZ's current
    content) and must raise if anything has drifted since this artifact
    was written. A file whose OWN bytes are self-consistent but whose
    checkpoint/predecessor has since changed underneath it must never be
    silently reused. Returns `None` if the path does not exist yet (the
    caller must build and write it)."""
    if not path.exists():
        return None
    self_hash, payload = sio.read_and_verify_json_artifact(
        path, expected_kind=expected_kind, expected_keys=expected_keys)
    if predecessor_check is not None:
        predecessor_check(payload)
    return dict(path=path, self_hash=self_hash, payload=payload, resumed=True)


def _check_checkpoint_hash(checkpoint_path: str, expected_sha256: str, context: str) -> None:
    p = Path(checkpoint_path)
    if not p.exists():
        raise RuntimeError(f"RESUME_CHECKPOINT_MISSING: {checkpoint_path} ({context}) no longer exists")
    actual = sio.file_sha256(p)
    if actual != expected_sha256:
        raise RuntimeError(f"RESUME_CHECKPOINT_DRIFT: {checkpoint_path} ({context}) hash changed since this "
                           f"record was built ({expected_sha256} -> {actual})")


def _check_predecessor_hash(payload: dict, field: str, current_hash: str, context: str) -> None:
    if payload[field] != current_hash:
        raise RuntimeError(f"RESUME_PREDECESSOR_MISMATCH: {context}'s recorded {field} no longer matches the "
                           "CURRENT upstream artifact -- refusing to silently reuse a record built against "
                           "different provenance")


def _battery_record_predecessor_check(setup_hash: str, ckpt_path: str, ckpt_sha256: str, context: str):
    def _check(payload: dict) -> None:
        _check_predecessor_hash(payload, "predecessor_setup_manifest_sha256", setup_hash, context)
        _check_checkpoint_hash(ckpt_path, ckpt_sha256, context)
    return _check


def _input_record_hashes(records: dict) -> dict:
    """`records`: {(name, seed): record dict with a `self_hash` field}.
    Flattened to a JSON-friendly `{"name__seedN": self_hash}` map -- a
    Summary Manifest's OWN provenance record of exactly which input
    records it was built from, re-checked on resume (Section 5's
    aggregation must never silently go stale if any one input record was
    rebuilt/changed after the Summary was written)."""
    return {f"{name}__seed{seed}": rec["self_hash"] for (name, seed), rec in records.items()}


def _summary_predecessor_check(records: dict, recorded_field: str = "input_record_hashes",
                               companion_npz_fields: tuple = ()):
    """`companion_npz_fields`: `((path_field, sha256_field), ...)` -- e.g.
    `(("settling_time_ecdf_path", "settling_time_ecdf_sha256"),)`. Without
    this, a resumed Summary Manifest's own JSON self-hash/input-record
    hashes could still check out while its ECDF/resample-array companion
    NPZ was silently deleted or altered underneath it -- re-verified here
    exactly like every other companion-file check in this codebase."""
    fresh_hashes = _input_record_hashes(records)

    def _check(payload: dict) -> None:
        if payload[recorded_field] != fresh_hashes:
            raise RuntimeError(
                "RESUME_STALE_SUMMARY: one or more input battery records no longer match what this "
                "Summary Manifest was built from -- refusing to silently reuse a stale aggregation"
            )
        for path_field, sha256_field in companion_npz_fields:
            npz_path = ROOT / payload[path_field]
            if not npz_path.exists():
                raise RuntimeError(f"RESUME_SUMMARY_COMPANION_MISSING: {payload[path_field]} no longer exists")
            if sio.file_sha256(npz_path) != payload[sha256_field]:
                raise RuntimeError(f"RESUME_SUMMARY_COMPANION_DRIFT: {payload[path_field]} hash changed since "
                                   "this Summary Manifest was built")
    return _check


# ---------------------------------------------------------------------------
# Setup Manifest (Section 4): single-CSTR (batteries 1-4)
# ---------------------------------------------------------------------------
def build_setup_manifest(ablation_freeze_path: Path, ablation_m1_path: Path,
                          ablation_m2_paths: dict, ablation_m3_path: Path, pool: dict,
                          solver_audit_freeze_hash: str,
                          out_dir: Path = NAMESPACE, dest_path: Path = None) -> dict:
    """`ablation_m2_paths`: {seed: path} for every seed this run of the
    audit will cover. Verifies (not merely hashes) the upstream chain:
    freeze -> M1 -> every M2 -> M3, AND that `pool`'s actual content
    matches what M1 recorded (Section 4: "normalization/scaling
    provenance"), resolving and hash-checking every (condition, seed)
    checkpoint this run will need up front (fail fast, not deep inside a
    later battery). `solver_audit_freeze_hash`: THIS protocol's OWN
    Freeze Manifest hash (from `verify_freeze_manifest()`, called by the
    real entry point before this function) -- recorded here so a
    produced Setup Manifest can be traced back to the exact frozen
    protocol version that produced it, not just the (separate) real
    ablation's own freeze."""
    dest_path = dest_path if dest_path is not None else (out_dir / "solver_audit_setup_manifest_2026_09_01.json")

    # Every value below is computed FRESH regardless of whether `dest_
    # path` already exists -- this is what makes resume evidence-aware
    # rather than a bare self-hash check: an existing file is only reused
    # if its RECORDED values match what re-verifying the CURRENT upstream
    # chain (freeze/M1/M2/M3/checkpoints/pool) produces right now, not
    # merely because the file's own bytes are internally self-consistent.
    ablation_freeze_hash = adrv.verify_freeze_manifest(ablation_freeze_path)
    # Reuses `verify_query_manifest`'s OWN pool-content and freeze-chain
    # checks (passing `pool=pool, freeze_manifest_path=ablation_freeze_path`)
    # rather than re-deriving the same comparison independently -- a
    # second, slightly different implementation of the same check would
    # risk silently drifting from the frozen ablation driver's own logic.
    m1_verified = adrv.verify_query_manifest(ablation_m1_path, pool=pool, freeze_manifest_path=ablation_freeze_path)
    pool_matches_m1 = True

    m2_hashes = {}
    m2_payloads = {}
    for seed, path in ablation_m2_paths.items():
        self_hash, payload = sio.read_and_verify_json_artifact(path, expected_kind="ablation_training_manifest")
        if payload["predecessor_query_manifest_sha256"] != m1_verified["self_hash"]:
            raise RuntimeError(f"RESUME_PROVENANCE_MISMATCH: seed {seed}'s M2 was built against a different M1")
        m2_hashes[str(seed)] = self_hash
        m2_payloads[seed] = payload

    m3_verified = adrv.verify_results_manifest(ablation_m3_path)
    expected_m2_hashes = {str(s): m2_hashes[str(s)] for s in ablation_m2_paths}
    if m3_verified["payload"]["predecessor_training_manifest_sha256"] != expected_m2_hashes:
        raise RuntimeError(
            "RESUME_PROVENANCE_MISMATCH: the real ablation's M3 does not match the M2 hashes for "
            "the requested seed set (a different seed subset than what M3 covers)"
        )

    resolved_checkpoints = {}
    for condition in ALL_CONDITIONS:
        resolved_checkpoints[condition] = {}
        for seed, m2_payload in m2_payloads.items():
            prefix = slib.checkpoint_prefix_for_condition(condition, m2_payload)
            ckpt_path = slib.find_checkpoint_path(m2_payload, prefix, seed)  # re-verifies hash internally
            resolved_checkpoints[condition][str(seed)] = dict(
                path=ckpt_path, sha256=m2_payload["checkpoint_sha256"][ckpt_path])

    # Actual tensor placement, not `torch.backends.mps.is_available()`
    # (confirmed True on the development machine despite every real
    # computation running on CPU, per the grep-confirmed absence of any
    # `.to(...)`/`device=`/cuda/mps call in budget_sweep_solver.py,
    # ggn_mpc_probe.py, or ablation_lib.py).
    device = str(torch.zeros(1).device)
    # Full software/hardware environment (Section 4's "software/hardware
    # environment" requirement) -- device/thread-count alone are not
    # enough to make a wall-clock number reproducible/citable in a
    # response letter without also pinning what actually produced it.
    environment = dict(
        os_platform=platform.platform(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        torch_version=torch.__version__,
        cpu_architecture=platform.machine(),
        processor=platform.processor(),
    )

    fresh_payload_fields = dict(
        predecessor_ablation_freeze_sha256=ablation_freeze_hash,
        predecessor_solver_audit_freeze_sha256=solver_audit_freeze_hash,
        predecessor_ablation_m1_sha256=m1_verified["self_hash"],
        predecessor_ablation_m2_sha256=m2_hashes,
        predecessor_ablation_m3_sha256=m3_verified["self_hash"],
        pool_content_sha256_matches_m1=pool_matches_m1,
        resolved_checkpoints=resolved_checkpoints,
        compute_device=device,
        environment=environment,
    )
    existing = _resume_or_verify(dest_path, "solver_audit_setup_manifest", EXPECTED_SETUP_MANIFEST_PAYLOAD_KEYS)
    if existing is not None:
        for field, fresh_value in fresh_payload_fields.items():
            if existing["payload"][field] != fresh_value:
                raise RuntimeError(
                    f"RESUME_STALE_SETUP_MANIFEST: recorded {field!r} no longer matches the CURRENT "
                    "upstream evidence -- refusing to silently reuse a Setup Manifest built against "
                    "different provenance"
                )
        return existing

    payload = dict(
        kind="solver_audit_setup_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        **fresh_payload_fields,
        schema_version="solver_audit_v11_2026_09_02",
        simulator_file_sha256=sio.file_sha256(ROOT / "src" / "cstr" / "lcnn_paper_simulator.py"),
        ggn_mechanics_file_sha256=sio.file_sha256(ROOT / "scripts" / "ggn_mpc_probe.py"),
        ggn_objective=GGN_OBJECTIVE,
        torch_num_threads=sio.TIMING_TORCH_NUM_THREADS,
        timing_warmup_steps=sio.TIMING_WARMUP_STEPS,
        isclose_rtol=sio.ISCLOSE_RTOL, isclose_atol=sio.ISCLOSE_ATOL,
        survivorship_threshold=sio.SURVIVORSHIP_COMPLETION_RATE_THRESHOLD,
        coverage=dict(
            battery1_conditions=list(ALL_CONDITIONS), battery1_seeds=sorted(int(s) for s in m2_hashes),
            battery2_3_conditions=list(DETAILED_CONDITIONS), battery2_3_seeds=sorted(int(s) for s in m2_hashes),
            n_ic=5,
        ),
        horizon=HORIZON, steps=STEPS, lr=LR, rho_u=RHO_U, success_v=SUCCESS_V, budget=BUDGET,
    )
    return dict(path=dest_path, self_hash=sio.write_with_self_verification(dest_path, payload), payload=payload, resumed=False)


def verify_setup_manifest(path: Path) -> dict:
    self_hash, payload = sio.read_and_verify_json_artifact(
        path, expected_kind="solver_audit_setup_manifest", expected_keys=EXPECTED_SETUP_MANIFEST_PAYLOAD_KEYS)
    return dict(self_hash=self_hash, payload=payload)


def build_two_cstr_setup_manifest(solver_audit_freeze_hash: str, out_dir: Path = NAMESPACE,
                                  dest_path: Path = None) -> dict:
    """`solver_audit_freeze_hash`: THIS protocol's own Freeze Manifest
    hash (see `build_setup_manifest`'s identical rationale) -- two-CSTR
    Battery 5 genuinely retrains 20 models, so tracing which frozen
    protocol version produced a given real run matters here just as much
    as for the single-CSTR batteries."""
    dest_path = dest_path if dest_path is not None else (
        out_dir / "solver_audit_two_cstr_setup_manifest_2026_09_01.json")
    legacy_log = ROOT / "results" / "interim" / "logs" / "two_cstr_lgrad.json"
    # Computed fresh regardless of resume (same evidence-aware discipline
    # as `build_setup_manifest`): a resumed file is only reused if these
    # CURRENT file hashes/environment still match what it recorded.
    # Full software/hardware environment (matching `build_setup_
    # manifest`'s single-CSTR side) -- Battery 5 genuinely retrains 20
    # models, so its own wall-clock/reproducibility claims need the same
    # environment snapshot the single-CSTR Setup Manifest already pins.
    environment = dict(
        os_platform=platform.platform(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        torch_version=torch.__version__,
        cpu_architecture=platform.machine(),
        processor=platform.processor(),
    )
    device = str(torch.zeros(1).device)
    fresh_fields = dict(
        predecessor_solver_audit_freeze_sha256=solver_audit_freeze_hash,
        two_cstr_series_file_sha256=sio.file_sha256(ROOT / "src" / "cstr" / "two_cstr_series.py"),
        two_cstr_lgrad_file_sha256=sio.file_sha256(ROOT / "scripts" / "two_cstr_lgrad.py"),
        legacy_log_sha256=sio.file_sha256(legacy_log),
        compute_device=device,
        environment=environment,
        torch_num_threads=sio.TIMING_TORCH_NUM_THREADS,
    )

    def _predecessor_check(payload: dict) -> None:
        for field, fresh_value in fresh_fields.items():
            if payload[field] != fresh_value:
                raise RuntimeError(
                    f"RESUME_STALE_TWO_CSTR_SETUP_MANIFEST: recorded {field!r} no longer matches the "
                    "CURRENT file content"
                )

    existing = _resume_or_verify(dest_path, "solver_audit_two_cstr_setup_manifest",
                                 EXPECTED_TWO_CSTR_SETUP_MANIFEST_PAYLOAD_KEYS,
                                 predecessor_check=_predecessor_check)
    if existing is not None:
        return existing
    payload = dict(
        kind="solver_audit_two_cstr_setup_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        **fresh_fields,
        legacy_log_path=str(legacy_log.relative_to(ROOT)),
        seeds=slib.TWO_CSTR_SEEDS, configs=sorted(slib.TWO_CSTR_CONFIGS.keys()),
        schema_version="solver_audit_v11_2026_09_02_two_cstr",
    )
    return dict(path=dest_path, self_hash=sio.write_with_self_verification(dest_path, payload), payload=payload, resumed=False)


def verify_two_cstr_setup_manifest(path: Path) -> dict:
    self_hash, payload = sio.read_and_verify_json_artifact(
        path, expected_kind="solver_audit_two_cstr_setup_manifest",
        expected_keys=EXPECTED_TWO_CSTR_SETUP_MANIFEST_PAYLOAD_KEYS)
    return dict(self_hash=self_hash, payload=payload)


_NORM_CACHE = {}


def _shared_norm(sim):
    """`make_norm`'s output is a pure function of the real pool's
    normalization constants -- computed once and cached (not re-derived
    per rollout), matching `run_seed()`'s own `make_norm(data_tuple)`
    call in the real ablation driver."""
    key = id(sim)
    if key in _NORM_CACHE:
        return _NORM_CACHE[key]
    from disambiguate_landscape_vs_gradient import make_norm
    pool = alib.load_pool()
    data_tuple = (pool["XU"], pool["Y"], pool["xm"], pool["xs"], pool["ym"], pool["ys"],
                  pool["train_idx"], pool["test_idx"])
    result = make_norm(data_tuple)
    _NORM_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Battery 1 (+4 piggybacked): one record per (condition, seed). B/E are
# NOT independently run here (Section 2's dedup contract) -- their
# rollout-summary/state-constraint records are MERGED from Battery 2's
# already-collected data via `battery1_record_from_battery2`.
# ---------------------------------------------------------------------------
def run_battery1_condition_seed(condition: str, seed: int, m2_path: Path, sim, out_dir: Path,
                                 setup_hash: str) -> dict:
    assert condition in BATTERY1_OWN_CONDITIONS, (
        f"condition {condition!r} is a Battery-2/3 detailed condition (B/E) -- its Battery-1-level "
        f"record must be built via battery1_record_from_battery2(), not run_battery1_condition_seed()"
    )
    import budget_sweep_solver as bs
    path = out_dir / f"solver_audit_battery1_{condition}_seed{seed}_2026_09_01.json"
    # Cheap up front (JSON read + dict lookup + a hash check `find_
    # checkpoint_path` already does internally): resolved BEFORE the
    # resume check so a resumed record's predecessor/checkpoint identity
    # can actually be re-verified against CURRENT evidence, not merely
    # its own self-hash.
    _, m2_payload = sio.read_and_verify_json_artifact(m2_path, expected_kind="ablation_training_manifest")
    ckpt_prefix = slib.checkpoint_prefix_for_condition(condition, m2_payload)
    ckpt_path = slib.find_checkpoint_path(m2_payload, ckpt_prefix, seed)
    ckpt_sha256 = m2_payload["checkpoint_sha256"][ckpt_path]

    existing = _resume_or_verify(
        path, "solver_audit_battery1_record", EXPECTED_BATTERY1_RECORD_PAYLOAD_KEYS,
        predecessor_check=_battery_record_predecessor_check(
            setup_hash, ckpt_path, ckpt_sha256, f"Battery1 {condition} seed={seed}"))
    if existing is not None:
        return existing

    seq = slib.load_condition_seq(condition, seed, m2_payload)
    norm, un_lo, un_hi = _shared_norm(sim)

    per_ic = []
    for ic_idx, x0 in enumerate(bs.ICS):
        rollout = slib.single_cstr_rollout(
            seq, norm, x0, HORIZON, BUDGET, STEPS, LR, RHO_U, SUCCESS_V, un_lo, un_hi, sim,
            bs.INPUT_LO, bs.INPUT_HI, record_iterations=False)
        original = bs.closed_loop(seq, norm, x0, HORIZON, BUDGET, STEPS, LR, RHO_U, SUCCESS_V, un_lo, un_hi)
        sanity = slib.battery1_sanity_compare(rollout["summary"], original)
        constraint_stats = slib.state_constraint_stats(
            rollout["x_traj"], rollout["us"], sim, bs.INPUT_LO, bs.INPUT_HI)
        per_ic.append(dict(
            initial_condition_index=ic_idx, summary=rollout["summary"],
            state_constraint=constraint_stats, sanity_check_vs_frozen_closed_loop=sanity,
        ))
        if not sanity["passed"]:
            raise RuntimeError(
                f"BATTERY1_SANITY_CHECK_FAILED: condition={condition} seed={seed} ic={ic_idx} "
                f"mismatches={sanity['mismatches']}"
            )

    seed_level_success_rate = float(np.mean([r["summary"]["success"] for r in per_ic]))
    gate2 = slib.gate2_compare(seed_level_success_rate, m2_payload["closed_loop_success"][condition])

    payload = dict(
        kind="solver_audit_battery1_record",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_setup_manifest_sha256=setup_hash,
        condition=condition, seed=seed, source="battery1_independent",
        checkpoint_path=ckpt_path, checkpoint_sha256=m2_payload["checkpoint_sha256"][ckpt_path],
        per_ic=per_ic, seed_level_success_rate=seed_level_success_rate,
        gate2_result=gate2,
    )
    return dict(path=path, self_hash=sio.write_with_self_verification(path, payload), payload=payload, resumed=False)


def battery1_record_from_battery2(condition: str, seed: int, battery2_record_payload: dict, out_dir: Path,
                                  setup_hash: str) -> dict:
    """B/E's Battery-1-level record (Section 2's dedup contract): rollout-
    summary and state-constraint fields are MERGED from Battery 2's
    already-collected `per_ic_summary`, never independently re-run.
    Writes to the SAME `solver_audit_battery1_{condition}_seed{seed}_...`
    path Battery 1's own records use, so downstream aggregation (Section
    5) can treat all 8 conditions uniformly without special-casing B/E."""
    assert condition in DETAILED_CONDITIONS
    path = out_dir / f"solver_audit_battery1_{condition}_seed{seed}_2026_09_01.json"

    def _predecessor_check(payload: dict) -> None:
        _check_predecessor_hash(payload, "predecessor_setup_manifest_sha256", setup_hash,
                                f"Battery1(merged) {condition} seed={seed}")
        if payload["checkpoint_sha256"] != battery2_record_payload["checkpoint_sha256"]:
            raise RuntimeError(
                f"RESUME_PREDECESSOR_MISMATCH: Battery1(merged) {condition} seed={seed}'s recorded checkpoint "
                "hash no longer matches the CURRENT Battery 2 record it was merged from"
            )

    existing = _resume_or_verify(path, "solver_audit_battery1_record", EXPECTED_BATTERY1_RECORD_PAYLOAD_KEYS,
                                 predecessor_check=_predecessor_check)
    if existing is not None:
        return existing

    per_ic = [
        dict(initial_condition_index=r["initial_condition_index"], summary=r["summary"],
             state_constraint=r["state_constraint"], sanity_check_vs_frozen_closed_loop=None)
        for r in battery2_record_payload["per_ic_summary"]
    ]
    seed_level_success_rate = float(np.mean([r["summary"]["success"] for r in per_ic]))
    payload = dict(
        kind="solver_audit_battery1_record",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_setup_manifest_sha256=setup_hash,
        condition=condition, seed=seed, source="merged_from_battery2",
        checkpoint_path=battery2_record_payload["checkpoint_path"],
        checkpoint_sha256=battery2_record_payload["checkpoint_sha256"],
        per_ic=per_ic, seed_level_success_rate=seed_level_success_rate,
        gate2_result=battery2_record_payload["gate2_result"],
    )
    return dict(path=path, self_hash=sio.write_with_self_verification(path, payload), payload=payload, resumed=False)


# ---------------------------------------------------------------------------
# Battery 2 (detailed Adam audit, B/E only): one record per (condition, seed).
# ---------------------------------------------------------------------------
def run_battery2_condition_seed(condition: str, seed: int, m2_path: Path, sim, out_dir: Path,
                                 setup_hash: str) -> dict:
    import budget_sweep_solver as bs
    assert condition in DETAILED_CONDITIONS
    path = out_dir / f"solver_audit_battery2_{condition}_seed{seed}_2026_09_01.json"
    _, m2_payload = sio.read_and_verify_json_artifact(m2_path, expected_kind="ablation_training_manifest")
    ckpt_prefix = slib.checkpoint_prefix_for_condition(condition, m2_payload)
    ckpt_path = slib.find_checkpoint_path(m2_payload, ckpt_prefix, seed)
    ckpt_sha256 = m2_payload["checkpoint_sha256"][ckpt_path]

    def _predecessor_check(payload: dict) -> None:
        _check_predecessor_hash(payload, "predecessor_setup_manifest_sha256", setup_hash,
                                f"Battery2 {condition} seed={seed}")
        _check_checkpoint_hash(ckpt_path, ckpt_sha256, f"Battery2 {condition} seed={seed}")
        npz_path = ROOT / payload["iteration_arrays_path"]
        if not npz_path.exists():
            raise RuntimeError(f"RESUME_NPZ_MISSING: {payload['iteration_arrays_path']} no longer exists")
        if sio.file_sha256(npz_path) != payload["iteration_arrays_sha256"]:
            raise RuntimeError(f"RESUME_NPZ_DRIFT: {payload['iteration_arrays_path']} hash changed since "
                               "this record was built")

    existing = _resume_or_verify(path, "solver_audit_battery2_record", EXPECTED_BATTERY2_RECORD_PAYLOAD_KEYS,
                                 predecessor_check=_predecessor_check)
    if existing is not None:
        return existing

    seq = slib.load_condition_seq(condition, seed, m2_payload)
    norm, un_lo, un_hi = _shared_norm(sim)
    xm, xs, ym, ys = norm
    un_lo_np, un_hi_np = un_lo.numpy(), un_hi.numpy()

    per_ic_summary = []
    gate1_results = []
    iteration_arrays = {}
    step_arrays = {}
    for ic_idx, x0 in enumerate(bs.ICS):
        # Clean-timing pass FIRST (Section 3.2/6 item 3): zero per-
        # iteration logging overhead, so its OWN `rollout_solver_core_
        # wall_clock_seconds` is the trustworthy "core" number -- the
        # detailed pass below still measures its own timing, but that
        # number is relabeled diagnostic-only and never used as "core".
        light_rollout = slib.single_cstr_rollout(
            seq, norm, x0, HORIZON, BUDGET, STEPS, LR, RHO_U, SUCCESS_V, un_lo, un_hi, sim,
            bs.INPUT_LO, bs.INPUT_HI, record_iterations=False)

        rollout = slib.single_cstr_rollout(
            seq, norm, x0, HORIZON, BUDGET, STEPS, LR, RHO_U, SUCCESS_V, un_lo, un_hi, sim,
            bs.INPUT_LO, bs.INPUT_HI, record_iterations=True)
        original = bs.closed_loop(seq, norm, x0, HORIZON, BUDGET, STEPS, LR, RHO_U, SUCCESS_V, un_lo, un_hi)
        gate1 = slib.gate1_compare(rollout["summary"], original)
        gate1_results.append(dict(initial_condition_index=ic_idx, **gate1))
        sio.require_gate_pass("Gate1", gate1, context=f"{condition} seed={seed} ic={ic_idx}")
        gate1_light = slib.gate1_compare(light_rollout["summary"], original)
        sio.require_gate_pass("Gate1-clean-timing-pass", gate1_light,
                              context=f"{condition} seed={seed} ic={ic_idx}")

        # Section 3.4/3.5 enrichment is audit instrumentation, timed
        # SEPARATELY from the Adam solver core (Section 3.2/6 item 3) --
        # measured here and patched into the rollout summary afterward,
        # never left as an unmeasured/false-zero timing field.
        t_audit0 = time.perf_counter()
        for step_record in rollout["step_records"]:
            slib.enrich_step_record_with_residual_bridge(
                seq, sim, step_record, un_lo_np, un_hi_np, xm, xs, ym, ys, bs.INPUT_LO, bs.INPUT_HI)
            slib.enrich_step_record_with_counterfactual(sim, step_record, HORIZON, RHO_U, xm, xs, ym, ys)
        t_audit1 = time.perf_counter()
        # This pass's OWN timing (with logging overhead) is relabeled
        # diagnostic-only BEFORE `apply_measured_audit_wall_clock` fills
        # in the audit component; the CLEAN core timing from `light_
        # rollout` above is what actually gets reported as "core".
        rollout["summary"]["rollout_instrumented_wall_clock_with_logging_overhead_seconds"] = (
            rollout["summary"].pop("rollout_solver_core_wall_clock_seconds"))
        rollout["summary"]["rollout_solver_core_wall_clock_seconds"] = (
            light_rollout["summary"]["rollout_solver_core_wall_clock_seconds"])
        slib.apply_measured_audit_wall_clock(rollout["summary"], t_audit1 - t_audit0)

        constraint_stats = slib.state_constraint_stats(rollout["x_traj"], rollout["us"], sim,
                                                        bs.INPUT_LO, bs.INPUT_HI)
        per_ic_summary.append(dict(initial_condition_index=ic_idx, summary=rollout["summary"],
                                    state_constraint=constraint_stats))
        _flatten_step_records_into_arrays(rollout["step_records"], ic_idx, iteration_arrays)
        _flatten_step_level_arrays(rollout["step_records"], ic_idx, step_arrays)

    iteration_arrays_path = out_dir / f"solver_audit_battery2_{condition}_seed{seed}_iterations_2026_09_01.npz"
    iteration_arrays_sha256 = sio.write_npz_verify_if_exists(
        iteration_arrays_path, {**iteration_arrays, **step_arrays})

    seed_level_success_rate = float(np.mean([r["summary"]["success"] for r in per_ic_summary]))
    gate2 = slib.gate2_compare(seed_level_success_rate, m2_payload["closed_loop_success"][condition])

    payload = dict(
        kind="solver_audit_battery2_record",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_setup_manifest_sha256=setup_hash,
        condition=condition, seed=seed,
        checkpoint_path=ckpt_path, checkpoint_sha256=m2_payload["checkpoint_sha256"][ckpt_path],
        per_ic_summary=per_ic_summary, gate1_results=gate1_results, gate2_result=gate2,
        iteration_arrays_path=str(iteration_arrays_path.relative_to(ROOT)),
        iteration_arrays_sha256=iteration_arrays_sha256,
    )
    return dict(path=path, self_hash=sio.write_with_self_verification(path, payload), payload=payload, resumed=False)


def _flatten_step_records_into_arrays(step_records: list, ic_idx: int, arrays: dict) -> None:
    """Packs every logged iteration (across every control step of one IC's
    rollout) into flat per-field arrays, prefixed `ic{ic_idx}_`, for the
    Battery 2 NPZ companion (Section 4: "compressed NPZ arrays"). Retains
    `grad_hat`/`grad_true` (per-coordinate, 2 columns each), the FULL
    `un_before`/`un_raw`/`un_projected` horizon plans (flattened per
    `(horizon, action-dim)` coordinate -- Section 3.4's own raw/projected
    step data, not just derived scalars), the counterfactual-objective
    BEFORE/AFTER pair, and the FD diagnostic `kind` per coordinate
    (encoded as a short string array) so one-sided/exclusion statistics
    (Section 3.5's boundary/exclusion reporting) and the full before ->
    raw -> projected change per iteration are reconstructable from this
    artifact alone, without needing the original in-memory `step_records`."""
    rows = []
    for sr in step_records:
        for it_idx, it in enumerate(sr["iterations"]):
            rb = it.get("residual_bridge", {})
            fd_diag = rb.get("fd_diagnostics", [{"kind": "unknown"}, {"kind": "unknown"}])
            row = dict(
                step_index=sr["step_index"], iteration_index=it_idx,
                j_learned=it["j_learned"], j_learned_after=it.get("j_learned_after", float("nan")),
                counterfactual_true_objective_before=it.get("counterfactual_true_objective_before", float("nan")),
                counterfactual_true_objective_after=it.get("counterfactual_true_objective_after", float("nan")),
                n_clamped_coords=it["n_clamped_coords"],
                n_clamped_coords_block0=it["n_clamped_coords_block0"],
                g_hat=rb.get("g_hat", float("nan")), g_true=rb.get("g_true", float("nan")),
                grad_hat_0=rb.get("grad_hat", [float("nan")] * 2)[0],
                grad_hat_1=rb.get("grad_hat", [float("nan")] * 2)[1],
                grad_true_0=rb.get("grad_true", [float("nan")] * 2)[0],
                grad_true_1=rb.get("grad_true", [float("nan")] * 2)[1],
                cosine=rb.get("cosine", float("nan")), raw_inner_product=rb.get("raw_inner_product", float("nan")),
                excluded=rb.get("excluded", True),
                directional_derivative_true_raw_step=rb.get("directional_derivative_true_raw_step", float("nan")),
                directional_derivative_true_projected_step=rb.get(
                    "directional_derivative_true_projected_step", float("nan")),
                projection_active=rb.get("projection_active", False),
                fd_kind_0=fd_diag[0]["kind"], fd_kind_1=fd_diag[1]["kind"],
            )
            n_h, n_d = it["un_before"].shape
            for h in range(n_h):
                for d in range(n_d):
                    row[f"un_before_h{h}_d{d}"] = float(it["un_before"][h, d])
                    row[f"un_raw_h{h}_d{d}"] = float(it["un_raw"][h, d])
                    row[f"un_projected_h{h}_d{d}"] = float(it["un_projected"][h, d])
            rows.append(row)
    prefix = f"ic{ic_idx}_"
    if not rows:
        return
    for key in rows[0]:
        values = [r[key] for r in rows]
        arrays[prefix + key] = np.array(values) if not isinstance(values[0], str) else np.array(values, dtype="U32")


def _flatten_step_level_arrays(step_records: list, ic_idx: int, arrays: dict) -> None:
    """Packs every logged CONTROL STEP (not iteration) of one IC's rollout
    into flat per-field arrays, prefixed `ic{ic_idx}_perstep_` (distinct
    from `_flatten_step_records_into_arrays`'s per-ITERATION prefix, which
    also happens to store a `step_index` column) -- Section 3.3's
    per-control-step fields (warm-start plan, final plan, predicted vs.
    REALIZED next state/V, realized decrease) were previously only held
    in the in-memory `step_records` and discarded once this function
    returned; now published to the same NPZ companion."""
    if not step_records:
        return
    horizon, action_dim = step_records[0]["warm_start_un"].shape
    rows = []
    for sr in step_records:
        row = dict(
            step_index=sr["step_index"],
            x_before_0=float(sr["x_before"][0]), x_before_1=float(sr["x_before"][1]),
            u0_0=float(sr["u0"][0]), u0_1=float(sr["u0"][1]),
            predicted_next_state_0=float(sr["predicted_next_state"][0]),
            predicted_next_state_1=float(sr["predicted_next_state"][1]),
            predicted_V=float(sr["predicted_V"]),
            V_before=float(sr["V_before"]), V_after_realized=float(sr["V_after_realized"]),
            realized_decrease=float(sr["realized_decrease"]),
            solve_wall_clock_seconds=float(sr["solve_wall_clock_seconds"]),
        )
        for h in range(horizon):
            for d in range(action_dim):
                row[f"warm_start_un_h{h}_d{d}"] = float(sr["warm_start_un"][h, d])
                row[f"final_plan_un_h{h}_d{d}"] = float(sr["final_plan_un"][h, d])
        rows.append(row)
    prefix = f"ic{ic_idx}_perstep_"
    for key in rows[0]:
        arrays[prefix + key] = np.array([r[key] for r in rows])


# ---------------------------------------------------------------------------
# Battery 3 (GGN descriptive audit, B/E only): one record per (condition, seed).
# ---------------------------------------------------------------------------
def run_battery3_condition_seed(condition: str, seed: int, m2_path: Path, sim, out_dir: Path,
                                 setup_hash: str) -> dict:
    import ggn_mpc_probe as ggn
    assert condition in DETAILED_CONDITIONS
    path = out_dir / f"solver_audit_battery3_{condition}_seed{seed}_2026_09_01.json"
    _, m2_payload = sio.read_and_verify_json_artifact(m2_path, expected_kind="ablation_training_manifest")
    ckpt_prefix = slib.checkpoint_prefix_for_condition(condition, m2_payload)
    ckpt_path = slib.find_checkpoint_path(m2_payload, ckpt_prefix, seed)
    ckpt_sha256 = m2_payload["checkpoint_sha256"][ckpt_path]

    existing = _resume_or_verify(
        path, "solver_audit_battery3_record", EXPECTED_BATTERY3_RECORD_PAYLOAD_KEYS,
        predecessor_check=_battery_record_predecessor_check(
            setup_hash, ckpt_path, ckpt_sha256, f"Battery3 {condition} seed={seed}"))
    if existing is not None:
        return existing

    seq = slib.load_condition_seq(condition, seed, m2_payload)
    norm, un_lo, un_hi = _shared_norm(sim)

    per_ic = []
    gate3_results = []
    for ic_idx, x0 in enumerate(ggn.ICS):
        original = ggn.ggn_closed_loop(seq, norm, x0, HORIZON, BUDGET, STEPS, RHO_U, SUCCESS_V,
                                        un_lo, un_hi, GGN_OBJECTIVE)
        # Clean-timing pass (log_iterations=False): zero per-iteration
        # bookkeeping overhead -- its OWN timing is the trustworthy "core"
        # number (Section 3.2/6 item 3).
        clean = slib.ggn_closed_loop_instrumented(seq, norm, x0, HORIZON, BUDGET, STEPS, RHO_U,
                                                  SUCCESS_V, un_lo, un_hi, GGN_OBJECTIVE, sim,
                                                  log_iterations=False)
        gate3_clean = slib.gate3_compare(clean, original)
        sio.require_gate_pass("Gate3-clean-timing-pass", gate3_clean, context=f"{condition} seed={seed} ic={ic_idx}")

        instrumented = slib.ggn_closed_loop_instrumented(seq, norm, x0, HORIZON, BUDGET, STEPS, RHO_U,
                                                          SUCCESS_V, un_lo, un_hi, GGN_OBJECTIVE, sim,
                                                          log_iterations=True)
        gate3 = slib.gate3_compare(instrumented, original)
        gate3_results.append(dict(initial_condition_index=ic_idx, **gate3))
        sio.require_gate_pass("Gate3", gate3, context=f"{condition} seed={seed} ic={ic_idx}")
        # Merge: the "core" timing comes from the clean pass; the fully-
        # logged pass supplies everything else (its own timing renamed
        # diagnostic-only, never claimed as core -- already the key
        # `ggn_closed_loop_instrumented` gives it when `log_iterations=True`).
        instrumented["rollout_solver_core_wall_clock_seconds"] = clean["rollout_solver_core_wall_clock_seconds"]
        per_ic.append(dict(initial_condition_index=ic_idx, summary=instrumented))

    payload = dict(
        kind="solver_audit_battery3_record",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_setup_manifest_sha256=setup_hash,
        condition=condition, seed=seed,
        checkpoint_path=ckpt_path, checkpoint_sha256=m2_payload["checkpoint_sha256"][ckpt_path],
        per_ic=per_ic, gate3_results=gate3_results, objective=GGN_OBJECTIVE,
    )
    return dict(path=path, self_hash=sio.write_with_self_verification(path, payload), payload=payload, resumed=False)


# ---------------------------------------------------------------------------
# Battery 5 (two-CSTR minimum audit), Section 8's four PHASES, run as
# their own top-level stages (not fused per-(config,seed)): phase 1+2
# reconstruct+gate-check every (config, seed) pair; ONLY if ALL 20 pass
# does phase 3 freeze the checkpoint set; phase 4 then RELOADS from the
# frozen checkpoint files (never the in-memory training object) to run
# the instrumented audit.
# ---------------------------------------------------------------------------
def battery5_phase1_and_2_reconstruct_and_gate(config_name: str, seed: int, out_dir: Path,
                                               checkpoint_dir: Path) -> dict:
    """Phases 1-2 only: reconstruct + Gate 4. Does NOT run the phase-4
    audit -- callers must not proceed to phase 4 for ANY (config, seed)
    until every pair in the batch has passed this AND phase 3's Replica
    Manifest has been built (`build_replica_manifest`)."""
    replica = slib.reconstruct_two_cstr_replica(config_name, seed, checkpoint_dir)
    legacy = slib.load_two_cstr_legacy_metrics(config_name, seed)
    gate4 = slib.gate4_compare(replica["metrics"], legacy)
    if not gate4["passed"]:
        raise sio.TwoCstrReconstructionIncompatible(
            f"TWO_CSTR_RECONSTRUCTION_INCOMPATIBLE: config={config_name} seed={seed} "
            f"mismatches={gate4['mismatches']}"
        )
    return dict(replica=replica, legacy=legacy, gate4=gate4)


EXPECTED_TWO_CSTR_REPLICA_MANIFEST_PAYLOAD_KEYS = {
    "kind", "generated_at_utc", "predecessor_two_cstr_setup_manifest_sha256", "replicas",
}


def _replica_key(config_name: str, seed: int) -> str:
    return f"{config_name}__seed{seed}"


def build_replica_manifest(reconstructions: dict, out_dir: Path, two_cstr_setup_hash: str) -> dict:
    """Phase 3's REAL freeze artifact -- a self-verifying JSON pinning
    every (config, seed) pair's reconstructed metrics, Gate 4 result, and
    checkpoint path+hash, built only once ALL pairs have passed Gate 4
    (an all-or-none contract: any failure already raised
    `TwoCstrReconstructionIncompatible` before `reconstructions` could be
    fully populated). Phase 4 verifies against THIS manifest -- never
    against whatever a caller happens to hold in memory."""
    path = out_dir / "solver_audit_two_cstr_replica_manifest_2026_09_01.json"
    replicas = {}
    for (config_name, seed), recon in reconstructions.items():
        replicas[_replica_key(config_name, seed)] = dict(
            config_name=config_name, seed=seed,
            checkpoint_path=recon["replica"]["checkpoint_path"],
            checkpoint_sha256=recon["replica"]["checkpoint_sha256"],
            metrics=recon["replica"]["metrics"], legacy_metrics=recon["legacy"], gate4_result=recon["gate4"],
        )

    def _predecessor_check(payload: dict) -> None:
        _check_predecessor_hash(payload, "predecessor_two_cstr_setup_manifest_sha256", two_cstr_setup_hash,
                                "two-CSTR Replica Manifest")
        if payload["replicas"] != replicas:
            raise RuntimeError(
                "RESUME_STALE_REPLICA_MANIFEST: recorded replica set no longer matches the CURRENT "
                "reconstructions (metrics/gate4/checkpoint identity) -- refusing to silently reuse it"
            )

    existing = _resume_or_verify(path, "solver_audit_two_cstr_replica_manifest",
                                 EXPECTED_TWO_CSTR_REPLICA_MANIFEST_PAYLOAD_KEYS,
                                 predecessor_check=_predecessor_check)
    if existing is not None:
        return existing
    payload = dict(
        kind="solver_audit_two_cstr_replica_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_two_cstr_setup_manifest_sha256=two_cstr_setup_hash,
        replicas=replicas,
    )
    return dict(path=path, self_hash=sio.write_with_self_verification(path, payload), payload=payload, resumed=False)


def verify_replica_manifest(path: Path) -> dict:
    self_hash, payload = sio.read_and_verify_json_artifact(
        path, expected_kind="solver_audit_two_cstr_replica_manifest",
        expected_keys=EXPECTED_TWO_CSTR_REPLICA_MANIFEST_PAYLOAD_KEYS)
    for key, r in payload["replicas"].items():
        p = Path(r["checkpoint_path"])
        if not p.exists():
            raise RuntimeError(f"REPLICA_CHECKPOINT_MISSING: {r['checkpoint_path']} (key={key}) no longer exists")
        if sio.file_sha256(p) != r["checkpoint_sha256"]:
            raise RuntimeError(
                f"REPLICA_CHECKPOINT_DRIFT: {r['checkpoint_path']} (key={key}) hash changed since "
                "the Replica Manifest was built"
            )
    return dict(self_hash=self_hash, payload=payload)


def battery5_phase4_instrumented_audit(config_name: str, seed: int, out_dir: Path,
                                       replica_manifest_hash: str, replica_manifest_payload: dict) -> dict:
    """Phase 4: verifies against the REAL Replica Manifest (never a
    caller-supplied in-memory dict) and RELOADS the model from the FROZEN
    checkpoint file it references (never the in-memory training object
    from phase 1) -- proves the persisted checkpoint is actually what the
    audit ran against, not merely what training produced in memory. Also
    runs a per-IC equivalence check of `two_cstr_rollout_instrumented`
    against the ORIGINAL, unmodified `two_cstr_lgrad.closed_loop()` on the
    reloaded model (mirroring Battery 1's own "cheap sanity check ...
    worth including" spirit, Section 2 item 4) before trusting the
    instrumented audit's output."""
    path = out_dir / f"solver_audit_battery5_{config_name}_seed{seed}_2026_09_01.json"
    replica_entry = replica_manifest_payload["replicas"][_replica_key(config_name, seed)]
    checkpoint_path = replica_entry["checkpoint_path"]
    reconstructed_metrics = replica_entry["metrics"]
    legacy_metrics = replica_entry["legacy_metrics"]

    def _predecessor_check(payload: dict) -> None:
        _check_predecessor_hash(payload, "predecessor_replica_manifest_sha256", replica_manifest_hash,
                                f"Battery5 {config_name} seed={seed}")
        _check_checkpoint_hash(checkpoint_path, replica_entry["checkpoint_sha256"],
                               f"Battery5 {config_name} seed={seed}")

    existing = _resume_or_verify(path, "solver_audit_battery5_record", EXPECTED_BATTERY5_RECORD_PAYLOAD_KEYS,
                                 predecessor_check=_predecessor_check)
    if existing is not None:
        return existing
    gate4_result = replica_entry["gate4_result"]
    if sio.file_sha256(Path(checkpoint_path)) != replica_entry["checkpoint_sha256"]:
        raise RuntimeError(f"REPLICA_CHECKPOINT_DRIFT: {checkpoint_path} hash changed since the Replica Manifest "
                           "was built")

    import two_cstr_lgrad as T
    lam_val, lam_grad = slib.TWO_CSTR_CONFIGS[config_name]
    torch.manual_seed(seed)
    reloaded_model = T.build_lcnn(seed)
    state = torch.load(checkpoint_path, map_location="cpu")
    reloaded_model.load_state_dict(state)
    reloaded_model.eval()
    seq = T.freeze(reloaded_model)

    data = T.make_data(n=12000)
    norm = T.make_norm(data)
    un_lo = torch.tensor((T.BOX_LO - data[2][4:]) / data[3][4:], dtype=torch.float32)
    un_hi = torch.tensor((T.BOX_HI - data[2][4:]) / data[3][4:], dtype=torch.float32)

    # Two-CSTR's own lr/rho_u/success_V convention (`two_cstr_lgrad.main()`:
    # horizon, steps, lr, rho_u, success_V = 3, 120, 0.2, 0.01, 2.0) --
    # distinct from the single-CSTR audit's lr=0.1 (Gate-2-matched
    # endpoint); budget=20 chosen to match the higher-budget "b20"
    # convention already reported in `two_cstr_lgrad.json`.
    TWO_CSTR_LR = 0.2

    audit_rows = []
    reload_equivalence_gate_results = []
    for ic_idx, x0 in enumerate(T.ICS):
        original = T.closed_loop(seq, norm, x0, HORIZON, BUDGET, STEPS, TWO_CSTR_LR, 0.01, SUCCESS_V,
                                 un_lo, un_hi)
        rollout = slib.two_cstr_rollout_instrumented(
            seq, norm, x0, HORIZON, BUDGET, STEPS, TWO_CSTR_LR, 0.01, SUCCESS_V, un_lo, un_hi)
        equivalence = sio.gate_compare_dict(
            dict(final_V=rollout["final_V"], max_V=rollout["max_V"], success=rollout["success"]),
            original, exact_keys=("success",), isclose_keys=("final_V", "max_V"))
        reload_equivalence_gate_results.append(dict(initial_condition_index=ic_idx, **equivalence))
        sio.require_gate_pass("Battery5-reload-equivalence", equivalence,
                              context=f"{config_name} seed={seed} ic={ic_idx}")

        stats = slib.state_constraint_stats(rollout["x_traj"], rollout["us"], T.SIM, T.BOX_LO, T.BOX_HI,
                                            temperature_state_indices=(1, 3))
        audit_rows.append(dict(initial_condition_index=ic_idx,
                                summary=dict(final_V=rollout["final_V"], max_V=rollout["max_V"],
                                            success=rollout["success"],
                                            solver_completed=rollout["solver_completed"]),
                                state_constraint=stats))

    payload = dict(
        kind="solver_audit_battery5_record",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_replica_manifest_sha256=replica_manifest_hash,
        config_name=config_name, seed=seed,
        replica_checkpoint_path=checkpoint_path,
        replica_checkpoint_sha256=sio.file_sha256(Path(checkpoint_path)),
        reconstructed_metrics=reconstructed_metrics, legacy_metrics=legacy_metrics, gate4_result=gate4_result,
        reload_equivalence_gate_results=reload_equivalence_gate_results,
        state_constraint_audit=audit_rows,
    )
    return dict(path=path, self_hash=sio.write_with_self_verification(path, payload), payload=payload, resumed=False)


# ---------------------------------------------------------------------------
# Section 5 -- statistical contract (reuses `alib`'s own bootstrap-CI /
# exact-permutation machinery, never reimplemented).
# ---------------------------------------------------------------------------
def paired_diff(metric_a: dict, metric_b: dict, seeds: list) -> np.ndarray:
    return np.array([metric_a[str(s)] - metric_b[str(s)] for s in seeds], dtype=np.float64)


def _safe_evaluate_contrast(diffs: np.ndarray) -> tuple:
    """Guards against a NaN silently propagating through `bootstrap_ci`/
    `exact_permutation_pvalue` (percentile/mean of an array containing
    NaN is NaN, with no error raised) whenever fewer than the full
    pre-registered seed count contributes a FINITE paired difference --
    e.g. a condition had ZERO completed ICs for one seed (Section 3.7's
    completed-only denominator leaves that seed's metric as NaN), or
    (Battery 2) every logged iteration for a seed was `excluded`. Rather
    than silently reporting a meaningless NaN CI/p-value as if it were a
    real (if wide) interval, returns an EXPLICIT
    `CONTRAST_UNAVAILABLE_INCOMPLETE_EVIDENCE` status with `n_valid_seeds`
    disclosed. Returns `(result_dict, resample_idx_or_None)`."""
    finite_mask = np.isfinite(diffs)
    if not np.all(finite_mask):
        return dict(point_estimate=float("nan"), ci_lower=float("nan"), ci_upper=float("nan"),
                    p_value=float("nan"), status="CONTRAST_UNAVAILABLE_INCOMPLETE_EVIDENCE",
                    n_valid_seeds=int(finite_mask.sum()), n_seeds=int(len(diffs))), None
    raw = alib.evaluate_contrast(diffs)
    resample_idx = raw.pop("resample_idx", None)
    result = dict(raw)
    result["status"] = "OK"
    result["n_valid_seeds"] = int(len(diffs))
    result["n_seeds"] = int(len(diffs))
    return result, resample_idx


def contrast_with_survivorship_rule(metric_a: dict, metric_b: dict, completion_a: dict, completion_b: dict,
                                    seeds: list) -> tuple:
    """Returns `(clean_result_dict, resample_idx_array)` -- the bootstrap
    resample-index array is NOT JSON-serializable and is stripped from
    the dict a caller stores in the Summary Manifest's own JSON, but is
    returned separately so the caller can persist it in a companion NPZ
    (matching the real ablation's own `_build_resample_arrays`
    reproducibility convention, Section 10 of that protocol) rather than
    silently discarding it.

    Survivorship gap, fixed to match the protocol's own text (Section 5):
    the OVERALL per-condition completion rate (mean across ALL seeds),
    not `max` of PER-SEED completion-rate differences -- the latter lets
    a single IC failing in one seed (a per-seed gap of up to 0.2 for 5
    ICs) trigger the same downgrade as a genuine condition-wide
    completion-rate difference, when the actual aggregate difference
    might be as small as 1/50 = 0.02."""
    diffs = paired_diff(metric_a, metric_b, seeds)
    result, resample_idx = _safe_evaluate_contrast(diffs)
    overall_completion_a = float(np.mean([completion_a[str(s)] for s in seeds]))
    overall_completion_b = float(np.mean([completion_b[str(s)] for s in seeds]))
    gap = abs(overall_completion_a - overall_completion_b)
    result["survivorship_downgraded_to_descriptive"] = (
        result["status"] != "OK" or gap > sio.SURVIVORSHIP_COMPLETION_RATE_THRESHOLD)
    result["completion_rate_gap"] = gap
    return result, resample_idx


def _extract_field(r: dict, path: tuple):
    cur = r
    for p in path:
        cur = cur[p]
    return cur


# Every continuous-endpoint metric Section 5 actually names (Battery 1/4's
# rollout summary + Section 3.6 state-constraint fields) -- `path` walks
# into one `per_ic` entry (`{"summary": {...}, "state_constraint": {...}}`);
# `full_denominator=True` means EVERY attempted IC counts (failed ICs
# already carry the correct sentinel -- `success=False`, `settling_time=
# steps+1` -- so they must NOT be dropped from this metric's own
# denominator); `False` means completed-rollouts-only (Section 3.7).
BATTERY1_METRIC_SPECS = {
    "success": (("summary", "success"), True),
    "settling_time": (("summary", "settling_time"), True),
    "final_V": (("summary", "final_V"), False),
    "max_V": (("summary", "max_V"), False),
    "tv": (("summary", "tv"), False),
    "lyapunov_increase_count": (("summary", "lyapunov_increase_count"), False),
    "solver_core_wall_clock_median_seconds": (
        ("summary", "rollout_solver_core_wall_clock_seconds", "median"), False),
    "solver_core_wall_clock_p95_seconds": (
        ("summary", "rollout_solver_core_wall_clock_seconds", "p95"), False),
    "input_tv_per_actuator_physical_0": (("summary", "input_tv_per_actuator_physical", 0), False),
    "input_tv_per_actuator_physical_1": (("summary", "input_tv_per_actuator_physical", 1), False),
    "input_tv_per_actuator_normalized_0": (("summary", "input_tv_per_actuator_normalized", 0), False),
    "input_tv_per_actuator_normalized_1": (("summary", "input_tv_per_actuator_normalized", 1), False),
    "max_temperature_deviation_K": (
        ("state_constraint", "per_temperature_index", "1", "max_temperature_deviation_K"), False),
    "max_temperature_absolute_K": (
        ("state_constraint", "per_temperature_index", "1", "max_temperature_absolute_K"), False),
    "temperature_violation_frequency": (
        ("state_constraint", "per_temperature_index", "1", "temperature_violation_frequency"), False),
    "temperature_violation_integral": (
        ("state_constraint", "per_temperature_index", "1", "temperature_violation_integral"), False),
    "input_saturation_frequency_actuator0": (
        ("state_constraint", "input_saturation_frequency", "0"), False),
    "input_saturation_frequency_actuator1": (
        ("state_constraint", "input_saturation_frequency", "1"), False),
    "temperature_cap_satisfied_posthoc": (("state_constraint", "temperature_cap_satisfied_posthoc"), False),
    "input_box_satisfied": (("state_constraint", "input_box_satisfied"), False),
    "input_never_saturated": (("state_constraint", "input_never_saturated"), False),
}
BATTERY1_CONTRAST_PAIRS = dict(
    E_vs_B=("E", "B"), F_vs_E=("F", "E"), C_vs_B=("C", "B"), EFNN_vs_BFNN=("E-FNN", "B-FNN"))


def _battery1_seed_level_metric(records: dict, condition: str, seeds: list, metric_name: str) -> dict:
    """`records`: {(condition, seed): battery1-record `dict(path=..,
    self_hash=.., payload=..)`}. Mean over the 5 ICs, matching Section 5's
    "aggregate the 5 ICs within each seed FIRST" -- denominator rule
    (full vs. completed-only) per `BATTERY1_METRIC_SPECS`."""
    path, full_denominator = BATTERY1_METRIC_SPECS[metric_name]
    out = {}
    for seed in seeds:
        payload = records[(condition, seed)]["payload"]
        per_ic = payload["per_ic"]
        selected = per_ic if full_denominator else [r for r in per_ic if r["summary"].get("solver_completed", True)]
        vals = [float(_extract_field(r, path)) for r in selected]
        out[str(seed)] = float(np.mean(vals)) if vals else float("nan")
    return out


def _battery1_completion_rate(records: dict, condition: str, seeds: list) -> dict:
    out = {}
    for seed in seeds:
        payload = records[(condition, seed)]["payload"]
        completions = [bool(r["summary"].get("solver_completed", True)) for r in payload["per_ic"]]
        out[str(seed)] = float(np.mean(completions))
    return out


def _battery1_pooled_settling_time(records: dict, condition: str, seeds: list) -> np.ndarray:
    """Raw (unaveraged) settling_time across every (seed, IC) pair for one
    condition -- the pooled sample Section 6 item 1's ECDF is reported
    over, distinct from the seed-level MEAN `_battery1_seed_level_metric`
    computes for the contrast family."""
    vals = []
    for seed in seeds:
        payload = records[(condition, seed)]["payload"]
        vals.extend(float(r["summary"]["settling_time"]) for r in payload["per_ic"])
    return np.array(vals, dtype=np.float64)


def build_battery1_summary_manifest(battery1_records: dict, seeds: list, out_dir: Path,
                                    setup_hash: str) -> dict:
    """Section 5's continuous-endpoint contrast family, for EVERY metric
    in `BATTERY1_METRIC_SPECS` (not `success` alone): E-B (primary), F-E
    (ceiling separation), C-B (symmetric residual), EFNN-BFNN
    (architecture interaction), each with the survivorship-downgrade rule
    applied per contrast. D is descriptive-only (Section 5), not
    contrasted -- its own per-seed values are reported alongside. The
    pooled (unaveraged) `settling_time` sample per condition is written
    to a companion NPZ for Section 6 item 1's ECDF requirement."""
    path = out_dir / "solver_audit_battery1_summary_2026_09_01.json"
    existing = _resume_or_verify(
        path, "solver_audit_battery1_summary", EXPECTED_BATTERY1_SUMMARY_PAYLOAD_KEYS,
        predecessor_check=_summary_predecessor_check(
            battery1_records,
            companion_npz_fields=(("settling_time_ecdf_path", "settling_time_ecdf_sha256"),
                                  ("resample_arrays_path", "resample_arrays_sha256"))))
    if existing is not None:
        return existing

    completion = {c: _battery1_completion_rate(battery1_records, c, seeds) for c in ALL_CONDITIONS}
    contrasts = {}
    descriptive_d = {}
    resample_arrays = {}
    for metric_name in BATTERY1_METRIC_SPECS:
        seed_level = {c: _battery1_seed_level_metric(battery1_records, c, seeds, metric_name)
                     for c in ALL_CONDITIONS}
        contrasts[metric_name] = {}
        for name, (a, b) in BATTERY1_CONTRAST_PAIRS.items():
            clean, resample_idx = contrast_with_survivorship_rule(
                seed_level[a], seed_level[b], completion[a], completion[b], seeds)
            contrasts[metric_name][name] = clean
            if resample_idx is not None:
                resample_arrays[f"{metric_name}__{name}_resample_idx"] = resample_idx
        descriptive_d[metric_name] = seed_level["D"]

    ecdf_arrays = {f"{c}_settling_time_pooled": _battery1_pooled_settling_time(battery1_records, c, seeds)
                  for c in ALL_CONDITIONS}
    ecdf_path = out_dir / "solver_audit_battery1_settling_time_ecdf_2026_09_01.npz"
    ecdf_sha256 = sio.write_npz_verify_if_exists(ecdf_path, ecdf_arrays)

    resample_path = out_dir / "solver_audit_battery1_resample_arrays_2026_09_01.npz"
    resample_sha256 = sio.write_npz_verify_if_exists(resample_path, resample_arrays)

    payload = dict(
        kind="solver_audit_battery1_summary",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_setup_manifest_sha256=setup_hash,
        seeds=seeds, contrasts=contrasts, descriptive_d=descriptive_d,
        completion_rates=completion,
        settling_time_ecdf_path=str(ecdf_path.relative_to(ROOT)), settling_time_ecdf_sha256=ecdf_sha256,
        resample_arrays_path=str(resample_path.relative_to(ROOT)), resample_arrays_sha256=resample_sha256,
        input_record_hashes=_input_record_hashes(battery1_records),
    )
    return dict(path=path, self_hash=sio.write_with_self_verification(path, payload), payload=payload, resumed=False)


def _battery2_npz_arrays(record: dict) -> dict:
    with np.load(ROOT / record["payload"]["iteration_arrays_path"]) as npz:
        return {k: npz[k] for k in npz.files}


def _b2_mean_excluding(arrays: dict, ic_idx: int, key: str):
    k, ek = f"ic{ic_idx}_{key}", f"ic{ic_idx}_excluded"
    if k not in arrays:
        return None
    vals = arrays[k]
    excl = arrays[ek].astype(bool) if ek in arrays else np.zeros(len(vals), dtype=bool)
    valid = vals[~excl]
    return float(np.mean(valid)) if len(valid) else None


def _b2_rate(arrays: dict, ic_idx: int, key: str, predicate) -> float:
    k = f"ic{ic_idx}_{key}"
    if k not in arrays:
        return None
    vals = arrays[k]
    return float(np.mean([predicate(v) for v in vals])) if len(vals) else None


def _b2_fd_rate(arrays: dict, ic_idx: int, predicate) -> float:
    k0, k1 = f"ic{ic_idx}_fd_kind_0", f"ic{ic_idx}_fd_kind_1"
    if k0 not in arrays:
        return None
    combined = np.concatenate([arrays[k0], arrays[k1]])
    return float(np.mean([predicate(k) for k in combined])) if len(combined) else None


def _b2_mean_signed_decrease(arrays: dict, ic_idx: int, before_key: str, after_key: str):
    """Mean SIGNED `before - after` over every logged iteration (positive
    = objective genuinely DECREASED, negative = it got WORSE) -- fixed
    from an earlier version that took `abs(diff)` between consecutive
    raw BEFORE values, which cannot distinguish improvement from
    regression and also silently dropped each step's LAST iteration
    (which has no "next before" to diff against). Uses the stored
    before/after PAIR directly (`j_learned`/`j_learned_after`,
    `counterfactual_true_objective_before/after`), which exists for
    EVERY logged iteration including the last one in each step."""
    bk, ak = f"ic{ic_idx}_{before_key}", f"ic{ic_idx}_{after_key}"
    if bk not in arrays or ak not in arrays:
        return None
    diffs = arrays[bk] - arrays[ak]
    valid = diffs[np.isfinite(diffs)]
    return float(np.mean(valid)) if len(valid) else None


# Section 5's gradient-bridge metric family (Section 3.5's full quantity
# set, not `cosine` alone): each `fn(arrays, ic_idx) -> float | None`.
BATTERY2_METRIC_FNS = {
    "cosine": lambda a, i: _b2_mean_excluding(a, i, "cosine"),
    "g_hat": lambda a, i: _b2_mean_excluding(a, i, "g_hat"),
    "g_true": lambda a, i: _b2_mean_excluding(a, i, "g_true"),
    "raw_inner_product": lambda a, i: _b2_mean_excluding(a, i, "raw_inner_product"),
    "directional_derivative_true_raw_step": lambda a, i: _b2_mean_excluding(
        a, i, "directional_derivative_true_raw_step"),
    "directional_derivative_true_projected_step": lambda a, i: _b2_mean_excluding(
        a, i, "directional_derivative_true_projected_step"),
    "projection_active_rate": lambda a, i: _b2_rate(a, i, "projection_active", bool),
    "fd_one_sided_rate": lambda a, i: _b2_fd_rate(a, i, lambda k: k in ("one_sided_forward", "one_sided_backward")),
    "fd_excluded_rate": lambda a, i: _b2_fd_rate(a, i, lambda k: k == "excluded"),
    "j_learned_decrease": lambda a, i: _b2_mean_signed_decrease(a, i, "j_learned", "j_learned_after"),
    "counterfactual_objective_decrease": lambda a, i: _b2_mean_signed_decrease(
        a, i, "counterfactual_true_objective_before", "counterfactual_true_objective_after"),
}


def _battery2_seed_level_metric(battery2_records: dict, condition: str, seeds: list, metric_name: str) -> dict:
    """Section 5's two-stage aggregation for Battery 2 (pinned in protocol
    v8): within each IC, the mean over non-excluded logged iterations;
    then averaged across the IC's 5 initial conditions."""
    fn = BATTERY2_METRIC_FNS[metric_name]
    out = {}
    for seed in seeds:
        arrays = _battery2_npz_arrays(battery2_records[(condition, seed)])
        per_ic_vals = [v for v in (fn(arrays, ic) for ic in range(5)) if v is not None and np.isfinite(v)]
        out[str(seed)] = float(np.mean(per_ic_vals)) if per_ic_vals else float("nan")
    return out


def build_battery2_summary_manifest(battery2_records: dict, seeds: list, out_dir: Path,
                                    setup_hash: str) -> dict:
    """Section 5's Proposition/gradient-bridge contrast family: E-B ONLY
    (Battery 2 covers only B/E, so no F/C/FNN gradient-bridge data exists
    to contrast against) -- for EVERY metric in `BATTERY2_METRIC_FNS`
    (residual value, raw inner product, directional derivatives,
    projection/FD usage rates, learned/counterfactual-objective
    iteration-to-iteration change -- not `cosine` alone)."""
    path = out_dir / "solver_audit_battery2_summary_2026_09_01.json"
    existing = _resume_or_verify(
        path, "solver_audit_battery2_summary", EXPECTED_BATTERY2_SUMMARY_PAYLOAD_KEYS,
        predecessor_check=_summary_predecessor_check(
            battery2_records, companion_npz_fields=(("resample_arrays_path", "resample_arrays_sha256"),)))
    if existing is not None:
        return existing
    contrasts = {}
    resample_arrays = {}
    for metric_name in BATTERY2_METRIC_FNS:
        e_vals = _battery2_seed_level_metric(battery2_records, "E", seeds, metric_name)
        b_vals = _battery2_seed_level_metric(battery2_records, "B", seeds, metric_name)
        diffs = paired_diff(e_vals, b_vals, seeds)
        result, resample_idx = _safe_evaluate_contrast(diffs)
        if resample_idx is not None:
            resample_arrays[f"{metric_name}__E_vs_B_resample_idx"] = resample_idx
        contrasts[metric_name] = dict(E_vs_B=result)

    resample_path = out_dir / "solver_audit_battery2_resample_arrays_2026_09_01.npz"
    resample_sha256 = sio.write_npz_verify_if_exists(resample_path, resample_arrays)

    payload = dict(
        kind="solver_audit_battery2_summary",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_setup_manifest_sha256=setup_hash,
        seeds=seeds, contrasts=contrasts,
        resample_arrays_path=str(resample_path.relative_to(ROOT)), resample_arrays_sha256=resample_sha256,
        input_record_hashes=_input_record_hashes(battery2_records),
    )
    return dict(path=path, self_hash=sio.write_with_self_verification(path, payload), payload=payload, resumed=False)


def _battery3_ic_iteration_stats(r_summary: dict) -> dict:
    """Per-IC damping/backtracking/rejection stats extracted from the
    fully-logged (`log_iterations=True`) `step_records` (Section 3.4) --
    `lam_before` (over EVERY iteration, INCLUDING singular-solve attempts
    -- `lam_before` is always defined even then, Section 3.4's own record
    construction sets it to the pre-multiplication damping value, and
    excluding singular attempts would silently drop the hardest damping
    cases from the mean), `accepted_alpha` (only over ACCEPTED
    iterations), `rejection_frac` (denominator EXCLUDES singular-solve
    iterations, since those never reached the accept/reject decision at
    all), and `singular_solve_frac` (reported separately so the singular-
    solve rate itself is visible, not merely folded silently into other
    denominators)."""
    step_records = r_summary.get("step_records")
    if not step_records:
        return dict(mean_lam_before=None, mean_accepted_alpha=None, rejection_frac=None,
                    singular_solve_frac=None)
    lams, alphas = [], []
    total = considered = rejected = singular = 0
    for sr in step_records:
        for it in sr["iterations"]:
            total += 1
            lams.append(it["lam_before"])
            if it.get("singular_solve"):
                singular += 1
                continue
            considered += 1
            if it["accepted"]:
                alphas.append(it["accepted_alpha"])
            else:
                rejected += 1
    return dict(
        mean_lam_before=float(np.mean(lams)) if lams else None,
        mean_accepted_alpha=float(np.mean(alphas)) if alphas else None,
        singular_solve_frac=float(singular / total) if total else None,
        rejection_frac=float(rejected / considered) if considered else None,
    )


BATTERY3_SCALAR_METRIC_KEYS = ("accepted_frac", "true_incr_frac", "mean_pred_dec_applied", "mean_true_dec",
                              "mean_counterfactual_dec_applied")
BATTERY3_TIMING_METRIC_KEYS = ("median", "p95")
BATTERY3_ITERATION_METRIC_KEYS = ("mean_lam_before", "mean_accepted_alpha", "rejection_frac",
                                 "singular_solve_frac")


def build_battery3_summary_manifest(battery3_records: dict, seeds: list, out_dir: Path,
                                    setup_hash: str) -> dict:
    """Section 5: GGN quantities are DESCRIPTIVE only -- no formal
    contrast/CI, consistent with Battery 3 never having been scoped as a
    causal comparison (Section 2). Aggregated SEED-FIRST (5 ICs averaged
    within each seed, THEN mean/std/n taken over the resulting per-seed
    values) -- fixed from an earlier version that pooled all 50 (seed,
    IC) pairs directly, which is not what Section 5's own "aggregate the
    5 ICs within each seed FIRST" convention specifies. Also reports
    damping (`lam_before`), backtracking `accepted_alpha`, the rejection
    fraction, and the CLEAN core-timing median/P95 (Section 7's
    solver-instrumentation applicability note: GGN, unlike Adam, genuinely
    has line-search/accept-reject mechanics worth summarizing)."""
    path = out_dir / "solver_audit_battery3_summary_2026_09_01.json"
    existing = _resume_or_verify(path, "solver_audit_battery3_summary", EXPECTED_BATTERY3_SUMMARY_PAYLOAD_KEYS,
                                 predecessor_check=_summary_predecessor_check(battery3_records))
    if existing is not None:
        return existing
    descriptive = {}
    all_metric_keys = (BATTERY3_SCALAR_METRIC_KEYS + BATTERY3_TIMING_METRIC_KEYS
                       + BATTERY3_ITERATION_METRIC_KEYS)
    for condition in DETAILED_CONDITIONS:
        seed_level = {k: {} for k in all_metric_keys}
        for seed in seeds:
            payload = battery3_records[(condition, seed)]["payload"]
            per_ic_values = {k: [] for k in all_metric_keys}
            for r in payload["per_ic"]:
                for k in BATTERY3_SCALAR_METRIC_KEYS:
                    v = r["summary"].get(k)
                    if v is not None and np.isfinite(v):
                        per_ic_values[k].append(v)
                for k in BATTERY3_TIMING_METRIC_KEYS:
                    v = r["summary"]["rollout_solver_core_wall_clock_seconds"][k]
                    if np.isfinite(v):
                        per_ic_values[k].append(v)
                iter_stats = _battery3_ic_iteration_stats(r["summary"])
                for k in BATTERY3_ITERATION_METRIC_KEYS:
                    v = iter_stats[k]
                    if v is not None and np.isfinite(v):
                        per_ic_values[k].append(v)
            for k in all_metric_keys:
                if per_ic_values[k]:
                    seed_level[k][seed] = float(np.mean(per_ic_values[k]))
        descriptive[condition] = {
            k: (dict(mean=float(np.mean(list(v.values()))), n=len(v)) if v else dict(mean=None, n=0))
            for k, v in seed_level.items()
        }
    payload = dict(
        kind="solver_audit_battery3_summary",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_setup_manifest_sha256=setup_hash,
        seeds=seeds, descriptive=descriptive,
        input_record_hashes=_input_record_hashes(battery3_records),
    )
    return dict(path=path, self_hash=sio.write_with_self_verification(path, payload), payload=payload, resumed=False)


def build_battery5_summary_manifest(battery5_records: dict, seeds: list, out_dir: Path,
                                    two_cstr_setup_hash: str) -> dict:
    """Section 5: two-CSTR Battery 5 is explicitly NOT a causal contrast
    (no equal-query control exists for two-CSTR, Section 0) -- reports
    `value_only`/`value+grad` DESCRIPTIVELY (mean/min/max across the 10
    reconstructed seeds), explicitly labeled as not a rigorously powered
    comparison. Metrics: per-reactor state-constraint stats (indices 1
    and 3) averaged over the 5 ICs first, then mean/min/max over seeds,
    plus the seed-level `success` rate."""
    path = out_dir / "solver_audit_battery5_summary_2026_09_01.json"
    existing = _resume_or_verify(path, "solver_audit_battery5_summary", EXPECTED_BATTERY5_SUMMARY_PAYLOAD_KEYS,
                                 predecessor_check=_summary_predecessor_check(battery5_records))
    if existing is not None:
        return existing
    descriptive = {}
    # Derived from `battery5_records`' ACTUAL keys, not the global
    # `slib.TWO_CSTR_CONFIGS` -- the real run always populates both
    # configs, but this function must not assume so (a caller exercising
    # a subset, e.g. a smoke/dev run, must not KeyError on a config it
    # never actually built a record for).
    present_configs = sorted({config_name for (config_name, _seed) in battery5_records})
    reactor_indices = ("1", "3")
    actuator_indices = ("0", "1", "2", "3")
    worst_case = {}
    for config_name in present_configs:
        seed_level_keys = (
            ["success", "solver_completed_rate", "temperature_cap_satisfied_posthoc",
             "input_box_satisfied", "input_never_saturated"]
            + [f"max_temperature_deviation_reactor{r}" for r in reactor_indices]
            + [f"max_temperature_absolute_reactor{r}" for r in reactor_indices]
            + [f"temperature_violation_frequency_reactor{r}" for r in reactor_indices]
            + [f"temperature_violation_integral_reactor{r}" for r in reactor_indices]
            + [f"input_saturation_frequency_actuator{a}" for a in actuator_indices]
        )
        seed_level = {k: [] for k in seed_level_keys}
        # TRUE worst case across every one of the 50 (seed, IC) rollouts
        # directly -- NOT derived from the seed-first descriptive mean/
        # min/max above, which averages the 5 ICs within a seed FIRST and
        # so dilutes a single IC's genuine worst excursion. Reported as a
        # SEPARATE top-level field, alongside (never replacing) the
        # seed-first descriptive statistics Section 5 otherwise pins.
        raw_max_dev = {r: [] for r in reactor_indices}
        raw_max_abs = {r: [] for r in reactor_indices}
        for seed in seeds:
            payload = battery5_records[(config_name, seed)]["payload"]
            rows = payload["state_constraint_audit"]
            seed_level["success"].append(float(np.mean([r["summary"]["success"] for r in rows])))
            seed_level["solver_completed_rate"].append(
                float(np.mean([bool(r["summary"]["solver_completed"]) for r in rows])))
            seed_level["temperature_cap_satisfied_posthoc"].append(float(np.mean(
                [bool(r["state_constraint"]["temperature_cap_satisfied_posthoc"]) for r in rows])))
            box_vals = [r["state_constraint"]["input_box_satisfied"] for r in rows
                       if r["state_constraint"]["input_box_satisfied"] is not None]
            if box_vals:
                seed_level["input_box_satisfied"].append(float(np.mean([bool(v) for v in box_vals])))
            never_sat_vals = [r["state_constraint"]["input_never_saturated"] for r in rows
                              if r["state_constraint"]["input_never_saturated"] is not None]
            if never_sat_vals:
                seed_level["input_never_saturated"].append(float(np.mean([bool(v) for v in never_sat_vals])))
            for reactor_idx in reactor_indices:
                pti = [r["state_constraint"]["per_temperature_index"][reactor_idx] for r in rows]
                dev_vals = [p["max_temperature_deviation_K"] for p in pti]
                abs_vals = [p["max_temperature_absolute_K"] for p in pti]
                seed_level[f"max_temperature_deviation_reactor{reactor_idx}"].append(float(np.mean(dev_vals)))
                seed_level[f"max_temperature_absolute_reactor{reactor_idx}"].append(float(np.mean(abs_vals)))
                raw_max_dev[reactor_idx].extend(dev_vals)
                raw_max_abs[reactor_idx].extend(abs_vals)
                seed_level[f"temperature_violation_frequency_reactor{reactor_idx}"].append(
                    float(np.mean([p["temperature_violation_frequency"] for p in pti])))
                seed_level[f"temperature_violation_integral_reactor{reactor_idx}"].append(
                    float(np.mean([p["temperature_violation_integral"] for p in pti])))
            for actuator_idx in actuator_indices:
                sat = [r["state_constraint"]["input_saturation_frequency"][actuator_idx] for r in rows]
                seed_level[f"input_saturation_frequency_actuator{actuator_idx}"].append(float(np.mean(sat)))
        descriptive[config_name] = {
            k: (dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), n_seeds=len(v))
                if v else dict(mean=None, min=None, max=None, n_seeds=0))
            for k, v in seed_level.items()
        }
        worst_case[config_name] = {
            f"reactor{r}": dict(
                worst_max_temperature_deviation_K=float(np.max(raw_max_dev[r])),
                worst_max_temperature_absolute_K=float(np.max(raw_max_abs[r])),
                n_rollouts=len(raw_max_dev[r]),
            )
            for r in reactor_indices
        }
    payload = dict(
        kind="solver_audit_battery5_summary",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_two_cstr_setup_manifest_sha256=two_cstr_setup_hash,
        seeds=seeds, descriptive=descriptive, worst_case=worst_case,
        input_record_hashes=_input_record_hashes(battery5_records),
    )
    return dict(path=path, self_hash=sio.write_with_self_verification(path, payload), payload=payload, resumed=False)


# ---------------------------------------------------------------------------
# Real top-level entry point -- NOT invoked anywhere in this file. Gated
# on this protocol's OWN Freeze Manifest (not yet built) exactly as
# `run_real_protocol()` gates on the ablation's.
# ---------------------------------------------------------------------------
def run_real_solver_audit(out_dir: Path = NAMESPACE, seeds: list = None,
                          freeze_manifest_path: Path = None) -> dict:
    """The canonical production entry point for the single-CSTR batteries
    (1-4) + Section 5's Summary Manifests. Verifies THIS protocol's own
    freeze first (before any real data access), pins single-threaded CPU
    execution (Section 3.2's timing methodology -- BEFORE any timed work
    starts), then runs Batteries 1-3 for every requested seed (B/E's
    Battery-1-level record is MERGED from Battery 2, never re-run), then
    the three Summary Manifests. Every step is resume-or-verify. Not
    called anywhere in this codebase yet -- real execution only happens
    after this file, the synthetic smoke suite, and a full re-review are
    all complete, per Section 0."""
    resolved_freeze_path = freeze_manifest_path if freeze_manifest_path is not None else FREEZE_MANIFEST_PATH
    solver_audit_freeze_hash = verify_freeze_manifest(resolved_freeze_path)
    slib.configure_deterministic_single_thread()

    resolved_seeds = seeds if seeds is not None else alib.SEEDS
    if list(resolved_seeds) != list(alib.SEEDS):
        raise ValueError(
            f"run_real_solver_audit requires the full pre-registered seed set {list(alib.SEEDS)}, "
            f"not an arbitrary subset ({list(resolved_seeds)})."
        )

    import budget_sweep_solver as bs
    import probe_action_gradient as pr
    ablation_out_dir = adrv.NAMESPACE
    ablation_freeze_path = adrv.FREEZE_MANIFEST_PATH
    ablation_m1_path = ablation_out_dir / "ablation_query_manifest_2026_08_31.json"
    ablation_m2_paths = {
        s: ablation_out_dir / f"ablation_training_manifest_seed{s}_2026_08_31.json" for s in resolved_seeds
    }
    for s, p in ablation_m2_paths.items():
        if not p.exists():
            raise RuntimeError(f"UPSTREAM_M2_MISSING: seed {s}'s real-ablation Training Manifest not found at {p}")
    seed_set_tag = hashlib.sha256(repr(sorted(resolved_seeds)).encode()).hexdigest()[:12]
    ablation_m3_path = ablation_out_dir / f"ablation_results_manifest_{seed_set_tag}_2026_08_31.json"
    if not ablation_m3_path.exists():
        raise RuntimeError(f"UPSTREAM_M3_MISSING: expected Results Manifest not found at {ablation_m3_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    sim = bs.SIM
    pool = alib.load_pool()
    setup = build_setup_manifest(ablation_freeze_path, ablation_m1_path, ablation_m2_paths, ablation_m3_path,
                                 pool, solver_audit_freeze_hash, out_dir=out_dir)
    del pr  # imported only to make the real-simulator/oracle dependency chain explicit at this call site

    battery1_records = {}
    battery2_records = {}
    battery3_records = {}
    for seed in resolved_seeds:
        for condition in DETAILED_CONDITIONS:
            battery2_records[(condition, seed)] = run_battery2_condition_seed(
                condition, seed, ablation_m2_paths[seed], sim, out_dir, setup["self_hash"])
            battery3_records[(condition, seed)] = run_battery3_condition_seed(
                condition, seed, ablation_m2_paths[seed], sim, out_dir, setup["self_hash"])
            battery1_records[(condition, seed)] = battery1_record_from_battery2(
                condition, seed, battery2_records[(condition, seed)]["payload"], out_dir, setup["self_hash"])
        for condition in BATTERY1_OWN_CONDITIONS:
            battery1_records[(condition, seed)] = run_battery1_condition_seed(
                condition, seed, ablation_m2_paths[seed], sim, out_dir, setup["self_hash"])

    battery1_summary = build_battery1_summary_manifest(battery1_records, list(resolved_seeds), out_dir,
                                                        setup["self_hash"])
    battery2_summary = build_battery2_summary_manifest(battery2_records, list(resolved_seeds), out_dir,
                                                        setup["self_hash"])
    battery3_summary = build_battery3_summary_manifest(battery3_records, list(resolved_seeds), out_dir,
                                                        setup["self_hash"])

    return dict(setup=setup, battery1=battery1_records, battery2=battery2_records, battery3=battery3_records,
                battery1_summary=battery1_summary, battery2_summary=battery2_summary,
                battery3_summary=battery3_summary)


def run_real_two_cstr_minimum_audit(out_dir: Path = NAMESPACE, seeds: list = None) -> dict:
    """Section 8's four-phase two-CSTR minimum audit. Not gated on the
    single-CSTR freeze (Battery 5 has its own, separate Setup Manifest,
    Section 4), but IS gated on this protocol's own freeze existing
    first, matching the same design-review discipline. Enforces the full
    pre-registered 10-seed set for the same reason
    `run_real_solver_audit` does -- an arbitrary subset is a smoke/dev-
    only concept. Phases 1-2 run for ALL (config, seed) pairs FIRST;
    phase 3 (freeze) only happens if every pair passed Gate 4; phase 4
    then runs against the frozen checkpoints."""
    solver_audit_freeze_hash = verify_freeze_manifest(FREEZE_MANIFEST_PATH)
    slib.configure_deterministic_single_thread()

    resolved_seeds = seeds if seeds is not None else slib.TWO_CSTR_SEEDS
    if list(resolved_seeds) != list(slib.TWO_CSTR_SEEDS):
        raise ValueError(
            f"run_real_two_cstr_minimum_audit requires the full pre-registered seed set "
            f"{list(slib.TWO_CSTR_SEEDS)}, not an arbitrary subset ({list(resolved_seeds)})."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "two_cstr_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    setup = build_two_cstr_setup_manifest(solver_audit_freeze_hash, out_dir=out_dir)

    # Phases 1-2: reconstruct + Gate 4 for ALL 20 (config, seed) pairs.
    reconstructions = {}
    for config_name in sorted(slib.TWO_CSTR_CONFIGS.keys()):
        for seed in resolved_seeds:
            reconstructions[(config_name, seed)] = battery5_phase1_and_2_reconstruct_and_gate(
                config_name, seed, out_dir, checkpoint_dir)
    # Phase 3: freeze -- a REAL manifest artifact, reached only if every
    # pair above passed (an exception from phase 1-2 propagates out
    # before this line, per `TwoCstrReconstructionIncompatible`'s
    # all-or-none contract). Explicit defense-in-depth: the PRODUCTION
    # path must freeze exactly 20 (config, seed) pairs (2 configs x 10
    # seeds) -- `build_replica_manifest` itself stays generic (the
    # synthetic smoke suite legitimately exercises it with a single
    # pair), so this count is asserted here, at the one call site that is
    # supposed to be exhaustive.
    expected_n_pairs = len(slib.TWO_CSTR_CONFIGS) * len(resolved_seeds)
    if len(reconstructions) != expected_n_pairs:
        raise RuntimeError(
            f"TWO_CSTR_REPLICA_COUNT_MISMATCH: expected {expected_n_pairs} (config, seed) pairs "
            f"({len(slib.TWO_CSTR_CONFIGS)} configs x {len(resolved_seeds)} seeds), got {len(reconstructions)}"
        )
    replica_manifest = build_replica_manifest(reconstructions, out_dir, setup["self_hash"])
    replica_manifest_verified = verify_replica_manifest(replica_manifest["path"])

    # Phase 4: reload from the FROZEN checkpoint files (per the verified
    # Replica Manifest, never the in-memory `reconstructions` dict) and audit.
    records = {}
    for (config_name, seed) in reconstructions:
        records[(config_name, seed)] = battery5_phase4_instrumented_audit(
            config_name, seed, out_dir, replica_manifest_verified["self_hash"],
            replica_manifest_verified["payload"])

    battery5_summary = build_battery5_summary_manifest(records, list(resolved_seeds), out_dir, setup["self_hash"])

    return dict(setup=setup, replica_manifest=replica_manifest, battery5=records, battery5_summary=battery5_summary)


# ---------------------------------------------------------------------------
# Freeze Manifest (this protocol's own -- separate from the ablation's)
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
        raise RuntimeError("cannot freeze: smoke results file is internally inconsistent (n_total mismatch)")
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
    for p in REQUIRED_UPSTREAM_FILES:
        if not p.exists():
            raise RuntimeError(f"cannot freeze: required file missing: {p}")
        file_hashes[str(p.relative_to(ROOT))] = sio.file_sha256(p)
    payload = dict(
        kind="solver_audit_freeze_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        file_sha256=file_hashes,
        smoke_n_pass=smoke["n_pass"], smoke_n_total=smoke["n_total"],
    )
    sio.write_with_self_verification(dest_path, payload)
    return dest_path


def verify_freeze_manifest(path: Path = FREEZE_MANIFEST_PATH) -> str:
    self_hash, payload = sio.read_and_verify_json_artifact(
        path, expected_kind="solver_audit_freeze_manifest",
        expected_keys={"kind", "generated_at_utc", "file_sha256", "smoke_n_pass", "smoke_n_total"})
    expected_keys = {str(p.relative_to(ROOT)) for p in REQUIRED_UPSTREAM_FILES}
    actual_keys = set(payload["file_sha256"].keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"FREEZE_KEYSET_MISMATCH: manifest covers {sorted(actual_keys)} but the current "
            f"REQUIRED_UPSTREAM_FILES set is {sorted(expected_keys)}"
        )
    for rel_path, expected_hash in payload["file_sha256"].items():
        actual = sio.file_sha256(ROOT / rel_path)
        if actual != expected_hash:
            raise RuntimeError(f"FREEZE_DRIFT: {rel_path} hash changed since freeze ({expected_hash} -> {actual})")
    return self_hash
