#!/usr/bin/env python3
"""N-family / N-scale / FD / L-grad sensitivity driver
(`docs/JPC_SENSITIVITY_PROTOCOL_2026_09_03.md`, v5). Canonical entry
point: `run_real_sensitivity_protocol()`. Never modifies
`ablation_execution_2026_08_31/` or `solver_audit_execution_2026_09_01/`
-- all new artifacts live under `results/ablation_sensitivity_2026_09_03/`.
"""
from __future__ import annotations

import ast
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

import cstr.ablation_io as aio            # noqa: E402
import cstr.ablation_lib as alib          # noqa: E402
import cstr.sensitivity_lib as slib       # noqa: E402
import cstr.solver_audit_io as sio        # noqa: E402
import ablation_driver_2026_08_31 as orig # noqa: E402

RESULTS_DIR = ROOT / "results"
NAMESPACE = RESULTS_DIR / "ablation_sensitivity_2026_09_03"
SMOKE_RESULTS_PATH = RESULTS_DIR / "sensitivity_smoke_results_2026_09_03.json"
COMPAT_RESULT_PATH = RESULTS_DIR / "sensitivity_compatibility_reconstruction_2026_09_03.json"
FREEZE_MANIFEST_PATH = ROOT / "docs" / "JPC_SENSITIVITY_PROTOCOL_FREEZE_MANIFEST_2026_09_03.json"
PROTOCOL_DOC = ROOT / "docs" / "JPC_SENSITIVITY_PROTOCOL_2026_09_03.md"
THIS_DRIVER = ROOT / "scripts" / "sensitivity_driver_2026_09_03.py"
THIS_LIB = ROOT / "src" / "cstr" / "sensitivity_lib.py"

ORIGINAL_ABLATION_FREEZE = orig.FREEZE_MANIFEST_PATH
ORIGINAL_ABLATION_NAMESPACE = orig.NAMESPACE
SOLVER_AUDIT_FREEZE = ROOT / "docs" / "JPC_SOLVER_AUDIT_PROTOCOL_FREEZE_MANIFEST_2026_09_01.json"
SOLVER_AUDIT_SETUP = RESULTS_DIR / "solver_audit_execution_2026_09_01" / "solver_audit_setup_manifest_2026_09_01.json"
SOLVER_AUDIT_NAMESPACE = RESULTS_DIR / "solver_audit_execution_2026_09_01"

SEEDS = [0, 1, 2, 3, 4]
N_EVIDENCE_SLOTS = 80

EXPECTED_FREEZE_KEYS = {
    "kind", "generated_at_utc", "file_sha256", "smoke_n_pass", "smoke_n_total",
    "predecessor_original_ablation_freeze_sha256", "predecessor_solver_audit_freeze_sha256",
    "compatibility_reconstruction_result_path", "compatibility_reconstruction_result_sha256",
}
EXPECTED_SETUP_KEYS = {
    "kind", "generated_at_utc", "predecessor_freeze_sha256",
    "predecessor_original_ablation_m1_sha256", "pool_content_sha256",
    "clean_rebuild_paths", "clean_rebuild_sha256", "common_support_mask_path",
    "common_support_mask_sha256", "fd_reference_ug_scale", "oracle_fixed_idx_path",
    "oracle_fixed_idx_sha256", "environment",
}


# ---------------------------------------------------------------------------
# Environment recording (v4->v5 item 8, mirrors the solver-audit schema)
# ---------------------------------------------------------------------------

def get_environment_info() -> dict:
    return dict(
        cpu_architecture=platform.machine(),
        numpy_version=np.__version__,
        os_platform=platform.platform(),
        processor=platform.processor(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
    )


# ---------------------------------------------------------------------------
# Freeze Manifest: transitive local-import closure (v4->v5 item 9)
# ---------------------------------------------------------------------------

def _module_dotted_to_path(mod: str) -> Path | None:
    if mod.startswith("cstr."):
        rel = mod[len("cstr."):].replace(".", "/")
        p = ROOT / "src" / "cstr" / f"{rel}.py"
        return p if p.exists() else None
    p_scripts = ROOT / "scripts" / f"{mod}.py"
    if p_scripts.exists():
        return p_scripts
    p_cstr_bare = ROOT / "src" / "cstr" / f"{mod}.py"
    return p_cstr_bare if p_cstr_bare.exists() else None


def _extract_imported_modules(path: Path) -> set:
    tree = ast.parse(path.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module)
    return mods


def compute_local_import_closure(entry_files: list) -> set:
    """Repo-internal (`src/cstr`, `scripts`) transitive import closure of
    `entry_files`, computed via `ast` (not a hand-curated list)."""
    seen_files = set(Path(f).resolve() for f in entry_files)
    frontier = list(seen_files)
    while frontier:
        f = frontier.pop()
        for mod in _extract_imported_modules(f):
            p = _module_dotted_to_path(mod)
            if p is not None:
                rp = p.resolve()
                if rp not in seen_files:
                    seen_files.add(rp)
                    frontier.append(rp)
    return seen_files


# ---------------------------------------------------------------------------
# Discarded compatibility reconstruction
# ---------------------------------------------------------------------------

def _frozen_e_seed0_record():
    m2_path = ORIGINAL_ABLATION_NAMESPACE / "ablation_training_manifest_seed0_2026_08_31.json"
    payload = json.loads(m2_path.read_text())
    return payload


def run_compatibility_reconstruction(pool: dict, clean: dict, scale: dict, sim, out_dir: Path) -> dict:
    """Section 6: ONE discarded training reconstruction + ONE discarded
    `clean` reconstruction. Both are compared byte-for-byte against the
    frozen originals and then discarded -- never used as reported data."""
    checks = {}
    tmp_dir = out_dir / "_compat_reconstruction_scratch"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # (a) Training path.
    frozen = _frozen_e_seed0_record()
    r = slib.train_variant_condition("E", 0, pool, clean, scale, family="cauchy",
                                     noise_scale=slib.NOISE_SCALE_BASELINE,
                                     checkpoint_dir=tmp_dir, checkpoint_prefix="compat_recon_E")
    checks["val_mse_match"] = bool(np.isclose(r["val_mse"], frozen["val_mse"]["E"], rtol=0, atol=0)
                                   or r["val_mse"] == frozen["val_mse"]["E"])
    checks["test_mse_match"] = (r["test_mse"] == frozen["test_mse"]["E"])
    checks["init_state_hash_match"] = (r["init_state_hash"] == frozen["init_state_hashes"]["E"])

    frozen_ckpt_path = None
    frozen_ckpt_hash = None
    for path_str, h in frozen["checkpoint_sha256"].items():
        if Path(path_str).name.startswith("E_seed0_ep"):
            frozen_ckpt_path, frozen_ckpt_hash = path_str, h
    recon_ckpt_path = r["checkpoints"][-1]["path"] if r["checkpoints"] else None
    if frozen_ckpt_path is not None and recon_ckpt_path is not None:
        recon_state = torch.load(recon_ckpt_path, map_location="cpu")
        frozen_state = torch.load(ROOT / frozen_ckpt_path if not Path(frozen_ckpt_path).is_absolute()
                                  else frozen_ckpt_path, map_location="cpu")
        checks["final_checkpoint_state_dict_match"] = all(
            torch.equal(recon_state[k], frozen_state[k]) for k in frozen_state
        ) and set(recon_state.keys()) == set(frozen_state.keys())
    else:
        checks["final_checkpoint_state_dict_match"] = False

    # (b) `clean`-builder path.
    import budget_sweep_solver as bs   # noqa: PLC0415
    import probe_action_gradient as pr  # noqa: PLC0415
    recon_clean = slib.build_clean_query_table_with_eps(pool, sim, pr.precompute_phi, alib.FD_EPS)
    array_keys = ["Phi_all", "Yphi_all", "valid_mask", "un_plus", "un_minus",
                  "y_plus_n", "y_minus_n", "true_df_du", "true_ug"]
    checks["clean_rebuild_byte_identical"] = all(
        np.array_equal(recon_clean[k], clean[k]) for k in array_keys
    )

    all_passed = all(checks.values())
    payload = dict(
        kind="sensitivity_compatibility_reconstruction",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        checks=checks, all_passed=all_passed,
        environment=get_environment_info(),
    )
    self_hash = aio.write_with_self_verification(COMPAT_RESULT_PATH, payload)
    return dict(path=COMPAT_RESULT_PATH, self_hash=self_hash, payload=payload, all_passed=all_passed)


# ---------------------------------------------------------------------------
# Freeze Manifest
# ---------------------------------------------------------------------------

def build_freeze_manifest(dest_path: Path = FREEZE_MANIFEST_PATH) -> Path:
    if not SMOKE_RESULTS_PATH.exists():
        raise RuntimeError(f"cannot freeze: {SMOKE_RESULTS_PATH} does not exist -- run the smoke suite first")
    smoke = json.loads(SMOKE_RESULTS_PATH.read_text())
    if not smoke.get("all_passed"):
        raise RuntimeError(f"cannot freeze: smoke all_passed=False ({smoke.get('n_pass')}/{smoke.get('n_total')})")
    if not COMPAT_RESULT_PATH.exists():
        raise RuntimeError("cannot freeze: compatibility reconstruction result JSON does not exist")
    compat_self_hash, compat_payload = aio.read_and_verify_json_artifact(COMPAT_RESULT_PATH)
    if not compat_payload.get("all_passed"):
        raise RuntimeError(
            f"cannot freeze: compatibility reconstruction did not pass: {compat_payload.get('checks')}"
        )

    original_ablation_hash = orig.verify_freeze_manifest(ORIGINAL_ABLATION_FREEZE)
    solver_audit_hash = _verify_solver_audit_freeze()

    entry_files = [PROTOCOL_DOC, THIS_DRIVER, THIS_LIB, SMOKE_RESULTS_PATH]
    closure = compute_local_import_closure([THIS_DRIVER, THIS_LIB])
    all_files = set(entry_files) | closure
    file_hashes = {str(p.relative_to(ROOT)): aio.file_sha256(p) for p in all_files}

    payload = dict(
        kind="sensitivity_freeze_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        file_sha256=file_hashes,
        smoke_n_pass=smoke["n_pass"], smoke_n_total=smoke["n_total"],
        predecessor_original_ablation_freeze_sha256=original_ablation_hash,
        predecessor_solver_audit_freeze_sha256=solver_audit_hash,
        compatibility_reconstruction_result_path=str(COMPAT_RESULT_PATH.relative_to(ROOT)),
        compatibility_reconstruction_result_sha256=compat_self_hash,
    )
    aio.write_with_self_verification(dest_path, payload)
    return dest_path


def _verify_solver_audit_freeze() -> str:
    # The solver-audit freeze verifier lives in its own driver module,
    # which this protocol does not import wholesale (avoiding an
    # unrelated dependency surface) -- verify by direct file-hash
    # recomputation against the manifest's own recorded set instead.
    self_hash, payload = aio.read_and_verify_json_artifact(SOLVER_AUDIT_FREEZE)
    for rel_path, expected_hash in payload["file_sha256"].items():
        actual = aio.file_sha256(ROOT / rel_path)
        if actual != expected_hash:
            raise RuntimeError(f"SOLVER_AUDIT_FREEZE_DRIFT: {rel_path} hash changed since freeze")
    return self_hash


def verify_freeze_manifest(path: Path = FREEZE_MANIFEST_PATH) -> str:
    self_hash, payload = aio.read_and_verify_json_artifact(
        path, expected_kind="sensitivity_freeze_manifest", expected_keys=EXPECTED_FREEZE_KEYS)
    for rel_path, expected_hash in payload["file_sha256"].items():
        actual = aio.file_sha256(ROOT / rel_path)
        if actual != expected_hash:
            raise RuntimeError(f"FREEZE_DRIFT: {rel_path} hash changed since freeze")
    if orig.verify_freeze_manifest(ORIGINAL_ABLATION_FREEZE) != payload["predecessor_original_ablation_freeze_sha256"]:
        raise RuntimeError("ORIGINAL_ABLATION_FREEZE_CHAIN_STALE")
    if _verify_solver_audit_freeze() != payload["predecessor_solver_audit_freeze_sha256"]:
        raise RuntimeError("SOLVER_AUDIT_FREEZE_CHAIN_STALE")
    return self_hash


# ---------------------------------------------------------------------------
# Setup Manifest
# ---------------------------------------------------------------------------

def load_frozen_inputs():
    orig.verify_freeze_manifest(ORIGINAL_ABLATION_FREEZE)
    import budget_sweep_solver as bs   # noqa: PLC0415
    import probe_action_gradient as pr  # noqa: PLC0415

    sim = bs.SIM
    pool = alib.load_pool()
    clean = alib.build_clean_query_table(pool, sim, pr.precompute_phi)
    scale = alib.shared_scale_constants(pool, clean)
    m1_path = ORIGINAL_ABLATION_NAMESPACE / "ablation_query_manifest_2026_08_31.json"
    m1_verified = orig.verify_query_manifest(m1_path, pool=pool, clean=clean,
                                             freeze_manifest_path=ORIGINAL_ABLATION_FREEZE)
    return dict(pool=pool, clean=clean, scale=scale, sim=sim, m1_self_hash=m1_verified["self_hash"])


def _load_pinned_oracle_idx(out_dir: Path) -> tuple:
    """v4->v5 item: pins the ORIGINAL Results Manifest's own
    `oracle_fixed_idx_path`/`_sha256`, never independently recomputed."""
    m3_paths = list(ORIGINAL_ABLATION_NAMESPACE.glob("ablation_results_manifest_*_2026_08_31.json"))
    if len(m3_paths) != 1:
        raise RuntimeError(f"expected exactly 1 original M3, found {len(m3_paths)}")
    m3_self_hash, m3_payload = aio.read_and_verify_json_artifact(m3_paths[0])
    idx_path = ROOT / m3_payload["oracle_fixed_idx_path"]
    actual_hash = aio.file_sha256(idx_path)
    if actual_hash != m3_payload["oracle_fixed_idx_sha256"]:
        raise RuntimeError("ORACLE_FIXED_IDX_DRIFT")
    with np.load(idx_path) as npz:
        idx = npz[npz.files[0]]
    return idx, str(idx_path.relative_to(ROOT)), actual_hash


def build_setup_manifest(out_dir: Path, dest_path: Path, freeze_self_hash: str) -> dict:
    loaded = load_frozen_inputs()
    pool, clean, scale, sim = loaded["pool"], loaded["clean"], loaded["scale"], loaded["sim"]

    import budget_sweep_solver as bs   # noqa: PLC0415
    import probe_action_gradient as pr  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_rebuild_paths = {}
    clean_rebuild_sha256 = {}
    rebuilt = {}
    for eps in (1e-4, 1e-2, 0.02):
        c = slib.build_clean_query_table_with_eps(pool, sim, pr.precompute_phi, eps)
        rebuilt[eps] = c
        npz_path = out_dir / f"sensitivity_clean_eps{eps}_2026_09_03.npz"
        arrays = {k: v for k, v in c.items() if isinstance(v, np.ndarray)}
        h = aio.write_npz_verify_if_exists(npz_path, arrays)
        clean_rebuild_paths[str(eps)] = str(npz_path.relative_to(ROOT))
        clean_rebuild_sha256[str(eps)] = h
    rebuilt[1e-3] = clean  # native (unmasked) 1e-3 == the frozen original clean

    common_mask = clean["valid_mask"].copy()
    for eps in (1e-4, 1e-2, 0.02):
        common_mask &= rebuilt[eps]["valid_mask"]
    mask_npz_path = out_dir / "sensitivity_common_support_mask_2026_09_03.npz"
    mask_hash = aio.write_npz_verify_if_exists(mask_npz_path, dict(common_mask=common_mask))

    fd_reference_clean = slib.apply_common_support_mask(clean, common_mask)
    fd_reference_ug_scale = slib.compute_ug_scale(fd_reference_clean)

    oracle_idx, oracle_idx_path, oracle_idx_hash = _load_pinned_oracle_idx(out_dir)

    payload = dict(
        kind="sensitivity_setup_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_freeze_sha256=freeze_self_hash,
        predecessor_original_ablation_m1_sha256=loaded["m1_self_hash"],
        pool_content_sha256=aio.pool_content_sha256(pool),
        clean_rebuild_paths=clean_rebuild_paths,
        clean_rebuild_sha256=clean_rebuild_sha256,
        common_support_mask_path=str(mask_npz_path.relative_to(ROOT)),
        common_support_mask_sha256=mask_hash,
        fd_reference_ug_scale=fd_reference_ug_scale,
        oracle_fixed_idx_path=oracle_idx_path,
        oracle_fixed_idx_sha256=oracle_idx_hash,
        environment=get_environment_info(),
    )
    if dest_path.exists():
        # Resume-or-verify: the Setup Manifest JSON itself has no
        # verify-if-exists writer (unlike its npz companions above, which
        # already are verify-if-exists) -- re-verify the existing file's
        # self-hash and reuse it rather than attempting a second write
        # (which would raise on the no-overwrite path, and whose payload
        # would differ from the first solely by `generated_at_utc` even
        # if every substantive field matched).
        self_hash, existing_payload = aio.read_and_verify_json_artifact(dest_path)
        if existing_payload["predecessor_freeze_sha256"] != freeze_self_hash:
            raise RuntimeError("RESUME_PROVENANCE_MISMATCH: existing Setup Manifest's predecessor freeze hash differs")
    else:
        self_hash = aio.write_with_self_verification(dest_path, payload)
    return dict(path=dest_path, self_hash=self_hash, payload=payload,
               pool=pool, clean=clean, scale=scale, sim=sim,
               rebuilt_clean=rebuilt, common_mask=common_mask,
               fd_reference_clean=fd_reference_clean, fd_reference_ug_scale=fd_reference_ug_scale,
               oracle_idx=oracle_idx)


# ---------------------------------------------------------------------------
# Evaluator-compatibility gate (v4->v5 items 1-2)
# ---------------------------------------------------------------------------

def run_evaluator_compatibility_gate() -> dict:
    import budget_sweep_solver as bs                    # noqa: PLC0415
    import cstr.solver_audit_lib as sal                  # noqa: PLC0415
    from disambiguate_landscape_vs_gradient import make_norm  # noqa: PLC0415

    setup_path = SOLVER_AUDIT_SETUP
    audit_self_hash, audit_setup = aio.read_and_verify_json_artifact(setup_path)
    horizon, budget, steps = audit_setup["horizon"], audit_setup["budget"], audit_setup["steps"]
    lr, rho_u, success_v = audit_setup["lr"], audit_setup["rho_u"], audit_setup["success_v"]

    pool = alib.load_pool()
    data_tuple = (pool["XU"], pool["Y"], pool["xm"], pool["xs"], pool["ym"], pool["ys"],
                 pool["train_idx"], pool["test_idx"])
    norm, un_lo, un_hi = make_norm(data_tuple)
    input_lo, input_hi = bs.INPUT_LO, bs.INPUT_HI

    m2_path = ORIGINAL_ABLATION_NAMESPACE / "ablation_training_manifest_seed0_2026_08_31.json"
    m2_payload = json.loads(m2_path.read_text())

    results = {}
    for condition in ("B", "E"):
        seq = sal.load_condition_seq(condition, 0, m2_payload)
        battery1_path = (SOLVER_AUDIT_NAMESPACE
                        / f"solver_audit_battery1_{condition}_seed0_2026_09_01.json")
        _, battery1_payload = aio.read_and_verify_json_artifact(battery1_path)
        per_ic_ok = []
        for ic_index, x0 in enumerate(bs.ICS):
            new_summary = slib.rollout_per_ic_summary(
                seq, norm, x0, bs.SIM, input_lo, input_hi, un_lo, un_hi,
                horizon, budget, steps, lr, rho_u, success_v)
            old_summary = battery1_payload["per_ic"][ic_index]["summary"]
            cmp = slib.compare_per_ic_to_battery1(new_summary, old_summary,
                                                  sio.ISCLOSE_RTOL, sio.ISCLOSE_ATOL)
            per_ic_ok.append(cmp["all_match"])
        results[condition] = dict(all_ic_match=all(per_ic_ok), per_ic=per_ic_ok)

    all_passed = all(results[c]["all_ic_match"] for c in results)
    return dict(all_passed=all_passed, per_condition=results)


# ---------------------------------------------------------------------------
# Per-model training loop + Training Records + Results Manifest
# ---------------------------------------------------------------------------

def all_slots() -> list:
    slots = []
    for family in ("noiseless", "gaussian", "correlated"):
        for condition in ("B", "E"):
            for seed in SEEDS:
                slots.append(dict(axis="n_family", level=family, condition=condition, seed=seed))
    for scale_val in (0.1, 0.4):
        for condition in ("B", "E"):
            for seed in SEEDS:
                slots.append(dict(axis="n_scale", level=scale_val, condition=condition, seed=seed))
    for eps in slib.FD_EPS_LEVELS:
        for seed in SEEDS:
            slots.append(dict(axis="fd", level=eps, condition="E", seed=seed))
    for lam in (0.025, 0.1):
        for seed in SEEDS:
            slots.append(dict(axis="lgrad", level=lam, condition="E", seed=seed))
    assert len(slots) == N_EVIDENCE_SLOTS, f"expected {N_EVIDENCE_SLOTS} slots, got {len(slots)}"
    return slots


def _slot_id(slot: dict) -> str:
    return f"{slot['axis']}_{slot['level']}_{slot['condition']}_seed{slot['seed']}"


def _record_path(out_dir: Path, slot: dict) -> Path:
    return out_dir / f"sensitivity_record_{_slot_id(slot)}_2026_09_03.json"


def _find_fresh_attempt_dir(checkpoint_root: Path, slot_id: str) -> tuple:
    attempt_n = 1
    d = checkpoint_root / slot_id / f"attempt{attempt_n}"
    while d.exists():
        attempt_n += 1
        d = checkpoint_root / slot_id / f"attempt{attempt_n}"
    return d, attempt_n


def _frozen_baseline_init_hash(condition: str, seed: int) -> str:
    m2_path = ORIGINAL_ABLATION_NAMESPACE / f"ablation_training_manifest_seed{seed}_2026_08_31.json"
    payload = json.loads(m2_path.read_text())
    return payload["init_state_hashes"][condition]


def train_one_slot(slot: dict, setup: dict, checkpoint_root: Path) -> dict:
    pool, clean, scale, sim = setup["pool"], setup["clean"], setup["scale"], setup["sim"]
    seed, condition = slot["seed"], slot["condition"]
    attempt_dir, attempt_id = _find_fresh_attempt_dir(checkpoint_root, _slot_id(slot))
    attempt_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_slot_id(slot)}_a{attempt_id}"

    assert alib.shuffle_seed(seed) == seed + 5000

    try:
        if slot["axis"] == "n_family":
            r = slib.train_variant_condition(condition, seed, pool, clean, scale,
                                             family=slot["level"], noise_scale=slib.NOISE_SCALE_BASELINE,
                                             checkpoint_dir=attempt_dir, checkpoint_prefix=prefix)
        elif slot["axis"] == "n_scale":
            r = slib.train_variant_condition(condition, seed, pool, clean, scale,
                                             family="cauchy", noise_scale=slot["level"],
                                             checkpoint_dir=attempt_dir, checkpoint_prefix=prefix)
        elif slot["axis"] == "fd":
            eps = slot["level"]
            masked_clean = (setup["fd_reference_clean"] if eps == 1e-3
                           else slib.apply_common_support_mask(setup["rebuilt_clean"][eps], setup["common_mask"]))
            scale_fixed = dict(scale); scale_fixed["ug_scale"] = setup["fd_reference_ug_scale"]
            r = alib.train_base_condition("E", seed, pool, masked_clean, scale_fixed,
                                          checkpoint_dir=attempt_dir, checkpoint_prefix=prefix)
        elif slot["axis"] == "lgrad":
            r = slib.train_lam_grad_variant(seed, pool, clean, scale, lam_grad=slot["level"],
                                            checkpoint_dir=attempt_dir, checkpoint_prefix=prefix)
        else:
            raise ValueError(f"unknown axis: {slot['axis']!r}")
    except alib.TrainingDivergedError as e:
        return dict(outcome="TRAINING_DIVERGED", reason=str(e), attempt_id=attempt_id)

    init_ok = (r["init_state_hash"] == _frozen_baseline_init_hash(condition, seed))
    ckpt_path = r["checkpoints"][-1]["path"] if r["checkpoints"] else None
    ckpt_hash = aio.file_sha256(Path(ckpt_path)) if ckpt_path else None

    return dict(outcome="TRAINED", attempt_id=attempt_id, model=r["model"],
               val_mse=r["val_mse"], test_mse=r["test_mse"], init_state_hash=r["init_state_hash"],
               init_state_hash_matches_baseline=init_ok,
               checkpoint_path=ckpt_path, checkpoint_sha256=ckpt_hash,
               n_raw=r.get("n_raw"), n_applied=r.get("n_applied"), disp=r.get("disp"))


def evaluate_one_slot(slot: dict, model, setup: dict, norm, un_lo, un_hi, sim,
                      horizon: int, budget: int, steps: int, lr: float, rho_u: float, success_v: float,
                      input_lo, input_hi) -> dict:
    import budget_sweep_solver as bs  # noqa: PLC0415
    seq = alib.freeze_any(model)
    per_ic = []
    eval_failed = False
    for x0 in bs.ICS:
        try:
            s = slib.rollout_per_ic_summary(seq, norm, x0, sim, input_lo, input_hi, un_lo, un_hi,
                                            horizon, budget, steps, lr, rho_u, success_v)
        except alib.SIMULATOR_FAILURE_EXCEPTIONS:
            eval_failed = True
            s = dict(final_V=float("nan"), max_V=float("nan"), settling_time=steps + 1,
                     success=False, solver_completed=False, n_steps_completed=0,
                     failure_reason="EVALUATION_FAILED")
        per_ic.append(s)

    closed_loop_success = float(np.mean([p["success"] for p in per_ic]))
    final_v_mean = float(np.mean([p["final_V"] for p in per_ic if np.isfinite(p["final_V"])])
                        if any(np.isfinite(p["final_V"]) for p in per_ic) else float("nan"))
    max_v_mean = float(np.mean([p["max_V"] for p in per_ic if np.isfinite(p["max_V"])])
                      if any(np.isfinite(p["max_V"]) for p in per_ic) else float("nan"))
    settling_mean = float(np.mean([p["settling_time"] for p in per_ic]))
    incomplete_rollout = any(p["n_steps_completed"] < steps for p in per_ic)

    oracle = alib.diagnose_fixed_idx(model, setup["pool"], sim, setup["oracle_idx"])
    n_used = slib.oracle_n_used(model, setup["pool"], sim, setup["oracle_idx"])

    return dict(
        per_ic=per_ic, closed_loop_success=closed_loop_success,
        final_V_mean=final_v_mean, max_V_mean=max_v_mean, settling_time_mean=settling_mean,
        evaluation_failed=eval_failed, incomplete_rollout=incomplete_rollout,
        align_cos_median=oracle["align_cos"]["median"], wrong_sign_fraction=oracle["align_cos"]["frac_neg"],
        rel_grad_error_median=oracle["rel_grad_error"]["median"],
        excluded_fraction=oracle["excluded_fraction"], n_used=n_used,
    )


def build_training_record(slot: dict, out_dir: Path, train_result: dict, eval_result: dict = None) -> dict:
    path = _record_path(out_dir, slot)
    if path.exists():
        self_hash, payload = aio.read_and_verify_json_artifact(path)
        return dict(path=path, self_hash=self_hash, payload=payload)

    payload = dict(
        kind="sensitivity_training_record",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        axis=slot["axis"], level=slot["level"], condition=slot["condition"], seed=slot["seed"],
        outcome=train_result["outcome"],
    )
    if train_result["outcome"] == "TRAINING_DIVERGED":
        payload["reason"] = train_result["reason"]
        payload["attempt_id"] = train_result["attempt_id"]
    else:
        payload.update(
            attempt_id=train_result["attempt_id"],
            val_mse=train_result["val_mse"], test_mse=train_result["test_mse"],
            init_state_hash=train_result["init_state_hash"],
            init_state_hash_matches_baseline=train_result["init_state_hash_matches_baseline"],
            checkpoint_path=train_result["checkpoint_path"], checkpoint_sha256=train_result["checkpoint_sha256"],
        )
        if eval_result is not None:
            payload.update(
                evaluation_outcome=("EVALUATION_FAILED" if eval_result["evaluation_failed"]
                                    else ("INCOMPLETE_ROLLOUT" if eval_result["incomplete_rollout"] else "COMPLETE")),
                closed_loop_success=eval_result["closed_loop_success"],
                final_V_mean=eval_result["final_V_mean"], max_V_mean=eval_result["max_V_mean"],
                settling_time_mean=eval_result["settling_time_mean"],
                per_ic=eval_result["per_ic"],
                align_cos_median=eval_result["align_cos_median"],
                wrong_sign_fraction=eval_result["wrong_sign_fraction"],
                rel_grad_error_median=eval_result["rel_grad_error_median"],
                excluded_fraction=eval_result["excluded_fraction"], n_used=eval_result["n_used"],
                test_mse_endpoint=train_result["test_mse"],
            )
    if train_result.get("n_raw") is not None:
        noise_npz_path = out_dir / f"sensitivity_noise_{slot['axis']}_{slot['level']}_seed{slot['seed']}_2026_09_03.npz"
        noise_hash = aio.write_npz_verify_if_exists(
            noise_npz_path, dict(n_raw=train_result["n_raw"], n_applied=train_result["n_applied"],
                                 disp=train_result["disp"]))
        diag = slib.noise_diagnostics(train_result["n_raw"], train_result["n_applied"])
        payload["noise_companion_path"] = str(noise_npz_path.relative_to(ROOT))
        payload["noise_companion_sha256"] = noise_hash
        payload["noise_diagnostics"] = diag

    self_hash = aio.write_with_self_verification(path, payload)
    return dict(path=path, self_hash=self_hash, payload=payload)


REFERENCE_BY_AXIS = {"n_family": "frozen_2026_08_31_baseline", "n_scale": "frozen_2026_08_31_baseline",
                     "lgrad": "frozen_2026_08_31_baseline", "fd": "fd_common_support_1e-3_reference"}


def _endpoint_contrast(variant_vals: list, reference_vals: list, variant_ok: list, reference_ok: list) -> dict:
    paired_ok = [a and b for a, b in zip(variant_ok, reference_ok)]
    if not all(paired_ok):
        return dict(outcome="DESCRIPTIVE_CONTRAST_INCOMPLETE", n_complete=sum(paired_ok), n_total=len(paired_ok))
    diffs = np.array(variant_vals) - np.array(reference_vals)
    ci = alib.bootstrap_ci(diffs)
    ci.pop("resample_idx", None)
    return dict(outcome="COMPLETE", point_estimate=float(diffs.mean()),
               sd=float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0, ci=ci)


def run_real_sensitivity_protocol(out_dir: Path = NAMESPACE) -> dict:
    freeze_self_hash = verify_freeze_manifest(FREEZE_MANIFEST_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_path = out_dir / "sensitivity_setup_manifest_2026_09_03.json"
    # `build_setup_manifest` internally recomputes pool/clean/scale/rebuilds
    # fresh (cheap, deterministic given the frozen pool) and only WRITES a
    # new Setup Manifest file if `dest_path` doesn't already exist -- the
    # underlying `write_with_self_verification`/`write_npz_verify_if_exists`
    # calls are themselves verify-if-exists, so calling this again on a
    # resume reuses (and self-verifies) the existing manifest/companions
    # rather than raising a no-overwrite error.
    setup = build_setup_manifest(out_dir, setup_path, freeze_self_hash)
    setup_self_hash = setup["self_hash"]

    gate = run_evaluator_compatibility_gate()
    if not gate["all_passed"]:
        raise RuntimeError(f"EVALUATOR_WIRING_FAILURE: {gate}")

    import budget_sweep_solver as bs                    # noqa: PLC0415
    from disambiguate_landscape_vs_gradient import make_norm  # noqa: PLC0415
    audit_self_hash, audit_setup = aio.read_and_verify_json_artifact(SOLVER_AUDIT_SETUP)
    horizon, budget, steps = audit_setup["horizon"], audit_setup["budget"], audit_setup["steps"]
    lr, rho_u, success_v = audit_setup["lr"], audit_setup["rho_u"], audit_setup["success_v"]
    data_tuple = (setup["pool"]["XU"], setup["pool"]["Y"], setup["pool"]["xm"], setup["pool"]["xs"],
                 setup["pool"]["ym"], setup["pool"]["ys"], setup["pool"]["train_idx"], setup["pool"]["test_idx"])
    norm, un_lo, un_hi = make_norm(data_tuple)
    checkpoint_root = out_dir / "checkpoints"

    slots = all_slots()
    records = {}
    for slot in slots:
        path = _record_path(out_dir, slot)
        if path.exists():
            self_hash, payload = aio.read_and_verify_json_artifact(path)
            records[_slot_id(slot)] = payload
            continue
        train_result = train_one_slot(slot, setup, checkpoint_root)
        eval_result = None
        if train_result["outcome"] == "TRAINED":
            eval_result = evaluate_one_slot(slot, train_result["model"], setup, norm, un_lo, un_hi,
                                            setup["sim"], horizon, budget, steps, lr, rho_u, success_v,
                                            bs.INPUT_LO, bs.INPUT_HI)
        rec = build_training_record(slot, out_dir, train_result, eval_result)
        records[_slot_id(slot)] = rec["payload"]

    completeness = {sid: r["outcome"] for sid, r in records.items()}
    n_complete = sum(1 for v in completeness.values() if v == "TRAINED")

    results_path = out_dir / "sensitivity_results_manifest_2026_09_03.json"
    payload = dict(
        kind="sensitivity_results_manifest",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_setup_sha256=setup_self_hash,
        n_evidence_slots=N_EVIDENCE_SLOTS, n_complete=n_complete,
        completeness=completeness,
        reference_by_axis=REFERENCE_BY_AXIS,
    )
    assert len(records) == N_EVIDENCE_SLOTS
    if results_path.exists():
        self_hash, _ = aio.read_and_verify_json_artifact(results_path)
    else:
        self_hash = aio.write_with_self_verification(results_path, payload)
    return dict(setup=setup, records=records, results_path=results_path, results_self_hash=self_hash)


if __name__ == "__main__":
    print("This module's real entry points (run_compatibility_reconstruction, "
          "build_freeze_manifest, build_setup_manifest, run_evaluator_compatibility_gate, "
          "run_real_sensitivity_protocol) are invoked from the smoke suite and the "
          "orchestration steps described in docs/JPC_SENSITIVITY_PROTOCOL_2026_09_03.md, "
          "not by running this file directly with no arguments.")
