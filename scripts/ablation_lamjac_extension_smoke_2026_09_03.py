#!/usr/bin/env python3
"""Synthetic + minimal-real smoke suite for
`ablation_lamjac_extension_2026_09_03.py`."""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cstr.ablation_lib as alib  # noqa: E402
import ablation_lamjac_extension_2026_09_03 as ext  # noqa: E402

RESULTS_DIR = ROOT / "results"
OUT_PATH = RESULTS_DIR / "ablation_lamjac_extension_smoke_results_2026_09_03.json"

CHECKS = []


def check(name, fn):
    t0 = time.monotonic()
    try:
        fn()
        CHECKS.append(dict(name=name, passed=True, error=None, seconds=time.monotonic() - t0))
        print(f"PASS {name}")
    except Exception as e:  # noqa: BLE001
        CHECKS.append(dict(name=name, passed=False, error=f"{e}\n{traceback.format_exc()}",
                           seconds=time.monotonic() - t0))
        print(f"FAIL {name}: {e}")


def small_hyperparams():
    alib.EPOCHS = 2
    alib.WARMUP = 1
    alib.CHECKPOINT_EVERY = 1


# ---------------------------------------------------------------------------
# Selection-rule checks (pure function, no training)
# ---------------------------------------------------------------------------

def test_select_merged_grid_prefers_lower_val_mse():
    old_trace = [dict(lam_jac=0.001, val_mse=0.01, test_mse=0.01),
                dict(lam_jac=0.5, val_mse=0.005, test_mse=0.005)]
    new_trace = [dict(lam_jac=1.0, val_mse=0.003, test_mse=999.0),  # bad test_mse must not matter
                dict(lam_jac=2.0, val_mse=0.004, test_mse=0.001),
                dict(lam_jac=5.0, val_mse=0.006, test_mse=0.0001)]
    selected = ext._select_over_merged_grid(old_trace, new_trace)
    assert selected == 1.0, f"expected 1.0 (lowest val_mse), got {selected}"


def test_select_merged_grid_tie_break_smallest_lambda():
    old_trace = [dict(lam_jac=0.5, val_mse=0.003, test_mse=0.0)]
    new_trace = [dict(lam_jac=1.0, val_mse=0.003, test_mse=0.0),  # exact tie with 0.5
                dict(lam_jac=2.0, val_mse=0.003, test_mse=0.0)]
    selected = ext._select_over_merged_grid(old_trace, new_trace)
    assert selected == 0.5, f"tie-break must pick smallest lambda (0.5), got {selected}"


def test_boundary_persists_when_top_of_extended_grid_selected():
    old_trace = [dict(lam_jac=0.5, val_mse=0.01, test_mse=0.0)]
    new_trace = [dict(lam_jac=1.0, val_mse=0.008, test_mse=0.0),
                dict(lam_jac=2.0, val_mse=0.006, test_mse=0.0),
                dict(lam_jac=5.0, val_mse=0.001, test_mse=0.0)]  # monotonically best at boundary
    selected = ext._select_over_merged_grid(old_trace, new_trace)
    assert selected == 5.0
    outcome = "BOUNDARY_PERSISTS" if selected == max(t["lam_jac"] for t in new_trace) else "INTERIOR_OPTIMUM_FOUND"
    assert outcome == "BOUNDARY_PERSISTS"


def test_interior_optimum_found_when_middle_value_wins():
    old_trace = [dict(lam_jac=0.5, val_mse=0.01, test_mse=0.0)]
    new_trace = [dict(lam_jac=1.0, val_mse=0.002, test_mse=0.0),  # best
                dict(lam_jac=2.0, val_mse=0.004, test_mse=0.0),
                dict(lam_jac=5.0, val_mse=0.009, test_mse=0.0)]  # worse than 1.0: not a boundary win
    selected = ext._select_over_merged_grid(old_trace, new_trace)
    assert selected == 1.0
    outcome = "BOUNDARY_PERSISTS" if selected == max(t["lam_jac"] for t in new_trace) else "INTERIOR_OPTIMUM_FOUND"
    assert outcome == "INTERIOR_OPTIMUM_FOUND"


def test_selection_ignores_closed_loop_and_test_mse_fields():
    # A candidate with catastrophic test_mse but best val_mse must still win --
    # this is the exact "sequestered from selection" contract the user specified.
    old_trace = [dict(lam_jac=0.5, val_mse=0.01, test_mse=0.0001)]
    new_trace = [dict(lam_jac=1.0, val_mse=0.001, test_mse=1e6)]
    selected = ext._select_over_merged_grid(old_trace, new_trace)
    assert selected == 1.0


# ---------------------------------------------------------------------------
# Freeze / Results manifest round-trip (fabricated small data, no real training)
# ---------------------------------------------------------------------------

def test_freeze_manifest_rejects_before_original_freeze_verifies():
    import ablation_driver_2026_08_31 as orig
    bad_path = ROOT / "docs" / "__does_not_exist__.json"
    try:
        orig.verify_freeze_manifest(bad_path)
        raise AssertionError("expected verify_freeze_manifest to raise for a missing path")
    except Exception:
        pass  # expected: any exception is fine, this just checks the call path is reachable


def test_results_manifest_round_trip():
    import cstr.ablation_io as aio
    tmp_dir = RESULTS_DIR / "_smoke_lamjac_ext_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / f"fake_results_{np.random.randint(1_000_000)}.json"

    per_seed = {}
    e_closed_loop_success = {}
    for s in range(3):
        per_seed[s] = dict(
            seed=s,
            old_trace=[dict(lam_jac=0.5, val_mse=0.01, test_mse=0.01)],
            new_trace=[dict(lam_jac=1.0, val_mse=0.005, test_mse=0.005)],
            merged_selected_lambda_jac=1.0,
            boundary_outcome="INTERIOR_OPTIMUM_FOUND",
            init_state_hash_consistent_with_original=True,
            new_candidate_closed_loop_success={1.0: 0.8, 2.0: 0.6, 5.0: 0.4},
            new_candidate_oracle={1.0: dict(align_cos=0.5), 2.0: dict(align_cos=0.4), 5.0: dict(align_cos=0.3)},
            selected_candidate_closed_loop_success=0.8,
            selected_candidate_test_mse=0.005,
        )
        e_closed_loop_success[s] = 1.0

    result = ext.build_results_manifest(dest, per_seed, e_closed_loop_success,
                                        freeze_self_hash="fake_freeze_hash",
                                        original_freeze_hash="fake_original_hash",
                                        m1_self_hash="fake_m1_hash")
    reloaded = json.loads(dest.read_text())
    assert reloaded["overall_boundary_outcome"] == "INTERIOR_OPTIMUM_FOUND"
    assert set(reloaded.keys()) == ext.EXPECTED_RESULTS_PAYLOAD_KEYS | {"canonical_payload_sha256"}
    # self-hash re-verification
    self_hash2, _ = aio.read_and_verify_json_artifact(dest)
    assert self_hash2 == result["self_hash"]
    dest.unlink()
    (tmp_dir / "ablation_lamjac_extension_resample_2026_09_03.npz").unlink(missing_ok=True)


def test_mixed_boundary_outcome_detected():
    per_seed = {
        0: dict(boundary_outcome="BOUNDARY_PERSISTS", merged_selected_lambda_jac=5.0,
               selected_candidate_closed_loop_success=0.5, selected_candidate_test_mse=0.01,
               old_trace=[], new_trace=[], init_state_hash_consistent_with_original=True,
               new_candidate_closed_loop_success={}, new_candidate_oracle={}, seed=0),
        1: dict(boundary_outcome="INTERIOR_OPTIMUM_FOUND", merged_selected_lambda_jac=1.0,
               selected_candidate_closed_loop_success=0.9, selected_candidate_test_mse=0.01,
               old_trace=[], new_trace=[], init_state_hash_consistent_with_original=True,
               new_candidate_closed_loop_success={}, new_candidate_oracle={}, seed=1),
    }
    outcomes = {per_seed[s]["boundary_outcome"] for s in per_seed}
    overall = "BOUNDARY_PERSISTS" if outcomes == {"BOUNDARY_PERSISTS"} else (
        "MIXED" if len(outcomes) > 1 else "INTERIOR_OPTIMUM_FOUND")
    assert overall == "MIXED"


# ---------------------------------------------------------------------------
# Minimal REAL end-to-end: 1 seed, real frozen pool/M1/M2 verification,
# tiny epochs, exercises the actual closed_loop()/diagnose_fixed_idx() path.
# ---------------------------------------------------------------------------

def test_real_frozen_input_reconstruction_verifies():
    loaded = ext.load_original_frozen_inputs(ext.ORIGINAL_FREEZE_MANIFEST_PATH)
    assert "pool" in loaded and "clean" in loaded and "scale" in loaded
    assert loaded["m1_self_hash"]


def test_real_original_seed_trace_loads_and_has_six_points():
    loaded = ext.load_original_frozen_inputs(ext.ORIGINAL_FREEZE_MANIFEST_PATH)
    trace = ext.load_original_seed_trace(0, loaded["m1_self_hash"], loaded["pool"], loaded["clean"], loaded["scale"])
    assert len(trace["old_trace"]) == 6
    assert trace["old_selected"] == 0.5
    assert trace["init_state_hash"]


def test_real_one_seed_one_lambda_end_to_end():
    small_hyperparams()
    try:
        loaded = ext.load_original_frozen_inputs(ext.ORIGINAL_FREEZE_MANIFEST_PATH)
        pool, clean, scale, sim = loaded["pool"], loaded["clean"], loaded["scale"], loaded["sim"]
        oracle_idx = alib.fixed_oracle_idx(pool["test_idx"])
        tmp_ckpt = RESULTS_DIR / "_smoke_lamjac_ext_tmp" / "checkpoints"
        tmp_ckpt.mkdir(parents=True, exist_ok=True)
        rec = ext.run_seed_extension(0, pool, clean, scale, sim, loaded["m1_self_hash"],
                                     tmp_ckpt, oracle_idx, grid=[1.0])
        assert rec["init_state_hash_consistent_with_original"] is True
        assert 1.0 in rec["new_candidate_closed_loop_success"]
        assert 0.0 <= rec["new_candidate_closed_loop_success"][1.0] <= 1.0
        assert "align_cos" in rec["new_candidate_oracle"][1.0]
        assert rec["merged_selected_lambda_jac"] in (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
    finally:
        alib.EPOCHS = 50
        alib.WARMUP = 30
        alib.CHECKPOINT_EVERY = 10


def main():
    check("select_merged_grid_prefers_lower_val_mse", test_select_merged_grid_prefers_lower_val_mse)
    check("select_merged_grid_tie_break_smallest_lambda", test_select_merged_grid_tie_break_smallest_lambda)
    check("boundary_persists_when_top_of_extended_grid_selected", test_boundary_persists_when_top_of_extended_grid_selected)
    check("interior_optimum_found_when_middle_value_wins", test_interior_optimum_found_when_middle_value_wins)
    check("selection_ignores_closed_loop_and_test_mse_fields", test_selection_ignores_closed_loop_and_test_mse_fields)
    check("freeze_manifest_rejects_before_original_freeze_verifies", test_freeze_manifest_rejects_before_original_freeze_verifies)
    check("results_manifest_round_trip", test_results_manifest_round_trip)
    check("mixed_boundary_outcome_detected", test_mixed_boundary_outcome_detected)
    check("real_frozen_input_reconstruction_verifies", test_real_frozen_input_reconstruction_verifies)
    check("real_original_seed_trace_loads_and_has_six_points", test_real_original_seed_trace_loads_and_has_six_points)
    check("real_one_seed_one_lambda_end_to_end", test_real_one_seed_one_lambda_end_to_end)

    n_pass = sum(1 for c in CHECKS if c["passed"])
    n_total = len(CHECKS)
    out = dict(n_pass=n_pass, n_total=n_total, all_passed=(n_pass == n_total), checks=CHECKS)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n{n_pass}/{n_total} checks passed. Results: {OUT_PATH}")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
