#!/usr/bin/env python3
"""Synthetic + minimal-real smoke suite for the N-family/N-scale/FD/L-grad
sensitivity protocol (`docs/JPC_SENSITIVITY_PROTOCOL_2026_09_03.md`, v5)."""
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

import cstr.ablation_io as aio            # noqa: E402
import cstr.ablation_lib as alib          # noqa: E402
import cstr.sensitivity_lib as slib       # noqa: E402
import sensitivity_driver_2026_09_03 as drv  # noqa: E402

RESULTS_DIR = ROOT / "results"
OUT_PATH = RESULTS_DIR / "sensitivity_smoke_results_2026_09_03.json"

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


def restore_hyperparams():
    alib.EPOCHS = 50
    alib.WARMUP = 30
    alib.CHECKPOINT_EVERY = 10


# ---------------------------------------------------------------------------
# N-family / N-scale noise construction
# ---------------------------------------------------------------------------

def test_cauchy_baseline_reproduces_frozen_noise():
    """The parameterized builder at family=cauchy/scale=0.2 must produce
    Yn/disp byte-identical to the frozen `alib.build_noise_realized_table`."""
    pool = alib.load_pool()
    import budget_sweep_solver as bs  # noqa: PLC0415
    import probe_action_gradient as pr  # noqa: PLC0415
    clean = alib.build_clean_query_table(pool, bs.SIM, pr.precompute_phi)
    frozen = alib.build_noise_realized_table(pool, clean, seed=0, sensitivity=False)
    param = slib.build_noise_realized_table_parameterized(pool, clean, seed=0, family="cauchy",
                                                          scale=slib.NOISE_SCALE_BASELINE)
    assert np.array_equal(frozen["Yn"], param["Yn"])
    assert np.array_equal(frozen["disp"], param["disp"])
    # n_applied is exactly what got added to Yn at train_idx.
    train_idx = np.asarray(pool["train_idx"])
    ym, ys = pool["ym"], pool["ys"]
    clean_yn_train = (pool["Y"][train_idx] - ym) / ys
    reconstructed_n_applied = frozen["Yn"][train_idx] - clean_yn_train
    assert np.allclose(reconstructed_n_applied, param["n_applied"], atol=1e-6)


def test_noiseless_is_all_zero():
    pool = alib.load_pool()
    import budget_sweep_solver as bs  # noqa: PLC0415
    import probe_action_gradient as pr  # noqa: PLC0415
    clean = alib.build_clean_query_table(pool, bs.SIM, pr.precompute_phi)
    r = slib.build_noise_realized_table_parameterized(pool, clean, seed=0, family="noiseless")
    train_idx = np.asarray(pool["train_idx"])
    clean_yn_train = (pool["Y"][train_idx] - pool["ym"]) / pool["ys"]
    assert np.allclose(r["Yn"][train_idx], clean_yn_train)
    assert np.all(r["n_raw"] == 0) and np.all(r["n_applied"] == 0)


def test_gaussian_mad_matches_cauchy_baseline():
    pool = alib.load_pool()
    import budget_sweep_solver as bs  # noqa: PLC0415
    import probe_action_gradient as pr  # noqa: PLC0415
    clean = alib.build_clean_query_table(pool, bs.SIM, pr.precompute_phi)
    g = slib.build_noise_realized_table_parameterized(pool, clean, seed=0, family="gaussian")
    mad = np.median(np.abs(g["n_applied"] - np.median(g["n_applied"], axis=0)), axis=0)
    # Cauchy(0.2)'s MAD is exactly 0.2; large-sample Gaussian MAD should be close.
    assert np.all(np.abs(mad - 0.2) < 0.02), f"MAD not close to 0.2: {mad}"


def test_gaussian_and_correlated_share_base_draw():
    """v4->v5 item 1: correlated-gaussian must be a Cholesky transform of
    the EXACT SAME z as plain gaussian, not an independent draw."""
    pool = alib.load_pool()
    import budget_sweep_solver as bs  # noqa: PLC0415
    import probe_action_gradient as pr  # noqa: PLC0415
    clean = alib.build_clean_query_table(pool, bs.SIM, pr.precompute_phi)
    g = slib.build_noise_realized_table_parameterized(pool, clean, seed=0, family="gaussian")
    c = slib.build_noise_realized_table_parameterized(pool, clean, seed=0, family="correlated")
    z_from_gaussian = g["n_raw"] / slib.GAUSSIAN_SIGMA_MAD_MATCHED
    L = np.linalg.cholesky(np.array([[1.0, slib.RHO_CORRELATED], [slib.RHO_CORRELATED, 1.0]]))
    with np.errstate(all="ignore"):
        z_from_correlated = (c["n_raw"] / slib.GAUSSIAN_SIGMA_MAD_MATCHED) @ np.linalg.inv(L).T
    assert np.allclose(z_from_gaussian, z_from_correlated, atol=1e-9)


def test_correlated_noise_actually_correlated():
    pool = alib.load_pool()
    import budget_sweep_solver as bs  # noqa: PLC0415
    import probe_action_gradient as pr  # noqa: PLC0415
    clean = alib.build_clean_query_table(pool, bs.SIM, pr.precompute_phi)
    c = slib.build_noise_realized_table_parameterized(pool, clean, seed=0, family="correlated")
    corr = np.corrcoef(c["n_applied"][:, 0], c["n_applied"][:, 1])[0, 1]
    assert abs(corr - slib.RHO_CORRELATED) < 0.05, f"empirical correlation {corr} far from rho={slib.RHO_CORRELATED}"


def test_noise_diagnostics_definitions():
    n_applied = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]])
    n_raw = np.array([[1.0, 2.0], [6.0, 4.0], [-1.0, -2.0]])  # row 1 dim0 would be clipped at 5
    diag = slib.noise_diagnostics(n_raw, n_applied)
    assert diag["clipping_fraction"] == 1 / 6  # exactly one of 6 scalar entries exceeds 5 in abs value
    assert diag["correlation"] is not None


def test_noise_diagnostics_noiseless_correlation_is_none():
    z = np.zeros((10, 2))
    diag = slib.noise_diagnostics(z, z)
    assert diag["correlation"] is None


# ---------------------------------------------------------------------------
# FD common-support masking
# ---------------------------------------------------------------------------

def test_common_support_mask_matches_eps002_nesting():
    pool = alib.load_pool()
    import budget_sweep_solver as bs  # noqa: PLC0415
    import probe_action_gradient as pr  # noqa: PLC0415
    clean_1e3 = alib.build_clean_query_table(pool, bs.SIM, pr.precompute_phi)
    clean_002 = slib.build_clean_query_table_with_eps(pool, bs.SIM, pr.precompute_phi, 0.02)
    common = clean_1e3["valid_mask"] & clean_002["valid_mask"]
    assert np.array_equal(common, clean_002["valid_mask"]), "nesting result: intersection should equal eps=0.02's own mask"


def test_apply_common_support_mask_zeroes_newly_dropped():
    clean = dict(
        valid_mask=np.array([[True, True], [True, False]]),
        true_ug=np.array([[1.0, 2.0], [3.0, 0.0]]),
        true_df_du=np.ones((2, 2, 2)),
    )
    common = np.array([[True, False], [True, False]])
    out = slib.apply_common_support_mask(clean, common)
    assert not out["valid_mask"][0, 1]
    assert out["true_ug"][0, 1] == 0.0
    assert np.all(out["true_df_du"][0, :, 1] == 0.0)
    # untouched entries preserved
    assert out["valid_mask"][0, 0] and out["true_ug"][0, 0] == 1.0


def test_compute_ug_scale_matches_shared_scale_constants_formula():
    rng = np.random.default_rng(0)
    true_ug = rng.standard_normal((100, 2))
    valid_mask = np.ones((100, 2), dtype=bool)
    clean = dict(true_ug=true_ug, valid_mask=valid_mask)
    computed = slib.compute_ug_scale(clean)
    expected = max(true_ug[valid_mask].std(ddof=1), 1.0)
    assert abs(computed - expected) < 1e-12


# ---------------------------------------------------------------------------
# Evaluator-compatibility comparison logic
# ---------------------------------------------------------------------------

def test_compare_per_ic_isclose_vs_exact():
    new = dict(final_V=1.0000001, max_V=5.0, settling_time=42, success=True)
    old = dict(final_V=1.0000000, max_V=5.0, settling_time=42, success=True)
    cmp = slib.compare_per_ic_to_battery1(new, old, 1e-5, 1e-8)
    assert cmp["all_match"]

    new2 = dict(final_V=1.0, max_V=5.0, settling_time=41, success=True)  # off-by-one settling_time
    cmp2 = slib.compare_per_ic_to_battery1(new2, old, 1e-5, 1e-8)
    assert not cmp2["all_match"] and not cmp2["per_field"]["settling_time"]

    new3 = dict(final_V=1.0, max_V=5.0, settling_time=42, success=False)  # wrong success
    cmp3 = slib.compare_per_ic_to_battery1(new3, old, 1e-5, 1e-8)
    assert not cmp3["all_match"] and not cmp3["per_field"]["success"]


# ---------------------------------------------------------------------------
# Import closure
# ---------------------------------------------------------------------------

def test_import_closure_includes_known_dependencies():
    closure = drv.compute_local_import_closure([drv.THIS_DRIVER, drv.THIS_LIB])
    names = {p.name for p in closure}
    for expected in ("ablation_lib.py", "sensitivity_lib.py", "ablation_io.py",
                     "budget_sweep_solver.py", "solver_audit_lib.py", "solver_audit_io.py"):
        assert expected in names, f"import closure missing expected dependency {expected}"


# ---------------------------------------------------------------------------
# Slot enumeration
# ---------------------------------------------------------------------------

def test_all_slots_are_80_and_unique():
    slots = drv.all_slots()
    assert len(slots) == 80
    ids = [drv._slot_id(s) for s in slots]
    assert len(set(ids)) == 80, "duplicate slot IDs found"


def test_reference_by_axis():
    assert drv.REFERENCE_BY_AXIS["n_family"] == "frozen_2026_08_31_baseline"
    assert drv.REFERENCE_BY_AXIS["fd"] == "fd_common_support_1e-3_reference"


# ---------------------------------------------------------------------------
# oracle_n_used cross-check
# ---------------------------------------------------------------------------

def test_oracle_n_used_matches_diagnose_fixed_idx_excluded_fraction():
    small_hyperparams()
    try:
        pool = alib.load_pool()
        import budget_sweep_solver as bs  # noqa: PLC0415
        import probe_action_gradient as pr  # noqa: PLC0415
        clean = alib.build_clean_query_table(pool, bs.SIM, pr.precompute_phi)
        scale = alib.shared_scale_constants(pool, clean)
        r = alib.train_base_condition("E", 0, pool, clean, scale)
        idx = alib.fixed_oracle_idx(pool["test_idx"])[:50]  # small subset for speed
        oracle = alib.diagnose_fixed_idx(r["model"], pool, bs.SIM, idx)
        n_used = slib.oracle_n_used(r["model"], pool, bs.SIM, idx)
        expected_n_used = round(oracle["n_points"] * (1 - oracle["excluded_fraction"]))
        assert n_used == expected_n_used, f"{n_used} != {expected_n_used}"
        assert isinstance(n_used, int)
    finally:
        restore_hyperparams()


# ---------------------------------------------------------------------------
# Real minimal end-to-end: one slot per axis, tiny epochs
# ---------------------------------------------------------------------------

def test_real_one_slot_per_axis_end_to_end():
    small_hyperparams()
    try:
        pool = alib.load_pool()
        import budget_sweep_solver as bs  # noqa: PLC0415
        import probe_action_gradient as pr  # noqa: PLC0415
        from disambiguate_landscape_vs_gradient import make_norm  # noqa: PLC0415
        clean = alib.build_clean_query_table(pool, bs.SIM, pr.precompute_phi)
        scale = alib.shared_scale_constants(pool, clean)
        data_tuple = (pool["XU"], pool["Y"], pool["xm"], pool["xs"], pool["ym"], pool["ys"],
                     pool["train_idx"], pool["test_idx"])
        norm, un_lo, un_hi = make_norm(data_tuple)
        oracle_idx = alib.fixed_oracle_idx(pool["test_idx"])[:50]
        import shutil  # noqa: PLC0415
        tmp = RESULTS_DIR / "_smoke_sensitivity_tmp"
        shutil.rmtree(tmp, ignore_errors=True)  # dev-loop convention: fresh scratch dir each run
        tmp.mkdir(parents=True, exist_ok=True)

        setup = dict(pool=pool, clean=clean, scale=scale, sim=bs.SIM, oracle_idx=oracle_idx)

        # n_family
        r1 = slib.train_variant_condition("E", 0, pool, clean, scale, family="gaussian",
                                          noise_scale=slib.NOISE_SCALE_BASELINE,
                                          checkpoint_dir=tmp, checkpoint_prefix="smoke_nfamily")
        assert r1["n_raw"] is not None

        # n_scale
        r2 = slib.train_variant_condition("B", 0, pool, clean, scale, family="cauchy",
                                          noise_scale=0.4, checkpoint_dir=tmp, checkpoint_prefix="smoke_nscale")
        assert r2["val_mse"] > 0

        # fd (common-support at eps=1e-4)
        clean_1e4 = slib.build_clean_query_table_with_eps(pool, bs.SIM, pr.precompute_phi, 1e-4)
        common_mask = clean["valid_mask"] & clean_1e4["valid_mask"]
        masked = slib.apply_common_support_mask(clean_1e4, common_mask)
        scale_fixed = dict(scale); scale_fixed["ug_scale"] = slib.compute_ug_scale(
            slib.apply_common_support_mask(clean, common_mask))
        r3 = alib.train_base_condition("E", 0, pool, masked, scale_fixed,
                                       checkpoint_dir=tmp, checkpoint_prefix="smoke_fd")
        assert r3["val_mse"] > 0

        # l-grad
        r4 = slib.train_lam_grad_variant(0, pool, clean, scale, lam_grad=0.1,
                                         checkpoint_dir=tmp, checkpoint_prefix="smoke_lgrad")
        assert r4["val_mse"] > 0

        # evaluate one of them end-to-end through the real evaluator path
        seq = alib.freeze_any(r4["model"])
        s = slib.rollout_per_ic_summary(seq, norm, bs.ICS[0], bs.SIM, bs.INPUT_LO, bs.INPUT_HI,
                                        un_lo, un_hi, 3, 20, 120, 0.1, 0.01, 2.0)
        assert set(s.keys()) == set(slib.PER_IC_FIELDS)
        oracle = alib.diagnose_fixed_idx(r4["model"], pool, bs.SIM, oracle_idx)
        assert "align_cos" in oracle
    finally:
        restore_hyperparams()


def test_real_evaluator_compatibility_gate():
    """Runs the actual gate (frozen checkpoints only, no new training) --
    cheap enough to run for real in smoke."""
    import sensitivity_driver_2026_09_03 as drv2  # noqa: PLC0415
    result = drv2.run_evaluator_compatibility_gate()
    assert result["all_passed"], f"evaluator compatibility gate failed: {result}"


def main():
    check("cauchy_baseline_reproduces_frozen_noise", test_cauchy_baseline_reproduces_frozen_noise)
    check("noiseless_is_all_zero", test_noiseless_is_all_zero)
    check("gaussian_mad_matches_cauchy_baseline", test_gaussian_mad_matches_cauchy_baseline)
    check("gaussian_and_correlated_share_base_draw", test_gaussian_and_correlated_share_base_draw)
    check("correlated_noise_actually_correlated", test_correlated_noise_actually_correlated)
    check("noise_diagnostics_definitions", test_noise_diagnostics_definitions)
    check("noise_diagnostics_noiseless_correlation_is_none", test_noise_diagnostics_noiseless_correlation_is_none)
    check("common_support_mask_matches_eps002_nesting", test_common_support_mask_matches_eps002_nesting)
    check("apply_common_support_mask_zeroes_newly_dropped", test_apply_common_support_mask_zeroes_newly_dropped)
    check("compute_ug_scale_matches_shared_scale_constants_formula", test_compute_ug_scale_matches_shared_scale_constants_formula)
    check("compare_per_ic_isclose_vs_exact", test_compare_per_ic_isclose_vs_exact)
    check("import_closure_includes_known_dependencies", test_import_closure_includes_known_dependencies)
    check("all_slots_are_80_and_unique", test_all_slots_are_80_and_unique)
    check("reference_by_axis", test_reference_by_axis)
    check("oracle_n_used_matches_diagnose_fixed_idx_excluded_fraction", test_oracle_n_used_matches_diagnose_fixed_idx_excluded_fraction)
    check("real_one_slot_per_axis_end_to_end", test_real_one_slot_per_axis_end_to_end)
    check("real_evaluator_compatibility_gate", test_real_evaluator_compatibility_gate)

    n_pass = sum(1 for c in CHECKS if c["passed"])
    n_total = len(CHECKS)
    out = dict(n_pass=n_pass, n_total=n_total, all_passed=(n_pass == n_total), checks=CHECKS)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n{n_pass}/{n_total} checks passed. Results: {OUT_PATH}")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
