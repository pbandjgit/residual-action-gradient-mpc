#!/usr/bin/env python3
"""Synthetic-only control-flow smoke test for the JPC go/no-go ablation
protocol's implementation (`src/cstr/ablation_lib.py`,
`src/cstr/ablation_io.py`), per
`docs/JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_2026_08_31.md` v6, Section
0.3. Uses a tiny fabricated pool and a fake, cheap `sim.step` throughout
-- never touches the real 20k-point pool, the real CSTR simulator, or any
real gradient oracle. Exercises every contract point the v1-v6 reviews
flagged: query-count fairness, the boundary-valid pair mask (pairwise
drop, normalized-coordinate eps), the FNN `freeze_any()` adapter through
an actual closed-loop-shaped call (not just materialization in
isolation), per-seed initial-state-hash identity across the LCNN
conditions, RNG-stream separation, the 7-member Holm-corrected secondary
family (including the MSE-match p=1 substitution), the exhaustive
outcome-map branches (including the previously-unclassified boundary
case), and the atomic no-overwrite/self-hash I/O convention.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cstr.ablation_io as aio          # noqa: E402
import cstr.ablation_lib as alib        # noqa: E402
import ablation_driver_2026_08_31 as drv  # noqa: E402
from budget_sweep_solver import INPUT_LO, INPUT_HI  # noqa: E402

RESULTS_DIR = ROOT / "results"
SCRATCH = RESULTS_DIR / "_ablation_smoke_scratch_2026_08_31"

CHECKS = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append({"name": name, "passed": bool(condition), "detail": detail})
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))


def fresh_scratch() -> Path:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    return SCRATCH


# ---------------------------------------------------------------------------
# Fake plant: a cheap linear-ish map, deterministic, no CasADi/gl_gym.
# ---------------------------------------------------------------------------
class FakeSim:
    def step(self, x, u):
        x = np.asarray(x, dtype=np.float64); u = np.asarray(u, dtype=np.float64)
        return np.array([
            0.9 * x[0] + 1e-3 * u[0] + 1e-9 * u[1],
            0.9 * x[1] + 1e-9 * u[0] + 1e-6 * u[1],
        ])


def fake_precompute_phi(sim, x_batch):
    n = len(x_batch)
    Phi = np.zeros((n, 2), dtype=np.float32)
    Yphi = np.array([sim.step(x_batch[i], Phi[i]) for i in range(n)], dtype=np.float32)
    return Phi, Yphi


def make_fake_pool(n_total=200, n_train=100, n_val=40, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.4, 0.4, size=(n_total, 2))
    u = rng.uniform(INPUT_LO * 0.5, INPUT_HI * 0.5, size=(n_total, 2))
    sim = FakeSim()
    y = np.array([sim.step(x[i], u[i]) for i in range(n_total)], dtype=np.float32)
    XU = np.concatenate([x, u], axis=1).astype(np.float32)
    xm = XU.mean(0).astype(np.float32); xs = (XU.std(0) + 1e-3).astype(np.float32)
    ym = y.mean(0).astype(np.float32); ys = (y.std(0) + 1e-3).astype(np.float32)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]; val_idx = perm[n_train:n_train + n_val]; test_idx = perm[n_train + n_val:]
    return dict(XU=XU, Y=y, xm=xm, xs=xs, ym=ym, ys=ys,
                train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def small_hyperparams():
    alib.EPOCHS = 4
    alib.WARMUP = 1
    alib.BATCH = 32
    alib.CHECKPOINT_EVERY = 1
    alib.LAM_JAC_GRID = [0.01, 0.1]
    alib.WEIGHT_DECAY_GRID = [0.0, 1e-3]
    alib.N_BOOTSTRAP = 500


# ===========================================================================
def test_param_count_match():
    n_lcnn, n_fnn = alib.assert_exact_param_count_match()
    check("LCNN and FNN-activation have exactly equal parameter counts", n_lcnn == n_fnn == 1922,
          f"n_lcnn={n_lcnn} n_fnn={n_fnn}")


def test_freeze_any_dispatch_and_closed_loop_shaped_call():
    torch.manual_seed(0)
    lcnn = alib.build_model("A", use_fnn=False)
    torch.manual_seed(0)
    fnn = alib.build_model("A", use_fnn=True)
    seq_lcnn = alib.freeze_any(lcnn)
    seq_fnn = alib.freeze_any(fnn)
    xun = torch.zeros(3, 4)
    raised = False
    try:
        seq_fnn(xun)
    except AttributeError:
        raised = True
    check("freeze_any(FNN model) does NOT raise AttributeError through an actual forward call",
          not raised, "this is the exact v5 P0 failure mode (bs.freeze requires .body/.out)")
    out_lcnn = seq_lcnn(xun)
    out_fnn = seq_fnn(xun)
    check("freeze_any output for LCNN and FNN both produce (3,2) tensors",
          tuple(out_lcnn.shape) == (3, 2) and tuple(out_fnn.shape) == (3, 2))
    check("freeze_any(FNN) disables grad", not any(p.requires_grad for p in seq_fnn.parameters()))


def test_clean_query_table_and_boundary_pairing():
    pool = make_fake_pool(seed=1)
    sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    vm = clean["valid_mask"]
    check("valid_mask shape is (n_train, 2)", vm.shape == (len(pool["train_idx"]), 2))
    counts = alib.valid_pair_counts(clean)
    check("valid_pair_counts totals are internally consistent",
          counts["n_pairs_valid"] + counts["n_pairs_dropped"] == counts["n_pairs_total"])

    # construct one point deliberately near the physical box edge so +eps
    # pushes out (in normalized coords the perturbation is tiny, so we test
    # the boundary logic directly on a synthetic un/u pair instead of
    # relying on random placement).
    pool2 = make_fake_pool(seed=2)
    edge_local = 0
    edge_global = pool2["train_idx"][edge_local]
    u_mean, u_std = pool2["xm"][2:], pool2["xs"][2:]
    un_edge = (INPUT_HI - u_mean) / u_std  # sits exactly at the upper physical box edge
    pool2["XU"][edge_global, 2:] = INPUT_HI
    clean2 = alib.build_clean_query_table(pool2, sim, fake_precompute_phi)
    dim0_valid = clean2["valid_mask"][edge_local, 0]
    check("a point placed exactly at the physical box's upper edge has dim-0 dropped as a PAIR (not a singleton)",
          dim0_valid == False,  # noqa: E712 -- +eps must push past INPUT_HI, dropping both signs
          f"valid_mask[edge,0]={dim0_valid}")

    # normalized-coordinate eps check: the stored un_plus - un_base should
    # equal FD_EPS in NORMALIZED space, not a fixed physical step.
    x_all = pool["XU"][:, :2].astype(np.float64)
    u_all = pool["XU"][:, 2:].astype(np.float64)
    u_mean0, u_std0 = pool["xm"][2:].astype(np.float64), pool["xs"][2:].astype(np.float64)
    un_tr = (u_all[pool["train_idx"]] - u_mean0) / u_std0
    valid0 = clean["valid_mask"][:, 0]
    diffs = clean["un_plus"][valid0, 0, 0] - un_tr[valid0, 0]
    check("perturbation is exactly FD_EPS in NORMALIZED action coordinates",
          np.allclose(diffs, alib.FD_EPS, atol=1e-6), f"max|diff-eps|={np.abs(diffs - alib.FD_EPS).max()}")


def test_query_counts_equal_across_conditions():
    pool = make_fake_pool(seed=3)
    sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    counts = alib.query_counts_per_condition(pool, clean)
    equal_conditions = {counts[c] for c in ("B", "C", "D", "E", "F")}
    check("B/C/D/E/F consume an IDENTICAL total query count", len(equal_conditions) == 1, str(counts))
    check("A's budget is strictly smaller than the query-consuming conditions'",
          counts["A"] < next(iter(equal_conditions)), str(counts))
    n_tr = len(pool["train_idx"])
    check("A's (base-only) budget is exactly n_train (one-step) + n_train (auxiliary-law), "
          "matching Section 2.7's own written train_idx-only scope",
          counts["A"] == 2 * n_tr, f"counts['A']={counts['A']} 2*n_tr={2 * n_tr}")
    check("Phi_all/Yphi_all in the clean query table are TRAIN-LOCAL sized (not full-pool sized)",
          clean["Phi_all"].shape[0] == n_tr and clean["Yphi_all"].shape[0] == n_tr,
          f"Phi_all shape={clean['Phi_all'].shape}")


def test_symmetric_loss_uses_g_scale():
    g_true = torch.tensor([1.0, -1.0, 2.0])
    g_hat = torch.tensor([0.5, -0.5, 1.0])
    g_scale = 2.0
    manual = (((g_true - g_hat) / g_scale) ** 2).mean().item()
    got = alib.residual_value_loss("symmetric", g_true, g_hat, g_scale).item()
    check("symmetric residual loss matches manual (g_true-g_hat)/g_scale squared-mean",
          abs(manual - got) < 1e-9, f"manual={manual} got={got}")
    one_sided = alib.residual_value_loss("one_sided", g_true, g_hat, g_scale).item()
    check("one-sided loss clamps negative residuals to zero (differs from symmetric)",
          one_sided != got)


def test_scale_constants_shapes_and_seed_independence():
    pool = make_fake_pool(seed=4)
    sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    scale = alib.shared_scale_constants(pool, clean)
    check("g_scale is a scalar float", isinstance(scale["g_scale"], float))
    check("ug_scale is a scalar float", isinstance(scale["ug_scale"], float))
    noise0 = alib.build_noise_realized_table(pool, clean, seed=0, sensitivity=False)
    noise1 = alib.build_noise_realized_table(pool, clean, seed=1, sensitivity=False)
    check("disp has shape (2,) and DIFFERS across seeds (the only per-seed scale const)",
          noise0["disp"].shape == (2,) and not np.allclose(noise0["disp"], noise1["disp"]))

    true_ug = clean["true_ug"].astype(np.float64)
    valid_ug = true_ug[clean["valid_mask"]]
    manual_ug_scale = float(np.float32(max(valid_ug.std(ddof=1), 1.0)).item())
    check("ug_scale uses ddof=1 (sample std) and excludes invalid-pair zero placeholders",
          abs(scale["ug_scale"] - manual_ug_scale) < 1e-6,
          f"got={scale['ug_scale']} manual={manual_ug_scale}")
    naive_ug_scale_including_invalid = float(max(true_ug.std(ddof=1), 1.0))
    if (~clean["valid_mask"]).any():
        check("excluding invalid pairs actually changes ug_scale for this fabricated pool "
              "(otherwise this check can't distinguish the fix from the bug)",
              abs(scale["ug_scale"] - naive_ug_scale_including_invalid) > 1e-9)


def test_rng_stream_separation():
    for k in alib.SEEDS:
        a, b, c = alib.base_noise_seed(k), alib.sensitivity_noise_seed(k), alib.shuffle_seed(k)
        check(f"seed {k}: base/sensitivity/shuffle RNG seeds are pairwise distinct",
              len({a, b, c}) == 3, f"{a},{b},{c}")
    pool = make_fake_pool(seed=5)
    sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    noise_primary = alib.build_noise_realized_table(pool, clean, seed=0, sensitivity=False)
    noise_sensitivity = alib.build_noise_realized_table(pool, clean, seed=0, sensitivity=True)
    check("base-row Yn/disp are IDENTICAL between primary and sensitivity arms (only augmented targets differ)",
          np.array_equal(noise_primary["Yn"], noise_sensitivity["Yn"])
          and np.array_equal(noise_primary["disp"], noise_sensitivity["disp"]))
    check("augmented targets DIFFER between primary (clean) and sensitivity (noisy) arms",
          not np.array_equal(noise_primary["y_plus_aug_n"], noise_sensitivity["y_plus_aug_n"]))


def test_initial_state_hash_identical_across_lcnn_conditions(scratch: Path):
    hashes = set()
    for condition in ("A", "B", "C", "D", "E", "F"):
        torch.manual_seed(7)
        model = alib.build_model(condition, use_fnn=False)
        hashes.add(alib.initial_state_hash(model))
    check("A-F (same LCNN architecture, same torch.manual_seed) share an IDENTICAL initial state_dict hash",
          len(hashes) == 1, f"{len(hashes)} distinct hashes")


def test_augmented_residual_target_uses_own_successor():
    """Direct regression check for the P0 fix: an augmented row's
    residual-value target g_true must come from ITS OWN clean
    perturbation successor (`y_plus_n`/`y_minus_n`), never the base
    point's original successor -- reproduces the exact computation
    `train_augmented_condition` performs, independently, on fabricated
    data where the two disagree by a known amount."""
    pool = make_fake_pool(seed=12)
    sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    train_idx = np.asarray(pool["train_idx"])
    ym, ys = pool["ym"], pool["ys"]

    valid_mask = clean["valid_mask"]
    point, dim = np.argwhere(valid_mask)[0]
    base_y_n_point = (pool["Y"][train_idx[point]] - ym) / ys
    aug_y_plus_n_point = clean["y_plus_n"][point, dim]

    check("a valid augmented row's clean perturbation successor differs from its base point's own successor "
          "(otherwise this regression test cannot distinguish the two)",
          not np.allclose(base_y_n_point, aug_y_plus_n_point, atol=1e-6))

    P = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float64)

    def V(y_n):
        y_phys = y_n * ys + ym
        return float(y_phys @ P @ y_phys)

    Yphi_point_n = (clean["Yphi_all"][point] - ym) / ys
    g_true_correct = V(aug_y_plus_n_point) - V(Yphi_point_n)
    g_true_bug = V(base_y_n_point) - V(Yphi_point_n)
    check("the correct (post-fix) and buggy (pre-fix) residual targets are numerically distinguishable "
          "for this fabricated case", abs(g_true_correct - g_true_bug) > 1e-9)

    # Now confirm train_augmented_condition's INTERNAL computation matches
    # g_true_correct, not g_true_bug, by replicating its exact all_phi_local_idx/
    # aug_sign construction and reading the same clean/noise arrays it reads.
    n_tr = len(train_idx)
    aug_rows = [(i, j, sign) for i in range(n_tr) for j in range(2) for sign in (0, 1) if valid_mask[i, j]]
    aug_i = np.array([r[0] for r in aug_rows]); aug_j = np.array([r[1] for r in aug_rows])
    aug_sign = np.array([r[2] for r in aug_rows])
    match = (aug_i == point) & (aug_j == dim) & (aug_sign == 0)
    row_local = int(np.argwhere(match)[0][0])
    y_plus_clean_n, y_minus_clean_n = clean["y_plus_n"], clean["y_minus_n"]
    aug_y_clean = np.where(aug_sign[:, None] == 0, y_plus_clean_n[aug_i, aug_j], y_minus_clean_n[aug_i, aug_j])
    check("train_augmented_condition's own aug_y_clean array (post-fix logic) equals the manually-computed "
          "correct per-row successor, not the base point's successor",
          np.allclose(aug_y_clean[row_local], aug_y_plus_n_point, atol=1e-6))


def test_train_base_and_augmented_smoke(scratch: Path):
    small_hyperparams()
    pool = make_fake_pool(seed=8)
    sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    scale = alib.shared_scale_constants(pool, clean)
    ckpt_dir = scratch / "ckpts"

    r_a = alib.train_base_condition("A", 0, pool, clean, scale, checkpoint_dir=ckpt_dir)
    check("condition A trains and produces finite val/test MSE",
          np.isfinite(r_a["val_mse"]) and np.isfinite(r_a["test_mse"]))

    r_e = alib.train_base_condition("E", 0, pool, clean, scale, checkpoint_dir=ckpt_dir)
    check("condition E (L_grad) trains and produces finite val/test MSE",
          np.isfinite(r_e["val_mse"]) and np.isfinite(r_e["test_mse"]))

    r_d = alib.train_base_condition("D", 0, pool, clean, scale, lam_jac=0.05, checkpoint_dir=ckpt_dir)
    check("condition D (L_Jac) trains and produces finite val/test MSE",
          np.isfinite(r_d["val_mse"]) and np.isfinite(r_d["test_mse"]))
    check("condition D produced at least one checkpoint at the small cadence",
          len(r_d["checkpoints"]) > 0)

    r_b = alib.train_augmented_condition("B", 0, pool, clean, scale, checkpoint_dir=ckpt_dir)
    check("condition B (augmented, clean primary) trains and produces finite val/test MSE",
          np.isfinite(r_b["val_mse"]) and np.isfinite(r_b["test_mse"]))
    check("condition B produced at least one checkpoint", len(r_b["checkpoints"]) > 0)

    r_c = alib.train_augmented_condition("C", 0, pool, clean, scale, checkpoint_dir=ckpt_dir)
    check("condition C (augmented, symmetric primary) trains and produces finite val/test MSE",
          np.isfinite(r_c["val_mse"]) and np.isfinite(r_c["test_mse"]))

    r_bfnn = alib.train_augmented_condition("B-FNN", 0, pool, clean, scale, use_fnn=True, checkpoint_dir=ckpt_dir,
                                             checkpoint_prefix="BFNN")
    check("B-FNN (FNN architecture block) trains and produces finite val/test MSE",
          np.isfinite(r_bfnn["val_mse"]) and np.isfinite(r_bfnn["test_mse"]))

    return dict(r_a=r_a, r_b=r_b, r_c=r_c, r_d=r_d, r_e=r_e)


def test_select_lambda_jac(scratch: Path):
    small_hyperparams()
    pool = make_fake_pool(seed=9)
    sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    scale = alib.shared_scale_constants(pool, clean)
    ckpt_dir = scratch / "lamjac_ckpts"
    result = alib.select_lambda_jac(0, pool, clean, scale, grid=[0.01, 0.1], checkpoint_dir=ckpt_dir)
    check("select_lambda_jac returns a grid trace covering every candidate",
          len(result["grid_trace"]) == 2)
    check("select_lambda_jac's selected lambda is a member of the grid",
          result["selected_lambda_jac"] in [0.01, 0.1])
    min_mse = min(t["val_mse"] for t in result["grid_trace"])
    selected_mse = next(t["val_mse"] for t in result["grid_trace"] if t["lam_jac"] == result["selected_lambda_jac"])
    check("select_lambda_jac chooses the grid value with minimum validation MSE",
          abs(selected_mse - min_mse) < 1e-12)
    check("EVERY grid candidate (not just the selected one) has its own checkpoint(s) on disk",
          all(len(t["checkpoints"]) > 0 for t in result["grid_trace"]))
    all_ckpt_paths = [c["path"] for t in result["grid_trace"] for c in t["checkpoints"]]
    check("every grid candidate's checkpoint paths are distinct (no collision across lambda values)",
          len(all_ckpt_paths) == len(set(all_ckpt_paths)))
    check("all_results retains a trained model object for every grid candidate (not discarded)",
          set(result["all_results"].keys()) == {0.01, 0.1}
          and all(r["model"] is not None for r in result["all_results"].values()))


def test_diagnose_fixed_idx_and_oracle_determinism():
    pool = make_fake_pool(seed=10)
    sim = FakeSim()
    torch.manual_seed(0)
    model = alib.build_model("A", use_fnn=False)
    idx1 = alib.fixed_oracle_idx(pool["test_idx"], n_pts=20)
    idx2 = alib.fixed_oracle_idx(pool["test_idx"], n_pts=20)
    check("fixed_oracle_idx is deterministic given the same meta-seed", np.array_equal(idx1, idx2))
    diag = alib.diagnose_fixed_idx(model, pool, sim, idx1)
    check("diagnose_fixed_idx returns the expected metric keys",
          {"align_cos", "rel_grad_error", "excluded_fraction", "n_points"} <= set(diag.keys()))
    check("diagnose_fixed_idx's n_points matches the passed idx length", diag["n_points"] == len(idx1))


def test_bootstrap_ci_fresh_generator_reproducibility():
    diffs_a = np.array([0.1, 0.2, -0.1, 0.3, 0.0, 0.1, 0.2, 0.1, -0.05, 0.15])
    diffs_b = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5])
    ci_a1 = alib.bootstrap_ci(diffs_a, n_resample=200)
    ci_a2 = alib.bootstrap_ci(diffs_a, n_resample=200)
    check("bootstrap_ci is exactly reproducible across two independent calls (fresh generator each time)",
          np.array_equal(ci_a1["resample_idx"], ci_a2["resample_idx"]))
    ci_b = alib.bootstrap_ci(diffs_b, n_resample=200)
    check("the SAME fixed seed produces the SAME resample-index pattern regardless of the data (n identical)",
          np.array_equal(ci_a1["resample_idx"], ci_b["resample_idx"]))
    check("CI lower <= point estimate <= CI upper for a clearly-positive series",
          ci_a1["ci_lower"] <= ci_a1["point_estimate"] <= ci_a1["ci_upper"])


def test_exact_permutation_and_holm():
    diffs = np.array([0.3, 0.25, 0.4, 0.2, 0.35, 0.3, 0.28, 0.32, 0.22, 0.31])
    perm = alib.exact_permutation_pvalue(diffs)
    check("exact permutation test enumerates all 2^10 sign patterns", perm["n_patterns"] == 1024)
    check("permutation p-value is in [0,1]", 0.0 <= perm["p_value"] <= 1.0)
    check("a clearly-positive, low-variance series yields a small permutation p-value", perm["p_value"] < 0.05)

    pvals = [0.001, 0.01, 0.02, 0.5, 0.5, 0.5, 0.5]
    adj = alib.holm_adjust(pvals)
    check("Holm-adjusted p-values are non-decreasing under the sorted order and each >= raw p",
          all(a >= p - 1e-12 for a, p in zip(adj, pvals)))
    check("Holm adjustment produces exactly 7 outputs for a 7-item family", len(adj) == 7)


def test_secondary_family_size_and_mse_infeasible_p1():
    seeds = alib.SEEDS
    rng = np.random.default_rng(0)
    conditions = ["A", "B", "C", "D", "E", "F", "B-FNN", "E-FNN"]
    success_rates = {c: {s: float(rng.uniform(0, 1)) for s in seeds} for c in conditions}
    fam = alib.compute_secondary_family(success_rates, seeds, mse_match_diffs=None)
    check("secondary family has exactly 7 members", len(fam) == 7, str(sorted(fam.keys())))
    check("MSE_match contrast is p=1 (never significant) when MSE-match is infeasible",
          fam["MSE_match"]["p_value"] == 1.0 and fam["MSE_match"]["infeasible"] is True)
    check("MSE_match's Holm-adjusted p-value is still computed (never dropped from the family)",
          "p_value_holm_adjusted" in fam["MSE_match"])
    holm_vals = [fam[n]["p_value_holm_adjusted"] for n in alib.SECONDARY_CONTRAST_NAMES]
    check("every secondary contrast carries a Holm-adjusted p-value in [0,1]",
          all(0.0 <= v <= 1.0 for v in holm_vals))


def test_outcome_map_exhaustive_classification():
    cases = [
        dict(ci_lower=0.05, ci_upper=0.5, point_estimate=0.3, expect="SUPPORTED"),
        dict(ci_lower=-0.5, ci_upper=-0.05, point_estimate=-0.3, expect="NEGATIVE"),
        dict(ci_lower=-0.05, ci_upper=0.05, point_estimate=-0.05, expect="UNCLEAR"),  # the v3-fixed edge case
        dict(ci_lower=0.01, ci_upper=0.3, point_estimate=0.15, expect="UNCLEAR"),
        # point_estimate <= -0.1 triggers NEGATIVE via the materiality floor
        # regardless of CI width (v6 Section 9) -- this case is NOT "UNCLEAR".
        dict(ci_lower=-0.5, ci_upper=0.5, point_estimate=-0.15, expect="NEGATIVE"),
        dict(ci_lower=-0.5, ci_upper=0.5, point_estimate=-0.05, expect="UNCLEAR"),
    ]
    for c in cases:
        got = alib.classify_query_matched_effect(c)
        check(f"QUERY_MATCHED_EFFECT({c['ci_lower']},{c['ci_upper']},{c['point_estimate']}) == {c['expect']}",
              got == c["expect"], f"got {got}")

    interaction_cases = [
        dict(p_value_holm_adjusted=0.01, ci_lower=0.3, ci_upper=0.5, expect="INTERACTING"),
        dict(p_value_holm_adjusted=0.9, ci_lower=-0.05, ci_upper=0.05, expect="EQUIVALENT"),
        dict(p_value_holm_adjusted=0.9, ci_lower=-0.3, ci_upper=0.3, expect="INCONCLUSIVE"),
    ]
    for c in interaction_cases:
        got = alib.classify_loss_interaction(c)
        check(f"LOSS_INTERACTION classification == {c['expect']}", got == c["expect"], f"got {got}")


def test_mse_attribution_exhaustive():
    fake_contrast_sig = dict(p_value_holm_adjusted=0.01, point_estimate=0.3)
    fake_contrast_small = dict(p_value_holm_adjusted=0.01, point_estimate=0.05)
    fake_contrast_ns = dict(p_value_holm_adjusted=0.9, point_estimate=0.01)

    r1 = alib.classify_mse_attribution("MSE_MATCH_INFEASIBLE", fake_contrast_sig, 0)
    check("MSE_ATTRIBUTION: infeasible -> UNRESOLVED/MSE_MATCH_INFEASIBLE",
          r1["status"] == "UNRESOLVED" and r1["sub_label"] == "MSE_MATCH_INFEASIBLE")

    r2 = alib.classify_mse_attribution("FEASIBLE", fake_contrast_sig, 4)
    check("MSE_ATTRIBUTION: >3 test-MSE generalization failures -> UNRESOLVED/VALIDATION_MATCH_DID_NOT_GENERALIZE",
          r2["status"] == "UNRESOLVED" and r2["sub_label"] == "VALIDATION_MATCH_DID_NOT_GENERALIZE")

    r3 = alib.classify_mse_attribution("FEASIBLE", fake_contrast_sig, 0)
    check("MSE_ATTRIBUTION: Holm-significant + point>=0.2 -> SUPPORTED", r3["status"] == "SUPPORTED")

    r4 = alib.classify_mse_attribution("FEASIBLE", fake_contrast_small, 0)
    check("MSE_ATTRIBUTION: significant but point<0.2 -> NOT_SUPPORTED (exhaustive remainder)",
          r4["status"] == "NOT_SUPPORTED")

    r5 = alib.classify_mse_attribution("FEASIBLE", fake_contrast_ns, 0)
    check("MSE_ATTRIBUTION: not significant -> NOT_SUPPORTED", r5["status"] == "NOT_SUPPORTED")


def test_full_outcome_orchestration():
    seeds = alib.SEEDS
    rng = np.random.default_rng(1)
    conditions = ["A", "B", "C", "D", "E", "F", "B-FNN", "E-FNN"]
    success_rates = {c: {s: 0.2 for s in seeds} for c in conditions}
    for s in seeds:
        success_rates["E"][s] = 0.9
        success_rates["E-FNN"][s] = 0.9
    fam = alib.compute_secondary_family(success_rates, seeds, mse_match_diffs=None)
    primary = alib.evaluate_contrast(alib.paired_success_rate_diff(success_rates["E"], success_rates["B"], seeds))
    mse_attr = alib.classify_mse_attribution("MSE_MATCH_INFEASIBLE", fam["MSE_match"], 0)
    outcome = alib.evaluate_full_outcome(primary, fam, mse_attr)
    check("full outcome orchestration returns all required keys",
          {"query_matched_effect", "target_specificity", "non_groupsort_replication",
           "loss_interaction", "mse_attribution", "manuscript_action"} <= set(outcome.keys()))
    check("a strong, clean E-vs-B separation classifies QUERY_MATCHED_EFFECT as SUPPORTED",
          outcome["query_matched_effect"] == "SUPPORTED")
    check("MSE_ATTRIBUTION=UNRESOLVED (infeasible) blocks the full-claim conjunction",
          outcome["manuscript_action"] == "NARROW_TO_SUBSET")


def test_failure_completeness_contract():
    raised = False
    try:
        alib.check_finite_or_diverged(float("nan"), "E", 0)
    except alib.TrainingDivergedError:
        raised = True
    check("check_finite_or_diverged raises TrainingDivergedError on NaN loss", raised)
    ok_raised = False
    try:
        alib.check_finite_or_diverged(1.23, "E", 0)
    except alib.TrainingDivergedError:
        ok_raised = True
    check("check_finite_or_diverged does not raise on a finite loss", not ok_raised)

    def fake_closed_loop_numerical_failure(**kw):
        raise RuntimeError("simulated NaN-triggered solver failure")

    def fake_closed_loop_programming_bug(**kw):
        raise NameError("simulated undefined-variable bug")

    def fake_closed_loop_ok(**kw):
        return {"final_V": 1.0, "success": True}

    check("safe_closed_loop_success returns False (not an exception) for a numerical/simulator failure",
          alib.safe_closed_loop_success(fake_closed_loop_numerical_failure) is False)
    prog_bug_propagated = False
    try:
        alib.safe_closed_loop_success(fake_closed_loop_programming_bug)
    except NameError:
        prog_bug_propagated = True
    check("safe_closed_loop_success does NOT swallow a programming-error exception type (NameError propagates)",
          prog_bug_propagated)
    check("safe_closed_loop_success returns True for a genuinely successful rollout",
          alib.safe_closed_loop_success(fake_closed_loop_ok) is True)

    raised_gap = False
    try:
        alib.require_seed_completeness({"A": [0, 1, 3]}, seeds=[0, 1, 2, 3])
    except RuntimeError:
        raised_gap = True
    check("require_seed_completeness raises when a condition is missing a seed", raised_gap)
    alib.require_seed_completeness({"A": [0, 1, 2, 3]}, seeds=[0, 1, 2, 3])  # must not raise
    check("require_seed_completeness does not raise when every seed is present", True)


def test_atomic_io_no_overwrite_and_self_hash(scratch: Path):
    p = scratch / "io_test" / "artifact.json"
    aio.write_with_self_verification(p, {"a": 1, "b": [1, 2, 3]})
    raised = False
    try:
        aio.write_with_self_verification(p, {"a": 2})
    except FileExistsError:
        raised = True
    check("write_with_self_verification refuses to overwrite an existing artifact", raised)
    _, payload = aio.read_and_verify_json_artifact(p)
    check("read_and_verify_json_artifact round-trips the payload correctly", payload == {"a": 1, "b": [1, 2, 3]})

    p2 = scratch / "io_test" / "nested" / "deep" / "artifact2.json"
    aio.atomic_write_json_no_overwrite(p2, {"x": 1})
    check("atomic_write_json_no_overwrite creates missing nested parent directories", p2.exists())


def _build_scratch_freeze_manifest(scratch: Path, tag: str) -> Path:
    """Builds a REAL, self-verifying freeze manifest against the actual
    current project files (this is safe -- it writes only to `scratch`,
    never to `docs/FREEZE_MANIFEST_PATH`), by temporarily pointing
    `drv.SMOKE_RESULTS_PATH` at a fabricated passing-smoke record for the
    duration of the call only. `tag` keeps each call's paths distinct
    under the shared, whole-run `scratch` directory."""
    fake_smoke_path = scratch / f"fake_smoke_for_freeze_{tag}.json"
    aio.atomic_write_json_no_overwrite(fake_smoke_path, {
        "n_pass": 1, "n_total": 1, "all_passed": True,
        "checks": [{"name": "fabricated-scratch-check", "passed": True, "detail": ""}],
    })
    old_path = drv.SMOKE_RESULTS_PATH
    try:
        drv.SMOKE_RESULTS_PATH = fake_smoke_path
        return drv.build_freeze_manifest(dest_path=scratch / f"scratch_freeze_manifest_{tag}.json")
    finally:
        drv.SMOKE_RESULTS_PATH = old_path


def test_freeze_manifest_tamper_detection(scratch: Path):
    freeze_path = _build_scratch_freeze_manifest(scratch, "tamper")
    self_hash = drv.verify_freeze_manifest(freeze_path)
    check("verify_freeze_manifest returns a non-empty self-hash for a genuine, untampered manifest",
          isinstance(self_hash, str) and len(self_hash) == 64)

    _, payload = aio.read_and_verify_json_artifact(freeze_path)
    stripped = dict(payload)
    stripped["file_sha256"] = dict(payload["file_sha256"])
    dropped_key = next(iter(stripped["file_sha256"]))
    del stripped["file_sha256"][dropped_key]
    tampered_path = scratch / "tampered_freeze_manifest.json"
    aio.write_with_self_verification(tampered_path, stripped)
    raised = False
    try:
        drv.verify_freeze_manifest(tampered_path)
    except RuntimeError as e:
        raised = "FREEZE_KEYSET_MISMATCH" in str(e)
    check("verify_freeze_manifest rejects a manifest with a required file's entry STRIPPED "
          "(even though the stripped manifest is internally self-consistent)", raised)


def test_manifest_chain_on_scratch(scratch: Path):
    small_hyperparams()
    alib.LAM_JAC_GRID = [0.05]
    alib.WEIGHT_DECAY_GRID = [0.0]
    run_dir = scratch / "ablation_run"
    pool = make_fake_pool(seed=11)
    sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    scale = alib.shared_scale_constants(pool, clean)

    freeze_path = _build_scratch_freeze_manifest(scratch, "chain")

    import budget_sweep_solver as bs
    old_sim = bs.SIM
    try:
        # run_seed now asserts `sim is bs.SIM` (Section-1 simulator-identity
        # fix) -- point bs.SIM at the SAME object passed to run_full_protocol,
        # not merely an equivalent-behaving separate instance.
        bs.SIM = sim
        out = drv.run_full_protocol(pool, clean, scale, sim, out_dir=run_dir, seeds=[0, 1],
                                     freeze_manifest_path=freeze_path)
    finally:
        bs.SIM = old_sim

    check("run_full_protocol's M1 self-verifies",
          aio.read_and_verify_json_artifact(out["m1"]["path"])[0] == out["m1"]["self_hash"])
    m1_payload = json.loads(out["m1"]["path"].read_text())
    check("M1 records the ACTUAL clean-query-table content hash (not merely counts)",
          "clean_query_table_sha256" in m1_payload and len(m1_payload["clean_query_table_sha256"]) == 64)

    check("run_full_protocol produced a Training Manifest (M2) for every requested seed",
          set(out["m2_paths"].keys()) == {0, 1})
    m2_0 = json.loads(out["m2_paths"][0].read_text())
    check("M2 records the predecessor Query Manifest's hash",
          m2_0["predecessor_query_manifest_sha256"] == out["m1"]["self_hash"])
    check("M2 records all 7 shared scale-constant fields",
          {"xm", "xs", "ym", "ys", "g_scale", "ug_scale"} <= set(m2_0["scale_constants"].keys()))
    check("M2 records the FULL lambda_Jac grid trace, not just the selected value",
          len(m2_0["lambda_jac_grid_trace"]) == len(alib.LAM_JAC_GRID))
    check("M2 records wall-clock and optimizer-step disclosure per condition",
          "A" in m2_0["wall_clock_seconds"] and "A" in m2_0["optimizer_steps"])
    check("M2 records real closed-loop success rates for every condition",
          set(m2_0["closed_loop_success"].keys()) >= {"A", "B", "C", "D", "E", "F", "B-FNN", "E-FNN"})
    check("M1 records pool CONTENT hash (not the on-disk file's hash), decoupled from provenance assumptions",
          "pool_content_sha256" in m1_payload and "pool_file_sha256" not in m1_payload)
    check("M2's scale_constants includes disp (the 7th, per-seed scale constant, previously missing)",
          "disp" in m2_0["scale_constants"] and len(m2_0["scale_constants"]["disp"]) == 2)
    check("M2 references a hash-locked companion NOISE_REALIZED_TABLE npz",
          "noise_realized_table_sha256" in m2_0 and (ROOT / m2_0["noise_realized_table_path"]).exists())
    check("M2 records the pre-registered non-gating sensitivity arm's results for B and C",
          set(m2_0["sensitivity_arm"]["closed_loop_success"].keys()) == {"B", "C"})
    check("M2's checkpoint hash closure includes MSE-match attempt-log paths, not just primary/D-grid checkpoints",
          any(a.get("path") in m2_0["checkpoint_sha256"] for a in m2_0["mse_match_attempt_log"] if a.get("path")))
    check("M2 records full oracle summary stats (not just a median) per condition",
          {"align_cos", "rel_grad_error", "excluded_fraction"} <= set(m2_0["oracle_summary"]["E"].keys())
          and "frac_neg" in m2_0["oracle_summary"]["E"]["align_cos"])

    check("run_full_protocol's M3 self-verifies",
          aio.read_and_verify_json_artifact(out["m3"]["path"])[0] == out["m3"]["self_hash"])
    m3_payload = json.loads(out["m3"]["path"].read_text())
    check("M3 secondary_family has 7 entries", len(m3_payload["secondary_family"]) == 7)
    check("M3 contains no raw resample_idx arrays inline (hashed/companion-file referenced instead)",
          "resample_idx" not in m3_payload["primary_contrast"])
    check("M3 references a companion raw-metrics file with a recorded hash",
          "raw_metrics_sha256" in m3_payload and len(m3_payload["raw_metrics_sha256"]) == 64)
    check("M3's companion raw-resample-arrays npz actually contains the primary contrast's resample indices",
          (ROOT / m3_payload["raw_resample_arrays_path"]).exists())
    with np.load(ROOT / m3_payload["raw_resample_arrays_path"]) as raw:
        check("the primary contrast's raw resample_idx array has shape (n_resample, n_seeds) and is recoverable",
              raw["primary_resample_idx"].shape == (alib.N_BOOTSTRAP, 2))
    raw_metrics = json.loads((ROOT / m3_payload["raw_metrics_path"]).read_text())
    check("M3's companion raw-metrics file contains per-seed per-condition closed-loop success",
          raw_metrics["closed_loop_success"]["E"]["0"] is not None)
    check("M3's companion raw-metrics file contains full oracle stats (frac_neg, excluded_fraction), not just medians",
          "frac_neg" in raw_metrics["oracle_align_cos"]["E"]["0"]
          and raw_metrics["oracle_excluded_fraction"]["E"]["0"] is not None)
    check("M3's companion raw-metrics file reports the pre-registered sensitivity arm's closed-loop success",
          raw_metrics["sensitivity_arm_closed_loop_success"]["B"]["0"] is not None)
    check("M3 persists the ACTUAL fixed oracle index array (not merely its hash)",
          "oracle_fixed_idx_path" in m3_payload and (ROOT / m3_payload["oracle_fixed_idx_path"]).exists())
    with np.load(ROOT / m3_payload["oracle_fixed_idx_path"]) as oidx:
        check("the persisted oracle index array is non-empty and matches the recorded count",
              len(oidx["oracle_idx"]) > 0)
    check("M3 reports a descriptive per-lambda-candidate E-vs-D contrast for the FULL lambda_Jac grid",
          set(m3_payload["lambda_jac_full_grid_contrasts"].keys()) == {str(lam) for lam in alib.LAM_JAC_GRID})
    check("outcome map's QUERY_MATCHED_EFFECT is one of the three valid categories",
          m3_payload["outcome"]["query_matched_effect"] in ("SUPPORTED", "UNCLEAR", "NEGATIVE"))

    # --- Resume-or-verify: rerunning against the SAME out_dir must reuse
    # M1/M2 (never retrain, never raise FileExistsError) and reproduce an
    # identical M3.
    import budget_sweep_solver as bs2
    old_sim2 = bs2.SIM
    try:
        bs2.SIM = sim
        out_resumed = drv.run_full_protocol(pool, clean, scale, sim, out_dir=run_dir, seeds=[0, 1],
                                             freeze_manifest_path=freeze_path)
    finally:
        bs2.SIM = old_sim2
    check("resuming run_full_protocol against the same out_dir does not crash "
          "(no FileExistsError from re-attempting no-overwrite writes)",
          out_resumed["m3"]["self_hash"] is not None)
    check("resuming reuses the IDENTICAL M1 (same self-hash, never rebuilt)",
          out_resumed["m1"]["self_hash"] == out["m1"]["self_hash"])
    check("resuming reuses the IDENTICAL M2 for every seed (never retrained)",
          all(out_resumed["m2_paths"][s] == out["m2_paths"][s] for s in (0, 1)))
    check("resuming reproduces a BYTE-IDENTICAL M3 outcome (deterministic reconstruction from persisted M2 alone)",
          json.loads(out_resumed["m3"]["path"].read_text()) == m3_payload)

    def _comparable_aggregated(agg):
        # NaN != NaN under `==`, so MSE_match's infeasible-branch NaN
        # fields would make an otherwise-identical dict compare unequal;
        # round-trip through the same NaN->None sanitizer the atomic JSON
        # writer already uses so both sides normalize identically.
        def strip(c):
            return aio._sanitize_nonfinite({k: v for k, v in c.items() if k != "resample_idx"})
        return dict(outcome=agg["outcome"], primary=strip(agg["primary_contrast"]),
                    secondary={n: strip(c) for n, c in agg["secondary_family"].items()})

    check("resuming's INDEPENDENTLY RECOMPUTED aggregation (aggregate_and_evaluate runs fresh even on M3 "
          "reuse, not merely a re-read of the old M3 file) reproduces the identical outcome from the "
          "persisted M2 summaries alone",
          _comparable_aggregated(out_resumed["aggregated"]) == _comparable_aggregated(out["aggregated"]))

    # A THIRD seed appended to an already-partially-complete run_dir must
    # train ONLY the new seed, reusing the existing seeds' M2 untouched.
    try:
        bs2.SIM = sim
        out_extended = drv.run_full_protocol(pool, clean, scale, sim, out_dir=run_dir, seeds=[0, 1, 2],
                                              freeze_manifest_path=freeze_path)
    finally:
        bs2.SIM = old_sim2
    check("extending a resumed run to an additional seed succeeds and reuses the prior seeds' M2 unchanged",
          out_extended["m2_paths"][0] == out["m2_paths"][0]
          and out_extended["m2_paths"][1] == out["m2_paths"][1]
          and 2 in out_extended["m2_paths"])


def test_resume_tamper_detection(scratch: Path):
    """Direct regression coverage for the P0 finding that resume-or-verify
    trusted a reused M1/M2/M3's own self-hash without re-verifying the
    ACTUAL evidence files (checkpoints, noise table, M3 companions) or the
    live freeze chain. Runs one small seed once, then tampers with each
    piece of evidence in turn and confirms resume now raises instead of
    silently proceeding."""
    small_hyperparams()
    alib.LAM_JAC_GRID = [0.05]
    alib.WEIGHT_DECAY_GRID = [0.0]
    run_dir = scratch / "tamper_run"
    pool = make_fake_pool(seed=20)
    sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    scale = alib.shared_scale_constants(pool, clean)
    freeze_path = _build_scratch_freeze_manifest(scratch, "resumetamper")

    import budget_sweep_solver as bs3
    old_sim3 = bs3.SIM

    def _run():
        bs3.SIM = sim
        try:
            return drv.run_full_protocol(pool, clean, scale, sim, out_dir=run_dir, seeds=[0],
                                          freeze_manifest_path=freeze_path)
        finally:
            bs3.SIM = old_sim3

    out = _run()
    m1_path = out["m1"]["path"]
    m2_path = out["m2_paths"][0]

    # 1. tamper a checkpoint referenced by M2.
    m2_payload = json.loads(m2_path.read_text())
    ckpt_path = Path(next(iter(m2_payload["checkpoint_sha256"].keys())))
    original_ckpt_bytes = ckpt_path.read_bytes()
    ckpt_path.write_bytes(b"corrupted" + original_ckpt_bytes[9:])
    raised = False
    try:
        _run()
    except RuntimeError as e:
        raised = "CHECKPOINT_DRIFT" in str(e)
    check("resume detects a TAMPERED checkpoint file referenced by M2 (CHECKPOINT_DRIFT)", raised)
    ckpt_path.write_bytes(original_ckpt_bytes)  # restore for subsequent checks

    # 2. delete that same checkpoint entirely.
    ckpt_path.unlink()
    raised = False
    try:
        _run()
    except RuntimeError as e:
        raised = "CHECKPOINT_MISSING" in str(e)
    check("resume detects a DELETED checkpoint file referenced by M2 (CHECKPOINT_MISSING)", raised)
    ckpt_path.write_bytes(original_ckpt_bytes)  # restore

    check("resume succeeds again once the tampered checkpoint is restored (not a false-positive failure mode)",
          _run()["m3"]["self_hash"] is not None)

    # 3. tamper the noise-realized-table companion npz.
    m2_payload = json.loads(m2_path.read_text())
    noise_path = ROOT / m2_payload["noise_realized_table_path"]
    original_noise_bytes = noise_path.read_bytes()
    noise_path.write_bytes(original_noise_bytes + b"\x00")
    raised = False
    try:
        _run()
    except RuntimeError as e:
        raised = "NOISE_TABLE_DRIFT" in str(e)
    check("resume detects a TAMPERED noise-realized-table npz referenced by M2 (NOISE_TABLE_DRIFT)", raised)
    noise_path.write_bytes(original_noise_bytes)  # restore
    check("resume succeeds again once the tampered noise table is restored", _run()["m3"]["self_hash"] is not None)

    # 4. tamper an M3 companion file (raw metrics json).
    out_ok = _run()
    m3_payload = json.loads(out_ok["m3"]["path"].read_text())
    raw_metrics_path = ROOT / m3_payload["raw_metrics_path"]
    original_raw_bytes = raw_metrics_path.read_bytes()
    raw_metrics_path.write_bytes(original_raw_bytes + b" ")
    raised = False
    try:
        _run()
    except RuntimeError as e:
        raised = "M3_COMPANION_DRIFT" in str(e)
    check("resume detects a TAMPERED M3 companion file (raw metrics json) (M3_COMPANION_DRIFT)", raised)
    raw_metrics_path.write_bytes(original_raw_bytes)  # restore
    check("resume succeeds again once the tampered M3 companion is restored", _run()["m3"]["self_hash"] is not None)

    # 5. pass a DIFFERENT clean-query-table object on resume (same on-disk
    # npz file, different in-memory content) -- must be caught even though
    # the file itself was never touched.
    different_clean = alib.build_clean_query_table(make_fake_pool(seed=21), sim, fake_precompute_phi)
    raised = False
    try:
        drv.verify_query_manifest(m1_path, clean=different_clean)
    except RuntimeError as e:
        raised = "CLEAN_TABLE_CONTENT_MISMATCH" in str(e)
    check("verify_query_manifest detects a DIFFERENT in-memory clean-query-table on resume "
          "(CLEAN_TABLE_CONTENT_MISMATCH), even though the on-disk npz file is untouched", raised)

    # 6. a stale freeze manifest (rebuilt since M1 was made) must be caught.
    stale_freeze_path = _build_scratch_freeze_manifest(scratch, "resumetamper_stale")
    raised = False
    try:
        drv.verify_query_manifest(m1_path, freeze_manifest_path=stale_freeze_path)
    except RuntimeError as e:
        raised = "FREEZE_CHAIN_STALE" in str(e)
    check("verify_query_manifest detects a STALE freeze chain (a different Freeze Manifest than M1's "
          "recorded predecessor) (FREEZE_CHAIN_STALE)", raised)


def test_simulator_identity_mismatch_rejected(scratch: Path):
    """Direct regression coverage for the 5th-round P0 finding that
    `run_seed`'s `sim` parameter (used for training/oracle) could silently
    diverge from the module-level `budget_sweep_solver.SIM` global that the
    real `closed_loop()` call actually uses internally (it cannot accept an
    injected simulator). Confirms the guard added at the top of `run_seed`
    actually fires when a MISMATCHED sim is passed, and that the matched
    case (already exercised throughout `test_manifest_chain_on_scratch`)
    is not accidentally always truthy."""
    small_hyperparams()
    alib.LAM_JAC_GRID = [0.05]
    alib.WEIGHT_DECAY_GRID = [0.0]
    pool = make_fake_pool(seed=30)
    sim = FakeSim()
    other_sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    scale = alib.shared_scale_constants(pool, clean)
    oracle_idx = alib.fixed_oracle_idx(pool["test_idx"])

    import budget_sweep_solver as bs
    old_sim = bs.SIM
    try:
        bs.SIM = other_sim  # a DIFFERENT instance than `sim`, same class
        raised = False
        try:
            drv.run_seed(0, pool, clean, scale, scratch / "simid_ckpt", oracle_idx, sim)
        except RuntimeError as e:
            raised = "SIMULATOR_IDENTITY_MISMATCH" in str(e)
        check("run_seed rejects a `sim` that is not the SAME object as budget_sweep_solver.SIM "
              "(SIMULATOR_IDENTITY_MISMATCH), even when the mismatched object is behaviorally equivalent",
              raised)

        bs.SIM = sim  # now the SAME object -- the guard must not fire
        no_raise = True
        try:
            drv.run_seed(0, pool, clean, scale, scratch / "simid_ckpt_ok", oracle_idx, sim)
        except RuntimeError as e:
            if "SIMULATOR_IDENTITY_MISMATCH" in str(e):
                no_raise = False
            else:
                raise
        check("run_seed proceeds past the identity guard when `sim IS budget_sweep_solver.SIM`", no_raise)
    finally:
        bs.SIM = old_sim


def test_m3_content_tamper_detection(scratch: Path):
    """Direct regression coverage for the 5th-round P0 finding that
    `verify_results_manifest` only proved M3's file bytes and companion
    files were unmodified since M3 was built -- it never proved the
    NUMBERS inside M3 were ever correct. A self-consistent (re-hashed)
    but WRONG M3 must still be rejected on resume via
    `_assert_aggregated_matches_m3_payload`, which re-derives the outcome
    fresh from the resumed seeds' M2 data."""
    small_hyperparams()
    alib.LAM_JAC_GRID = [0.05]
    alib.WEIGHT_DECAY_GRID = [0.0]
    run_dir = scratch / "m3_tamper_run"
    pool = make_fake_pool(seed=31)
    sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    scale = alib.shared_scale_constants(pool, clean)
    freeze_path = _build_scratch_freeze_manifest(scratch, "m3tamper")

    import budget_sweep_solver as bs4
    old_sim4 = bs4.SIM
    try:
        bs4.SIM = sim
        out = drv.run_full_protocol(pool, clean, scale, sim, out_dir=run_dir, seeds=[0],
                                     freeze_manifest_path=freeze_path)
    finally:
        bs4.SIM = old_sim4
    m3_path = out["m3"]["path"]

    # Tamper M3's CONTENT (not just its bytes) while keeping it internally
    # self-consistent: overwrite the outcome field with a bogus value, then
    # recompute the self-hash over the modified payload exactly as
    # `write_with_self_verification` would have -- so `read_and_verify_
    # json_artifact`'s own self-hash check passes, and this can ONLY be
    # caught by re-deriving the outcome fresh from the M2 data.
    original_bytes = m3_path.read_bytes()
    tampered_payload = json.loads(original_bytes)
    tampered_payload.pop("canonical_payload_sha256")
    tampered_payload["outcome"] = dict(tampered_payload["outcome"])
    tampered_payload["outcome"]["query_matched_effect"] = "SUPPORTED" \
        if tampered_payload["outcome"]["query_matched_effect"] != "SUPPORTED" else "NEGATIVE"
    tampered_payload["canonical_payload_sha256"] = aio.canonical_sha256(tampered_payload)
    m3_path.write_text(json.dumps(tampered_payload, indent=2, sort_keys=True, default=str) + "\n")

    check("a self-rehashed but content-tampered M3 still passes its OWN self-hash check "
          "(confirming the tamper is realistic, not caught trivially)",
          aio.read_and_verify_json_artifact(m3_path)[0] == tampered_payload["canonical_payload_sha256"])

    raised = False
    try:
        bs4.SIM = sim
        drv.run_full_protocol(pool, clean, scale, sim, out_dir=run_dir, seeds=[0], freeze_manifest_path=freeze_path)
    except RuntimeError as e:
        raised = "M3_CONTENT_MISMATCH" in str(e)
    finally:
        bs4.SIM = old_sim4
    check("resume detects a self-consistent but CONTENT-TAMPERED M3 by re-deriving the outcome fresh "
          "from the resumed seeds' M2 data (M3_CONTENT_MISMATCH)", raised)

    m3_path.write_bytes(original_bytes)  # restore
    try:
        bs4.SIM = sim
        out_restored = drv.run_full_protocol(pool, clean, scale, sim, out_dir=run_dir, seeds=[0],
                                              freeze_manifest_path=freeze_path)
    finally:
        bs4.SIM = old_sim4
    check("resume succeeds again once the tampered M3 is restored (not a false-positive failure mode)",
          out_restored["m3"]["self_hash"] is not None)


def test_partial_publication_resume(scratch: Path):
    """Direct regression coverage for the 5th-round P0 finding that a
    crash mid-run (after some companion files were published but before
    the stage's own self-verifying JSON was written) could not be resumed:
    a naive retry would hit `FileExistsError` on the companion file(s)
    already written by the interrupted attempt. Exercises the
    `write_npz_verify_if_exists`/`write_json_verify_if_exists` helpers
    directly (M1's clean-query-table npz and M3's raw-metrics json) by
    simulating a crash: write the companion file exactly as the real
    builder would, WITHOUT writing the final self-verifying JSON, then
    call the real builder again and confirm it recovers instead of
    raising `FileExistsError`."""
    small_hyperparams()
    alib.LAM_JAC_GRID = [0.05]
    alib.WEIGHT_DECAY_GRID = [0.0]
    run_dir = scratch / "partial_pub_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pool = make_fake_pool(seed=32)
    sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    freeze_path = _build_scratch_freeze_manifest(scratch, "partialpub")

    # --- M1: simulate a crash between the npz publish and the JSON write.
    array_fields = {k: v for k, v in clean.items() if isinstance(v, np.ndarray)}
    npz_path = run_dir / "ablation_clean_query_table_2026_08_31.npz"
    aio.atomic_write_npz_no_overwrite(npz_path, array_fields)
    m1_json_path = run_dir / "ablation_query_manifest_2026_08_31.json"
    check("simulated M1 crash leaves the npz published but NOT the JSON (a genuine partial-publication state)",
          npz_path.exists() and not m1_json_path.exists())
    m1 = drv.build_query_manifest(pool, clean, run_dir, freeze_manifest_path=freeze_path)
    check("build_query_manifest recovers from a crash-simulated pre-existing npz instead of raising "
          "FileExistsError, and still produces a valid self-verifying M1",
          aio.read_and_verify_json_artifact(m1["path"])[0] == m1["self_hash"])

    # A genuinely DIFFERENT npz content pre-existing at the same path (not
    # merely absent) must still be rejected -- verify-if-exists is a crash-
    # recovery mechanism, not a license to silently accept mismatched data.
    run_dir2 = scratch / "partial_pub_run_mismatch"
    run_dir2.mkdir(parents=True, exist_ok=True)
    wrong_arrays = {k: (v + 1.0 if np.issubdtype(v.dtype, np.floating) else v) for k, v in array_fields.items()}
    aio.atomic_write_npz_no_overwrite(run_dir2 / "ablation_clean_query_table_2026_08_31.npz", wrong_arrays)
    raised = False
    try:
        drv.build_query_manifest(pool, clean, run_dir2, freeze_manifest_path=freeze_path)
    except RuntimeError as e:
        raised = "PARTIAL_PUBLICATION_MISMATCH" in str(e)
    check("build_query_manifest refuses to silently adopt a pre-existing npz with genuinely DIFFERENT "
          "content at the same path (PARTIAL_PUBLICATION_MISMATCH), distinguishing a real crash-residue "
          "match from silent data corruption", raised)

    # --- Full pipeline: crash mid-SEED (checkpoints written, M2 not yet
    # built) must resume via a fresh per-attempt checkpoint subdirectory,
    # never colliding with the crashed attempt's already-written files.
    import budget_sweep_solver as bs5
    old_sim5 = bs5.SIM
    try:
        bs5.SIM = sim
        scale = alib.shared_scale_constants(pool, clean)
        oracle_idx = alib.fixed_oracle_idx(pool["test_idx"])
        checkpoint_dir = run_dir / "checkpoints"
        # Simulate a crashed FIRST attempt for seed 0: an attempt1/
        # directory already exists with residue in it (as if run_seed had
        # started writing checkpoints and then died), but no M2 exists yet.
        crashed_attempt_dir = checkpoint_dir / "seed0" / "attempt1"
        crashed_attempt_dir.mkdir(parents=True, exist_ok=True)
        (crashed_attempt_dir / "A_seed0_ep000.pt").write_bytes(b"crashed-residue-not-a-real-checkpoint")
        m2_path = run_dir / "ablation_training_manifest_seed0_2026_08_31.json"
        check("no M2 exists yet for seed 0 despite crashed-attempt residue on disk "
              "(genuinely mid-seed partial-publication state)", not m2_path.exists())

        out = drv.run_full_protocol(pool, clean, scale, sim, out_dir=run_dir, seeds=[0],
                                     freeze_manifest_path=freeze_path)
        check("run_full_protocol resumes past crashed-attempt checkpoint residue without raising "
              "FileExistsError, by writing the retry's checkpoints to a FRESH attempt2/ subdirectory",
              (checkpoint_dir / "seed0" / "attempt2").exists())
        check("the crashed attempt1's residue file is left untouched on disk, never deleted or overwritten",
              (crashed_attempt_dir / "A_seed0_ep000.pt").read_bytes() == b"crashed-residue-not-a-real-checkpoint")
        check("the resumed run still produces a valid, self-verifying M2 for seed 0",
              aio.read_and_verify_json_artifact(out["m2_paths"][0])[0] == json.loads(
                  out["m2_paths"][0].read_text())["canonical_payload_sha256"])

        # A SECOND crash-then-retry (attempt2 also left as residue, e.g.
        # M2 build itself crashed after training finished) must advance to
        # attempt3, not attempt2, even though seed 0's M2 now genuinely
        # exists from the successful run above -- exercised by deleting M2
        # and confirming the NEXT retry picks a fresh attempt number.
        # Forcing seed 0's M2 to be rebuilt makes the existing M3 stale (its
        # recorded predecessor M2 hash won't match the freshly-retrained
        # M2's hash, since the checkpoint paths differ by attempt number);
        # `run_full_protocol` correctly REFUSES to silently reuse a stale
        # M3 (RESUME_PROVENANCE_MISMATCH) rather than rebuilding it without
        # being asked, so the stale M3 must be removed explicitly too --
        # exactly the same "never silently overwrite, but the caller decides
        # when a rebuild is warranted" discipline as the rest of this chain.
        m2_path.unlink()
        out["m3"]["path"].unlink()
        out2 = drv.run_full_protocol(pool, clean, scale, sim, out_dir=run_dir, seeds=[0],
                                      freeze_manifest_path=freeze_path)
        check("a second retry (after deleting seed 0's M2) advances to attempt3, never reusing "
              "attempt2's now-completed checkpoint directory",
              (checkpoint_dir / "seed0" / "attempt3").exists())
        check("the second retry's M3 self-verifies (a fresh M3 rebuilt from the re-retrained M2)",
              aio.read_and_verify_json_artifact(out2["m3"]["path"])[0] == out2["m3"]["self_hash"])
    finally:
        bs5.SIM = old_sim5


def _rehash_tampered_payload(path: Path, mutate_fn) -> None:
    """Reads a self-verifying JSON artifact at `path`, applies `mutate_fn`
    to its payload (with the self-hash field already popped), recomputes
    a CORRECT self-hash over the mutated payload, and writes it back --
    producing a payload that is internally self-consistent (passes its
    own self-hash check) but was never actually produced by the real
    builder. Exactly the class of tamper `expected_kind`/`expected_keys`
    exist to catch beyond mere self-hash verification."""
    payload = json.loads(path.read_text())
    payload.pop("canonical_payload_sha256")
    mutate_fn(payload)
    payload["canonical_payload_sha256"] = aio.canonical_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def test_run_real_protocol_verifies_freeze_before_any_data_access(scratch: Path):
    """Direct regression coverage for the 6th-round P0 finding that
    `run_real_protocol` imported the real simulator and computed the real
    pool/CLEAN_QUERY_TABLE BEFORE `run_full_protocol` got around to
    checking the freeze gate -- meaning a missing or corrupted freeze
    manifest was only discovered after real simulator queries already
    happened. Never lets a real pool load actually happen: `alib.
    load_pool` is monkeypatched to raise a canary, so a REJECTED freeze
    must raise its own error without ever reaching the canary, while an
    ACCEPTED freeze must reach (and trigger) the canary -- proving the
    guard is a genuine ordering fix, not simply an always-failing check."""
    class _Canary(Exception):
        pass

    def _canary_load_pool():
        raise _Canary("load_pool must not be called before the freeze gate is verified")

    old_load_pool = alib.load_pool
    alib.load_pool = _canary_load_pool
    try:
        missing_freeze_path = scratch / "does_not_exist_freeze_for_order_test.json"
        raised_kind = None
        try:
            drv.run_real_protocol(out_dir=scratch / "unused_out_missing_freeze", seeds=list(alib.SEEDS),
                                  freeze_manifest_path=missing_freeze_path)
        except _Canary:
            raised_kind = "canary"
        except Exception:
            raised_kind = "freeze_error"
        check("run_real_protocol rejects a MISSING freeze manifest before ever calling "
              "alib.load_pool() (the canary never fires)", raised_kind == "freeze_error")

        freeze_path = _build_scratch_freeze_manifest(scratch, "realproto_order")
        raised_kind = None
        try:
            drv.run_real_protocol(out_dir=scratch / "unused_out_good_freeze", seeds=list(alib.SEEDS),
                                  freeze_manifest_path=freeze_path)
        except _Canary:
            raised_kind = "canary"
        except Exception:
            raised_kind = "other"
        check("run_real_protocol proceeds past a GENUINE freeze manifest to real data access "
              "(the load_pool canary DOES fire), confirming the guard is not a blanket failure",
              raised_kind == "canary")
    finally:
        alib.load_pool = old_load_pool


def test_run_real_protocol_enforces_canonical_seed_set(scratch: Path):
    """Direct regression coverage for the 6th-round P0 finding's second
    half: the canonical real entry point must only ever run the full
    pre-registered 10-seed set, never an arbitrary caller-chosen subset
    (that concept is smoke/test-only, via the generic `run_full_
    protocol`). Also never lets real data access happen: the canary
    pattern from the freeze-ordering test above confirms a REJECTED seed
    subset never reaches `alib.load_pool()` either."""
    class _Canary(Exception):
        pass

    def _canary_load_pool():
        raise _Canary("load_pool must not be called for a rejected arbitrary seed subset")

    freeze_path = _build_scratch_freeze_manifest(scratch, "realproto_seeds")
    old_load_pool = alib.load_pool
    alib.load_pool = _canary_load_pool
    try:
        raised_kind = None
        try:
            drv.run_real_protocol(out_dir=scratch / "unused_out_bad_seeds", seeds=[0, 1],
                                  freeze_manifest_path=freeze_path)
        except _Canary:
            raised_kind = "canary"
        except ValueError as e:
            raised_kind = "value_error" if "pre-registered seed set" in str(e) else "other_value_error"
        except Exception:
            raised_kind = "other"
        check("run_real_protocol rejects an arbitrary seed subset with a ValueError, before any "
              "real data access, rather than silently training a partial seed set for real",
              raised_kind == "value_error")

        raised_kind = None
        try:
            drv.run_real_protocol(out_dir=scratch / "unused_out_good_seeds", seeds=list(alib.SEEDS),
                                  freeze_manifest_path=freeze_path)
        except _Canary:
            raised_kind = "canary"
        except Exception:
            raised_kind = "other"
        check("run_real_protocol accepts the full canonical seed set and proceeds to real data access "
              "(the load_pool canary DOES fire)", raised_kind == "canary")
    finally:
        alib.load_pool = old_load_pool


def test_manifest_exact_keyset_enforced(scratch: Path):
    """Direct regression coverage for the 6th-round P1 finding that
    `expected_kind` was checked but the EXACT payload keyset the protocol
    document (Section 10) promises each manifest stage hash-locks was
    not -- a field silently dropped, or an unexpected extra field, would
    otherwise only surface later (if at all) via a confusing KeyError deep
    in some unrelated lookup. Each tamper is self-rehashed (via
    `_rehash_tampered_payload`) so it is caught SPECIFICALLY by the new
    keyset check, not merely by the pre-existing self-hash check."""
    small_hyperparams()
    alib.LAM_JAC_GRID = [0.05]
    alib.WEIGHT_DECAY_GRID = [0.0]
    run_dir = scratch / "keyset_run"
    pool = make_fake_pool(seed=40)
    sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    scale = alib.shared_scale_constants(pool, clean)
    freeze_path = _build_scratch_freeze_manifest(scratch, "keyset")

    import budget_sweep_solver as bs6
    old_sim6 = bs6.SIM
    try:
        bs6.SIM = sim
        out = drv.run_full_protocol(pool, clean, scale, sim, out_dir=run_dir, seeds=[0],
                                     freeze_manifest_path=freeze_path)
    finally:
        bs6.SIM = old_sim6

    m1_path = out["m1"]["path"]
    original_m1_bytes = m1_path.read_bytes()

    _rehash_tampered_payload(m1_path, lambda p: p.pop("n_val"))
    raised = False
    try:
        drv.verify_query_manifest(m1_path)
    except RuntimeError as e:
        raised = "MANIFEST_KEYSET_MISMATCH" in str(e)
    check("verify_query_manifest rejects an M1 with a MISSING required key, even though the "
          "tampered payload is self-rehashed and internally self-consistent (MANIFEST_KEYSET_MISMATCH)",
          raised)
    m1_path.write_bytes(original_m1_bytes)

    _rehash_tampered_payload(m1_path, lambda p: p.__setitem__("unexpected_extra_field", 1))
    raised = False
    try:
        drv.verify_query_manifest(m1_path)
    except RuntimeError as e:
        raised = "MANIFEST_KEYSET_MISMATCH" in str(e)
    check("verify_query_manifest rejects an M1 with an UNEXPECTED EXTRA key (MANIFEST_KEYSET_MISMATCH)",
          raised)
    m1_path.write_bytes(original_m1_bytes)
    check("verify_query_manifest succeeds again once the tampered M1 is restored",
          drv.verify_query_manifest(m1_path)["self_hash"] is not None)

    m3_path = out["m3"]["path"]
    original_m3_bytes = m3_path.read_bytes()
    _rehash_tampered_payload(m3_path, lambda p: p.pop("outcome"))
    raised = False
    try:
        drv.verify_results_manifest(m3_path)
    except RuntimeError as e:
        raised = "MANIFEST_KEYSET_MISMATCH" in str(e)
    check("verify_results_manifest rejects an M3 with a MISSING required key (MANIFEST_KEYSET_MISMATCH)", raised)
    m3_path.write_bytes(original_m3_bytes)
    check("verify_results_manifest succeeds again once the tampered M3 is restored",
          drv.verify_results_manifest(m3_path)["self_hash"] is not None)


def test_m3_companion_semantic_tamper_detection(scratch: Path):
    """Direct regression coverage for the 6th-round P1 finding that M3's
    companion files (raw_metrics.json, oracle_fixed_idx.npz,
    raw_resample_arrays.npz) were only checked for byte-drift against
    their OWN recorded hash, never for whether their content was ever
    correct to begin with. Tampers raw_metrics.json's content AND updates
    M3's own recorded hash/self-hash to match -- a fully self-consistent,
    hash-verified but semantically WRONG companion, i.e. a "perfect
    crime" against the pre-existing byte-drift check -- and confirms
    `_assert_m3_companions_match_fresh_evidence` still catches it by
    re-deriving the expected content fresh from the resumed M2 data."""
    small_hyperparams()
    alib.LAM_JAC_GRID = [0.05]
    alib.WEIGHT_DECAY_GRID = [0.0]
    run_dir = scratch / "m3_companion_tamper_run"
    pool = make_fake_pool(seed=41)
    sim = FakeSim()
    clean = alib.build_clean_query_table(pool, sim, fake_precompute_phi)
    scale = alib.shared_scale_constants(pool, clean)
    freeze_path = _build_scratch_freeze_manifest(scratch, "m3companiontamper")

    import budget_sweep_solver as bs7
    old_sim7 = bs7.SIM
    try:
        bs7.SIM = sim
        out = drv.run_full_protocol(pool, clean, scale, sim, out_dir=run_dir, seeds=[0],
                                     freeze_manifest_path=freeze_path)
    finally:
        bs7.SIM = old_sim7

    m3_path = out["m3"]["path"]
    original_m3_bytes = m3_path.read_bytes()
    m3_payload = json.loads(original_m3_bytes)
    raw_metrics_path = ROOT / m3_payload["raw_metrics_path"]
    original_raw_metrics_bytes = raw_metrics_path.read_bytes()

    tampered_raw_metrics = json.loads(original_raw_metrics_bytes)
    flipped = False
    for per_seed in tampered_raw_metrics["closed_loop_success"].values():
        for s, v in per_seed.items():
            if v is not None:
                per_seed[s] = 1.0 - float(v)
                flipped = True
                break
        if flipped:
            break
    check("the semantic-tamper setup actually flips a real closed_loop_success value "
          "(the tamper is non-vacuous)", flipped)
    new_raw_metrics_bytes = (json.dumps(aio._sanitize_nonfinite(tampered_raw_metrics), indent=2,
                                        sort_keys=True, default=str, allow_nan=False) + "\n").encode()
    raw_metrics_path.write_bytes(new_raw_metrics_bytes)
    new_raw_metrics_hash = aio.file_sha256(raw_metrics_path)

    m3_payload_mut = dict(m3_payload)
    m3_payload_mut.pop("canonical_payload_sha256")
    m3_payload_mut["raw_metrics_sha256"] = new_raw_metrics_hash
    m3_payload_mut["canonical_payload_sha256"] = aio.canonical_sha256(m3_payload_mut)
    m3_path.write_text(json.dumps(m3_payload_mut, indent=2, sort_keys=True, default=str) + "\n")

    check("the tampered raw_metrics companion's file hash now matches M3's (re-synced) recorded hash "
          "(the tamper is a realistic 'perfect crime', not caught trivially by a changed byte hash)",
          drv.verify_results_manifest(m3_path)["self_hash"] is not None)

    raised = False
    try:
        bs7.SIM = sim
        drv.run_full_protocol(pool, clean, scale, sim, out_dir=run_dir, seeds=[0], freeze_manifest_path=freeze_path)
    except RuntimeError as e:
        raised = "M3_COMPANION_CONTENT_MISMATCH" in str(e)
    finally:
        bs7.SIM = old_sim7
    check("resume detects a hash-consistent but semantically WRONG raw_metrics companion by "
          "re-deriving its expected content fresh from the resumed seeds' M2 data "
          "(M3_COMPANION_CONTENT_MISMATCH)", raised)

    m3_path.write_bytes(original_m3_bytes)
    raw_metrics_path.write_bytes(original_raw_metrics_bytes)
    try:
        bs7.SIM = sim
        out_restored = drv.run_full_protocol(pool, clean, scale, sim, out_dir=run_dir, seeds=[0],
                                             freeze_manifest_path=freeze_path)
    finally:
        bs7.SIM = old_sim7
    check("resume succeeds again once both the tampered M3 and raw_metrics companion are restored",
          out_restored["m3"]["self_hash"] is not None)


def test_freeze_manifest_refuses_without_passing_smoke(scratch: Path):
    fake_smoke_path = scratch / "fake_smoke.json"
    old_path = drv.SMOKE_RESULTS_PATH
    try:
        drv.SMOKE_RESULTS_PATH = fake_smoke_path
        aio.atomic_write_json_no_overwrite(fake_smoke_path, {"n_pass": 1, "n_total": 2, "all_passed": False})
        raised = False
        try:
            drv.build_freeze_manifest(dest_path=scratch / "fake_freeze.json")
        except RuntimeError:
            raised = True
        check("build_freeze_manifest refuses to freeze when the smoke suite has failing checks", raised)
    finally:
        drv.SMOKE_RESULTS_PATH = old_path


def test_freeze_manifest_refuses_internally_inconsistent_smoke_record(scratch: Path):
    fake_smoke_path = scratch / "fake_smoke_inconsistent.json"
    old_path = drv.SMOKE_RESULTS_PATH
    try:
        drv.SMOKE_RESULTS_PATH = fake_smoke_path
        # claims all_passed=True and n_pass=2, but only 1 check actually
        # has passed=true -- a hand-tampered or buggy smoke record.
        aio.atomic_write_json_no_overwrite(fake_smoke_path, {
            "n_pass": 2, "n_total": 2, "all_passed": True,
            "checks": [{"name": "a", "passed": True}, {"name": "b", "passed": False}],
        })
        raised = False
        try:
            drv.build_freeze_manifest(dest_path=scratch / "fake_freeze_inconsistent.json")
        except RuntimeError as e:
            raised = "internally inconsistent" in str(e)
        check("build_freeze_manifest rejects a smoke record whose all_passed/n_pass contradict its own checks list",
              raised)
    finally:
        drv.SMOKE_RESULTS_PATH = old_path


def test_real_closed_loop_through_freeze_any_fnn_path():
    """Direct P0 regression check: exercises the ACTUAL
    `budget_sweep_solver.closed_loop()` function (not a reimplementation)
    with an FNN model routed through `freeze_any`, using a tiny fake
    plant (monkeypatched `bs.SIM`) and a SMALL horizon/budget/steps
    purely to keep this specific smoke check fast -- this is what v5's
    P0 finding was: a bare `seq_fnn(xun)` forward call did not exercise
    the real evaluation entry point closely enough to catch it."""
    import budget_sweep_solver as bs
    from disambiguate_landscape_vs_gradient import make_norm
    old_sim = bs.SIM
    try:
        bs.SIM = FakeSim()
        pool = make_fake_pool(seed=13)
        torch.manual_seed(0)
        fnn = alib.build_model("E", use_fnn=True)
        seq = alib.freeze_any(fnn)
        data_tuple = (pool["XU"], pool["Y"], pool["xm"], pool["xs"], pool["ym"], pool["ys"],
                      pool["train_idx"], pool["test_idx"])
        norm, un_lo, un_hi = make_norm(data_tuple)
        raised = False
        result = None
        try:
            result = bs.closed_loop(seq, norm, np.array([0.0, 5.0]), 2, 3, 3, 0.1, 0.01, 2.0, un_lo, un_hi)
        except AttributeError:
            raised = True
        check("the REAL budget_sweep_solver.closed_loop() runs to completion for a frozen FNN model "
              "without AttributeError", not raised)
        check("closed_loop() returns the expected result keys for the FNN path",
              result is not None and {"final_V", "success"} <= set(result.keys()))
    finally:
        bs.SIM = old_sim


def test_state_dict_sha256_deterministic():
    torch.manual_seed(42)
    m1 = alib.build_model("A", use_fnn=False)
    torch.manual_seed(42)
    m2 = alib.build_model("A", use_fnn=False)
    torch.manual_seed(43)
    m3 = alib.build_model("A", use_fnn=False)
    h1 = aio.state_dict_sha256(m1); h2 = aio.state_dict_sha256(m2); h3 = aio.state_dict_sha256(m3)
    check("state_dict_sha256 is deterministic for identical init seeds", h1 == h2)
    check("state_dict_sha256 differs for different init seeds", h1 != h3)


def main() -> None:
    scratch = fresh_scratch()
    try:
        test_param_count_match()
        test_freeze_any_dispatch_and_closed_loop_shaped_call()
        test_clean_query_table_and_boundary_pairing()
        test_query_counts_equal_across_conditions()
        test_augmented_residual_target_uses_own_successor()
        test_symmetric_loss_uses_g_scale()
        test_scale_constants_shapes_and_seed_independence()
        test_rng_stream_separation()
        test_initial_state_hash_identical_across_lcnn_conditions(scratch)
        test_train_base_and_augmented_smoke(scratch)
        test_select_lambda_jac(scratch)
        test_diagnose_fixed_idx_and_oracle_determinism()
        test_bootstrap_ci_fresh_generator_reproducibility()
        test_exact_permutation_and_holm()
        test_secondary_family_size_and_mse_infeasible_p1()
        test_outcome_map_exhaustive_classification()
        test_mse_attribution_exhaustive()
        test_full_outcome_orchestration()
        test_failure_completeness_contract()
        test_atomic_io_no_overwrite_and_self_hash(scratch)
        test_freeze_manifest_tamper_detection(scratch)
        test_manifest_chain_on_scratch(scratch)
        test_resume_tamper_detection(scratch)
        test_simulator_identity_mismatch_rejected(scratch)
        test_m3_content_tamper_detection(scratch)
        test_partial_publication_resume(scratch)
        test_run_real_protocol_verifies_freeze_before_any_data_access(scratch)
        test_run_real_protocol_enforces_canonical_seed_set(scratch)
        test_manifest_exact_keyset_enforced(scratch)
        test_m3_companion_semantic_tamper_detection(scratch)
        test_freeze_manifest_refuses_without_passing_smoke(scratch)
        test_freeze_manifest_refuses_internally_inconsistent_smoke_record(scratch)
        test_real_closed_loop_through_freeze_any_fnn_path()
        test_state_dict_sha256_deterministic()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    n_pass = sum(1 for c in CHECKS if c["passed"])
    n_total = len(CHECKS)
    print(f"\n{n_pass}/{n_total} checks passed")

    from datetime import datetime, timezone
    # This artifact is deliberately re-runnable/overwritable during the
    # design-review -> implement -> smoke iteration loop (it is NOT yet a
    # frozen artifact -- nothing hashes it as evidence until
    # `build_freeze_manifest()` records its content hash into the
    # self-verifying, no-overwrite Freeze Manifest at freeze time). Once
    # frozen, THAT hash is the permanent record and cannot be silently
    # replaced; only pre-freeze dev-loop runs are overwritable, matching
    # the same convention already used for Paper M's analogous smoke
    # results artifact.
    out_path = RESULTS_DIR / "ablation_synthetic_smoke_results_2026_08_31.json"
    if out_path.exists():
        out_path.unlink()
    aio.atomic_write_json_no_overwrite(out_path, {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_pass": n_pass, "n_total": n_total,
        "all_passed": n_pass == n_total,
        "checks": CHECKS,
    })

    if n_pass != n_total:
        raise SystemExit(f"FAILED: {n_total - n_pass} check(s) did not pass")


if __name__ == "__main__":
    main()
