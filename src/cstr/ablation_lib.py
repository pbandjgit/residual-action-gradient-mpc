"""Core library for the JPC go/no-go equal-query ablation protocol
(`docs/JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_2026_08_31.md`, v6).

Implements, in order: Section 2 (query pool construction), Section 3
(condition loss formulas), Section 4 (FNN variant + freeze_any adapter),
Section 5 (training loop, RNG stream separation, initial-state hash),
Section 6 (fixed-index oracle + relative gradient error), Section 7
(primary/secondary contrasts, statistics), Section 8 (MSE-matched
comparison), Section 9 (outcome-map decision logic).

No plant/simulator/CasADi import happens at module scope beyond what
`probe_action_gradient`/`budget_sweep_solver` already need -- the
synthetic smoke test exercises this module with fabricated arrays and a
fake `sim`/`model`, never the real CSTR.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cstr.models_onestep import build_onestep, cauchy_nll          # noqa: E402

# ---------------------------------------------------------------------------
# Fixed literals (protocol Sections 2, 5, 6, 7 -- pre-registered, not tuned)
# ---------------------------------------------------------------------------
POOL_PATH = ROOT / "data" / "processed" / "lcnn_paper_cstr_onestep_20k.npz"
FD_EPS = 1e-3
SEEDS = list(range(10))
LAM_VAL = 0.003
LAM_G = 0.05
LAM_JAC_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
EPOCHS = 50
WARMUP = 30
BATCH = 256
LR = 2e-3
CHECKPOINT_EVERY = 10
WEIGHT_DECAY_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
MSE_MATCH_TOL = 0.10
ORACLE_META_SEED = 90210
ORACLE_N_PTS = 800
BOOTSTRAP_SEED = 31415
N_BOOTSTRAP = 10_000
N_PERMUTATION = 1024  # 2^10
MIN_EFFECT = 0.2
EQUIV_MARGIN = 0.1
LCNN_CFG = {"model": "lcnn", "hidden": 40, "lipschitz_bound": 4.0, "n_layers": 2}
FNN_ACTIVATION = "softsign"


def base_noise_seed(k: int) -> int:
    return k + 1000


def sensitivity_noise_seed(k: int) -> int:
    return k + 4000


def shuffle_seed(k: int) -> int:
    return k + 5000


# ---------------------------------------------------------------------------
# Section 2: query pool
# ---------------------------------------------------------------------------

def load_pool(pool_path: Path = POOL_PATH) -> dict:
    d = np.load(pool_path)
    return dict(
        XU=d["XU"].astype(np.float32), Y=d["Y"].astype(np.float32),
        xm=d["x_mean"].astype(np.float32), xs=d["x_std"].astype(np.float32),
        ym=d["y_mean"].astype(np.float32), ys=d["y_std"].astype(np.float32),
        train_idx=d["train_idx"], val_idx=d["val_idx"], test_idx=d["test_idx"],
    )


def build_clean_query_table(pool: dict, sim, precompute_phi_fn) -> dict:
    """Section 2.2 (normalized-coordinate perturbation, physical-box
    boundary check), 2.4 (pair-wise drop), 2.7 (auxiliary-law queries),
    2.8 (Jacobian/gradient targets over the valid-pair mask). `sim` must
    expose `.step(x, u)`; `precompute_phi_fn(sim, x_batch)` must return
    `(Phi, Yphi)` matching `probe_action_gradient.precompute_phi`'s
    contract. Pure function of the pool + sim, seed-independent.

    Fixed: `Phi_all`/`Yphi_all` are computed ONLY at `train_idx` (TRAIN-
    LOCAL sized, `(n_train, 2)`), matching Section 2.7's own written scope
    ("for every training point") exactly -- the query_counts_per_condition
    accounting depends on this, and computing it over the full pool (as an
    earlier draft did) silently under-reported the auxiliary-law query
    count relative to what the protocol's own text pre-registers. Every
    downstream index into `Phi_all`/`Yphi_all` is therefore a LOCAL train
    index (0..n_train-1), never a global pool index."""
    XU, Y = pool["XU"], pool["Y"]
    xm, xs, ym, ys = pool["xm"], pool["xs"], pool["ym"], pool["ys"]
    train_idx = np.asarray(pool["train_idx"])
    x_all = XU[:, :2].astype(np.float64)
    u_all = XU[:, 2:].astype(np.float64)
    u_mean, u_std = xm[2:].astype(np.float64), xs[2:].astype(np.float64)
    from budget_sweep_solver import INPUT_LO, INPUT_HI  # noqa: PLC0415 (avoid hard dep at import time for smoke)

    Phi_all, Yphi_all = precompute_phi_fn(sim, x_all[train_idx])

    n_tr = len(train_idx)
    valid_mask = np.zeros((n_tr, 2), dtype=bool)
    y_plus = np.zeros((n_tr, 2, 2), dtype=np.float64)   # [point, dim, output], physical
    y_minus = np.zeros((n_tr, 2, 2), dtype=np.float64)
    un_plus = np.zeros((n_tr, 2, 2), dtype=np.float64)  # [point, dim, input_dim], normalized action
    un_minus = np.zeros((n_tr, 2, 2), dtype=np.float64)

    un_tr = (u_all[train_idx] - u_mean) / u_std
    x_tr = x_all[train_idx]
    for j in range(2):
        up = un_tr.copy(); up[:, j] += FD_EPS
        dn = un_tr.copy(); dn[:, j] -= FD_EPS
        uu_p = up * u_std + u_mean
        uu_m = dn * u_std + u_mean
        in_box_p = np.all((uu_p >= INPUT_LO) & (uu_p <= INPUT_HI), axis=1)
        in_box_m = np.all((uu_m >= INPUT_LO) & (uu_m <= INPUT_HI), axis=1)
        pair_valid = in_box_p & in_box_m
        valid_mask[:, j] = pair_valid
        idx_valid = np.where(pair_valid)[0]
        for i in idx_valid:
            y_plus[i, j] = sim.step(x_tr[i], uu_p[i])
            y_minus[i, j] = sim.step(x_tr[i], uu_m[i])
        un_plus[idx_valid, j] = up[idx_valid]
        un_minus[idx_valid, j] = dn[idx_valid]

    y_plus_n = (y_plus - ym) / ys
    y_minus_n = (y_minus - ym) / ys
    true_df_du = np.zeros((n_tr, 2, 2), dtype=np.float64)  # [point, output, input_dim]
    for j in range(2):
        true_df_du[:, :, j] = np.where(
            valid_mask[:, j:j + 1], (y_plus_n[:, j, :] - y_minus_n[:, j, :]) / (2 * FD_EPS), 0.0
        )

    def Vnp(y):
        P = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float64)
        return np.einsum("ni,ij,nj->n", y, P, y)

    true_ug = np.zeros((n_tr, 2), dtype=np.float64)
    for j in range(2):
        Vp = Vnp(y_plus[:, j, :])
        Vm = Vnp(y_minus[:, j, :])
        true_ug[:, j] = np.where(valid_mask[:, j], (Vp - Vm) / (2 * FD_EPS), 0.0)

    return dict(
        Phi_all=Phi_all, Yphi_all=Yphi_all,
        valid_mask=valid_mask,
        un_plus=un_plus.astype(np.float32), un_minus=un_minus.astype(np.float32),
        y_plus_n=y_plus_n.astype(np.float32), y_minus_n=y_minus_n.astype(np.float32),
        true_df_du=true_df_du.astype(np.float32), true_ug=true_ug.astype(np.float32),
    )


def valid_pair_counts(clean: dict) -> dict:
    vm = clean["valid_mask"]
    return dict(n_points=int(vm.shape[0]), n_pairs_total=int(vm.shape[0] * 2),
                n_pairs_valid=int(vm.sum()), n_pairs_dropped=int((~vm).sum()))


def query_counts_per_condition(pool: dict, clean: dict) -> dict:
    n_tr = len(pool["train_idx"])
    n_pairs_valid = int(clean["valid_mask"].sum())
    base = n_tr + n_tr  # one-step base query + auxiliary-law query (Section 2.7), per training point
    extra = 2 * n_pairs_valid  # each valid pair = 2 perturbation queries (+eps, -eps)
    return {c: base + extra for c in ("B", "C", "D", "E", "F")} | {"A": base}


def build_noise_realized_table(pool: dict, clean: dict, seed: int, sensitivity: bool) -> dict:
    """Section 2.5. `sensitivity=False` -> primary (CLEAN augmented B/C
    targets, normalized-space `y_plus_n`/`y_minus_n` passed through
    unchanged). `sensitivity=True` -> the non-primary Cauchy-noised
    augmented-target sensitivity arm, using the SAME noise scale/clip
    convention as the base pool's own noise (`noise_scale=0.2`,
    `noise_clip=5.0`, both calibrated in NORMALIZED-output space, matching
    `lgrad_experiment.py:69-70` exactly)."""
    Y = pool["Y"]; ym = pool["ym"]; ys = pool["ys"]
    train_idx = np.asarray(pool["train_idx"])
    noise_scale, noise_clip = 0.2, 5.0

    Yn = ((Y - ym) / ys).copy()
    rng_base = np.random.default_rng(base_noise_seed(seed))
    n = noise_scale * rng_base.standard_cauchy(size=Yn[train_idx].shape)
    Yn[train_idx] += np.clip(n, -noise_clip, noise_clip).astype(np.float32)
    disp = np.median(np.abs(Yn[train_idx] - np.median(Yn[train_idx], 0)), 0) * 1.4826

    y_plus_aug_n = clean["y_plus_n"].copy()
    y_minus_aug_n = clean["y_minus_n"].copy()
    if sensitivity:
        rng_sens = np.random.default_rng(sensitivity_noise_seed(seed))
        vm = clean["valid_mask"]
        for arr in (y_plus_aug_n, y_minus_aug_n):
            noise = noise_scale * rng_sens.standard_cauchy(size=arr.shape)
            noise = np.clip(noise, -noise_clip, noise_clip).astype(np.float32)
            arr += noise * vm[:, :, None]

    return dict(Yn=Yn.astype(np.float32), disp=disp.astype(np.float32),
                y_plus_aug_n=y_plus_aug_n, y_minus_aug_n=y_minus_aug_n)


def shared_scale_constants(pool: dict, clean: dict) -> dict:
    """Section 2.6: xm/xs/ym/ys/g_scale/ug_scale are seed-independent
    (g_scale/ug_scale computed once from the CLEAN pool arrays, at the
    fixed train_idx). `disp` is per-seed and lives in the noise-realized
    table instead."""
    Y = pool["Y"]; ym = pool["ym"]; ys = pool["ys"]
    train_idx = np.asarray(pool["train_idx"])
    Yphi_all = clean["Yphi_all"]  # train-local sized (n_train, 2)

    def Vnp(y):
        P = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float64)
        return np.einsum("ni,ij,nj->n", y, P, y)

    # ddof=1 (sample std, Bessel's correction) to match the reference
    # implementation's torch `.std()` convention, which defaults to
    # unbiased=True -- numpy's default ddof=0 (population std) would be a
    # real formula mismatch, not just a precision difference. Computed in
    # float64 for numerical stability, then cast to float32 for the final
    # stored scale constant, matching Section 2.6's stated float32 dtype
    # (the protocol doc's own convention for these scalars).
    g_true_train = Vnp(Y[train_idx].astype(np.float64)) - Vnp(Yphi_all.astype(np.float64))
    g_scale = np.float32(max(g_true_train.std(ddof=1), 1.0)).item()
    # ug_scale must exclude invalid (dropped-pair) entries, which are
    # stored as placeholder 0.0 in true_ug -- including them would bias
    # the scale downward (toward the zero placeholder), not reflect the
    # genuine gradient-label distribution. This valid-pair-only exclusion
    # is a real design decision this protocol introduces (the reference
    # `lgrad_experiment.py` convention this formula otherwise mirrors has
    # no boundary-drop/invalid-pair concept at all) -- pre-registered in
    # the protocol document's Section 2.6 addendum, not an undocumented
    # implementation choice.
    true_ug = clean["true_ug"].astype(np.float64)
    valid_ug = true_ug[clean["valid_mask"]]
    ug_scale = np.float32(max(valid_ug.std(ddof=1), 1.0) if len(valid_ug) > 1 else 1.0).item()
    return dict(xm=pool["xm"], xs=pool["xs"], ym=ym, ys=ys, g_scale=g_scale, ug_scale=ug_scale)


# ---------------------------------------------------------------------------
# Section 7: statistics (bootstrap CI, exact permutation test, Holm)
# ---------------------------------------------------------------------------

def bootstrap_ci(diffs: np.ndarray, seed: int = None, n_resample: int = None,
                  alpha: float = 0.05) -> dict:
    """Section 7.3/7.5: a FRESH `np.random.default_rng(seed)` instantiated
    for THIS call only (never a shared, sequentially-consumed generator
    across contrasts) -- 95% two-sided percentile-method CI. Returns the
    generated resample-index array too, for the Results-Manifest hash
    (Section 10, M3). `seed`/`n_resample` default to the CURRENT module
    globals `BOOTSTRAP_SEED`/`N_BOOTSTRAP`, resolved at CALL time (not
    bound at function-definition time) so that test/dev-time overrides of
    the module globals actually take effect."""
    seed = seed if seed is not None else BOOTSTRAP_SEED
    n_resample = n_resample if n_resample is not None else N_BOOTSTRAP
    diffs = np.asarray(diffs, dtype=np.float64)
    n = len(diffs)
    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, n, size=(n_resample, n))
    resampled_means = diffs[resample_idx].mean(axis=1)
    lower = float(np.percentile(resampled_means, 100 * alpha / 2))
    upper = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return dict(point_estimate=float(diffs.mean()), ci_lower=lower, ci_upper=upper,
                confidence=1 - alpha, n_resample=n_resample, resample_idx=resample_idx)


def exact_permutation_pvalue(diffs: np.ndarray) -> dict:
    """Section 7.3: exact two-sided paired sign-flip test, statistic =
    `|mean paired difference|`, exhaustive over all `2^n` sign patterns."""
    diffs = np.asarray(diffs, dtype=np.float64)
    n = len(diffs)
    observed = abs(diffs.mean())
    n_patterns = 2 ** n
    count_ge = 0
    for bits in itertools.product([1, -1], repeat=n):
        signed_mean = abs((diffs * np.array(bits)).mean())
        if signed_mean >= observed - 1e-12:
            count_ge += 1
    return dict(p_value=count_ge / n_patterns, n_patterns=n_patterns, statistic=float(observed))


def holm_adjust(pvalues: list) -> list:
    """Standard Holm step-down. Returns adjusted p-values in the ORIGINAL
    input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [None] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        candidate = (m - rank) * pvalues[i]
        running_max = max(running_max, candidate)
        adjusted[i] = min(1.0, running_max)
    return adjusted


def paired_success_rate_diff(success_a: dict, success_b: dict, seeds: list) -> np.ndarray:
    """`success_a`/`success_b`: {seed: success_rate}. Returns the per-seed
    paired difference array (a - b), in `seeds` order."""
    return np.array([success_a[s] - success_b[s] for s in seeds], dtype=np.float64)


def evaluate_contrast(diffs: np.ndarray, bootstrap_seed: int = None) -> dict:
    ci = bootstrap_ci(diffs, seed=bootstrap_seed)
    perm = exact_permutation_pvalue(diffs)
    return dict(point_estimate=ci["point_estimate"], ci_lower=ci["ci_lower"], ci_upper=ci["ci_upper"],
                p_value=perm["p_value"], resample_idx=ci["resample_idx"])


SECONDARY_CONTRAST_NAMES = ["E_vs_D", "C_vs_B", "F_vs_E", "F_vs_C", "interaction", "EFNN_vs_BFNN", "MSE_match"]


def compute_secondary_family(success_rates: dict, seeds: list, mse_match_diffs: np.ndarray = None) -> dict:
    """Section 7.4/7.5: `success_rates` = {condition: {seed: rate}} for
    A-F plus B-FNN/E-FNN. `mse_match_diffs` = per-seed E-minus-matched-B
    success diffs (may have `< 10` entries if some seeds hit
    `MSE_MATCH_INFEASIBLE_FOR_SEED`), or `None` if the whole comparison is
    `MSE_MATCH_INFEASIBLE` (-> fixed conservative p=1, family size stays
    7 always). Contrasts 1-6 always use the full 10-seed permutation;
    only `MSE_match` may use a reduced `2^n_valid`."""
    contrasts = {}
    contrasts["E_vs_D"] = evaluate_contrast(paired_success_rate_diff(success_rates["E"], success_rates["D"], seeds))
    contrasts["C_vs_B"] = evaluate_contrast(paired_success_rate_diff(success_rates["C"], success_rates["B"], seeds))
    contrasts["F_vs_E"] = evaluate_contrast(paired_success_rate_diff(success_rates["F"], success_rates["E"], seeds))
    contrasts["F_vs_C"] = evaluate_contrast(paired_success_rate_diff(success_rates["F"], success_rates["C"], seeds))
    fe = paired_success_rate_diff(success_rates["F"], success_rates["E"], seeds)
    cb = paired_success_rate_diff(success_rates["C"], success_rates["B"], seeds)
    contrasts["interaction"] = evaluate_contrast(fe - cb)
    contrasts["EFNN_vs_BFNN"] = evaluate_contrast(
        paired_success_rate_diff(success_rates["E-FNN"], success_rates["B-FNN"], seeds))
    if mse_match_diffs is None:
        contrasts["MSE_match"] = dict(point_estimate=float("nan"), ci_lower=float("nan"), ci_upper=float("nan"),
                                       p_value=1.0, resample_idx=None, infeasible=True)
    else:
        contrasts["MSE_match"] = evaluate_contrast(mse_match_diffs)
        contrasts["MSE_match"]["infeasible"] = False

    names = SECONDARY_CONTRAST_NAMES
    pvals = [contrasts[n]["p_value"] for n in names]
    adjusted = holm_adjust(pvals)
    for name, adj in zip(names, adjusted):
        contrasts[name]["p_value_holm_adjusted"] = adj
    return contrasts


# ---------------------------------------------------------------------------
# Section 7.7: failure and completeness contract
# ---------------------------------------------------------------------------

class TrainingDivergedError(RuntimeError):
    pass


def check_finite_or_diverged(loss_value: float, condition: str, seed: int) -> None:
    if not np.isfinite(loss_value):
        raise TrainingDivergedError(f"TRAINING_DIVERGED: condition={condition} seed={seed} loss={loss_value}")


def check_model_finite_or_diverged(model: nn.Module, condition: str, seed: int) -> None:
    """New: the loss-value check alone cannot catch divergence introduced
    by the optimizer step itself (e.g. a NaN gradient producing a NaN
    weight update on the LAST batch, with no subsequent loss evaluation
    to trip `check_finite_or_diverged`) -- called after `opt.step()`."""
    for p in model.parameters():
        if not torch.isfinite(p).all():
            raise TrainingDivergedError(
                f"TRAINING_DIVERGED: condition={condition} seed={seed} non-finite parameter after optimizer step")


SIMULATOR_FAILURE_EXCEPTIONS = (RuntimeError, FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError)


def safe_closed_loop_success(closed_loop_fn, *args, **kwargs) -> bool:
    """A simulator/numerical exception or non-finite rollout is recorded
    as `success=False`, NEVER excluded from the success-rate denominator.
    Deliberately narrowed to `SIMULATOR_FAILURE_EXCEPTIONS` (numerical
    failure modes a real ODE/solver rollout can raise -- overflow,
    singular linear algebra, NaN-triggered `RuntimeError`) rather than a
    bare `except Exception`, which would also swallow programming errors
    (`AttributeError`, `NameError`, `TypeError`, `KeyError`) and silently
    misreport a code bug as a simulator failure."""
    try:
        result = closed_loop_fn(*args, **kwargs)
        if not np.isfinite(result.get("final_V", np.nan)):
            return False
        return bool(result["success"])
    except SIMULATOR_FAILURE_EXCEPTIONS:
        return False


def require_seed_completeness(per_condition_seeds: dict, seeds: list = SEEDS) -> None:
    """Section 7.7(c): every primary-battery/FNN-block condition must have
    ALL of `seeds` present before Section 9's outcome computation may run."""
    missing = {}
    for condition, present in per_condition_seeds.items():
        gap = set(seeds) - set(present)
        if gap:
            missing[condition] = sorted(gap)
    if missing:
        raise RuntimeError(f"INCOMPLETE_PRIMARY_BATTERY: missing seeds {missing}")


# ---------------------------------------------------------------------------
# Section 4: FNN variant + freeze_any adapter
# ---------------------------------------------------------------------------

def build_model(condition: str, use_fnn: bool = False) -> nn.Module:
    if use_fnn:
        return build_onestep({"model": "fnn_activation", "hidden": 40, "activation": FNN_ACTIVATION})
    return build_onestep(LCNN_CFG)


def freeze_any(model: nn.Module):
    """Section 4: dispatches on model shape. `bs.freeze()` (the function
    the actual primary evaluation path calls) requires `.body`/`.out`;
    the FNN variant has neither and needs no Bjorck materialization, so
    it is returned eval-mode, grad-disabled, unchanged."""
    if hasattr(model, "body") and hasattr(model, "out"):
        import budget_sweep_solver as bs  # noqa: PLC0415
        return bs.freeze(model)
    model = model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def assert_exact_param_count_match() -> tuple:
    lcnn = build_onestep(LCNN_CFG)
    fnn = build_onestep({"model": "fnn_activation", "hidden": 40, "activation": FNN_ACTIVATION})
    n_lcnn = sum(p.numel() for p in lcnn.parameters())
    n_fnn = sum(p.numel() for p in fnn.parameters())
    if n_lcnn != n_fnn:
        raise AssertionError(f"FNN/LCNN parameter count mismatch: {n_fnn} != {n_lcnn}")
    return n_lcnn, n_fnn


# ---------------------------------------------------------------------------
# Section 3 conditions table
# ---------------------------------------------------------------------------
CONDITIONS = {
    "A": dict(loss_family="one_sided", grad_term=None, augmented=False),
    "B": dict(loss_family="one_sided", grad_term=None, augmented=True),
    "C": dict(loss_family="symmetric", grad_term=None, augmented=True),
    "D": dict(loss_family="one_sided", grad_term="jac", augmented=False),
    "E": dict(loss_family="one_sided", grad_term="lgrad", augmented=False),
    "F": dict(loss_family="symmetric", grad_term="lgrad", augmented=False),
}


def condition_config(condition: str) -> dict:
    """FNN-block conditions (`B-FNN`, `E-FNN`) reuse B/E's loss-family/
    augmented config on a different architecture (Section 4: `B-FNN`
    reuses exactly condition B's data role, `E-FNN` condition E's)."""
    base = condition.split("-FNN")[0]
    return CONDITIONS[base]


def residual_value_loss(loss_family: str, g_true: torch.Tensor, g_hat: torch.Tensor, g_scale: float) -> torch.Tensor:
    e = (g_true - g_hat) / g_scale
    if loss_family == "one_sided":
        return torch.relu(e).pow(2).mean()
    if loss_family == "symmetric":
        return e.pow(2).mean()
    raise ValueError(f"unknown loss_family: {loss_family!r}")


# ---------------------------------------------------------------------------
# Section 5: training loop (base-row conditions A/D/E/F; augmented B/C)
# ---------------------------------------------------------------------------

def _Vt(y, Pt):
    return (y @ Pt * y).sum(-1)


def initial_state_hash(model: nn.Module) -> str:
    from cstr.ablation_io import state_dict_sha256  # noqa: PLC0415
    return state_dict_sha256(model)


def train_base_condition(condition: str, seed: int, pool: dict, clean: dict, scale: dict,
                          use_fnn: bool = False, lam_jac: float = None,
                          checkpoint_dir: Path = None, checkpoint_prefix: str = None) -> dict:
    """Conditions A, D, E, F: base `train_idx` rows only, no augmented
    rows, an optional gradient/Jacobian term added for `ep > WARMUP`."""
    cfg = condition_config(condition)
    assert not cfg["augmented"]
    P = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float32)
    Pt = torch.tensor(P)

    torch.manual_seed(seed)
    model = build_model(condition, use_fnn=use_fnn)
    init_hash = initial_state_hash(model)

    noise = build_noise_realized_table(pool, clean, seed, sensitivity=False)
    XU, ym, ys = pool["XU"], pool["ym"], pool["ys"]
    xm, xs = pool["xm"], pool["xs"]
    train_idx = np.asarray(pool["train_idx"])
    Xn = torch.tensor((XU - xm) / xs)
    Yn_t = torch.tensor(noise["Yn"])
    disp_t = torch.tensor(noise["disp"])
    Phi_all, Yphi_all = clean["Phi_all"], clean["Yphi_all"]  # train-local sized (n_train, 2)
    Phi_t = torch.tensor(Phi_all)
    g_true_all = _Vt(torch.tensor(pool["Y"][train_idx]), Pt) - _Vt(torch.tensor(Yphi_all), Pt)  # train-local
    g_scale = scale["g_scale"]
    ug_scale = scale["ug_scale"]
    xm_t, xs_t = torch.tensor(xm), torch.tensor(xs)
    ym_t, ys_t = torch.tensor(ym), torch.tensor(ys)

    valid_mask_t = torch.tensor(clean["valid_mask"])
    true_ug_t = torch.tensor(clean["true_ug"])
    true_df_du_t = torch.tensor(clean["true_df_du"])

    from cstr.models_onestep import cauchy_nll as _cauchy_nll  # noqa: PLC0415

    tr = np.asarray(train_idx)
    rng_shuffle = np.random.default_rng(shuffle_seed(seed))
    opt = torch.optim.Adam(model.parameters(), LR)
    checkpoints = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng_shuffle.permutation(len(tr))
        for k in range(0, len(tr), BATCH):
            b_local = order[k:k + BATCH]
            b = tr[b_local]
            opt.zero_grad()
            xb = Xn[b]
            pred = model(xb)
            loss = _cauchy_nll(pred, Yn_t[b], disp_t)
            if ep > WARMUP:
                y_hat = pred * ys_t + ym_t
                xphi = torch.cat([torch.tensor(XU[b, :2]), Phi_t[b_local]], 1)
                yphi_hat = model((xphi - xm_t) / xs_t) * ys_t + ym_t
                g_hat = _Vt(y_hat, Pt) - _Vt(yphi_hat, Pt)
                loss = loss + LAM_VAL * residual_value_loss(
                    cfg["loss_family"], g_true_all[b_local], g_hat, g_scale)
                if cfg["grad_term"] == "lgrad":
                    un_b = xb[:, 2:].detach().requires_grad_(True)
                    xun = torch.cat([xb[:, :2], un_b], 1)
                    y = model(xun) * ys_t + ym_t
                    Vsum = _Vt(y, Pt).sum()
                    (g_un,) = torch.autograd.grad(Vsum, un_b, create_graph=True)
                    vm_b = valid_mask_t[b_local]
                    err = ((g_un - true_ug_t[b_local]) / ug_scale) ** 2
                    err = err * vm_b
                    denom = vm_b.sum().clamp(min=1)
                    loss = loss + LAM_G * err.sum() / denom
                elif cfg["grad_term"] == "jac":
                    un_b = xb[:, 2:].detach().requires_grad_(True)
                    xun = torch.cat([xb[:, :2], un_b], 1)
                    y = model(xun) * ys_t + ym_t
                    yn = (y - ym_t) / ys_t
                    jac_hat = torch.zeros(len(b), 2, 2)
                    for out_k in range(2):
                        (g_k,) = torch.autograd.grad(yn[:, out_k].sum(), un_b, create_graph=True)
                        jac_hat[:, out_k, :] = g_k
                    vm_b = valid_mask_t[b_local]  # (batch,2) per input dim
                    err = (jac_hat - true_df_du_t[b_local]) ** 2  # (batch,2out,2in)
                    err = err * vm_b[:, None, :]
                    denom = vm_b.sum().clamp(min=1)
                    loss = loss + lam_jac * err.sum() / denom
            check_finite_or_diverged(loss.item(), condition, seed)
            loss.backward(); opt.step()
            check_model_finite_or_diverged(model, condition, seed)
            if hasattr(model, "apply_constraints"):
                model.apply_constraints()
        if checkpoint_dir is not None and (ep % CHECKPOINT_EVERY == 0 or ep == EPOCHS):
            from cstr.ablation_io import atomic_write_torch_no_overwrite  # noqa: PLC0415
            prefix = checkpoint_prefix if checkpoint_prefix is not None else condition
            ckpt_path = checkpoint_dir / f"{prefix}_seed{seed}_ep{ep:03d}.pt"
            atomic_write_torch_no_overwrite(ckpt_path, model.state_dict())
            checkpoints.append(dict(epoch=ep, path=str(ckpt_path)))
    model.eval()
    val_idx = np.asarray(pool["val_idx"]); test_idx = np.asarray(pool["test_idx"])
    with torch.no_grad():
        pv_val = (model(Xn[val_idx]) * ys_t + ym_t).numpy()
        pv_test = (model(Xn[test_idx]) * ys_t + ym_t).numpy()
    val_mse = float(np.mean(((pv_val - pool["Y"][val_idx]) / ys) ** 2))
    test_mse = float(np.mean(((pv_test - pool["Y"][test_idx]) / ys) ** 2))
    check_finite_or_diverged(val_mse, condition, seed)
    check_finite_or_diverged(test_mse, condition, seed)
    return dict(model=model, val_mse=val_mse, test_mse=test_mse,
                init_state_hash=init_hash, checkpoints=checkpoints)


def select_lambda_jac(seed: int, pool: dict, clean: dict, scale: dict,
                       grid: list = None, checkpoint_dir: Path = None) -> dict:
    """Section 5.3: choose the `\\lambda_Jac` grid value minimizing
    validation MSE only; deterministic tie-break by smallest `\\lambda`.
    Never uses the Section 6 gradient-fidelity oracle. Returns the full
    grid trace (Section 5.3's transparency requirement) plus every
    candidate's trained result -- corrected: EVERY grid candidate is
    checkpointed (a distinct `checkpoint_prefix` per `lam`), not only the
    selected one, since Section 5.3 requires reporting each candidate's
    resulting E-D closed-loop outcome, which is impossible if the other
    candidates' models were discarded."""
    grid = grid if grid is not None else LAM_JAC_GRID
    trace = []
    results = {}
    for lam in grid:
        r = train_base_condition("D", seed, pool, clean, scale, lam_jac=lam,
                                  checkpoint_dir=checkpoint_dir, checkpoint_prefix=f"D_lamjac{lam}")
        trace.append(dict(lam_jac=lam, val_mse=r["val_mse"], test_mse=r["test_mse"],
                          checkpoints=r["checkpoints"]))
        results[lam] = r
    min_mse = min(t["val_mse"] for t in trace)
    candidates = sorted(lam for lam, r in results.items() if r["val_mse"] == min_mse)
    selected_lam = candidates[0]
    selected = results[selected_lam]
    return dict(selected_lambda_jac=selected_lam, grid_trace=trace, result=selected, all_results=results)


def fixed_oracle_idx(test_idx: np.ndarray, n_pts: int = None, meta_seed: int = None) -> np.ndarray:
    """Section 6: ONE fixed 800-index subset of `test_idx`, drawn with the
    literal meta-seed, independent of the 10 training seeds -- applied
    identically to every condition/seed's oracle evaluation. Defaults
    resolved at call time from `ORACLE_N_PTS`/`ORACLE_META_SEED`."""
    n_pts = n_pts if n_pts is not None else ORACLE_N_PTS
    meta_seed = meta_seed if meta_seed is not None else ORACLE_META_SEED
    rng = np.random.default_rng(meta_seed)
    return rng.choice(np.asarray(test_idx), size=min(n_pts, len(test_idx)), replace=False)


def diagnose_fixed_idx(model: nn.Module, pool: dict, sim, idx: np.ndarray, delta: float = 0.05) -> dict:
    """Section 6: `probe_action_gradient.diagnose()`'s logic, parameterized
    by an EXPLICIT index array instead of internal `seed`-based resampling
    (that function re-samples its 800-point subset per `seed`, conflating
    training-seed variance with test-point-sampling variance -- see the
    protocol's v4 correction). Adds the relative gradient error norm field
    that does not exist in the original `diagnose()`. Never used inside
    any training loop or selection step -- final-evaluation-only, shared
    identically across every condition and seed via `idx`."""
    XU, Y = pool["XU"], pool["Y"]
    xm, xs, ym, ys = pool["xm"], pool["xs"], pool["ym"], pool["ys"]
    x = XU[idx, :2].astype(np.float64); u = XU[idx, 2:].astype(np.float64)
    ym_t, ys_t = torch.tensor(ym), torch.tensor(ys)
    u_mean, u_std = xm[2:].astype(np.float64), xs[2:].astype(np.float64)
    x_mean, x_std = xm[:2].astype(np.float64), xs[:2].astype(np.float64)

    def learned_u_grad(un_np):
        un = torch.tensor(un_np, dtype=torch.float32, requires_grad=True)
        xn = torch.tensor((x - x_mean) / x_std, dtype=torch.float32)
        xun = torch.cat([xn, un], 1)
        y = model(xun) * ys_t + ym_t
        P = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float32)
        Pt = torch.tensor(P)
        g = _Vt(y, Pt).sum()
        (grad,) = torch.autograd.grad(g, un)
        return grad.detach().numpy()

    un = (u - u_mean) / u_std
    g_hat_grad = learned_u_grad(un)

    def Vnp(y):
        P = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float64)
        return np.einsum("ni,ij,nj->n", y, P, y)

    eps = 1e-3
    true_grad = np.zeros_like(un)
    for j in range(2):
        up = un.copy(); up[:, j] += eps
        dn = un.copy(); dn[:, j] -= eps
        uu_p = up * u_std + u_mean; uu_m = dn * u_std + u_mean
        Vp = Vnp(np.array([sim.step(x[i], uu_p[i]) for i in range(len(x))]))
        Vm = Vnp(np.array([sim.step(x[i], uu_m[i]) for i in range(len(x))]))
        true_grad[:, j] = (Vp - Vm) / (2 * eps)

    true_norm = np.linalg.norm(true_grad, axis=1)
    hat_norm = np.linalg.norm(g_hat_grad, axis=1)
    excluded = (true_norm <= 1e-9) | (hat_norm <= 1e-9)
    ok = ~excluded

    def cos(a, b, mask):
        c = np.full(len(a), np.nan)
        na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
        c[mask] = (a[mask] * b[mask]).sum(1) / (na[mask] * nb[mask])
        return c

    align = cos(true_grad, g_hat_grad, ok)
    rel_err = np.full(len(true_grad), np.nan)
    rel_err[ok] = np.linalg.norm(g_hat_grad[ok] - true_grad[ok], axis=1) / true_norm[ok]

    def q(a):
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return dict(median=float("nan"), p90=float("nan"), mean=float("nan"))
        return dict(median=float(np.median(a)), p90=float(np.percentile(a, 90)), mean=float(np.mean(a)))

    return {
        "align_cos": {**q(align), "frac_neg": float(np.mean(align[ok] < 0)) if ok.any() else float("nan")},
        "rel_grad_error": q(rel_err),
        "excluded_fraction": float(excluded.mean()),
        "n_points": int(len(idx)),
    }


def train_augmented_condition(condition: str, seed: int, pool: dict, clean: dict, scale: dict,
                               use_fnn: bool = False, sensitivity: bool = False,
                               checkpoint_dir: Path = None, weight_decay: float = 0.0,
                               checkpoint_prefix: str = None) -> dict:
    """Conditions B, C: base rows + valid-pair augmented rows, both
    subject to the Cauchy fit loss AND the residual-value loss (Section
    2.8), no gradient term."""
    cfg = condition_config(condition)
    assert cfg["augmented"]
    P = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float32)
    Pt = torch.tensor(P)

    torch.manual_seed(seed)
    model = build_model(condition, use_fnn=use_fnn)
    init_hash = initial_state_hash(model)

    noise = build_noise_realized_table(pool, clean, seed, sensitivity=sensitivity)
    XU, ym, ys = pool["XU"], pool["ym"], pool["ys"]
    xm, xs = pool["xm"], pool["xs"]
    train_idx = np.asarray(pool["train_idx"])
    n_tr = len(train_idx)
    xm_t, xs_t = torch.tensor(xm), torch.tensor(xs)
    ym_t, ys_t = torch.tensor(ym), torch.tensor(ys)

    x_tr_n = torch.tensor((XU[train_idx, :2] - xm[:2]) / xs[:2])
    un_tr = torch.tensor((XU[train_idx, 2:] - xm[2:]) / xs[2:])
    base_xun_n = torch.cat([x_tr_n, un_tr], 1)
    base_y_n = torch.tensor(noise["Yn"][train_idx])

    valid_mask = clean["valid_mask"]
    aug_rows = [(i, j, sign) for i in range(n_tr) for j in range(2) for sign in (0, 1) if valid_mask[i, j]]
    if aug_rows:
        aug_i = np.array([r[0] for r in aug_rows]); aug_j = np.array([r[1] for r in aug_rows])
        aug_sign = np.array([r[2] for r in aug_rows])
        un_plus, un_minus = clean["un_plus"], clean["un_minus"]
        y_plus_n, y_minus_n = noise["y_plus_aug_n"], noise["y_minus_aug_n"]
        aug_un = np.where(aug_sign[:, None] == 0, un_plus[aug_i, aug_j], un_minus[aug_i, aug_j])
        aug_y = np.where(aug_sign[:, None] == 0, y_plus_n[aug_i, aug_j], y_minus_n[aug_i, aug_j])
        aug_x_n = x_tr_n.numpy()[aug_i]
        aug_xun_n = torch.tensor(np.concatenate([aug_x_n, aug_un], 1).astype(np.float32))
        aug_y_t = torch.tensor(aug_y.astype(np.float32))
        aug_phi_idx = aug_i  # same local-train-index as the base point it was perturbed from
    else:
        aug_xun_n = torch.zeros(0, 4); aug_y_t = torch.zeros(0, 2); aug_phi_idx = np.zeros(0, dtype=int)

    all_xun_n = torch.cat([base_xun_n, aug_xun_n], 0)
    all_y_n = torch.cat([base_y_n, aug_y_t], 0)
    all_phi_local_idx = np.concatenate([np.arange(n_tr), aug_phi_idx])
    Phi_all, Yphi_all = clean["Phi_all"], clean["Yphi_all"]  # train-local sized (n_train, 2)
    Phi_t = torch.tensor(Phi_all[all_phi_local_idx])
    Yphi_t = torch.tensor(Yphi_all[all_phi_local_idx])

    # Fixed (P0): the residual-value target for EVERY row must come from
    # THAT row's own CLEAN successor state, never the base point's
    # original successor. Base rows use the pool's clean Y at train_idx;
    # augmented rows use the CLEAN (never noise-injected, regardless of
    # the sensitivity arm) perturbation successor from clean_query_table,
    # matching how E/F's gradient target is always clean too.
    base_y_clean_n = (torch.tensor(pool["Y"][train_idx].astype(np.float32)) - ym_t) / ys_t
    if aug_rows:
        y_plus_clean_n, y_minus_clean_n = clean["y_plus_n"], clean["y_minus_n"]
        aug_y_clean = np.where(aug_sign[:, None] == 0, y_plus_clean_n[aug_i, aug_j], y_minus_clean_n[aug_i, aug_j])
        aug_y_clean_t = torch.tensor(aug_y_clean.astype(np.float32))
    else:
        aug_y_clean_t = torch.zeros(0, 2)
    all_y_clean_n = torch.cat([base_y_clean_n, aug_y_clean_t], 0)
    all_y_clean_phys = all_y_clean_n * ys_t + ym_t
    g_true_all = _Vt(all_y_clean_phys, Pt) - _Vt(Yphi_t, Pt)

    x_all_n = all_xun_n[:, :2]
    disp_t = torch.tensor(noise["disp"])
    g_scale = scale["g_scale"]

    n_total = all_xun_n.shape[0]
    rng_shuffle = np.random.default_rng(shuffle_seed(seed))
    opt = torch.optim.Adam(model.parameters(), LR, weight_decay=weight_decay)
    checkpoints = []
    prefix = checkpoint_prefix if checkpoint_prefix is not None else condition
    from cstr.models_onestep import cauchy_nll as _cauchy_nll
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng_shuffle.permutation(n_total)
        for k in range(0, n_total, BATCH):
            b = order[k:k + BATCH]
            opt.zero_grad()
            xb = all_xun_n[b]
            pred = model(xb)
            loss = _cauchy_nll(pred, all_y_n[b], disp_t)
            if ep > WARMUP:
                y_hat = pred * ys_t + ym_t
                xphi = torch.cat([x_all_n[b] * xs_t[:2] + xm_t[:2], Phi_t[b]], 1)
                yphi_hat = model((xphi - xm_t) / xs_t) * ys_t + ym_t
                g_hat = _Vt(y_hat, Pt) - _Vt(yphi_hat, Pt)
                loss = loss + LAM_VAL * residual_value_loss(cfg["loss_family"], g_true_all[b], g_hat, g_scale)
            check_finite_or_diverged(loss.item(), condition, seed)
            loss.backward(); opt.step()
            check_model_finite_or_diverged(model, condition, seed)
            if hasattr(model, "apply_constraints"):
                model.apply_constraints()
        if checkpoint_dir is not None and (ep % CHECKPOINT_EVERY == 0 or ep == EPOCHS):
            from cstr.ablation_io import atomic_write_torch_no_overwrite  # noqa: PLC0415
            ckpt_path = checkpoint_dir / f"{prefix}_seed{seed}_ep{ep:03d}.pt"
            atomic_write_torch_no_overwrite(ckpt_path, model.state_dict())
            checkpoints.append(dict(epoch=ep, path=str(ckpt_path), weight_decay=weight_decay))
    model.eval()
    val_idx = np.asarray(pool["val_idx"]); test_idx = np.asarray(pool["test_idx"])
    XU_all = pool["XU"]
    Xn_all = torch.tensor((XU_all - xm) / xs)
    with torch.no_grad():
        pv_val = (model(Xn_all[val_idx]) * ys_t + ym_t).numpy()
        pv_test = (model(Xn_all[test_idx]) * ys_t + ym_t).numpy()
    val_mse = float(np.mean(((pv_val - pool["Y"][val_idx]) / ys) ** 2))
    test_mse = float(np.mean(((pv_test - pool["Y"][test_idx]) / ys) ** 2))
    check_finite_or_diverged(val_mse, condition, seed)
    check_finite_or_diverged(test_mse, condition, seed)
    return dict(model=model, val_mse=val_mse, test_mse=test_mse,
                init_state_hash=init_hash, checkpoints=checkpoints)


# ---------------------------------------------------------------------------
# Section 8: MSE-matched comparison
# ---------------------------------------------------------------------------

def _eval_checkpoint_mse(condition: str, state_dict: dict, pool: dict, use_fnn: bool = False) -> tuple:
    model = build_model(condition, use_fnn=use_fnn)
    model.load_state_dict(state_dict)  # strict=True (PyTorch default): raises on any key/shape/dtype mismatch
    model.eval()
    XU, ym, ys, xm, xs = pool["XU"], pool["ym"], pool["ys"], pool["xm"], pool["xs"]
    Xn = torch.tensor((XU - xm) / xs)
    ym_t, ys_t = torch.tensor(ym), torch.tensor(ys)
    val_idx = np.asarray(pool["val_idx"]); test_idx = np.asarray(pool["test_idx"])
    with torch.no_grad():
        pv_val = (model(Xn[val_idx]) * ys_t + ym_t).numpy()
        pv_test = (model(Xn[test_idx]) * ys_t + ym_t).numpy()
    val_mse = float(np.mean(((pv_val - pool["Y"][val_idx]) / ys) ** 2))
    test_mse = float(np.mean(((pv_test - pool["Y"][test_idx]) / ys) ** 2))
    check_finite_or_diverged(val_mse, condition, -1)
    check_finite_or_diverged(test_mse, condition, -1)
    return val_mse, test_mse


def _relative_diff(a: float, b: float, floor: float = 1e-12) -> float:
    """`|a-b| / max(|b|, floor)` -- guards against division by (near-)zero
    when `b` (the reference MSE) is extremely small."""
    return abs(a - b) / max(abs(b), floor)


def mse_match_select(seed: int, pool: dict, clean: dict, scale: dict, e_val_mse: float,
                      b_primary_result: dict, checkpoint_dir: Path = None,
                      tol: float = MSE_MATCH_TOL, weight_decay_grid: list = None) -> dict:
    """Section 8.3/8.4: candidate grid = condition B's PRIMARY-run
    checkpoint cadence, falling back to a weight-decay retrain grid if no
    candidate is within `tol` (relative) of `e_val_mse`. Deterministic
    tie-break: (a) smallest relative MSE difference, (b) smallest weight
    decay, (c) earliest epoch. Also records the selected checkpoint's
    TEST MSE (never used for selection)."""
    weight_decay_grid = weight_decay_grid if weight_decay_grid is not None else WEIGHT_DECAY_GRID
    candidates = []
    for ckpt in b_primary_result["checkpoints"]:
        state = torch.load(ckpt["path"], map_location="cpu")
        val_mse, test_mse = _eval_checkpoint_mse("B", state, pool)
        candidates.append(dict(val_mse=val_mse, test_mse=test_mse,
                                rel_diff=_relative_diff(val_mse, e_val_mse),
                                weight_decay=0.0, epoch=ckpt["epoch"], path=ckpt["path"]))
    in_band = [c for c in candidates if c["rel_diff"] <= tol]
    attempt_log = list(candidates)
    if not in_band:
        for wd in weight_decay_grid:
            r = train_augmented_condition("B", seed, pool, clean, scale, weight_decay=wd,
                                           checkpoint_dir=checkpoint_dir,
                                           checkpoint_prefix=f"B_wd{wd}")
            rel_diff = _relative_diff(r["val_mse"], e_val_mse)
            entry = dict(val_mse=r["val_mse"], test_mse=r["test_mse"], rel_diff=rel_diff,
                         weight_decay=wd, epoch=EPOCHS,
                         path=r["checkpoints"][-1]["path"] if r["checkpoints"] else None)
            attempt_log.append(entry)
            if rel_diff <= tol:
                in_band.append(entry)
    if not in_band:
        return dict(outcome="MSE_MATCH_INFEASIBLE_FOR_SEED", attempt_log=attempt_log, selected=None)
    in_band_sorted = sorted(in_band, key=lambda c: (c["rel_diff"], c["weight_decay"], c["epoch"]))
    selected = in_band_sorted[0]
    return dict(outcome="MATCHED", attempt_log=attempt_log, selected=selected)


def mse_attribution_seed_status(seed_results: dict) -> str:
    """Section 9's `MSE_ATTRIBUTION` first two branches, per-seed input:
    aggregate `mse_match_select` outcomes across the 10 seeds to decide
    overall feasibility (>3/10 `MSE_MATCH_INFEASIBLE_FOR_SEED` ->
    `MSE_MATCH_INFEASIBLE`)."""
    infeasible = [s for s, r in seed_results.items() if r["outcome"] == "MSE_MATCH_INFEASIBLE_FOR_SEED"]
    if len(infeasible) > 3:
        return "MSE_MATCH_INFEASIBLE"
    return "FEASIBLE"


# ---------------------------------------------------------------------------
# Section 9: outcome-map decision logic
# ---------------------------------------------------------------------------

def classify_query_matched_effect(primary_contrast: dict) -> str:
    """Step 1. Evaluated in FIXED order (NEGATIVE, then SUPPORTED, then
    UNCLEAR as the exhaustive remainder) so every (CI, point estimate)
    pair lands in exactly one category."""
    ci_lower = primary_contrast["ci_lower"]; ci_upper = primary_contrast["ci_upper"]
    point = primary_contrast["point_estimate"]
    if ci_upper < 0 or point <= -0.1:
        return "NEGATIVE"
    if ci_lower > 0 and point >= MIN_EFFECT:
        return "SUPPORTED"
    return "UNCLEAR"


def _holm_gated_direction(contrast: dict) -> bool:
    return contrast["p_value_holm_adjusted"] <= 0.05 and contrast["point_estimate"] > 0


def classify_target_specificity(e_vs_d: dict) -> str:
    return "SUPPORTED" if _holm_gated_direction(e_vs_d) else "NOT_SPECIFIC"


def classify_non_groupsort_replication(efnn_vs_bfnn: dict) -> str:
    return "SUPPORTED" if _holm_gated_direction(efnn_vs_bfnn) else "GROUPSORT_SPECIFIC"


def classify_loss_interaction(interaction: dict) -> str:
    if interaction["p_value_holm_adjusted"] <= 0.05:
        return "INTERACTING"
    if interaction["ci_lower"] >= -EQUIV_MARGIN and interaction["ci_upper"] <= EQUIV_MARGIN:
        return "EQUIVALENT"
    return "INCONCLUSIVE"


def classify_mse_attribution(feasibility_status: str, mse_match_contrast: dict,
                              test_mse_generalization_fail_count: int) -> dict:
    """Fixed order: (1) UNRESOLVED if infeasible, (2) UNRESOLVED with
    sub-label `VALIDATION_MATCH_DID_NOT_GENERALIZE` if the matched
    checkpoints' test MSE drifted outside +/-10% of E's for more than
    3/10 seeds, (3) SUPPORTED if Holm-significant AND point estimate
    `>= 0.2`, (4) NOT_SUPPORTED as the exhaustive remainder."""
    if feasibility_status == "MSE_MATCH_INFEASIBLE":
        return dict(status="UNRESOLVED", sub_label="MSE_MATCH_INFEASIBLE")
    if test_mse_generalization_fail_count > 3:
        return dict(status="UNRESOLVED", sub_label="VALIDATION_MATCH_DID_NOT_GENERALIZE")
    if mse_match_contrast["p_value_holm_adjusted"] <= 0.05 and mse_match_contrast["point_estimate"] >= MIN_EFFECT:
        return dict(status="SUPPORTED", sub_label=None)
    return dict(status="NOT_SUPPORTED", sub_label=None)


def evaluate_full_outcome(primary_contrast: dict, secondary: dict, mse_attribution: dict) -> dict:
    """Top-level Section 9 orchestration: Step 1 + Step 2 (only if Step 1
    is SUPPORTED, else computed-but-exploratory) + the manuscript action."""
    step1 = classify_query_matched_effect(primary_contrast)
    target_specificity = classify_target_specificity(secondary["E_vs_D"])
    non_groupsort = classify_non_groupsort_replication(secondary["EFNN_vs_BFNN"])
    loss_interaction = classify_loss_interaction(secondary["interaction"])
    mse_attr = mse_attribution["status"]

    if step1 != "SUPPORTED":
        exploratory = True
        if step1 == "UNCLEAR":
            action = "NARROW_INCONCLUSIVE"
        else:
            action = "STOP_AND_REFRAME"
    else:
        exploratory = False
        full_claim = (target_specificity == "SUPPORTED" and non_groupsort == "SUPPORTED"
                      and mse_attr == "SUPPORTED")
        action = "FULL_CLAIM_STANDS" if full_claim else "NARROW_TO_SUBSET"

    return dict(
        query_matched_effect=step1, target_specificity=target_specificity,
        non_groupsort_replication=non_groupsort, loss_interaction=loss_interaction,
        mse_attribution=mse_attr, mse_attribution_sub_label=mse_attribution["sub_label"],
        step2_exploratory_only=exploratory, manuscript_action=action,
    )
