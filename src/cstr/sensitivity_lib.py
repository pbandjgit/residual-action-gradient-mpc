"""N-family / N-scale / FD / L-grad sensitivity library
(`docs/JPC_SENSITIVITY_PROTOCOL_2026_09_03.md`, v5). Reuses
`ablation_lib.train_base_condition`/`train_augmented_condition` verbatim
via temporary module-level monkeypatching of `build_noise_realized_table`/
`LAM_G` (the same override style this project's own smoke tests already
use for `EPOCHS`), never duplicating the training loop itself.

Nothing in this module has real side effects at import time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import cstr.ablation_lib as alib

# ---------------------------------------------------------------------------
# Constants (protocol Sections 2, v3->v4 item 3, v4->v5 item 3)
# ---------------------------------------------------------------------------

NOISE_SCALE_BASELINE = 0.2
NOISE_CLIP = 5.0
# 0.2 / Phi^-1(0.75), Phi^-1(0.75)=0.6744897501960817 (scipy.stats.norm.ppf(0.75)):
# MAD-matches a standard-normal draw to Cauchy(scale=0.2)'s MAD of exactly 0.2.
GAUSSIAN_SIGMA_MAD_MATCHED = 0.2965204437011204
RHO_CORRELATED = 0.7

NOISE_SCALE_LEVELS = [0.1, 0.2, 0.4]
FD_EPS_LEVELS = [1e-4, 1e-3, 1e-2, 0.02]
LAM_G_LEVELS = [0.025, 0.05, 0.1]

N_FAMILY_LEVELS = ("noiseless", "gaussian", "cauchy", "correlated")


def noise_family_seed(k: int) -> int:
    """Shared by `gaussian` and `correlated` (v4->v5 item 1's
    common-random-number fix) -- `correlated` must NEVER use a different
    seed than `gaussian`, only a Cholesky transform of the same draw."""
    return k + 7000


# ---------------------------------------------------------------------------
# N-family / N-scale: parameterized base label-noise construction
# ---------------------------------------------------------------------------

def build_noise_realized_table_parameterized(pool: dict, clean: dict, seed: int,
                                              family: str, scale: float = NOISE_SCALE_BASELINE) -> dict:
    """Drop-in replacement for `alib.build_noise_realized_table(pool, clean,
    seed, sensitivity=False)`, generalized over `family`/`scale`. The B/C
    augmented-target sensitivity arm (`y_plus_aug_n`/`y_minus_aug_n`) is
    untouched by this protocol -- always passed through clean, exactly the
    `sensitivity=False` behavior.

    Returns `Yn`, `disp`, `y_plus_aug_n`, `y_minus_aug_n` (same keys as
    the original, for drop-in compatibility) PLUS `n_raw`/`n_applied`
    (the actual injected-noise arrays, for diagnostics/companion storage,
    v4->v5 items 3-4)."""
    Y = pool["Y"]; ym = pool["ym"]; ys = pool["ys"]
    train_idx = np.asarray(pool["train_idx"])
    Yn = ((Y - ym) / ys).copy()
    shape = Yn[train_idx].shape  # (n_train, 2)

    if family == "noiseless":
        n_raw = np.zeros(shape, dtype=np.float64)
        n_applied = np.zeros(shape, dtype=np.float32)
    elif family == "cauchy":
        rng = np.random.default_rng(alib.base_noise_seed(seed))
        n_raw = scale * rng.standard_cauchy(size=shape)
        n_applied = np.clip(n_raw, -NOISE_CLIP, NOISE_CLIP).astype(np.float32)
    elif family == "gaussian":
        rng = np.random.default_rng(noise_family_seed(seed))
        z = rng.standard_normal(size=shape)
        n_raw = GAUSSIAN_SIGMA_MAD_MATCHED * z
        n_applied = n_raw.astype(np.float32)  # light-tailed: no clip, per protocol Section 2
    elif family == "correlated":
        # SAME seed/draw as "gaussian" (common random numbers) -- differs
        # ONLY by the Cholesky transform, never by a fresh draw.
        rng = np.random.default_rng(noise_family_seed(seed))
        z = rng.standard_normal(size=shape)
        L = np.linalg.cholesky(np.array([[1.0, RHO_CORRELATED], [RHO_CORRELATED, 1.0]]))
        # np.errstate suppresses a spurious (cosmetic-only, confirmed via
        # direct output-finiteness check) divide-by-zero/overflow warning
        # that some Accelerate-BLAS matmul code paths emit on this shape.
        with np.errstate(all="ignore"):
            n_raw = GAUSSIAN_SIGMA_MAD_MATCHED * (z @ L.T)
        n_applied = n_raw.astype(np.float32)
    else:
        raise ValueError(f"unknown noise family: {family!r}")

    Yn[train_idx] += n_applied
    disp = np.median(np.abs(Yn[train_idx] - np.median(Yn[train_idx], 0)), 0) * 1.4826

    return dict(
        Yn=Yn.astype(np.float32), disp=disp.astype(np.float32),
        y_plus_aug_n=clean["y_plus_n"].copy(), y_minus_aug_n=clean["y_minus_n"].copy(),
        n_raw=n_raw.astype(np.float64), n_applied=n_applied.astype(np.float64),
    )


def noise_diagnostics(n_raw: np.ndarray, n_applied: np.ndarray) -> dict:
    """v4->v5 items 3, 9: MAD/RMS/correlation on `n_applied` (what the
    model actually trains against), clipping fraction on `n_raw`."""
    mad = (np.median(np.abs(n_applied - np.median(n_applied, axis=0)), axis=0) * 1.0).tolist()
    rms = np.sqrt(np.mean(n_applied ** 2, axis=0)).tolist()
    clipping_fraction = float(np.mean(np.abs(n_raw) > NOISE_CLIP))
    if np.all(n_applied == 0):
        correlation = None
    else:
        c = np.corrcoef(n_applied[:, 0], n_applied[:, 1])
        correlation = float(c[0, 1]) if np.isfinite(c[0, 1]) else None
    return dict(mad_per_dim=mad, rms_per_dim=rms, clipping_fraction=clipping_fraction,
                correlation=correlation)


def train_variant_condition(condition: str, seed: int, pool: dict, clean: dict, scale: dict,
                             family: str, noise_scale: float,
                             checkpoint_dir: Path = None, checkpoint_prefix: str = None) -> dict:
    """B or E, with the base label noise replaced by `build_noise_
    realized_table_parameterized`. Monkeypatches `alib.build_noise_
    realized_table` for the duration of this call only (both `train_base_
    condition` and `train_augmented_condition` resolve that name from
    `ablation_lib`'s own module globals at CALL time, so this patch
    reaches whichever of the two `condition` dispatches to)."""
    original_fn = alib.build_noise_realized_table
    noise_holder = {}

    def _patched(pool_, clean_, seed_, sensitivity):
        table = build_noise_realized_table_parameterized(pool_, clean_, seed_, family, noise_scale)
        noise_holder["table"] = table
        return table

    alib.build_noise_realized_table = _patched
    try:
        if condition == "E":
            r = alib.train_base_condition("E", seed, pool, clean, scale,
                                          checkpoint_dir=checkpoint_dir, checkpoint_prefix=checkpoint_prefix)
        elif condition == "B":
            r = alib.train_augmented_condition("B", seed, pool, clean, scale,
                                               checkpoint_dir=checkpoint_dir, checkpoint_prefix=checkpoint_prefix)
        else:
            raise ValueError(f"train_variant_condition only supports B/E, got {condition!r}")
    finally:
        alib.build_noise_realized_table = original_fn

    table = noise_holder["table"]
    r["n_raw"] = table["n_raw"]
    r["n_applied"] = table["n_applied"]
    r["disp"] = table["disp"]
    return r


def train_lam_grad_variant(seed: int, pool: dict, clean: dict, scale: dict, lam_grad: float,
                            checkpoint_dir: Path = None, checkpoint_prefix: str = None) -> dict:
    """Condition E only, with `LAM_G` overridden for the duration of this
    call (`train_base_condition`'s `lgrad` branch reads the module global
    `LAM_G` at call time, so this override reaches it)."""
    original = alib.LAM_G
    alib.LAM_G = lam_grad
    try:
        return alib.train_base_condition("E", seed, pool, clean, scale,
                                         checkpoint_dir=checkpoint_dir, checkpoint_prefix=checkpoint_prefix)
    finally:
        alib.LAM_G = original


# ---------------------------------------------------------------------------
# FD common-support: parameterized `clean` rebuild + mask intersection
# ---------------------------------------------------------------------------

def build_clean_query_table_with_eps(pool: dict, sim, precompute_phi_fn, eps: float) -> dict:
    """Same as `alib.build_clean_query_table`, with `FD_EPS` replaced by
    `eps` (a parameter, not the module constant) -- a direct adaptation,
    not a call-through, since `alib.build_clean_query_table` has no `eps`
    parameter to inject."""
    XU, Y = pool["XU"], pool["Y"]
    xm, xs, ym, ys = pool["xm"], pool["xs"], pool["ym"], pool["ys"]
    train_idx = np.asarray(pool["train_idx"])
    x_all = XU[:, :2].astype(np.float64)
    u_all = XU[:, 2:].astype(np.float64)
    u_mean, u_std = xm[2:].astype(np.float64), xs[2:].astype(np.float64)
    from budget_sweep_solver import INPUT_LO, INPUT_HI  # noqa: PLC0415

    Phi_all, Yphi_all = precompute_phi_fn(sim, x_all[train_idx])

    n_tr = len(train_idx)
    valid_mask = np.zeros((n_tr, 2), dtype=bool)
    y_plus = np.zeros((n_tr, 2, 2), dtype=np.float64)
    y_minus = np.zeros((n_tr, 2, 2), dtype=np.float64)
    un_plus = np.zeros((n_tr, 2, 2), dtype=np.float64)
    un_minus = np.zeros((n_tr, 2, 2), dtype=np.float64)

    un_tr = (u_all[train_idx] - u_mean) / u_std
    x_tr = x_all[train_idx]
    for j in range(2):
        up = un_tr.copy(); up[:, j] += eps
        dn = un_tr.copy(); dn[:, j] -= eps
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
    true_df_du = np.zeros((n_tr, 2, 2), dtype=np.float64)
    for j in range(2):
        true_df_du[:, :, j] = np.where(
            valid_mask[:, j:j + 1], (y_plus_n[:, j, :] - y_minus_n[:, j, :]) / (2 * eps), 0.0
        )

    def Vnp(y):
        Pm = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float64)
        return np.einsum("ni,ij,nj->n", y, Pm, y)

    true_ug = np.zeros((n_tr, 2), dtype=np.float64)
    for j in range(2):
        Vp = Vnp(y_plus[:, j, :])
        Vm = Vnp(y_minus[:, j, :])
        true_ug[:, j] = np.where(valid_mask[:, j], (Vp - Vm) / (2 * eps), 0.0)

    return dict(
        Phi_all=Phi_all, Yphi_all=Yphi_all,
        valid_mask=valid_mask,
        un_plus=un_plus.astype(np.float32), un_minus=un_minus.astype(np.float32),
        y_plus_n=y_plus_n.astype(np.float32), y_minus_n=y_minus_n.astype(np.float32),
        true_df_du=true_df_du.astype(np.float32), true_ug=true_ug.astype(np.float32),
    )


def apply_common_support_mask(clean: dict, common_mask: np.ndarray) -> dict:
    """v3->v4 item 3 / v4->v5 item 11: masked-out pairs get `valid_mask=
    False` and their `true_ug`/`true_df_du` zeroed, using the SAME
    placeholder convention `build_clean_query_table` already uses for its
    own natively-invalid pairs -- not a new convention."""
    new_valid = clean["valid_mask"] & common_mask
    true_ug = clean["true_ug"].copy()
    true_df_du = clean["true_df_du"].copy()
    newly_dropped = clean["valid_mask"] & ~common_mask
    true_ug[newly_dropped] = 0.0
    for j in range(2):
        true_df_du[newly_dropped[:, j], :, j] = 0.0
    out = dict(clean)
    out["valid_mask"] = new_valid
    out["true_ug"] = true_ug
    out["true_df_du"] = true_df_du
    return out


def oracle_n_used(model, pool: dict, sim, idx: np.ndarray) -> int:
    """Mirrors ONLY `alib.diagnose_fixed_idx`'s exclusion-mask computation
    (same `1e-9` threshold), to obtain the literal INTEGER non-excluded
    count directly -- never derived as the floating-point product
    `n_points * (1 - excluded_fraction)` (v4->v5 item 7). The actual
    reported align_cos/frac_neg/rel_grad_error values always come from
    calling the real frozen `alib.diagnose_fixed_idx` itself; this
    function is a redundant integer-count cross-check, not a
    replacement."""
    XU = pool["XU"]
    xm, xs = pool["xm"], pool["xs"]
    x = XU[idx, :2].astype(np.float64); u = XU[idx, 2:].astype(np.float64)
    u_mean, u_std = xm[2:].astype(np.float64), xs[2:].astype(np.float64)
    x_mean, x_std = xm[:2].astype(np.float64), xs[:2].astype(np.float64)
    ym_t, ys_t = torch.tensor(pool["ym"]), torch.tensor(pool["ys"])

    un = (u - u_mean) / u_std
    un_t = torch.tensor(un, dtype=torch.float32, requires_grad=True)
    xn_t = torch.tensor((x - x_mean) / x_std, dtype=torch.float32)
    y = model(torch.cat([xn_t, un_t], 1)) * ys_t + ym_t
    Pm = torch.tensor(np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float32))
    Vsum = (y @ Pm * y).sum(-1).sum()
    (g_hat_grad_t,) = torch.autograd.grad(Vsum, un_t)
    g_hat_grad = g_hat_grad_t.detach().numpy()

    def Vnp(yv):
        Pm2 = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=np.float64)
        return np.einsum("ni,ij,nj->n", yv, Pm2, yv)

    true_grad = np.zeros_like(un)
    eps = 1e-3
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
    return int((~excluded).sum())


def compute_ug_scale(clean: dict) -> float:
    """Same formula as `alib.shared_scale_constants`'s `ug_scale`
    computation, applied standalone to a specific `clean` table (needed
    since the FD axis pins `ug_scale` from the common-support `1e-3`
    reference and reuses it across all four eps levels, v3->v4 item 3)."""
    true_ug = clean["true_ug"].astype(np.float64)
    valid_ug = true_ug[clean["valid_mask"]]
    return float(max(valid_ug.std(ddof=1), 1.0)) if len(valid_ug) > 1 else 1.0


# ---------------------------------------------------------------------------
# Continuous closed-loop evaluator + evaluator-compatibility gate
# ---------------------------------------------------------------------------

PER_IC_FIELDS = ("final_V", "max_V", "settling_time", "success",
                 "solver_completed", "n_steps_completed", "failure_reason")


def rollout_per_ic_summary(seq, norm, x0, sim, input_lo, input_hi, un_lo, un_hi,
                           horizon: int, budget: int, steps: int, lr: float, rho_u: float,
                           success_v: float) -> dict:
    import cstr.solver_audit_lib as sal  # noqa: PLC0415
    result = sal.single_cstr_rollout(seq, norm, x0, horizon, budget, steps, lr, rho_u,
                                     success_v, un_lo, un_hi, sim, input_lo, input_hi,
                                     record_iterations=False)
    s = result["summary"]
    return {k: s[k] for k in PER_IC_FIELDS}


def compare_per_ic_to_battery1(new: dict, old: dict, isclose_rtol: float, isclose_atol: float) -> dict:
    """v4->v5 items 1-2: `final_V`/`max_V` via isclose, `settling_time`/
    `success` via exact -- reusing the SAME tolerance values Battery 1
    itself uses (`solver_audit_io.ISCLOSE_RTOL`/`ISCLOSE_ATOL`)."""
    checks = {}
    for key in ("final_V", "max_V"):
        a, b = new[key], old[key]
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            checks[key] = (a is None and b is None) or (
                not np.isfinite(a) and not np.isfinite(b))
        else:
            checks[key] = bool(np.isclose(a, b, rtol=isclose_rtol, atol=isclose_atol))
    checks["settling_time"] = (new["settling_time"] == old["settling_time"])
    checks["success"] = (bool(new["success"]) == bool(old["success"]))
    return dict(all_match=all(checks.values()), per_field=checks)
