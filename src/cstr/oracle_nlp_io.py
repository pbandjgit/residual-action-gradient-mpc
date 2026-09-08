"""I/O + comparator helpers for the Oracle-Gradient / Converged-NLP
comparison protocol (`docs/JPC_ORACLE_NLP_COMPARISON_PROTOCOL_2026_09_03.md`,
v11).

Re-exports (never reimplements) `src/cstr/solver_audit_io.py`'s atomic-
write/self-verification/tolerance primitives, which themselves re-export
`ablation_io.py`. `values_match()`/`gate_compare_dict()`/`ISCLOSE_RTOL`/
`ISCLOSE_ATOL` are the SAME comparator this module's `flatten_leaves()`
wraps -- Sec. 5.2a's scope rule means list/nested-dict fields are
flattened to scalar leaves and then compared with the existing
`values_match()`, not a new tolerance algorithm.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

# Re-exported, not reimplemented.
from cstr.solver_audit_io import (  # noqa: F401
    ISCLOSE_ATOL,
    ISCLOSE_RTOL,
    TIMING_WARMUP_STEPS,
    TIMING_TORCH_NUM_THREADS,
    atomic_write_json_no_overwrite,
    atomic_write_npz_no_overwrite,
    atomic_write_torch_no_overwrite,
    canonical_sha256,
    dict_arrays_sha256,
    file_sha256,
    gate_compare_dict,
    pool_content_sha256,
    read_and_verify_json_artifact,
    require_gate_pass,
    values_match,
    write_json_verify_if_exists,
    write_npz_verify_if_exists,
    write_with_self_verification,
)

# ---------------------------------------------------------------------------
# Sec. 16 -- CONFIRMED, pre-registered numeric thresholds. Not to be
# relaxed post-hoc based on how real results turn out; any amendment must
# be a separate, explicit, append-only decision, mirroring this project's
# frozen-protocol discipline.
# ---------------------------------------------------------------------------
GATE12_ERR_TOL = 1e-8          # Sec. 5.3 gates 1-2 (symbolic-vs-numeric)
GATE3_ERR_TOL = 1e-4           # Sec. 5.3 gate 3 (AD-vs-FD)
FD_EPS = 1e-5                  # Sec. 5.3 gate 3's central-FD step size
GATE6_TIE_TOL = 1e-12          # unused directly (Gate 6 now uses values_match); retained as a documentation constant only
FLOOR_MARGIN_K = 1.0           # Sec. 6 floor-inactivity margin
TOL_DEDUP_INPUT = 1e-9         # Sec. 8, dimensionless normalized-input units
TOL_TIE_OBJECTIVE = 1e-9       # Sec. 8, cost/objective units -- DISTINCT constant from TOL_DEDUP_INPUT
ALPHA_KKT = 1.0                # Sec. 9 projected-gradient residual step size
TOL_KKT = 1e-6                 # Sec. 9 KKT/stationarity tolerance
TOL_PRIMAL = 1e-6              # Sec. 9 primal-feasibility slack. Empirically
                                # RELAXED (loosened) from the design doc's
                                # originally-proposed 1e-8 during implementation
                                # -- v12 fix: "adjusted DOWN" in an earlier
                                # version of this comment was backwards; 1e-6
                                # is a LARGER/looser tolerance than 1e-8, i.e.
                                # a relaxation, not a reduction. Real IPOPT
                                # solves at `ipopt.tol=1e-8` routinely leave
                                # ~1.7e-8 residual bound violation at an
                                # active box constraint (observed directly,
                                # condition B/IC0's near-boundary solution),
                                # which is JUST beyond a 1e-8 check -- a
                                # spurious certification failure on an
                                # otherwise perfectly good `Solve_Succeeded`
                                # solution. 1e-6 gives ~2 orders of magnitude
                                # margin over the observed slack while
                                # remaining a materially tight feasibility
                                # check (per this project's discipline of
                                # amending frozen-adjacent values only with
                                # real evidence, not by loosening until a
                                # result looks good). NOTE: this value was
                                # changed during implementation but the design
                                # doc (v11) was not updated to match until the
                                # v12 corrective revision -- see the design
                                # doc's Sec. 23 for the full account of that
                                # doc/code drift.
IPOPT_TOL = 1e-8               # Sec. 9 IPOPT's own convergence tolerance
IPOPT_ACCEPTABLE_TOL = 1e-6    # Sec. 9, distinct from IPOPT_TOL
IPOPT_ACCEPTABLE_ITER = 15     # Sec. 9, IPOPT default, explicitly pinned
IPOPT_MAX_ITER = 500           # Sec. 9


def to_jsonable(obj: Any) -> Any:
    """Recursively converts numpy arrays/scalars to native Python types
    (list/float/int/bool). Required before any payload built by this
    module's rollout-record/manifest writers is passed to `ablation_io.py`'s
    `write_with_self_verification`/`atomic_write_json_no_overwrite`: their
    own `_sanitize_nonfinite()` handles NaN/Inf floats and dict/list/tuple
    recursion, but returns a raw `np.ndarray` UNCHANGED -- `json.dumps`
    then stringifies it via `default=str` during serialization while the
    in-memory object still holds the ndarray, so the writer's own round-
    trip self-verification (`reread != sanitized`) raises `ValueError:
    truth value of an array is ambiguous` the first time any numpy array
    reaches it (confirmed empirically: every rollout record here carries
    `x_after`/`u0`/`final_plan_un` as ndarrays)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def err(a: Any, b: Any) -> float:
    """Sec. 5.2: `max_i ( |a_i - b_i| / max(1, |a_i|, |b_i|) )`, the ratio
    computed component-wise BEFORE the max is taken. Scoped (Sec. 5.2a) to
    gates 1-3's cross-implementation comparisons only -- never used for
    same-implementation regression/reproducibility checks, which use
    `values_match()` (via `flatten_leaves()` below) instead."""
    a_arr = np.atleast_1d(np.asarray(a, dtype=float))
    b_arr = np.atleast_1d(np.asarray(b, dtype=float))
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"err(): shape mismatch {a_arr.shape} vs {b_arr.shape}")
    denom = np.maximum(1.0, np.maximum(np.abs(a_arr), np.abs(b_arr)))
    return float(np.max(np.abs(a_arr - b_arr) / denom))


def flatten_leaves(obj: Any, prefix: str = "") -> Iterable[tuple]:
    """Sec. 5.3 gate 4: recursively walks a dict/list/tuple/ndarray-valued
    structure and yields `(dotted_key, scalar_value)` pairs at every leaf
    (a value that is not itself a dict/list/tuple/ndarray). Built ON TOP
    OF `values_match()`, not a replacement for it -- the only new logic
    here is the flattening walk; each yielded leaf pair is still compared
    with the existing `values_match()` by the caller (see
    `gate_compare_flattened` below)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten_leaves(v, new_prefix)
    elif isinstance(obj, (list, tuple, np.ndarray)):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}.{i}" if prefix else str(i)
            yield from flatten_leaves(v, new_prefix)
    else:
        yield (prefix, obj)


def _flatten_to_dict(obj: Any) -> dict:
    return dict(flatten_leaves(obj))


def gate_compare_flattened(new: dict, old: dict, exact_fields: Iterable[str],
                            isclose_fields: Iterable[str], field_map: dict = None) -> dict:
    """Generalizes `gate_compare_dict()` (`solver_audit_io.py:80-105`) to
    fields that may be scalar OR list/nested-dict-valued. Each named field
    is flattened (via `flatten_leaves`) on both `new` and `old`; every
    resulting leaf pair is compared with the EXISTING `values_match()` --
    `gate_compare_dict()` itself is not modified, this function only feeds
    it (conceptually) pre-flattened scalar leaves instead of the raw
    nested structure it was never designed to handle (Sec. 5.3 gate 4).
    A field entirely absent from both sides, or present on only one side,
    is reported as a single `MISSING_FIELD` mismatch -- it cannot
    silently vanish just because the flattened key set happened to be
    empty. `field_map` renames a field as it appears in `new` to its
    corresponding name in `old` (matching `gate_compare_dict()`'s own
    convention)."""
    field_map = field_map or {}
    flat_new = _flatten_to_dict(new)
    flat_old = _flatten_to_dict(old)
    mismatches = []

    def _compare_group(fields, exact):
        for field in fields:
            old_field = field_map.get(field, field)
            new_leaf_keys = sorted(k for k in flat_new if k == field or k.startswith(field + "."))
            old_leaf_keys = sorted(k for k in flat_old if k == old_field or k.startswith(old_field + "."))
            if not new_leaf_keys or not old_leaf_keys:
                mismatches.append(dict(field=field, reason="MISSING_FIELD",
                                        new_present=bool(new_leaf_keys), old_present=bool(old_leaf_keys)))
                continue
            new_suffixes = {k[len(field):] for k in new_leaf_keys}
            old_suffixes = {k[len(old_field):] for k in old_leaf_keys}
            if new_suffixes != old_suffixes:
                mismatches.append(dict(field=field, reason="LEAF_SHAPE_MISMATCH",
                                        new_leaves=new_leaf_keys, old_leaves=old_leaf_keys))
                continue
            for suffix in sorted(new_suffixes):
                nk, ok = field + suffix, old_field + suffix
                nv, ov = flat_new[nk], flat_old[ok]
                if not values_match(nv, ov, exact=exact):
                    mismatches.append(dict(field=nk, reason=("EXACT_MISMATCH" if exact else "ISCLOSE_MISMATCH"),
                                            new=nv, old=ov))

    _compare_group(exact_fields, True)
    _compare_group(isclose_fields, False)
    return dict(passed=len(mismatches) == 0, mismatches=mismatches)


class ReferenceNotEstablished(RuntimeError):
    """Sec. 10/13: raised when Oracle-20 or Certified-NLP fails
    reference validity for any IC -- the condition's outcome is
    `REFERENCE_NOT_ESTABLISHED`, no gap may be computed."""


class ImplementationOrValidationBlocked(RuntimeError):
    """Sec. 13's top-precedence outcome: a preflight gate, the freeze-time
    verification, a provenance/hash/resume-chain check, or the floor-
    inactivity invariant's own execution failed -- distinct from the
    invariant correctly detecting and flagging a floor-active point,
    which is a normal, non-blocking occurrence."""
