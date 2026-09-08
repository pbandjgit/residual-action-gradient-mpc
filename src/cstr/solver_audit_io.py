"""I/O + gate-comparison helpers for the single-CSTR query-matched solver
audit + two-plant state-constraint minimum audit
(`docs/JPC_SOLVER_AUDIT_PROTOCOL_2026_09_01.md`, v7).

Re-exports (never reimplements) `src/cstr/ablation_io.py`'s atomic-write /
self-verification primitives: that module is one of the 12 Freeze-
Manifest-covered files for the go/no-go ablation and must not be edited,
but nothing prevents importing it -- re-implementing the same atomic
create-only-publish / self-hash-round-trip logic a second time in this
file would only risk the two copies silently diverging, for no benefit
(this protocol lives in the same paper directory, not a separate
workspace the way Paper M's independently-implemented copy does).

Adds what `ablation_io.py` does not need: the compatibility-gate
tolerance contract (Section 4a) -- boolean/integer fields compared EXACT,
continuous float fields via `numpy.isclose(rtol=1e-5, atol=1e-8)`, with
optional field-name remapping (e.g. Gate 1's `violations` <->
`lyapunov_increase_count`, Section 1).
"""
from __future__ import annotations

import numbers
from typing import Any, Callable, Iterable

import numpy as np

# Re-exported, not reimplemented -- see module docstring.
from cstr.ablation_io import (  # noqa: F401
    atomic_write_json_no_overwrite,
    atomic_write_npz_no_overwrite,
    atomic_write_torch_no_overwrite,
    canonical_sha256,
    dict_arrays_sha256,
    file_sha256,
    pool_content_sha256,
    read_and_verify_json_artifact,
    state_dict_sha256,
    write_json_verify_if_exists,
    write_npz_verify_if_exists,
    write_with_self_verification,
)

# ---------------------------------------------------------------------------
# Tolerance contract (Section 4a, pinned as literals in v6 -- never chosen
# by looking at results, which would be circular).
# ---------------------------------------------------------------------------
ISCLOSE_RTOL = 1e-5
ISCLOSE_ATOL = 1e-8

# Section 5's survivorship-bias threshold, pinned as a literal in v6.
SURVIVORSHIP_COMPLETION_RATE_THRESHOLD = 0.10

# Section 3.2's timing-methodology literals, pinned in v5.
TIMING_WARMUP_STEPS = 2
TIMING_TORCH_NUM_THREADS = 1


def _is_bool(v: Any) -> bool:
    return isinstance(v, (bool, np.bool_))


def values_match(new: Any, old: Any, exact: bool) -> bool:
    """A single scalar comparison under the tolerance contract. `exact`
    selects EXACT (`==`, used for booleans/integers) vs. `numpy.isclose`
    (used for continuous floats) -- the caller decides which applies to a
    given field, per Section 4a's per-field split, never inferred from
    the Python type alone (an integer-valued float field like a raw
    iteration count must still compare exact)."""
    if exact:
        return bool(new == old)
    if new is None or old is None:
        return new is old
    if isinstance(new, numbers.Real) and isinstance(old, numbers.Real):
        if np.isnan(float(new)) and np.isnan(float(old)):
            return True
        return bool(np.isclose(float(new), float(old), rtol=ISCLOSE_RTOL, atol=ISCLOSE_ATOL))
    return bool(new == old)


def gate_compare_dict(new: dict, old: dict, exact_keys: Iterable[str], isclose_keys: Iterable[str],
                       field_map: dict = None) -> dict:
    """Compares `new` (the freshly-computed, instrumented side of a gate)
    against `old` (the frozen/legacy side) over exactly the fields named
    in `exact_keys`/`isclose_keys`. `field_map`, when given, maps a key
    name as it appears in `new` to the corresponding key name in `old`
    (Gate 1's `lyapunov_increase_count` -> `violations`, Section 1/4a) --
    required so a genuine naming difference (not an implementation bug)
    never spuriously fails the gate. Returns `{"passed": bool,
    "mismatches": [...]}`; a missing key on either side is itself a
    (reported, non-silent) mismatch rather than a KeyError."""
    field_map = field_map or {}
    mismatches = []
    for key in exact_keys:
        old_key = field_map.get(key, key)
        if key not in new or old_key not in old:
            mismatches.append(dict(field=key, reason="MISSING_FIELD",
                                    new_present=key in new, old_present=old_key in old))
            continue
        if not values_match(new[key], old[old_key], exact=True):
            mismatches.append(dict(field=key, reason="EXACT_MISMATCH", new=new[key], old=old[old_key]))
    for key in isclose_keys:
        old_key = field_map.get(key, key)
        if key not in new or old_key not in old:
            mismatches.append(dict(field=key, reason="MISSING_FIELD",
                                    new_present=key in new, old_present=old_key in old))
            continue
        if not values_match(new[key], old[old_key], exact=False):
            mismatches.append(dict(field=key, reason="ISCLOSE_MISMATCH", new=new[key], old=old[old_key]))
    return dict(passed=len(mismatches) == 0, mismatches=mismatches)


class TwoCstrReconstructionIncompatible(RuntimeError):
    """Named failure state (Section 4a/8): Gate 4 failed for at least one
    (config, seed) pair. The tolerance is never loosened to force a pass
    -- this exception is the only way that failure may be reported, never
    silently swallowed or retried with a wider tolerance."""


def require_gate_pass(gate_name: str, result: dict, context: str = "") -> None:
    if not result["passed"]:
        raise RuntimeError(
            f"COMPATIBILITY_GATE_FAILED: {gate_name}{f' ({context})' if context else ''} -- "
            f"mismatches: {result['mismatches']}"
        )
