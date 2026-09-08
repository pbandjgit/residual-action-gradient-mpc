"""I/O helpers for the JPC go/no-go ablation protocol
(`docs/JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_2026_08_31.md`, v7).

Independently implemented rather than imported from Paper M
(`paper_m_data_scaling_pilot/src/data_generation_io_2026_08_30.py`),
which this mirrors: cross-paper-directory imports would couple two
otherwise-independent workspaces, and Paper M's module is itself
explicitly documented as not to be reused outside its own protocol.
Same two properties: (1) atomic create-only publish via `os.link`
(raises on an existing destination, no TOCTOU window `os.replace`-based
rename has), (2) JSON artifacts self-verify via an embedded
`canonical_payload_sha256` field computed over the payload with that
field itself excluded.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path

import numpy as np
import torch


def _sanitize_nonfinite(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nonfinite(v) for v in obj]
    return obj


def canonical_sha256(payload: dict) -> str:
    sanitized = _sanitize_nonfinite(payload)
    return hashlib.sha256(
        json.dumps(sanitized, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def state_dict_sha256(model: torch.nn.Module) -> str:
    """Hash a model's `state_dict()` by concatenating each parameter's raw
    bytes in a fixed (sorted-key) order -- used for the per-(condition,seed)
    initial-weight identity check (Section 5.5)."""
    digest = hashlib.sha256()
    sd = model.state_dict()
    for key in sorted(sd.keys()):
        digest.update(key.encode())
        tensor = sd[key].detach().cpu().contiguous()
        digest.update(np.ascontiguousarray(tensor.numpy()).tobytes())
    return digest.hexdigest()


def dict_arrays_sha256(d: dict, keys: tuple = None) -> str:
    """Hashes a dict's array-valued entries (dtype+shape+raw bytes, in
    SORTED key order unless `keys` fixes an explicit order) -- a single
    generic content-hash used for both `pool` and `CLEAN_QUERY_TABLE`
    identity checks, so a manifest's provenance record reflects exactly
    what was passed to training, regardless of whether the source was
    loaded from disk or (for a smoke/dev run) fabricated in memory."""
    key_order = keys if keys is not None else tuple(sorted(k for k, v in d.items() if isinstance(v, np.ndarray)))
    digest = hashlib.sha256()
    for key in key_order:
        digest.update(key.encode())
        arr = np.ascontiguousarray(np.asarray(d[key]))
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def pool_content_sha256(pool: dict, keys: tuple = ("XU", "Y", "xm", "xs", "ym", "ys",
                                                    "train_idx", "val_idx", "test_idx")) -> str:
    """Hashes the ACTUAL array content of a `pool` dict, in a fixed key
    order -- used so a manifest's provenance record reflects exactly what
    was passed to training, regardless of whether `pool` was loaded from
    `POOL_PATH` or (for a smoke/dev run) fabricated in memory. Correcting
    a prior design where the manifest always hashed the on-disk pool
    FILE, decoupled from whatever `pool` dict a caller actually passed."""
    return dict_arrays_sha256(pool, keys=keys)


def _publish_atomic_create_only(tmp: Path, dest: Path) -> None:
    try:
        os.link(tmp, dest)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite {dest} (destination appeared during write)")
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json_no_overwrite(path: Path, payload: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_nonfinite(payload)
    tmp = path.with_name(f"{path.name}.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(json.dumps(sanitized, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")
        reread = json.loads(tmp.read_text())
        if reread != sanitized:
            raise RuntimeError(f"self-verification failed for {path}: round-trip mismatch")
        _publish_atomic_create_only(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_with_self_verification(path: Path, payload: dict, self_hash_field: str = "canonical_payload_sha256") -> str:
    if self_hash_field in payload:
        raise ValueError(
            f"payload already has a '{self_hash_field}' field; pass a "
            "different self_hash_field to avoid overwriting it"
        )
    result = dict(payload)
    result[self_hash_field] = canonical_sha256(result)
    atomic_write_json_no_overwrite(path, result)
    reread = json.loads(path.read_text())
    check = dict(reread)
    recorded = check.pop(self_hash_field)
    if canonical_sha256(check) != recorded:
        raise RuntimeError(f"self-verification failed for {path}")
    return recorded


def read_and_verify_json_artifact(path: Path, self_hash_field: str = "canonical_payload_sha256",
                                   expected_kind: str = None, expected_keys: set = None) -> tuple:
    """`expected_kind`, when given, is checked against the payload's own
    `kind` field -- every manifest in this chain records one, but nothing
    previously checked it, so e.g. accidentally passing an M2 path where an
    M1 was expected would pass self-verification (the JSON is internally
    consistent) and only fail later, deep inside whatever field lookup
    first differs between the two manifest kinds, with a confusing error.

    `expected_keys`, when given, is checked for an EXACT match (not merely
    a subset) against `set(payload.keys())` -- the protocol document
    promises each manifest stage hash-locks its own exact keyset; a field
    silently dropped (or an unexpected extra field written by a stray code
    path) would otherwise only be noticed if some later lookup happened to
    KeyError on it."""
    raw = json.loads(Path(path).read_text())
    payload = dict(raw)
    recorded = payload.pop(self_hash_field)
    if canonical_sha256(payload) != recorded:
        raise RuntimeError(f"RESUME_VERIFICATION_FAILED: {path} self-hash mismatch")
    if expected_kind is not None and payload.get("kind") != expected_kind:
        raise RuntimeError(
            f"MANIFEST_KIND_MISMATCH: {path} has kind={payload.get('kind')!r}, expected {expected_kind!r}"
        )
    if expected_keys is not None and set(payload.keys()) != set(expected_keys):
        missing = set(expected_keys) - set(payload.keys())
        extra = set(payload.keys()) - set(expected_keys)
        raise RuntimeError(
            f"MANIFEST_KEYSET_MISMATCH: {path} payload keys do not exactly match the expected "
            f"keyset (missing={sorted(missing)}, extra={sorted(extra)})"
        )
    return recorded, payload


def atomic_write_npz_no_overwrite(path: Path, arrays: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}.npz")
    try:
        np.savez(tmp, **arrays)
        with np.load(tmp, allow_pickle=True) as reloaded:
            reloaded_keys = set(reloaded.files)
            if reloaded_keys != set(arrays.keys()):
                raise RuntimeError(
                    f"self-verification failed for {path}: key mismatch "
                    f"{reloaded_keys} != {set(arrays.keys())}"
                )
            for key, value in arrays.items():
                value = np.asarray(value)
                reloaded_value = reloaded[key]
                if reloaded_value.shape != value.shape:
                    raise RuntimeError(
                        f"self-verification failed for {path}: shape mismatch "
                        f"on {key!r}: {reloaded_value.shape} != {value.shape}"
                    )
                if not np.array_equal(reloaded_value, value):
                    raise RuntimeError(
                        f"self-verification failed for {path}: value mismatch on {key!r}"
                    )
        _publish_atomic_create_only(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_npz_verify_if_exists(path: Path, arrays: dict) -> str:
    """Partial-publication crash recovery: a manifest builder that writes
    several companion files before its own self-verifying JSON (e.g. M1's
    query-table npz, M3's raw-metrics/resample/oracle-idx npz) can be
    interrupted AFTER a companion file is published but BEFORE the JSON
    is written. A naive retry would call the plain no-overwrite writer
    again and hit `FileExistsError` on the companion file that already
    exists, even though nothing is actually wrong. This checks for that
    case: if `path` already exists, its content is verified EXACTLY
    against `arrays` (never trusted blindly) and reused; otherwise it is
    written fresh via the normal atomic no-overwrite path."""
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=True) as existing:
            existing_keys = set(existing.files)
            if existing_keys != set(arrays.keys()):
                raise RuntimeError(
                    f"PARTIAL_PUBLICATION_MISMATCH: {path} already exists with a different "
                    f"keyset ({sorted(existing_keys)}) than expected ({sorted(arrays.keys())})"
                )
            for key, value in arrays.items():
                if not np.array_equal(existing[key], np.asarray(value)):
                    raise RuntimeError(
                        f"PARTIAL_PUBLICATION_MISMATCH: {path} already exists with different "
                        f"content for key {key!r} than the current in-memory value"
                    )
        return file_sha256(path)
    atomic_write_npz_no_overwrite(path, arrays)
    return file_sha256(path)


def write_json_verify_if_exists(path: Path, payload: dict) -> str:
    """Same partial-publication crash recovery as `write_npz_verify_if_
    exists`, for a plain (non-self-verifying) JSON companion file."""
    path = Path(path)
    sanitized = _sanitize_nonfinite(payload)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != sanitized:
            raise RuntimeError(
                f"PARTIAL_PUBLICATION_MISMATCH: {path} already exists with different content "
                "than the current in-memory payload"
            )
        return file_sha256(path)
    atomic_write_json_no_overwrite(path, payload)
    return file_sha256(path)


def atomic_write_torch_no_overwrite(path: Path, state_dict: dict) -> None:
    """Same create-only-publish discipline as the JSON/NPZ writers above,
    for the Section 8 disk-checkpoint cadence (torch has no built-in
    no-overwrite save)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    try:
        torch.save(state_dict, tmp)
        reloaded = torch.load(tmp, map_location="cpu")
        if set(reloaded.keys()) != set(state_dict.keys()):
            raise RuntimeError(f"self-verification failed for {path}: key mismatch")
        for k in state_dict:
            if not torch.equal(reloaded[k], state_dict[k].detach().cpu()):
                raise RuntimeError(f"self-verification failed for {path}: value mismatch on {k!r}")
        _publish_atomic_create_only(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
