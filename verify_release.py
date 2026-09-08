#!/usr/bin/env python3
"""Read-only verifier for the Paper H revision-release archives.

Frozen JSON records in this project reference their companion files (model
checkpoints, resample arrays, etc.) by an absolute path recorded at the time
of the original run, e.g.

    /Users/pbandj/Documents/mpc_phase1/phd/paper_h_smooth_lcnn_lmpc/results/
        ablation_execution_2026_08_31/checkpoints/seed0/attempt1/B_seed0_ep050.pt

Those frozen records are never edited (this project's own discipline), so a
reader who extracts the released tar.zst archives to a different machine and
a different directory will not find that literal path. This script does not
change any file; it re-derives an ARCHIVE-RELATIVE path from each recorded
absolute path by locating the known "/results/<archive_name>/" marker in the
string, then verifies (a) that the resulting file actually exists under the
extraction root, and (b) that its SHA-256 matches both the value recorded
inside the manifest itself (when the manifest records one, e.g.
`checkpoint_sha256` alongside `checkpoint_path`) and the independent
RELEASE_INDEX.json built directly from the original files before archiving.

Verification is therefore always by file CONTENT hash, never by comparing
path strings.

Usage:
    python3 verify_release.py --root <extraction_root> --index RELEASE_INDEX.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MARKER_RE = re.compile(r"/results/([^/]+)/(.+)$")


def sha256_of(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_index(index_path: Path) -> dict:
    idx = json.load(open(index_path))
    lookup = {}
    for dname, info in idx["archives"].items():
        for f in info["files"]:
            lookup[(dname, f["path"])] = f["sha256"]
    return lookup


def find_path_hash_pairs(obj, path_stack=()):
    """Yield (path_field_name, path_value, hash_field_name_or_None,
    hash_value_or_None, json_pointer) for every *_path string field found
    anywhere in a nested JSON structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.endswith("_path") and isinstance(v, str) and "/results/" in v:
                hash_key = k[: -len("_path")] + "_sha256"
                yield k, v, hash_key, obj.get(hash_key), path_stack + (k,)
            yield from find_path_hash_pairs(v, path_stack + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from find_path_hash_pairs(v, path_stack + (i,))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                     help="Directory containing the extracted release "
                          "archives (i.e. root/<archive_dir_name>/...). "
                          "All *_path references, wherever the manifest that "
                          "mentions them was found, are resolved against "
                          "this root.")
    ap.add_argument("--index", required=True, type=Path,
                     help="Path to RELEASE_INDEX.json")
    ap.add_argument("--manifests-dir", action="append", default=None, type=Path,
                     help="Additional directory to scan for manifest/summary "
                          "JSON files that reference companion files living "
                          "under --root (e.g. the repo's "
                          "results/revision_summaries/). May be given more "
                          "than once. If omitted, only --root itself is "
                          "scanned.")
    ap.add_argument("--json-glob", default="**/*.json",
                     help="Glob (relative to each scanned directory) for "
                          "manifest/record files to scan for *_path fields")
    args = ap.parse_args()

    scan_dirs = list(args.manifests_dir) if args.manifests_dir else []
    scan_dirs.append(args.root)

    index_lookup = load_index(args.index)

    n_json_scanned = 0
    n_fields_found = 0
    n_resolved = 0
    n_missing = 0
    n_hash_ok_manifest = 0
    n_hash_fail_manifest = 0
    n_hash_ok_index = 0
    n_hash_fail_index = 0
    n_not_in_index = 0
    failures = []

    all_json_files = set()
    for d in scan_dirs:
        all_json_files.update(d.glob(args.json_glob))

    for jf in sorted(all_json_files):
        try:
            data = json.load(open(jf))
        except Exception as exc:
            failures.append(f"UNREADABLE JSON: {jf}: {exc}")
            continue
        n_json_scanned += 1
        for path_key, path_val, hash_key, hash_val, pointer in find_path_hash_pairs(data):
            n_fields_found += 1
            m = MARKER_RE.search(path_val)
            if not m:
                failures.append(f"NO /results/ MARKER: {jf}:{'.'.join(map(str, pointer))} = {path_val}")
                continue
            archive_dir, rel_path = m.group(1), m.group(2)
            resolved = args.root / archive_dir / rel_path
            if not resolved.exists():
                n_missing += 1
                failures.append(f"MISSING FILE: {jf}:{'.'.join(map(str, pointer))} -> {resolved}")
                continue
            n_resolved += 1
            actual_hash = sha256_of(resolved)

            if hash_val:
                if actual_hash == hash_val:
                    n_hash_ok_manifest += 1
                else:
                    n_hash_fail_manifest += 1
                    failures.append(
                        f"HASH MISMATCH (manifest field {hash_key}): {resolved} "
                        f"expected {hash_val} got {actual_hash}"
                    )

            key = (archive_dir, rel_path)
            if key in index_lookup:
                if actual_hash == index_lookup[key]:
                    n_hash_ok_index += 1
                else:
                    n_hash_fail_index += 1
                    failures.append(
                        f"HASH MISMATCH (RELEASE_INDEX): {archive_dir}/{rel_path} "
                        f"expected {index_lookup[key]} got {actual_hash}"
                    )
            else:
                n_not_in_index += 1
                failures.append(f"NOT IN RELEASE_INDEX.json: {archive_dir}/{rel_path}")

    print("=== verify_release.py summary ===")
    print(f"JSON files scanned:            {n_json_scanned}")
    print(f"*_path fields found:           {n_fields_found}")
    print(f"resolved to an existing file:  {n_resolved}")
    print(f"missing (path did not exist):  {n_missing}")
    print(f"manifest-recorded hash OK:     {n_hash_ok_manifest}")
    print(f"manifest-recorded hash FAIL:   {n_hash_fail_manifest}")
    print(f"RELEASE_INDEX hash OK:         {n_hash_ok_index}")
    print(f"RELEASE_INDEX hash FAIL:       {n_hash_fail_index}")
    print(f"not present in RELEASE_INDEX:  {n_not_in_index}")

    if failures:
        print(f"\n{len(failures)} issue(s):")
        for line in failures[:200]:
            print(" -", line)
        if len(failures) > 200:
            print(f"   ... and {len(failures) - 200} more")
        return 1

    print("\nAll referenced companion files resolved and verified by SHA-256.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
