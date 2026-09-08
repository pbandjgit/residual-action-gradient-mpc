#!/usr/bin/env python3
"""R2.m4 correction (2026-09-04): failure-penalized settling time for the
manuscript's literal Table 4 (`tab:metrics`, two-CSTR).

R2.m4: "In Table 4, settling time is computed only over initial
conditions that successfully reach the target... Please additionally
report a failure-penalized metric, time-to-target distributions, or an
empirical cumulative distribution."

The existing `results/interim/logs/two_cstr_metrics.json` (already
real, already frozen) stores enough per-seed information to compute this
correctly with NO new rollout or training: each seed's `success`
(fraction of the 5 ICs that reached the target) and `settle` (the
CONDITIONAL mean settling time, averaged only over the successful ICs --
`nanmean` in the original `agg()`). This script re-derives the
failure-penalized per-seed value directly from those two fields:

    n_success = round(success * 5)
    t_penalized = (n_success * conditional_settle + (5 - n_success) * SENTINEL) / 5

with `SENTINEL = 121` (`steps + 1`, matching the same fixed-sentinel
convention already used in `single_cstr_rollout` and this session's
`two_cstr_lqr_operational_audit_2026_09_04.py` -- never imputed from a
partial trajectory).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LOG = ROOT / "results" / "interim" / "logs"
SOURCE_PATH = LOG / "two_cstr_metrics.json"
SENTINEL = 121  # steps (120) + 1
N_IC = 5


def failure_penalized_settle(success_fraction: float, conditional_settle: float) -> float:
    n_success = round(success_fraction * N_IC)
    cond = conditional_settle if n_success > 0 else 0.0
    return (n_success * cond + (N_IC - n_success) * SENTINEL) / N_IC


def main() -> dict:
    source_bytes = SOURCE_PATH.read_bytes()
    import hashlib
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    d = json.loads(source_bytes)

    out = dict(kind="two_cstr_failure_penalized_settling", predecessor_source_path=str(SOURCE_PATH.relative_to(ROOT)),
               predecessor_source_sha256=source_sha256, sentinel=SENTINEL, n_ic=N_IC,
               per_condition={})

    for tag in ("value_only", "value+grad"):
        per_seed_penalized = []
        for p in d["per_seed"][tag]:
            t_pen = failure_penalized_settle(p["success"], p["settle"])
            per_seed_penalized.append(dict(seed=p["seed"], success_fraction=p["success"],
                                           conditional_settle=p["settle"], failure_penalized_settle=t_pen))
        vals = np.array([r["failure_penalized_settle"] for r in per_seed_penalized])
        out["per_condition"][tag] = dict(
            per_seed=per_seed_penalized,
            mean=float(vals.mean()), sd_ddof0=float(vals.std(ddof=0)),
        )

    # auxiliary LQR: deterministic, already reported as 41.4 in tab:metrics/
    # two_cstr_metrics.json's aux_LQR entry -- unaffected by this correction
    # since it has no failures to penalize (success=1.0 on all 5 ICs).
    out["aux_LQR_settle_unchanged"] = d["controllers"]["aux_LQR"]["settle"]

    from cstr.ablation_io import write_with_self_verification  # noqa: PLC0415
    dest = LOG / "two_cstr_failure_penalized_settling_2026_09_04.json"
    self_hash = write_with_self_verification(dest, out)
    print("wrote", dest, "self_hash", self_hash)
    for tag in ("value_only", "value+grad"):
        c = out["per_condition"][tag]
        print(f"{tag}: mean={c['mean']:.2f} sd(ddof=0)={c['sd_ddof0']:.2f}")
    print("aux_LQR (unchanged):", out["aux_LQR_settle_unchanged"])
    return out


if __name__ == "__main__":
    main()
