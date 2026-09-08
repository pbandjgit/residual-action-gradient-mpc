#!/usr/bin/env python3
"""One-off launcher for the REAL execution of the frozen go/no-go ablation
protocol. NOT part of the Freeze Manifest's hash-locked file set (see
`docs/JPC_REVISION_GO_NO_GO_ABLATION_PROTOCOL_FREEZE_MANIFEST_2026_08_31.json`)
-- it exists only to invoke the frozen `run_real_protocol()` with default
arguments (real pool, canonical 10-seed set, current freeze manifest) and
print a progress/outcome summary. Never edits, and must never need to
edit, any of the 12 frozen files.
"""
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ablation_driver_2026_08_31 as drv  # noqa: E402


def main() -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] starting run_real_protocol()", flush=True)
    t0 = time.time()
    try:
        out = drv.run_real_protocol()
    except Exception:
        print(f"[{datetime.now(timezone.utc).isoformat()}] run_real_protocol() RAISED after "
              f"{time.time() - t0:.1f}s:", flush=True)
        traceback.print_exc()
        raise
    dt = time.time() - t0
    print(f"[{datetime.now(timezone.utc).isoformat()}] run_real_protocol() completed in {dt:.1f}s", flush=True)
    print("M1:", out["m1"]["path"], out["m1"]["self_hash"], flush=True)
    for seed, path in sorted(out["m2_paths"].items()):
        print("M2 seed", seed, ":", path, flush=True)
    print("M3:", out["m3"]["path"], out["m3"]["self_hash"], flush=True)
    print("outcome:", out["aggregated"]["outcome"], flush=True)


if __name__ == "__main__":
    main()
