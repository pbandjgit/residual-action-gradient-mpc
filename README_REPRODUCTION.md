# Reproducing the JPC major-revision results

This document covers the content added for the *Journal of Process Control*
major revision (JPROCONT-D-26-00624), on top of the original submission
described in `README.md`. It corresponds to five frozen experiment batches:
the equal-query single-CSTR ablation, the instrumented projected-Adam
solver-iterate analysis, the true-model/certified-local-NLP reference, the
extended action-Jacobian weight grid, and the finite-difference/corruption
sensitivity study.

## Repository vs. Release assets

Each frozen experiment batch recorded its complete per-seed, per-step
evidence (model checkpoints, bootstrap resample arrays, per-iteration solver
records). That raw evidence is the actual statistical basis for the paper's
reported confidence intervals and p-values, so it is published in full --
but not inside this repository's git history, since several of the raw
archives are tens of megabytes of largely-repetitive per-step JSON and
would make every clone of this repository unnecessarily heavy.

Instead:

- **This repository** contains the code that produced every reported number
  and figure, the top-level summary and manifest files for each batch
  (`results/revision_summaries/`, small, aggregate-level), an index of every
  file in the full raw archives with its SHA-256 (`RELEASE_INDEX.json` /
  `SHA256SUMS`), and a verifier (`verify_release.py`).
- **The GitHub Release** attached to this repository's revision tag carries
  the five full raw-evidence archives as binary assets:

  | Archive | Batch |
  |---|---|
  | `ablation_execution_2026_08_31.tar.zst` | Equal-query single-CSTR ablation (Table 1, Fig. 2-3, Table B.1) |
  | `solver_audit_execution_2026_09_01.tar.zst` | Instrumented projected-Adam solver-iterate analysis (Section 6.2, Supplementary Tables S4-S6) |
  | `oracle_nlp_execution_2026_09_03_corrective.tar.zst` | True-model Adam reference (TM-20) and certified local-NLP decomposition (Fig. 4, Supplementary Section S14) |
  | `ablation_lamjac_extension_2026_09_03.tar.zst` | Extended action-Jacobian weight grid (Table 1, Supplementary Section S2) |
  | `ablation_sensitivity_2026_09_03.tar.zst` | Finite-difference step / label-noise-family / gradient-weight sensitivity (Supplementary Sections S3, S16) |

  An earlier, uncorrected run of the oracle-NLP comparison
  (`oracle_nlp_execution_2026_09_03_preliminary`) is not published: it has a
  known documentation/provenance defect, is not the source of any number
  reported in the paper, and publishing it alongside the corrective batch
  would only create ambiguity about which archive is authoritative. It is
  retained in the authors' internal (non-public) records.

## Reproducing from a clean checkout

```bash
git clone https://github.com/pbandjgit/residual-action-gradient-mpc.git
cd residual-action-gradient-mpc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download the five archives from this repository's Release page for the
# revision tag, then extract them at the repository root:
for f in ablation_execution_2026_08_31 \
         solver_audit_execution_2026_09_01 \
         oracle_nlp_execution_2026_09_03_corrective \
         ablation_lamjac_extension_2026_09_03 \
         ablation_sensitivity_2026_09_03; do
  zstd -dc "${f}.tar.zst" | tar -xf - -C results/
done

# Verify: every companion file (checkpoints, resample arrays, per-iteration
# records) that any manifest references resolves under results/ and matches
# both the manifest's own recorded hash and RELEASE_INDEX.json.
python verify_release.py --root results \
    --manifests-dir results/revision_summaries \
    --index RELEASE_INDEX.json
```

A clean pass ends with "All referenced companion files resolved and
verified by SHA-256."

## Why a verifier, and not just paths that work out of the box

The frozen JSON records in `results/` were never edited after being written
(this project's own discipline for frozen experimental artifacts), and they
record their companion files by the absolute path in effect on the machine
that generated them, e.g.

```
/Users/.../results/ablation_execution_2026_08_31/checkpoints/seed0/attempt1/B_seed0_ep050.pt
```

That literal path will not exist on a different machine or under a
different checkout directory. `verify_release.py` never edits these frozen
files to "fix" the path. Instead, for every `<field>_path` string it finds
in any manifest, it locates the `/results/<batch>/` segment of the recorded
path, reinterprets everything after it as a path relative to wherever you
extracted the archives, and confirms two independent things about the file
found there: its SHA-256 matches the value the manifest itself recorded
(when the manifest records one, e.g. `checkpoint_path` next to
`checkpoint_sha256`), and its SHA-256 matches `RELEASE_INDEX.json`, which was
computed directly from the original files before they were archived.
Verification is therefore always by file content, never by comparing path
strings.

## New library modules and scripts

| Module / script | Role |
|---|---|
| `src/cstr/ablation_lib.py`, `ablation_io.py` | Equal-query training/evaluation library and its I/O contract |
| `scripts/ablation_driver_2026_08_31.py` | Runs the equal-query single-CSTR ablation (value-only vs. value+residual-gradient, ten seeds) |
| `src/cstr/solver_audit_lib.py`, `solver_audit_io.py` | Instrumented projected-Adam and Gauss-Newton solver-iterate recording |
| `scripts/solver_audit_driver_2026_09_01.py`, `run_real_protocol_launch_2026_09_01.py` | Runs the actual-solver-iterate analysis |
| `src/cstr/oracle_nlp_lib.py`, `oracle_nlp_io.py` | True-dynamics Adam reference (TM-20) and CasADi/IPOPT certified local-NLP reference |
| `scripts/oracle_nlp_driver_2026_09_03.py` | Runs the true-model/NLP decomposition |
| `scripts/ablation_lamjac_extension_2026_09_03.py` | Extends the action-Jacobian weight grid under a fixed, validation-MSE-only stopping rule |
| `src/cstr/sensitivity_lib.py` | Finite-difference-step / noise-family / gradient-weight sensitivity library |
| `scripts/sensitivity_driver_2026_09_03.py` | Runs the sensitivity study |
| `scripts/two_cstr_lqr_operational_audit_2026_09_04.py` | Auxiliary-LQR-only operational audit across all five two-CSTR scenarios (Table 4) |
| `scripts/two_cstr_failure_penalized_settling_2026_09_04.py` | Failure-penalized settling-time recomputation for the two-CSTR table |
| `scripts/two_cstr_actuator_tv_2026_09_07.py` | Per-actuator input total-variation breakdown (Supplementary Table S7) |
| `scripts/make_revision_visuals_2026_09_06.py` | Read-only: regenerates Figs. 1-4 directly from the archived JSON records, with no re-solving |

Each `*_synthetic_smoke_*.py` / `*_smoke_*.py` script is a fast, synthetic-data
correctness check for its corresponding driver and does not itself produce a
reported number.
