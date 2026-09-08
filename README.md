# Residual Action-Gradient Consistency for Finite-Budget Learned Lyapunov MPC of Nonlinear Processes

Reproduction code and data for the paper. A learned surrogate placed inside a
finite-budget Lyapunov MPC must be faithful in the **control-gradient** of the
Lyapunov residual (a one-step local derivative relevant to the online solver's
sensitivity to the input, not identical to the solver's actual multi-step
update direction), not only in the predicted residual value. We add a
training term, `L_grad`, that targets this, and evaluate it on two
open-loop-unstable reactor benchmarks (a CSTR and two CSTRs in series).

**This README describes the original submission's code layout (tag
`jpc-submission-2026-07-01`).** For the major-revision content added since
then (equal-query ablation, actual-solver-iterate audit, true-model/NLP
reference, sensitivity studies, and the two-CSTR LQR operational audit), see
`README_REPRODUCTION.md`.

## Requirements

- Python 3.11+ (results reported on 3.14)
- See `requirements.txt` (NumPy, SciPy, PyTorch, Matplotlib)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout

```
src/cstr/          plant simulators (single CSTR and two-CSTRs-in-series)
scripts/           experiments (produce JSON logs) and figure/table generators
data/processed/    single-CSTR one-step dataset (two-CSTR data is generated on the fly)
results/
  interim/logs/    experiment outputs (JSON) — figure scripts read these
  figures/paper/   generated figures (PDF/PNG) and LaTeX table fragments
```

## Two ways to reproduce

**A. Regenerate the figures/tables from the included logs (fast, no training).**
The `results/interim/logs/*.json` files are shipped, so the figure scripts run in
seconds:

```bash
python scripts/make_overview_figure.py      # Fig 1
python scripts/make_paper_figures.py        # Fig 2, Fig 3 (CSTR)
python scripts/make_control_figures.py      # Fig 4 (CSTR phase portrait)  [trains models]
python scripts/make_two_cstr_figures.py     # Fig 5, Fig 9; Tables 2, 3, 5
python scripts/make_jac_combined.py         # Fig 6 (Jacobian faithfulness)
python scripts/make_movesup_figure.py       # Fig 7 (move-suppression)
python scripts/make_roa_combined.py         # Fig 8 (region of attraction)  [trains models]
```

**B. Re-run the experiments from scratch (produces the JSON logs).**
Each script writes to `results/interim/logs/`. Seed counts match the paper
(main two-CSTR studies use 10 seeds; single-CSTR/supporting studies use 3).

```bash
# single CSTR
python scripts/lgrad_ablation.py            # CSTR ablation + dose sweep (Fig 2, Fig 3)
python scripts/reviewer_supplement_cstr.py  # offline-label + smooth-surrogate checks (Table 1)
python scripts/cstr_jac_faithful.py         # CSTR Jacobian faithfulness (Fig 6 top row)

# two CSTRs in series
python scripts/two_cstr_lgrad.py            # main ablation
python scripts/two_cstr_ggn.py V            # Gauss-Newton, rolled-out objective
python scripts/two_cstr_ggn.py residual     # Gauss-Newton, exact-residual objective
python scripts/two_cstr_dose.py             # lambda_grad dose-response
python scripts/two_cstr_budget.py           # budget sweep (B = 3..100)
python scripts/two_cstr_jac_faithful.py     # Jacobian faithfulness
python scripts/two_cstr_metrics.py          # closed-loop control metrics
python scripts/two_cstr_realism.py          # disturbance / plant-model mismatch
python scripts/two_cstr_tracking.py         # setpoint tracking
python scripts/two_cstr_roa.py              # region of attraction (two-CSTR grid)
python scripts/two_cstr_move_suppression.py # input-rate (move-suppression) sweep
```

## Paper figure / table map

| Item | Produced by | Reads / writes |
|------|-------------|----------------|
| Fig 1 (overview) | `make_overview_figure.py` | schematic, no data |
| Fig 2 (value≠gradient, CSTR) | `make_paper_figures.py` | `lgrad_ablation.json` |
| Fig 3 (λ_grad dose, CSTR) | `make_paper_figures.py` | `lgrad_ablation.json` |
| Fig 4 (CSTR phase portrait) | `make_control_figures.py` | trains CSTR models |
| Fig 5 (two-CSTR main, 4 panels) | `make_two_cstr_figures.py` | `two_cstr_lgrad.json`, `ggn_mpc_two_cstr_V.json`, `two_cstr_dose.json`, `two_cstr_jac_faithful.json` |
| Fig 6 (Jacobian faithfulness) | `make_jac_combined.py` | `cstr_jac_faithful.json`, `two_cstr_jac_faithful.json` |
| Fig 7 (move-suppression) | `make_movesup_figure.py` | `two_cstr_move_suppression.json` |
| Fig 8 (region of attraction) | `make_roa_combined.py` | `two_cstr_roa.json` (+ computes CSTR grid) |
| Fig 9 (process-operation robustness) | `make_two_cstr_figures.py` | `two_cstr_realism.json` |
| Table 1 (offline-label / smooth checks) | `reviewer_supplement_cstr.py` | `reviewer_supplement_cstr.json` |
| Table 2 (two-CSTR ablation) | `make_two_cstr_figures.py` | `two_cstr_lgrad.json`, `ggn_mpc_two_cstr_V.json` |
| Table 3 (Gauss-Newton, both objectives) | `make_two_cstr_figures.py` | `ggn_mpc_two_cstr_{V,residual}.json` |
| Table 4 (control metrics) | `two_cstr_metrics.py` | `two_cstr_metrics.json` |
| Table 5 (process operation) | `make_two_cstr_figures.py` | `two_cstr_realism.json`, `two_cstr_tracking.json` |

`make_two_cstr_figures.py` also writes the LaTeX table fragments
(`results/figures/paper/table_two_cstr*.tex`).

## Core modules

- `src/cstr/two_cstr_series.py` — two-CSTRs-in-series simulator (RK4, analytic Jacobian).
- `scripts/two_cstr_lgrad.py` — two-CSTR surrogate, `L_grad` training, and finite-budget MPC.
- `scripts/lgrad_experiment.py`, `budget_sweep_solver.py`, `probe_action_gradient.py`,
  `ggn_mpc_probe.py` — single-CSTR surrogate, training, first-order and Gauss-Newton MPC.

## Notes

- Runtimes are modest (CPU): each experiment is minutes; the 10-seed two-CSTR
  studies are the longest (~10–20 min each).
- `scripts/two_cstr_ipopt_baseline.py` (optional, needs CasADi) is an NLP stress
  check outside the main protocol. `scripts/two_cstr_constraint.py` explores a soft
  state-constraint penalty discussed as a scope boundary.

## Citation

```
Jangwoo Park, "Residual Action-Gradient Consistency for Finite-Budget Learned
Lyapunov Model Predictive Control of Nonlinear Processes," Journal of Process
Control (under review), 2026.
```

## License

Released under the MIT License (see `LICENSE`).
