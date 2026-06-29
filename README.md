# Residual Action-Gradient Consistency for Finite-Budget Learned Lyapunov MPC

Code and data to reproduce the experiments and figures in:

> J. Park, *Residual Action-Gradient Consistency for Finite-Budget Learned
> Lyapunov Model Predictive Control.*

The paper studies learned dynamics surrogates used inside finite-budget
Lyapunov-based MPC. It shows that **residual-value consistency** (matching the
predicted Lyapunov decrease) is not enough for a finite-budget solver, which
follows the **control-gradient** of the Lyapunov residual; a minimal training
term, `L_grad`, that matches this gradient to offline labels restores
closed-loop stabilization on two open-loop-unstable plants (a CSTR and a
cart-pole).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12+, PyTorch 2.10, NumPy 1.26 (CPU is sufficient; all
models are small).

## Layout

```
scripts/    experiment, solver, and figure-generation code (entry points below)
src/cstr/   CSTR simulator, Sontag auxiliary controller, one-step model utilities
data/       processed CSTR one-step dataset (20k pairs)
results/
  interim/logs/   JSON logs that back the paper figures and tables
  figures/paper/  rendered figures
```

## Reproducing the paper

Run from the repository root. Figures that read pre-computed logs are fast
and deterministic; figures and tables marked *(trains)* retrain the small
surrogates (a few minutes on CPU, fixed seeds).

### Figures

| Figure | Script | Notes |
|---|---|---|
| Overview schematic | `python scripts/make_block_diagram.py` | |
| Value/gradient decoupling, dose-response, Gauss-Newton diagnostics, scope | `python scripts/make_paper_figures.py` | from logs |
| Lyapunov convergence, CSTR/cart-pole trajectories | `python scripts/make_control_figures.py` | *(trains)* |
| Region of attraction | `python scripts/make_roa_figure.py` | *(trains)* |

### Tables

| Table | Script |
|---|---|
| Closed-loop control metrics | `python scripts/control_metrics.py` *(trains)* |
| Offline perturbation labels, smooth surrogate, cost | `python scripts/reviewer_supplement_cstr.py` *(trains)* |

### Regenerating the logs (optional)

The figure logs under `results/interim/logs/` are included so the figures can
be reproduced without retraining. To regenerate them from scratch:

```bash
python scripts/lgrad_ablation.py       # lgrad_ablation.json (decoupling, dose-response)
python scripts/ggn_mpc_probe.py        # ggn_mpc_probe_{V,residual}.json (CSTR GGN)
python scripts/ggn_mpc_cartpole.py     # ggn_mpc_cartpole_{V,residual}.json (cart-pole GGN)
python scripts/cartpole_campaign.py    # cartpole_campaign.json (multi-seed cart-pole)
python scripts/cartpole_dagger.py      # cartpole_dagger.json (on-policy refinement)
```

### Regenerating the dataset (optional)

```bash
python scripts/generate_lcnn_paper_cstr_data.py
```

## Notes

- The CSTR benchmark and Lipschitz-constrained network follow the
  process-control / robust learned-MPC literature; see the paper for references.
- "Finite budget" refers to a fixed small number of online solver iterations
  per control step; reported per-step timing is indicative, unoptimized PyTorch.

## License

Code and data are released under the MIT License (see `LICENSE`).
