# Darcy Flow Analytics

Physics-informed neural networks replace hours-long finite element loops in Darcy flow inverse problems. This repository holds the full workflow: we sample permeability tensors, solve the governing PDE with FEniCS, train Torch models that learn the mapping, and use Ray Tune plus rich diagnostics to decide when a new surrogate is ready for deployment. The goal is to show future collaborators how we de-risk scientific ML work with real tooling, not one-off notebooks.

## How to Run It

The Taskfile drives every workflow so new contributors can clone, install, and reproduce results without memorizing long commands.

```bash
# 1. Install the package + dependencies in editable mode
task install

# 2. Point CONFIG at a JSON spec (see "Configuration" below) and launch a training run
CONFIG=configs/darcy_primal.json task train

# 3. Reuse that spec to run Ray Tune hyper-parameter search
CONFIG=configs/darcy_primal.json task hp-search

# 4. Optional: run the regression tests against the reference FEniCS solutions
task test
```

Each task delegates to `python main.py <CONFIG> -tasks <train|hp_search>`, so you can always drop to the raw CLI if needed. Outputs are written alongside the JSON spec in `<config_dir>/<output_dir>`, making it easy to keep separate experiment folders in version control or object storage.

## Configuration Blueprint

Every experiment is described by a single JSON document so the same spec can drive data generation, model fitting, and Ray search. A minimal example:

```jsonc
{
  "output_dir": "artifacts/darcy_primal",
  "data_gen_params": {
    "num_data_points": {"training": 200, "validation": 50, "test": 50},
    "A_matrix_params": [[1.0, 10.0], [1.0, 10.0]]
  },
  "formulation_params": {
    "PDE": "Darcy_primal",
    "mesh_descr": "unitSquare10",
    "degree": 2,
    "f": "sin(pi*x[0]) * sin(pi*x[1])"
  },
  "nn_params": {
    "type": "dense_nn",
    "hidden_size_multiplier": 0.75,
    "num_layers": 6,
    "activation": "GeLU"
  },
  "training_params": {
    "epochs": 200,
    "learn_rate": 0.0005,
    "batch_size": 64,
    "losses_to_use": ["PDE", "data"]
  },
  "hp_search": {
    "type": "HyperOptSearch",
    "max_concurrent": 4,
    "num_samples": 40,
    "time_budget_hrs": 2,
    "scheduler": "ASHAScheduler",
    "training_params": {
      "epochs": [100, 250],
      "learn_rate": [1e-5, 1e-3],
      "batch_size": [32, 128]
    },
    "nn_architecture": {
      "nn_types": ["dense_nn"],
      "hidden_size_multipliers": [0.5, 1.0],
      "num_layers": [4, 10],
      "activations": ["GeLU", "ReLU"]
    }
  }
}
```

The parser in `main.py` feeds each block into the proper dataclass (see `src/FEM_solvers/FEM_solver.py`, `src/formulations/formulation.py`, and `src/AI/neural_networks.py`). Anything omitted falls back to sensible defaults, which keeps specs compact while still being fully type checked.

## System Overview

- **Finite Element Ground Truth (`src/FEM_solvers`)** — wraps the Darcy primal/dual formulations from `src/formulations/PDEs/Darcy.py`, solves them with FEniCS, and produces labeled datasets with configurable eigenvalue/eigenvector sampling strategies. All generated tensors are stored as CSVs for reproducibility.
- **Neural Surrogate Factory (`src/AI`)** — builds Torch modules on top of the PDE-specific factories in `src/AI/PDEs`. The abstractions keep Fenics-specific logic isolated while exposing friendly `.fit()`/`.multiple_net_eval()` APIs to the engine.
- **Training & Diagnostics (`src/engine.py`, `src/diagnostic_tools`)** — orchestrates ingestion, caching, model fitting, and evaluation. Loss and parity plots, R²/MSE summaries, boundary condition histograms, and PDE residual diagnostics are automatically emitted for both training and validation sets.
- **Hyper-parameter Search (`src/hp_tuning`)** — powers the `hp_search` task via Ray Tune, HyperOpt, and optional ASHA schedulers. The system persistently retries with fewer concurrent workers when GPUs are scarce and copies the best trial (weights, logs, plots) into `<output_dir>/hp_tuning/best_trial`.
- **Tests (`test/test_Darcy_generator.py`)** — sanity checks that both the primal and dual formulations converge to known analytic solutions as the mesh is refined, which protects us from silent regressions inside the discretization stack.

## Artifact Layout

For both `train` and `hp-search`, the engine writes structured output folders so downstream tools (MLFlow, dashboards, or notebooks) can read artifacts without spelunking:

- `training/` or `hp_tuning/` — root for a specific task, namespaced under `<CONFIG_DIR>/<output_dir>`.
- `training.csv`, `validation.csv`, `test.csv` — cached datasets to avoid regenerating FEniCS solves.
- `nets/` — serialized Torch weights plus the original mesh and NN hyperparameters for later reloads.
- `diagnosis/` — parity plots, PDE residual histograms, and CSV summaries for both train/validation.
- `hp_tuning/best_trial/` — self-contained copy of the winning Ray trial (weights, logs, loss curves).

Everything is plain files, so shipping experiments to cloud storage or sharing across teammates is trivial.

## Development Notes

- **Dependencies**: managed through `pyproject.toml`. `task install` installs the project in editable mode so that `python main.py ...` finds the `src` namespace automatically. Torch, FEniCS, Ray Tune, pandas, numpy, matplotlib, and scikit-learn form the core stack.
- **Style & Testing**: `task test` runs the Fenics regression suite via `pytest`, and you can plug in linters or type-checkers as needed; the project structure follows a standard `src/` layout to keep the interpreter path clean.
- **Extensibility**: add new PDEs by extending `src/formulations`, wiring them into `src/FEM_solvers`, and adding a bespoke NN factory/solver pair under `src/AI/PDEs`. Because everything funnels through the same JSON contracts, existing tasks keep working.
- **Operational focus**: every task writes deterministic artifacts (loss curves, parity plots, CSV summaries, cached datasets) so experiment lineage is auditable and deployment-ready.
- **Scientific rigor**: combining PDE residual losses with data parity, tracking boundary conditions separately, and validating convergence in `test/` prevents silent regressions.
- **Engineering discipline**: Taskfile workflows, type-backed configs, and Ray Tune search spaces ensure teammates can iterate predictably without spelunking through notebooks.

Clone it, point `CONFIG` at your scenario, and you have an end-to-end PDE surrogate lab that future teammates can trust.
