# Module map

## Entry points

- `reproduce.py`: supported checkpoint/full workflow; sets package paths and one-thread numerical environment.
- `configs/reproduction.yaml`: fixed targets, stable stream indices, budgets, sample sizes, and solver settings.
- `experiments/reproduction.py`: first-layer handoff, data generation, provenance, and reference comparisons.
- `run_experiments.py`, `run.py`, `config.py`: general experiment recipes.

## Numerical learner

All module paths below are relative to `src/aspire/`.

- `recovery/subspace.py`: original-real directional derivatives and column-space recovery.
- `recovery/sampling.py`: Hit-and-Run, independent endpoints, and chord bisection.
- `recovery/moments.py`: uncentered moments, spectral directions, normalization, and sampling diagnostics.
- `recovery/network.py`: stage orchestration and explicit completion of a prefix.
- `recovery/hessian_bank.py`: orthogonal probes and measured Hessians.
- `recovery/generalized_joint.py`: generalized eigendecomposition and candidate selection.
- `recovery/output.py`: fixed-hidden features and ordinary least squares.
- `oracles/`: interpolation, prefix inversion, suffix evaluations, and complex simulation where required.
- `query_ledger.py`: query charging at the lowest real-value interface.

## Evaluation and reporting

- `teacher.py`: evaluator-owned targets and analytic diagnostic derivatives.
- `evaluation/`: legal alignment, parameter errors, and prediction risks.
- `reporting/reproduction.py`: English offline HTML, CSV/JSON, and scientific figures.
- `plot.py`: English report for general recipes.
- `reference/`: small frozen targets, prefixes, checksums, and historical metrics.

The `diagnostics/`, `baselines/`, and `instances.py` modules retain additional numerical controls and target families. Optional neural baselines require PyTorch. Configuration availability does not imply that every grid was executed successfully.

Historical scripts and reports are retained in an ignored local archive and are not dependencies of the maintained reproduction workflow.
