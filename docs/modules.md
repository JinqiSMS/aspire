# Module guide

- `run_experiment.py`: command-line entry point and numerical-library thread settings.
- `configs/experiment_k*.yaml`: activation exponent, sample budgets, tolerances, and seeds.
- `data/experiment/`: fixed target arrays and their checksum.
- `src/aspire/experiments/experiment.py`: complete stage orchestration, query accounting, regression, and evaluation.
- `src/aspire/recovery/`: column recovery, Hit-and-Run sampling, ASPIRE moments, generalized eigendecomposition, and OLS.
- `src/aspire/oracles/`: counted real-query interpolation, recovered-prefix coordinate operations, and suffix evaluation.
- `src/aspire/evaluation/`: symmetry alignment and parameter/prediction metrics.
- `src/aspire/reporting/parameters.py`: complete parameter exports and Matplotlib figures.
- `scripts/summarize_activation.py`: checks and plots the three activation experiments.
- `scripts/diagnose_activation_moments.py`: evaluator-only population-moment diagnostics.
- `scripts/check_experiment.py`: parameter, metric, query, and output-file checks.
- `tests/`: algebra, sampling, generalized eigenproblems, information boundaries, and parameter exports.
- `examples/activation_comparison/`: recorded parameter values, metrics, execution metadata, and figures.

Ground truth belongs to target construction and evaluation. Recovery receives function values, public architecture/bounds, and earlier recovered layers. Figure generation reads saved parameter files and uses zero function queries.
