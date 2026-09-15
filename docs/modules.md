# Module guide

- `run_experiment.py`: command-line entry point and numerical-library thread settings.
- `configs/experiment_01.yaml`: network dimensions, sampling budgets, numerical tolerances, and seeds.
- `data/experiment_01/`: fixed target parameters, optional first-layer checkpoint, and checksums.
- `src/aspire/experiments/experiment_01.py`: stage orchestration, query accounting, regression, and evaluation.
- `src/aspire/recovery/`: column recovery, Hit-and-Run sampling, ASPIRE moments, generalized eigendecomposition, and OLS.
- `src/aspire/oracles/`: counted real-query interpolation, recovered-prefix coordinate operations, and suffix evaluation.
- `src/aspire/evaluation/`: symmetry alignment and parameter/prediction metrics.
- `src/aspire/reporting/parameters.py`: parameter exports and three Matplotlib figures.
- `tests/`: algebra, sampling, generalized eigenproblems, information boundaries, and parameter exports.
- `scripts/check_experiment.py`: numerical and output-file checks for the configured experiment.
- `examples/experiment_01/`: parameter values, metrics, query counts, and figures from the documented command with the stored first layer.

Ground truth belongs to target construction and evaluation. Recovery receives function values, public architecture/bounds, and already recovered layers. Figure generation reads saved parameter files and uses zero function queries.
