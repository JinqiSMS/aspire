# Recorded activation experiments

This directory contains the recorded weights, ground truth, metrics, and Matplotlib figures for the fixed 8-3-3-1 network at activation exponents 4, 6, and 8.

- [Comparison report](report.md) and [parameter-error figure](figures/activation_errors.png).
- Complete weight values: [k=4](k4/weights.txt), [k=6](k6/weights.txt), [k=8](k8/weights.txt).
- Each exponent's directory also contains CSV, JSON, and NPZ parameter files, numerical diagnostics, query counts, and PNG/PDF figures.

The recorded results were assembled in stages: each exponent's first-layer estimate came from a completed sampling run, and the Hessian and output-regression stages were rerun after correcting the second-layer normalization. The original `run.json`, `first_layer.json`, and `queries.json` records are retained. Their historical source paths identify the original execution; the full experiment commands in the repository README do not require those paths. Original first-layer costs are distinguished from queries made during the later-stage executions.

For new complete runs, use the configuration commands in the [repository README](../../README.md). This directory is a record of measured results, not an input to the learning algorithm.
