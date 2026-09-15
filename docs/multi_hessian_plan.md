# Multi-Hessian experiment plan

The implemented mathematics is documented in [algorithm.md](algorithm.md).

1. Preserve the first-layer column-space, independent-endpoint sampling, and moment algorithm.
2. Expose a stage boundary so a freshly estimated prefix can be passed to later stages without a historical run ID.
3. Measure one positive-definite anchor and multiple real-query probe Hessians. Save directions, centers, matrices, and exact query counters.
4. Construct random linear combinations and solve symmetric generalized eigenproblems. Read weights from the anchor times the generalized eigenvectors.
5. Select candidates without target parameters, labels from held-out directions, or target-error tuning.
6. Compare individual-probe and random-combination methods over probe counts and independent direction banks.
7. Freeze both hidden layers and fit unconstrained Gaussian ordinary least squares. Keep signed and absolute normalization separate.
8. Save explicit RNG streams, parameter arrays, physical and logical query accounting, reference comparisons, and English scientific reports.

The release uses conda with Python 3.10. Checkpoint and full modes share the same later-stage implementation. The maintained configuration is `configs/reproduction.yaml`; commands are in the root README.
