# Experiment design and scope

Installation and maintained commands are in [README.md](README.md).

## Information boundary

The learner receives only real-valued function evaluations, public bounds, architecture, and estimated earlier layers. Target weights and analytic derivatives belong to the evaluator. Recovery and oracle modules do not import target or evaluation modules.

## Stages

1. Recover a first-layer column-space basis using degree-16 real directional interpolation.
2. Sample the reduced sublevel body with independent Hit-and-Run chains; the best setting uses 16,777,216 endpoints and 32 steps each.
3. Estimate uncentered position and gradient moments, apply ASPIRE spectral recovery, and map directions back to the input coordinates.
4. Construct a suffix oracle from the estimated first layer and measure multiple Hessians through real queries.
5. Recover common directions with random symmetric generalized eigenproblems. Select by training residual and eigengap; no target or held-out error is used for selection.
6. Freeze hidden layers and fit the output using unconstrained, unweighted Gaussian ordinary least squares.

## Controls

Four historical prefixes, four Hessian probe counts, five direction banks, and five Gaussian training repetitions are retained. Compare a single pair, best single probe, one mixture, and 64 mixtures. Additional 8/32-candidate controls are included at the primary count. Signed and absolute normalization are reported separately.

Record held-out residuals, commutators, conditioning, rank failures, negative entries, aligned errors, and physical versus inherited query costs. Twenty-five crossed fits share one prefix and are not independent target networks.

## Reproducibility

Use the `aspire` conda environment with Python 3.10. Checkpoint mode regenerates later data from small frozen targets and prefixes. Full mode recomputes the first layer and hands it directly to the same later stages. Changed settings require a new output directory. Reports are offline with PNG/PDF/SVG exports.

## Limits

Numerical threshold success is empirical. Unknown finite-chain mixing constants and negative signed entries prevent interpreting it as a verification of every theorem assumption. Scalar-tail reductions, shallow base cases, and the full theoretical large-scale grid are outside the primary release target.
