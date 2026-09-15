# Activation exponent experiments

The network has architecture 8-3-3-1, with $z\mapsto z^k$ in both hidden layers. The target arrays, stage seeds, and numerical settings are fixed across $k=4,6,8$. Only the activation exponent changes.

The first layer uses column-space recovery plus ASPIRE. The second uses multiple Hessians and the paper's normalization

$$
\widehat w_{2,j}=\frac{|V_{:,j}|}{\mathbf 1^\top |V_{:,j}|},\qquad V=H_0C.
$$

This guarantees nonnegative entries and unit column sums for nonzero directions. Output coefficients are fitted by unconstrained ordinary least squares on standard Gaussian inputs.

## Recorded parameters

The following values are the first-layer operator error, second-layer operator error, and output coefficient L1 error, after hidden-unit permutation and first-layer sign alignment:

- $k=4$: **0.01303729, 0.02598736, 0.06898544**.
- $k=6$: **0.01271765, 0.02581270, 0.06030532**.
- $k=8$: **0.00627429, 0.01296156, 0.04586132**.

All three errors are below the configured threshold of 0.1 in each case. The true output coefficients are $a=(1/3,1/3,1/3)$; aligned estimates are

$$
\widehat a_{k=4}=(0.3586153112,\;0.3489443728,\;0.3052409090),
$$

$$
\widehat a_{k=6}=(0.3511328843,\;0.3700398663,\;0.3391325729),
$$

$$
\widehat a_{k=8}=(0.3302798380,\;0.3125748956,\;0.3553827164).
$$

Their L1 norms are 1.01280059, 1.06030532, and 0.99823745, respectively. These norms are distinct from recovery errors. The unconstrained output fit need not sum to one. All recorded second-layer entries are positive; their minima are 0.00769703, 0.01480840, and 0.00249922.

Complete weights, figures, and execution records are in [the recorded artifacts](../examples/activation_comparison/README.md). The [README](../README.md) gives the full commands for each configuration, output-file descriptions, query costs, and runtime estimates.

## Interpretation

These experiments demonstrate recovery for the fixed target and stage seeds. They do not establish a monotone improvement with exponent or an across-seed success rate.

The measured ASPIRE relative moment gaps are approximately 0.02705, 0.01806, and 0.01181. The scaled regression design condition numbers are approximately 1.006, 1.005, and 1.014. Gaussian polynomial labels are highly concentrated: the largest single squared label accounts for approximately 71.50%, 99.378%, and 99.9945% of training-label energy for the three exponents. Sensitivity to other sample streams remains unmeasured.

Optional population-moment diagnostics can be generated with `python scripts/diagnose_activation_moments.py`. They use analytic ground truth only for evaluation and do not supply derivatives or parameters to the learner.
