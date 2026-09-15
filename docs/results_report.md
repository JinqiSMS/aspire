# Current experimental results

The maintained experiment combines first-layer column recovery and ASPIRE moments, second-layer random multi-Hessian recovery, and Gaussian ordinary least squares for the output.

The main target has input dimension 8, hidden widths `[3,3]`, fourth-power activations, and a scalar output. Four historical learned prefixes are included in `reference/`.

## Best fixed-prefix group

`T32_large_seed0` uses 16,777,216 independent first-layer endpoints with 32 steps each and first-layer master seed 0. Its first-layer aligned operator error is approximately 0.01303729.

At 12 probe Hessians and 64 random combinations, five direction banks give a second-layer operator-error median of approximately 0.03510459. Keeping the original anchor output formula gives an output L1-error median of approximately 0.14996437, above the 0.1 target.

Replacing only the output estimate by Gaussian OLS gives 25 crossed combinations: five hidden recoveries and five training sets of 1,048,576 standard Gaussian inputs. The output L1-error median is approximately 0.07112770, with range `[0.05872970, 0.08604529]`. All 25 meet the joint numerical parameter threshold.

The best individual main case has errors:

```text
W1 operator error   0.013037289900738687
W2 operator error   0.030543022627354477
Output L1 error     0.058729695636741186
Coefficient L1 norm 0.991835706458432
```

## Other prefixes

- `T512_seed0`: first-layer error about 0.04102194; main OLS output-error median about 0.20709490; no crossed fit passes the joint threshold.
- `T512_seed1`: first-layer error about 0.11258445 already exceeds the target; main OLS median about 0.43096479.
- `T32_large_seed1`: first-layer error about 0.02977437; main OLS median about 0.15844016; no crossed fit passes.

Increasing the number of probe matrices or lowering the diagonalization residual does not monotonically improve parameter recovery. Prefix-induced model mismatch remains a limiting factor.

## Interpretation

The 25 crossed fits share one first-layer estimate, so they are not 25 independent target-network trials. Main signed normalization has some negative hidden entries. Absolute normalization is reported as a separate modification, not silently substituted based on target error.

The empirical success of the best modified pipeline does not establish stable success of the original algorithm across seeds or certify finite-chain mixing assumptions.

Use the commands in [README.md](../README.md) to regenerate results. See [validation.md](validation.md) for the Python 3.10 checks and [multi_hessian_generalized_report.md](multi_hessian_generalized_report.md) for the comparison protocol.
