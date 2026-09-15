# Release validation

The local conda environment is named `aspire`, using Python 3.10.21, NumPy 1.26.4, and SciPy 1.13.1. `conda activate aspire` resolves the correct interpreter, and `python -m pip check` reports no broken requirements.

The current test suite passes 56 tests, covering numerical oracles, information boundaries, sampling, checkpoints, generalized recovery, portable paths, frozen input checksums, and reference verification. The verification tests cover cached-result rechecks and rejection of failed, non-finite, unknown, or duplicate main records. The initial release had 51 tests; the five additional cases test verification behavior without changing recovery mathematics.

The best checkpoint preset regenerated Hessians and Gaussian samples and produced 55 records: five hidden recoveries, 25 main OLS fits, and 25 absolute-normalization controls. Its 30 main records match historical metrics with maximum absolute difference about `5.6e-15`. Computation took 36.01 seconds, excluding report generation.

The four-prefix checkpoint study generated 560 records: 360 hidden recoveries, 100 main OLS fits, and 100 absolute-normalization OLS controls. All 460 main records passed the historical comparison. Computation took 118.49 seconds, excluding report generation; exact metrics and physical counts are in [checkpoint_study.json](../reports/validation/checkpoint_study.json).

The small-budget smoke preset exercises first-layer sampling and explicit final-rank failure reporting. It is not a successful-accuracy test. The default checkpoint preset is the quick successful-reconstruction check.

A clean export of the staged Git files passed all 51 tests and regenerated the best checkpoint experiment. Its reference difference remained about `5.6e-15`, demonstrating that the published files do not depend on the local historical data directories. See [clean_checkout.json](../reports/validation/clean_checkout.json).

The offline report's JavaScript filters were executed and verified at 560 total records, 140 records for the selected prefix, and 25 main OLS records. All 560 individual record links and the static report links resolve. Scientific figures were inspected locally.

GitHub Actions also passed on both Ubuntu and Windows for release commit `c17b0384f0830afcd0f9a53e58364172374d19a3`: each job created the Python 3.10 conda environment, checked dependencies, ran the numerical tests, and regenerated the best checkpoint experiment with reference verification. See the [completed CI run](https://github.com/JinqiSMS/aspire/actions/runs/34986355727).

The fresh full best preset completed in 3501.31 seconds (58.36 minutes). Its first layer took 3474.86 seconds (57.91 minutes), using 16,777,216 independent chains with 32 steps each. The remaining stages took 26.41 seconds. The recovered first-layer entries differ from the historical checkpoint by at most `1.1102230246251565e-16`. All 30 main records passed reference verification with maximum metric difference `5.61644619700985e-11`; all 25 main OLS combinations met the parameter threshold. The output L1 error median is `0.07112770206042718`.

Physical costs for this full preset were 31,994,153,088 first-layer queries, 1,950 training-Hessian queries, 900 validation-Hessian queries, 5,242,880 Gaussian labels, and 20,000 prediction-evaluation queries: 31,999,418,818 in total. This includes five direction banks and five Gaussian training sets. It differs from the single-reconstruction learning cost quoted in the README.

The complete compact outputs, reconstructed parameters, stage ledger, and figures are in [full](../reports/validation/full/). See [full_run.json](../reports/validation/full_run.json) for the validation summary and separate computation/verification source hashes. The full job used the initial release's recovery implementation; subsequent changes affected reporting and reference verification. Its results were rechecked using the final verifier.

Timing measurements use the local Intel Core i5-13500H with one numerical-library thread and exclude installation and download time. Checkpoint timings exclude first-layer sampling; full timings include it.
