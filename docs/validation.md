# Release validation

The local conda environment is named `aspire`, using Python 3.10.21, NumPy 1.26.4, and SciPy 1.13.1. `conda activate aspire` resolves the correct interpreter, and `python -m pip check` reports no broken requirements.

The current test suite passes 56 tests, covering numerical oracles, information boundaries, sampling, checkpoints, generalized recovery, portable paths, frozen input checksums, and reference verification. The verification tests cover cached-result rechecks and rejection of failed, non-finite, unknown, or duplicate main records. The initial release had 51 tests; the five additional cases test verification behavior without changing recovery mathematics.

The best checkpoint preset regenerated Hessians and Gaussian samples and produced 55 records: five hidden recoveries, 25 main OLS fits, and 25 absolute-normalization controls. Its 30 main records match historical metrics with maximum absolute difference about `5.6e-15`. Computation took 36.01 seconds, excluding report generation.

The four-prefix checkpoint study generated 560 records: 360 hidden recoveries, 100 main OLS fits, and 100 absolute-normalization OLS controls. All 460 main records passed the historical comparison. Computation took 118.49 seconds, excluding report generation; exact metrics and physical counts are in [checkpoint_study.json](../reports/validation/checkpoint_study.json).

The small-budget smoke preset exercises first-layer sampling and explicit final-rank failure reporting. It is not a successful-accuracy test. The default checkpoint preset is the quick successful-reconstruction check.

A clean export of the staged Git files passed all 51 tests and regenerated the best checkpoint experiment. Its reference difference remained about `5.6e-15`, demonstrating that the published files do not depend on the local historical data directories. See [clean_checkout.json](../reports/validation/clean_checkout.json).

The offline report's JavaScript filters were executed and verified at 560 total records, 140 records for the selected prefix, and 25 main OLS records. All 560 individual record links and the static report links resolve. Scientific figures were inspected locally.

GitHub Actions also passed on both Ubuntu and Windows for release commit `c17b0384f0830afcd0f9a53e58364172374d19a3`: each job created the Python 3.10 conda environment, checked dependencies, ran the numerical tests, and regenerated the best checkpoint experiment with reference verification. See the [completed CI run](https://github.com/JinqiSMS/aspire/actions/runs/34986355727).

Full first-layer timing is being measured separately. The checkpoint results above do not include fresh first-layer sampling. Timing measurements use the local Intel Core i5-13500H with one numerical-library thread and exclude installation and download time.
