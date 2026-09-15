# ASPIRE: reproducible numerical experiments

ASPIRE studies layerwise parameter recovery for homogeneous polynomial networks using real-valued black-box queries. The main experimental pipeline is:

1. **First hidden layer:** column-space recovery, independent Hit-and-Run endpoints, gradient moments, and ASPIRE spectral recovery.
2. **Second hidden layer:** measured Hessians, random linear combinations, and symmetric generalized eigendecomposition.
3. **Output layer:** ordinary least squares on standard Gaussian inputs, keeping both recovered hidden layers fixed.

Multi-Hessian recovery and Gaussian output regression modify the original algorithm. The best fixed-prefix setting succeeds in 25 crossed repetitions; the other historical prefixes do not. These are empirical results, not a certification of the theorem or a claim that every seed succeeds. Signed normalization can produce negative hidden weights; an absolute-normalization control is reported separately.

## Install with conda

Create the **`aspire` environment with Python 3.10**:

```bash
git clone https://github.com/JinqiSMS/aspire.git
cd aspire
conda env create --file environment.yml
conda activate aspire
python --version
python -m pip check
```

The tested interpreter is Python 3.10.21. `environment.yml` selects Python 3.10 and installs the pinned dependencies in `requirements-lock.txt`. The main numerical versions are NumPy 1.26.4 and SciPy 1.13.1. No GPU or PyTorch is needed for the main pipeline; optional neural baselines need a separate PyTorch installation.

If `aspire` already exists, activate it, verify the Python version, and install the lock file:

```bash
conda activate aspire
python -c "import sys; assert sys.version_info[:2] == (3, 10)"
python -m pip install -r requirements-lock.txt
```

Run the commands below from the repository root. The entry point configures the package path and sets OpenBLAS, MKL, and OpenMP to one thread. No virtualenv or editable installation is required.

## Reproduce the best setting

The default route loads a small checked-in first-layer checkpoint and **regenerates the Hessian observations and Gaussian training data** before recomputing the later layers:

```bash
python reproduce.py --preset best --mode checkpoint --verify-reference
```

This produces five hidden recoveries, 25 main OLS fits, and 25 absolute-normalization OLS controls. No external dataset download or historical result directory is needed. Outputs go to `results/reproduction_best_checkpoint/`.

`--verify-reference` compares the 30 main records against historical metrics with absolute tolerance `1e-7`. The 25 absolute-normalization controls are separate. The initial Python 3.10 comparison differed by at most about `5.6e-15`.

When saved results already exist, `--verify-reference` rechecks their metrics without rerunning oracle calls. Failed main records and non-finite metrics fail verification. Use a new `--output` directory to regenerate the experiment itself.

## Recompute the first layer as well

```bash
python reproduce.py --preset best --mode full --verify-reference
```

Full mode loads only the fixed **target network**, not the estimated first-layer checkpoint. It runs column recovery and all chains, saves the new first-layer estimate, and passes it directly to multi-Hessian recovery and OLS.

The best setting uses **16,777,216 independent chains, exactly 32 steps per chain**, and 1,048,576 Gaussian samples per training repetition. This expensive command writes to `results/reproduction_best_full/`.

There is no checkpoint inside a sampling batch: an interrupted first layer restarts that layer. Once the layer completes, its saved estimate can be reused on a retry in the same output directory. To measure fresh computation, choose a new `--output` directory. Physical current-invocation costs are recorded separately from inherited costs.

## Reproduce the four-prefix study

```bash
python reproduce.py --preset study --mode checkpoint --verify-reference
```

The study compares a single Hessian pair, the best individual probe, one mixture, and 64 mixtures at `M = 3, 6, 12, 24`. It also includes 8- and 32-candidate controls at `M=12`, five direction banks per prefix, and five Gaussian training repetitions crossed with the five primary hidden recoveries.

Outputs contain **360 hidden-layer records, 100 main OLS records, and 100 absolute-normalization OLS controls**. Reference verification covers the first 460 records. All parents have explicit stable random-stream indices, independent of their positions in a filtered list.

Use `--preset study --mode full` to recompute all four first layers. This is much more expensive than checkpoint mode.

## View results

Open this file directly in a browser:

```text
results/reproduction_best_checkpoint/report/index.html
```

On Windows PowerShell:

```powershell
Start-Process ./results/reproduction_best_checkpoint/report/index.html
```

The report works offline, has run filters, links to individual records, and scientific figures in PNG/PDF/SVG.

```text
results/<run>/
  manifest.json                  # Settings, environment, provenance, physical queries
  records.json                   # All recovery and regression records
  prefixes/<parent>/             # First-layer parameters, configuration, query counts
  data/                          # Generated Hessians and Gaussian samples
  runs/<case>/
    metrics.json                 # Parameter errors and diagnostics
    parameters.npz               # W1, W2, a_raw
    decomposition.npz            # Generalized directions and candidates, when applicable
  report/
    index.html                   # Offline interactive report
    results.md                   # English summary
    records.csv                  # Flat records
    summary.json                 # Group medians and ranges
    *.png / *.pdf / *.svg        # Scientific figures
```

Rebuild figures without oracle calls:

```bash
python reproduce.py --report-only --output results/reproduction_best_checkpoint
```

Selected release outputs are included in [reports/reproduction](reports/reproduction/). The independently recomputed full run is in [reports/validation/full](reports/validation/full/); open its [offline report](reports/validation/full/report/index.html) after cloning. See [docs/algorithm.md](docs/algorithm.md) for the mathematics and [docs/modules.md](docs/modules.md) for the module map.

## Expected numerical results

The best historical main case uses first-layer seed `0`, direction repeat `1`, `M=12`, 64 candidates, and Gaussian training repeat `4`:

```text
First-layer aligned operator error     0.013037289900738687
Second-layer aligned operator error    0.030543022627354477
Output coefficient L1 error            0.058729695636741186
Output coefficient L1 norm             0.991835706458432
Joint parameter error                 0.058729695636741186
```

The output error and coefficient norm are different quantities. Across the 25 main OLS combinations for this prefix, the output L1 error has median `0.07112770` and range `[0.05872970, 0.08604529]`; all meet threshold `0.1`. They share one first-layer estimate and are not 25 independent full-network recoveries.

A fresh Python 3.10 full run reproduced these results: all 25 main OLS combinations passed, the first-layer entries differed from the historical checkpoint by at most `1.1e-16`, and the 30 compared main records differed by at most `5.7e-11`. See [full_run.json](reports/validation/full_run.json) for the measured values and query ledger.

## Randomness and settings

All settings are in [configs/reproduction.yaml](configs/reproduction.yaml). The target has input dimension 8, hidden widths `[3,3]`, activation `z -> z**4`, scalar output, and total polynomial degree 16.

The best first-layer master seed is `0`, and its algorithm sub-seed is `3141116543`. Historical seed manifests and input hashes are in [reference/manifest.json](reference/manifest.json).

Later stages use independent generators:

```python
def stream_seed(entropy, *parts):
    return int(np.random.SeedSequence([entropy, *parts]).generate_state(1)[0])

# parent_index is the explicit stream_index, not a list position.
direction_seed = stream_seed(2026091521, parent_index, direction_repeat, 0)
heldout_seed = stream_seed(2026091521, parent_index, direction_repeat, 1)
combination_seed = stream_seed(2026091522, parent_index, direction_repeat, M)
evaluation_seed = stream_seed(2026091523, parent_index)
gaussian_seed = stream_seed(2026091517, training_repeat)
```

The best parent's `stream_index` is **2**. Direction and Gaussian repetitions both use `0,1,2,3,4`. A call to `np.random.seed(0)` does not control these separate `default_rng` generators. Batch size is also part of the first-layer RNG layout and must remain fixed for the historical comparison.

Source hashes normalize relative path separators and line endings. Frozen arrays have SHA-256 checksums. Floating-point routines may differ slightly across operating systems and BLAS builds, so reference verification uses aligned numerical metrics with a tolerance.

## Query counts and runtime

One complete best-setting recovery, with one primary Hessian recovery and one Gaussian training set, costs:

```text
Column-space recovery                         2,176
Hit-and-Run membership queries        31,138,512,896
Gradient-moment queries                  855,638,016
First-layer total                    31,994,153,088
13 measured Hessians                            390
Gaussian labels                           1,048,576
Single-reconstruction learning total 31,995,202,054
```

The presets run multiple crossed repetitions, so their physical totals differ. Shared data is counted once in `manifest.json`. Do not sum inherited logical per-case costs across records. Held-out Hessians and prediction evaluation are separate.

On an Intel Core i5-13500H with one BLAS thread, Python 3.10 checkpoint computation took **28-36 seconds**, plus about 5-7 seconds for the report. The fresh full best preset took **58.36 minutes**: 57.91 minutes for the first layer and 26.41 seconds for the later stages. The four-prefix checkpoint study took 118.49 seconds before report generation. These are local wall-clock measurements, not guaranteed runtimes; installation and downloads are excluded. See [docs/validation.md](docs/validation.md) for the verification records.

Checkpoint mode generates several hundred MiB of data locally; study mode shares Gaussian inputs and needs roughly 400 MiB plus outputs. Full first-layer sampling holds large arrays in memory, so allow several GiB of free RAM. Environments and generated datasets are excluded from Git.

## Tests and development

```bash
python -m pytest tests -q
python reproduce.py --preset smoke
```

The small-budget smoke preset checks real-query execution and numerical-failure reporting. It is not an accuracy reproduction and can report an unresolved final Hessian rank. Use the default checkpoint preset for a successful complete reconstruction check.

The general driver retains other experiment recipes:

```bash
python run_experiments.py --config configs/strict_smoke.yaml --dry-run
python run_experiments.py --config configs/strict_smoke.yaml --no-plots
```

`configs/reproduction.yaml` belongs to `reproduce.py`, not the general driver. A configuration's existence is not evidence that its entire grid passed.

Public code, comments, configuration keys, documentation, and generated report labels use English. Historical local archives and scratch outputs are not runtime dependencies and are excluded from this release.
