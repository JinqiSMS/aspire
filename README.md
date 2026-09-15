# ASPIRE

## Experiment 1: layerwise parameter recovery

This experiment recovers an **8 → 3 → 3 → 1** polynomial network with activation $z\mapsto z^k$, for **$k=4,6,8$**, using noiseless real-valued function queries.

The first hidden layer uses column-space recovery and ASPIRE gradient moments. The second hidden layer uses multiple Hessians and symmetric generalized eigendecomposition, followed by coordinatewise absolute values and unit-column-sum normalization. The output coefficients are fitted by ordinary least squares on standard Gaussian inputs.

## Installation

```bash
git clone https://github.com/JinqiSMS/aspire.git
cd aspire
conda env create -f environment.yml
conda activate aspire
```

The conda environment is named `aspire` and uses **Python 3.10**. Dependencies are pinned in `requirements-lock.txt`, including NumPy 1.26.4 and SciPy 1.13.1. Computation runs on a CPU.

For an existing environment:

```bash
conda activate aspire
python -c "import sys; assert sys.version_info[:2] == (3, 10)"
python -m pip install -r requirements-lock.txt
```

## Run a configuration

Run commands from the repository root, using new output directories. On a fresh checkout, each command computes all three parameter blocks from function queries:

```bash
# Fourth-power activation
python run_experiment.py --config configs/experiment_01.yaml --output results/activation_comparison/k4

# Sixth-power activation
python run_experiment.py --config configs/experiment_01_k6.yaml --output results/activation_comparison/k6

# Eighth-power activation
python run_experiment.py --config configs/experiment_01_k8.yaml --output results/activation_comparison/k8
```

Run only the command for the exponent you need. The three configurations have identical target weights, numerical hyperparameters, and random seeds; only `architecture.k` changes.

Use an empty or new output directory for a full run. For another full execution, use a new path such as `--output results/run2/k6`.

Without arguments, `python run_experiment.py` runs `configs/experiment_01.yaml` and writes to `results/experiment_01/`.

After completing all three configurations, generate the comparison report and figures:

```bash
python scripts/summarize_activation.py --output results/activation_comparison
```

## Configuration and random seeds

The complete settings are in [experiment_01.yaml](configs/experiment_01.yaml), [experiment_01_k6.yaml](configs/experiment_01_k6.yaml), and [experiment_01_k8.yaml](configs/experiment_01_k8.yaml).

- First layer: **16,777,216 independent chains, 32 transitions per chain**, batch size 4,096, and 16 column-space probes; seed `3141116543`.
- Second layer: one anchor Hessian, 12 probe Hessians, and 64 random mixtures; direction seed `431061063`, mixture seed `2356887966`.
- Held-out Hessians: six probes; seed `2828169828`.
- Output regression: **1,048,576** standard Gaussian inputs; seed `1586144878`.
- Prediction evaluation: 20,000 angular directions; seed `1444349591`.

Each stage uses its own `numpy.random.default_rng(seed)`. The entry point fixes numerical-library thread counts to one. Batch size is part of the first-layer random-number layout. The target arrays in `data/experiment_01/ground_truth.npz` are checked by SHA-256. Floating-point values can vary slightly between numerical-library builds.

## View one experiment's results

The program prints the ground-truth weights, aligned estimates, and their differences in the terminal. For example, the sixth-power command writes:

```text
results/activation_comparison/k6/
  weights.txt               # Readable W1, W2, a, ground truth, and differences
  weights.csv               # One row per parameter coordinate
  weights.json              # Ground truth, raw estimates, aligned estimates
  weights.npz               # Full-precision parameter arrays
  metrics.json              # Parameter errors, success flag, diagnostics
  queries.json              # Stage query counts and totals
  run.json                  # Configuration, source hash, environment, timing
  first_layer.npz           # First-layer estimate
  first_layer.json          # First-layer diagnostics and query counts
  figures/
    weight_comparison.png   # Ground truth, estimates, and differences
    parameter_errors.png    # First layer, second layer, and output errors
    output_coefficients.png # Output coefficients versus ground truth
```

Every figure is also saved as a PDF. Open `weights.txt` for numerical comparisons, `metrics.json` for accuracy, and the PNG/PDF files for figures. `parameter_success` is true when both hidden-layer operator errors and the output coefficient L1 error are at most 0.1. `status: complete` means that execution finished; it is distinct from this accuracy criterion.

The parameter comparison aligns hidden-unit permutations and first-layer signs. Both raw and aligned arrays are preserved. To inspect them in Python:

```python
import numpy as np

with np.load("results/activation_comparison/k6/weights.npz") as weights:
    for name in ("W1", "W2", "a"):
        print(name, "ground truth:", weights[f"{name}_true"])
        print(name, "recovered:", weights[f"{name}_aligned"])
        print(name, "difference:", weights[f"{name}_difference"])
```

To redraw an existing experiment's figures without running the algorithm:

```bash
python run_experiment.py --figures-only --output results/activation_comparison/k6
```

## View the activation comparison

The summary command writes `report.md`, `summary.csv`, `summary.json`, and three comparison figures under `results/activation_comparison/figures/`: `activation_errors`, `activation_coefficients`, and `activation_diagnostics`, each as PNG and PDF.

Recorded results for the fixed target and stage seeds are listed below as **first-layer operator error, second-layer operator error, output coefficient L1 error**:

- **$k=4$:** 0.01303729, 0.02598736, 0.06898544.
- **$k=6$:** 0.01271765, 0.02581270, 0.06030532.
- **$k=8$:** 0.00627429, 0.01296156, 0.04586132.

Recorded artifacts and their stage-level execution metadata are in [examples/activation_comparison](examples/activation_comparison/README.md). Open the complete weights for [k=4](examples/activation_comparison/k4/weights.txt), [k=6](examples/activation_comparison/k6/weights.txt), or [k=8](examples/activation_comparison/k8/weights.txt). These are fixed-instance results, not estimates of an across-seed success rate.

![Activation parameter errors](examples/activation_comparison/figures/activation_errors.png)

See [docs/activation_experiment.md](docs/activation_experiment.md) for output coefficient values and [docs/algorithm.md](docs/algorithm.md) for the mathematical operations.

## Runtime and query costs

Allow approximately **one hour per full configuration** on an Intel Core i5-13500H with Python 3.10.21 and one numerical-library thread, or approximately three hours when running all three sequentially. Previous first-layer measurements were approximately 55–58 minutes. Timing depends on hardware and system load; allow several GiB of free RAM. Environment installation is not included.

Complete learning-query totals, including the first layer, training Hessians, and Gaussian regression labels, are:

- $k=4$: **31,995,202,054**; including held-out Hessians and prediction evaluation: **31,995,222,234**.
- $k=6$: **33,001,837,730**; including held-out Hessians and prediction evaluation: **33,001,857,982**.
- $k=8$: **33,337,385,790**; including held-out Hessians and prediction evaluation: **33,337,406,114**.

The network degree is $k^2$. First-layer gradient interpolation uses $k^2+1$ nodes per direction, and suffix-Hessian interpolation uses $k+1$. Degree-dependent node counts and radius bounds explain the different query costs. Exact stage counts are saved in `queries.json`.

## Checks and source layout

```bash
python -m pytest tests -q --basetemp tmp/pytest
python scripts/check_experiment.py results/activation_comparison/k6
```

The second command validates parameter files, symmetry alignment, nonnegative second-layer normalization, metrics, query accounting, and figure files. GitHub Actions runs numerical tests covering all three exponents and checks the recorded artifacts on Linux and Windows.

See [docs/modules.md](docs/modules.md) for the code map. Runtime outputs, caches, local archives, and environments are excluded from Git. Source code, comments, configuration, documentation, and output labels use English.
