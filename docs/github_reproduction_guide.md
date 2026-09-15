# Reproduction and publication

The maintained installation procedure and commands are in [README.md](../README.md).

The release includes source, entry points, conda configuration, pinned dependencies, experiment configurations, tests, English documentation, frozen instances, and the small `reference/` directory. It requires no sampled Gaussian data, old Hessian banks, or historical run folders.

```bash
conda env create --file environment.yml
conda activate aspire
python -m pytest tests -q
python reproduce.py --preset best --mode checkpoint --verify-reference
python reproduce.py --preset study --mode checkpoint --verify-reference
python reproduce.py --preset best --mode full --verify-reference
```

Checkpoint mode reuses the learned first layer and regenerates all later data. Full mode recomputes all estimated parameters from real queries to the fixed target. Use new output directories when measuring fresh executions; completed runs otherwise use their saved results.

The environment, local archives, caches, generated datasets, and full working result directories are excluded from Git. Selected release reports and verification records are included under `reports/`. Official commands write their own outputs under `results/`.

Code and documentation use English and LF text. Frozen instance JSON retains its pinned bytes. Source fingerprints normalize paths and line endings. Frozen numerical inputs have SHA-256 checksums.

Open `results/reproduction_best_checkpoint/report/index.html` directly in a browser. Inspect `records.json`, individual `parameters.npz` files, and `manifest.json` for metrics, parameters, configuration, timing, and physical query costs.
