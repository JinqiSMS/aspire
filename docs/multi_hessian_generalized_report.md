# Multi-Hessian recovery and Gaussian output regression

## Protocol

The experiment uses fixed target networks with dimension 8, hidden widths `[3,3]`, fourth-power activation, and no oracle noise. Four first-layer estimates are supplied for checkpoint comparisons; full mode recomputes them from real queries.

The first-layer settings are 1,048,576 endpoints with 512 steps for seeds 0 and 1, and 16,777,216 endpoints with 32 steps for seeds 0 and 1. Every endpoint comes from its own chain.

The second layer uses an anchor at the all-ones vector, perturbation scale 0.2, and Hessian interpolation radius 0.1. The probe counts are 3, 6, 12, and 24. There are five direction banks for each prefix and six held-out Hessians per bank.

The main solver uses 64 normalized Gaussian linear combinations and a symmetric generalized eigenproblem. It reads weight directions from $H_0C$, normalizes columns, and selects candidates using training residuals and eigengaps. Held-out matrices and target parameters are excluded from selection.

Controls include one pair, the best individual probe, one random mixture, and 8/32-candidate variants at the primary probe count 12. Gaussian output regression uses 1,048,576 independent standard Gaussian inputs per repetition, with five training repetitions crossed with five recovered hidden layers.

## Results

Only the most accurate first-layer prefix, `T32_large_seed0`, passes the complete numerical parameter threshold after adding OLS: 25/25 main crossed fits, output L1-error median about 0.07112770. Keeping the anchor output formula does not pass. The other three prefixes do not pass with main OLS.

The absolute-normalization control also passes 25/25 for that same best prefix, with output-error median about 0.08321944. It changes the weight readout and is reported separately.

Smaller joint residuals do not ensure smaller second-layer parameter errors. Approximate-prefix Hessians need not share an exact diagonalizing basis, and some measured probes need not be positive definite. The best main normalization has negative entries, so numerical error success does not establish every structural hypothesis of the paper.

## Reproduction

```bash
conda activate aspire
python reproduce.py --preset study --mode checkpoint --verify-reference
```

The maintained study regenerates its Hessians and Gaussian data and produces 360 hidden recoveries, 100 main OLS fits, and 100 absolute-normalization OLS fits. It compares the first 460 main records with the frozen historical metrics.

Open `results/reproduction_study_checkpoint/report/index.html`. Its CSV and JSON files expose individual and grouped errors, and every run links to its raw parameters. The full mathematical procedure is in [algorithm.md](algorithm.md), with environment and timing details in [validation.md](validation.md).
