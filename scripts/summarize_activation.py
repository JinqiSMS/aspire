"""Summarize the fixed-setting k=4,6,8 activation experiments with Matplotlib."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from aspire.io import inside, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/activation_comparison")
    parser.add_argument("--previous", help="Compare with an earlier normalization using identical first layers")
    args = parser.parse_args()
    root = inside(args.output)
    population_path = root / "population_moments.json"
    population = {row["k"]: row["estimates"][-1]["relative_gap"]
                  for row in read(population_path)["rows"]} if population_path.exists() else {}
    rows, parameter_files, specifications, commands = [], {}, [], []
    for k in (4, 6, 8):
        folder = root / f"k{k}"
        manifest = read(folder / "run.json")
        if manifest["status"] not in ("complete", "failed"):
            raise SystemExit(f"k={k} is still running")
        metrics = read(folder / "metrics.json")
        configuration = manifest["signature"]["configuration"]
        specifications.append(configuration)
        command = f"python run_experiment.py --config configs/experiment_k{k}.yaml"
        reuse = manifest.get("first_layer_reuse_source")
        command_output = (f"results/activation_comparison/k{k}" if root.is_relative_to(PROJECT_ROOT / "examples")
                          else folder.relative_to(PROJECT_ROOT).as_posix())
        commands.append(command + f" --output {command_output}")
        queries = read(folder / "queries.json")
        errors = metrics["per_layer_errors"]
        state = read(folder / "first_layer.json") if (folder / "first_layer.json").exists() else {}
        first_stage = state.get("stages", [{}])[0]
        hessian, regression = metrics.get("hessian_diagnostics", {}), metrics.get("regression_diagnostics", {})
        row = {"k": k, "status": manifest["status"], "first_layer_error": errors[0] if errors else None,
               "second_layer_error": errors[1] if len(errors) > 1 else None,
               "output_l1_error": metrics.get("output_l1_error"),
               "coefficient_l1_norm": metrics.get("coefficient_l1_norm"),
               "joint_parameter_error": metrics.get("joint_parameter_error"),
               "parameter_success": metrics["parameter_success"],
               "first_layer_origin": manifest.get("first_layer_origin"),
               "first_layer_source": reuse.get("directory") if reuse else None,
               "new_first_layer_queries": queries.get("current_execution", {}).get("first_layer"),
               "computation_seconds": manifest["computation_seconds"],
               "first_layer_sampling_seconds": state.get("wall_seconds"),
               "hessian_seconds": manifest.get("hessian_seconds"),
               "regression_seconds": manifest.get("regression_seconds"),
               "complete_learning_queries": queries.get("complete_learning_total"),
               "current_execution_queries": queries.get("current_execution_total"),
               "moment_gap": first_stage.get("moment_gap"),
               "population_moment_gap": population.get(k),
               "span_condition": first_stage.get("span_condition"),
               "hessian_relative_gap": hessian.get("final_gap"),
               "hessian_signal_residual": hessian.get("signal_residual"),
               "hessian_condition": hessian.get("final_condition"),
               "normalization": hessian.get("normalization"),
               "minimum_weight_entry": hessian.get("minimum_weight_entry"),
               "negative_weight_entries": hessian.get("negative_weight_entries"),
               "regression_condition": regression.get("scaled_design_condition"),
               "max_leverage": regression.get("max_leverage"),
               "max_label_energy_share": regression.get("max_label_energy_share"),
               "angular_gaussian_nmse": metrics.get("angular_gaussian_nmse"),
               "failure_reason": metrics.get("failure_reason")}
        rows.append(row)
        if (folder / "weights.npz").exists():
            with np.load(folder / "weights.npz", allow_pickle=False) as saved:
                parameter_files[k] = {name: saved[name].copy() for name in saved.files}
            parameters = parameter_files[k]
            if row["normalization"] == "coordinatewise_absolute_value_and_l1":
                assert np.all(parameters["W2_raw"] >= 0)
                np.testing.assert_allclose(parameters["W2_raw"].sum(axis=0), 1., atol=1e-14)
            calculated_errors = [np.linalg.norm(parameters[f"W{layer}_aligned"] -
                                                 parameters[f"W{layer}_true"], ord=2)
                                 for layer in (1, 2)]
            np.testing.assert_allclose(calculated_errors, errors, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(parameters["a_aligned"] - parameters["a_true"], ord=1),
                                       row["output_l1_error"], rtol=1e-10, atol=1e-12)
        if manifest["status"] == "complete":
            counts = queries["first_layer_breakdown"]
            assert counts["n_kept_samples"] == configuration["first_layer"]["samples"]
            assert counts["n_hr_transitions"] == (configuration["first_layer"]["samples"] *
                                                       configuration["first_layer"]["steps"])
    comparable = []
    for specification in specifications:
        normalized = json.loads(json.dumps(specification))
        normalized["architecture"].pop("k")
        comparable.append(normalized)
    assert all(specification == comparable[0] for specification in comparable), "Hyperparameters differ beyond k"
    if len(parameter_files) > 1:
        baseline = next(iter(parameter_files.values()))
        for parameters in parameter_files.values():
            for name in ("W1_true", "W2_true", "a_true"):
                np.testing.assert_array_equal(parameters[name], baseline[name])
    write_json(root / "summary.json", {"created_at": datetime.now(timezone.utc).isoformat(),
        "same_hyperparameters_except_k": True, "rows": rows,
        "baseline_note": "First-layer origins, sources, and new query counts are recorded separately for each exponent",
        "timing_note": "Computation time excludes figure export; saved first-layer time belongs to its original execution"})
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "savefig.dpi": 220})
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def save(figure, name):
        for extension in ("png", "pdf"):
            figure.savefig(figures / f"{name}.{extension}", bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for axis, key, label in zip(axes, ("first_layer_error", "second_layer_error", "output_l1_error"),
                               ("First-layer operator error", "Second-layer operator error", "Output coefficient L1 error")):
        available = [row for row in rows if row[key] is not None]
        axis.plot([row["k"] for row in available], [row[key] for row in available], "o-", color="#2878b5")
        for row in available:
            axis.annotate(f"{row[key]:.4g}", (row["k"], row[key]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
        for row in rows:
            if row[key] is None:
                axis.text(row["k"], .12, "Unavailable", transform=axis.get_xaxis_transform(), ha="center", fontsize=8)
        axis.axhline(specifications[0]["evaluation"]["parameter_delta"], color="#ae4436", ls="--", label="Threshold 0.1")
        axis.set_yscale("log"); axis.set_xticks([4, 6, 8]); axis.set_xlabel("Activation exponent k"); axis.set_title(label)
        axis.margins(x=.15, y=.3); axis.grid(alpha=.15)
    axes[0].legend(fontsize=9, frameon=False)
    figure.suptitle("8-3-3-1 network: identical target weights and hyperparameters")
    save(figure, "activation_errors")

    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")
    if parameter_files:
        ground_truth = next(iter(parameter_files.values()))["a_true"]
        positions = np.arange(len(ground_truth))
        width = .18
        sets = [("Ground truth", ground_truth, "#666666")] + [(f"k={k}", parameters["a_aligned"], color)
               for (k, parameters), color in zip(parameter_files.items(), ("#2878b5", "#57a16b", "#dd8748"))]
        for index, (label, coefficients, color) in enumerate(sets):
            bars = axis.bar(positions + (index-(len(sets)-1)/2)*width, coefficients, width, label=label, color=color)
            axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        axis.set_xticks(positions, [f"a[{index+1}]" for index in positions])
        axis.set_ylabel("Aligned output coefficient"); axis.legend(frameon=False, ncol=2); axis.margins(y=.3)
    axis.set_title("Output coefficients at each activation exponent")
    save(figure, "activation_coefficients")

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for axis, key, title in zip(axes, ("moment_gap", "hessian_relative_gap", "regression_condition"),
                               ("ASPIRE relative moment gap", "Selected Hessian relative gap", "Scaled regression condition")):
        available = [row for row in rows if row[key] is not None]
        axis.plot([row["k"] for row in available], [row[key] for row in available], "o-", color="#57a16b", label="Measured")
        if key == "moment_gap" and population:
            axis.plot(list(population), list(population.values()), "s--", color="#666666", label="Population quadrature")
            axis.legend(fontsize=8, frameon=False)
        if key == "regression_condition":
            axis.set_ylim(.99, max(1.03, max((row[key] for row in available), default=1) * 1.01))
            axis.axhline(1, color="#777777", ls=":", linewidth=1)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        else:
            axis.set_yscale("log")
        axis.set_xticks([4, 6, 8]); axis.set_xlabel("Activation exponent k"); axis.set_title(title)
        axis.grid(alpha=.15); axis.margins(x=.15, y=.3)
    save(figure, "activation_diagnostics")

    lines = ["# Activation exponent comparison", "",
             "The architecture is 8-3-3-1. Target weight matrices, coefficients, random seeds, and all numerical hyperparameters are fixed; only the activation exponent changes.", "",
             "Each exponent has its own first-layer estimate. The origins and any imported sources are recorded below. Saved first-layer sampling times and costs refer to the original computations, while current execution counts describe this run.", "",
             "## Full-run commands", "", "These commands start new complete experiments; existing results require a different --output directory.", "", "```bash",
             *commands,
             "python scripts/summarize_activation.py --output " + root.relative_to(PROJECT_ROOT).as_posix() +
             (f" --previous {args.previous}" if args.previous else ""), "```", "", "## Results", ""]
    lines[-2:] = ["## Fixed settings", "",
        "- Target: identical $W_1$, $W_2$, and $a=(1/3,1/3,1/3)$ for all exponents; noiseless value queries.",
        "- First layer: 16,777,216 independent chains, 32 transitions per chain, batch size 4,096, and 16 column-space probes.",
        "- First-layer oracle: original directional step 0.25; reduced-gradient step factor 0.5; bisection absolute tolerance $10^{-7}$ and at most 100 iterations.",
        "- Public bounds: $\\kappa_0=2$ and $\\mu=0.25$; first-layer limits are 40 billion queries and 14,400 seconds.",
        "- Second layer: one anchor plus 12 probe Hessians, 64 random mixtures, six held-out probes, $\\tau=0.2$, and interpolation step 0.1.",
        "- Output layer: ordinary least squares on 1,048,576 standard Gaussian inputs, batch size 65,536.",
        "- Evaluation: 20,000 angular directions; each hidden-layer operator error and output coefficient L1 error must be at most 0.1 to pass.",
        "- Seeds: first layer 3141116543; Hessian directions 431061063; mixtures 2356887966; held-out Hessians 2828169828; regression 1586144878; evaluation 1444349591.", "",
        "The total network degree is $k^2$: 16, 36, and 64. Degree-dependent interpolation node counts and radius bounds follow the same algorithm for each exponent.", "",
        "The corrected second-layer normalization is $\\widehat w_j=|v_j|/(\\mathbf{1}^\\top|v_j|)$, with coordinatewise absolute values. It enforces nonnegative entries and unit column sums. Output coefficients are refitted by unconstrained OLS after normalization.", "",
        "Errors use hidden-unit permutation alignment and first-layer sign alignment. The reported output error is $\\|\\widehat a-a\\|_1$, distinct from the coefficient norm $\\|\\widehat a\\|_1$.", "",
        "## Results", ""]
    for row in rows:
        lines += [f"### k={row['k']}", "", f"Status: {row['status']}; parameter threshold passed: {row['parameter_success']}.", ""]
        for key in ("first_layer_error", "second_layer_error", "output_l1_error", "coefficient_l1_norm", "normalization", "minimum_weight_entry", "negative_weight_entries", "first_layer_origin", "first_layer_source", "new_first_layer_queries", "moment_gap", "population_moment_gap", "hessian_relative_gap", "regression_condition", "max_leverage", "max_label_energy_share", "computation_seconds", "first_layer_sampling_seconds", "hessian_seconds", "regression_seconds", "complete_learning_queries", "current_execution_queries"):
            lines.append(f"- {key}: {row[key]}")
        if row["k"] in parameter_files:
            lines += ["", "Aligned output coefficients: `" +
                      np.array2string(parameter_files[row["k"]]["a_aligned"], precision=12) + "`."]
        if row["failure_reason"]:
            lines.append(f"- failure_reason: {row['failure_reason']}")
        lines += ["", f"Individual metrics and any recovered weight comparisons are in `k{row['k']}/`.", ""]
    if args.previous:
        previous = inside(args.previous)
        changes = []
        lines += ["## Normalization comparison", "",
            "The earlier signed-column results are retained separately. First-layer arrays are exactly equal before and after; only the later stages are rerun.", ""]
        for row, configuration in zip(rows, specifications):
            old_folder = previous / f"k{row['k']}"
            old = read(old_folder / "metrics.json")
            assert read(old_folder / "run.json")["signature"]["configuration"] == configuration
            with np.load(old_folder / "weights.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["W1_raw"], parameter_files[row["k"]]["W1_raw"])
                for name in ("W1_true", "W2_true", "a_true"):
                    np.testing.assert_array_equal(saved[name], parameter_files[row["k"]][name])
            assert row["new_first_layer_queries"] == 0
            change = {"k": row["k"], "first_layer_identical": True, "new_first_layer_queries": 0,
                "before_second_layer_error": old["per_layer_errors"][1],
                "after_second_layer_error": row["second_layer_error"],
                "before_output_l1_error": old["output_l1_error"], "after_output_l1_error": row["output_l1_error"],
                "before_negative_entries": old["hessian_diagnostics"]["negative_weight_entries"],
                "after_negative_entries": row["negative_weight_entries"]}
            changes.append(change)
            lines += [f"- k={row['k']}: W2 error {change['before_second_layer_error']:.8f} to {change['after_second_layer_error']:.8f}; "
                f"output L1 error {change['before_output_l1_error']:.8f} to {change['after_output_l1_error']:.8f}; "
                f"negative entries {change['before_negative_entries']} to {change['after_negative_entries']}."]
        lines += [""]
        write_json(root / "normalization_comparison.json", {"previous": args.previous, "rows": changes})
    lines += ["## Figures", "", "![Parameter errors](figures/activation_errors.png)", "",
              "![Output coefficients](figures/activation_coefficients.png)", "",
              "![Numerical diagnostics](figures/activation_diagnostics.png)", "",
              "The separate column-space diagnostic uses analytic gradients for evaluator-side comparison only; it does not supply derivatives or parameters to the learner. Its queries are recorded separately in column_diagnostics.json.", "",
              "Population moment gaps use deterministic sphere quadrature and analytic radial integration. Grid convergence is recorded in population_moments.json when that optional diagnostic has been generated. These diagnostics are not used by the learning algorithm.", "",
              "The batch value oracle evaluates even powers by multiplication chains, verified against general exponentiation. Original first-layer costs and current execution queries are reported separately above."]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
