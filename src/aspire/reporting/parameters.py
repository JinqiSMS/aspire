"""Export raw and aligned parameter values and draw static Matplotlib figures."""
import csv
import json
import time

import numpy as np

from ..evaluation.alignment import align
from ..io import save_npz, write_json


def save_comparison(output, true_weights, true_output, recovered_weights, recovered_output):
    aligned, coefficients, permutations = align(true_weights, recovered_weights, recovered_output)
    truth = {**{f"W{i + 1}": value for i, value in enumerate(true_weights)}, "a": true_output}
    raw = {**{f"W{i + 1}": value for i, value in enumerate(recovered_weights)}, "a": recovered_output}
    estimated = {**{f"W{i + 1}": value for i, value in enumerate(aligned)}, "a": coefficients}
    comparison = {"alignment": "Sequential hidden-unit permutations; sign alignment in the first layer only",
                  "permutations": permutations, "parameters": {}}
    arrays, entries = {}, []
    lines = ["EXPERIMENT 1: RECOVERED PARAMETERS AND GROUND TRUTH", "",
             "Recovered values below are aligned for hidden-unit permutation and first-layer sign symmetry.",
             "Raw estimates and aligned estimates are both retained in weights.json and weights.npz.", ""]
    for name, target in truth.items():
        difference = estimated[name] - target
        comparison["parameters"][name] = {"ground_truth": target, "recovered_raw": raw[name],
            "recovered_aligned": estimated[name], "difference": difference, "absolute_error": np.abs(difference)}
        arrays.update({f"{name}_true": target, f"{name}_raw": raw[name],
                       f"{name}_aligned": estimated[name], f"{name}_difference": difference})
        for index in np.ndindex(target.shape):
            entries.append({"parameter": name, "row": index[0] + 1,
                "column": index[1] + 1 if len(index) == 2 else 1,
                "ground_truth": float(target[index]), "recovered_aligned": float(estimated[name][index]),
                "difference": float(difference[index]), "absolute_error": float(abs(difference[index]))})
        lines += [name + " ground truth:", np.array2string(target, precision=10, suppress_small=False),
                  name + " recovered (aligned):", np.array2string(estimated[name], precision=10, suppress_small=False),
                  name + " difference:", np.array2string(difference, precision=10, suppress_small=False), ""]
    save_npz(output / "weights.npz", **arrays)
    write_json(output / "weights.json", comparison)
    with (output / "weights.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    (output / "weights.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return comparison


def create_figures(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    comparison = json.loads((output / "weights.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    configuration = json.loads((output / "run.json").read_text(encoding="utf-8"))["signature"]["configuration"]
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.dpi": 220})

    def save(figure, name):
        for extension in ("png", "pdf"):
            figure.savefig(folder / f"{name}.{extension}", bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(3, 3, figsize=(11.2, 11), gridspec_kw={"height_ratios": [8, 3, 1]}, layout="constrained")
    figure.suptitle("Experiment 1: parameter values after symmetry alignment", fontsize=15)
    for row, name in enumerate(("W1", "W2", "a")):
        values = comparison["parameters"][name]
        matrices = [np.atleast_2d(values[key]) for key in ("ground_truth", "recovered_aligned", "difference")]
        lower, upper = min(matrices[0].min(), matrices[1].min()), max(matrices[0].max(), matrices[1].max())
        for column, (matrix, title) in enumerate(zip(matrices, ("Ground truth", "Recovered", "Recovered - truth"))):
            axis = axes[row, column]
            limit = max(float(np.max(np.abs(matrix))), 1e-12)
            options = {"cmap": "RdBu_r", "vmin": -limit, "vmax": limit} if column == 2 else {"cmap": "viridis", "vmin": lower, "vmax": upper}
            artist = axis.imshow(matrix, aspect="auto", **options)
            for index in np.ndindex(matrix.shape):
                rgba = artist.cmap(artist.norm(matrix[index]))
                brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                axis.text(index[1], index[0], f"{matrix[index]:.4f}", ha="center", va="center",
                          color="black" if brightness > 0.6 else "white", fontsize=9)
            axis.set_title(f"{name}: {title}")
            axis.set_xticks(range(matrix.shape[1]), range(1, matrix.shape[1] + 1))
            axis.set_yticks(range(matrix.shape[0]), range(1, matrix.shape[0] + 1))
            figure.colorbar(artist, ax=axis, fraction=0.045, pad=0.025)
    save(figure, "weight_comparison")

    figure, axis = plt.subplots(figsize=(6.5, 4.3), layout="constrained")
    values = [*metrics["per_layer_errors"], metrics["output_l1_error"]]
    bars = axis.bar(["W1 operator error", "W2 operator error", "Output L1 error"], values,
                    color=["#2878b5", "#5c9c64", "#db8547"], width=0.55)
    axis.bar_label(bars, labels=[f"{value:.6f}" for value in values], padding=4)
    axis.axhline(configuration["evaluation"]["parameter_delta"], color="#af4035", ls="--", label="Threshold")
    axis.set_ylim(0, max(configuration["evaluation"]["parameter_delta"], *values) * 1.3)
    axis.set_ylabel("Parameter error")
    axis.set_title("Experiment 1: aligned parameter errors")
    axis.legend(frameon=False)
    save(figure, "parameter_errors")

    figure, axis = plt.subplots(figsize=(6.5, 4.3), layout="constrained")
    values = comparison["parameters"]["a"]
    indices = np.arange(len(values["ground_truth"]))
    for offset, key, label, color in ((-0.18, "ground_truth", "Ground truth", "#666666"),
                                      (0.18, "recovered_aligned", "Recovered", "#2878b5")):
        bars = axis.bar(indices + offset, values[key], width=0.34, label=label, color=color)
        axis.bar_label(bars, fmt="%.6f", padding=3, fontsize=9)
    axis.set_xticks(indices, [f"a[{index + 1}]" for index in indices])
    axis.set_ylabel("Coefficient value")
    axis.margins(y=0.25)
    axis.legend(frameon=False)
    axis.set_title("Experiment 1: output coefficients")
    save(figure, "output_coefficients")
    write_json(output / "figure_build.json", {"seconds": time.perf_counter() - started,
               "new_oracle_queries": 0, "files": [f"figures/{name}.{extension}"
               for name in ("weight_comparison", "parameter_errors", "output_coefficients")
               for extension in ("png", "pdf")]})
