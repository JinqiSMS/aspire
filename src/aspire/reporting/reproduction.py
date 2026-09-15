"""Build an English offline report without making any oracle queries."""
from pathlib import Path
import csv
import html
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from ..io import write_json


def aggregate(rows):
    groups = []
    keys = sorted({(row["parent_id"], row["method"], row["matrix_count"]) for row in rows})
    for parent, method, count in keys:
        selected = [row for row in rows if (row["parent_id"], row["method"], row["matrix_count"]) == (parent, method, count)]
        good = [row for row in selected if row["status"] == "complete"]
        group = {"parent_id": parent, "method": method, "matrix_count": count,
                 "runs": len(selected), "completed": len(good),
                 "successes": sum(row["parameter_success"] for row in good)}
        for name, getter in (("first_layer_error", lambda row: row["per_layer_errors"][0]),
                             ("second_layer_error", lambda row: row["per_layer_errors"][1]),
                             ("output_l1_error", lambda row: row["output_l1_error"]),
                             ("coefficient_l1_norm", lambda row: row["coefficient_l1_norm"]),
                             ("joint_parameter_error", lambda row: row["joint_parameter_error"])):
            values = [getter(row) for row in good]
            group[name] = {"median": float(np.median(values)), "min": min(values), "max": max(values)} if values else None
        groups.append(group)
    return groups


def save_figure(figure, folder, name):
    figure.tight_layout()
    for extension in ("png", "pdf", "svg"):
        path = folder / f"{name}.{extension}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        if extension == "svg":
            # SVG paths remain separated by newlines after trimming spaces.
            content = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n"
            path.write_text(content, encoding="utf-8", newline="\n")
    plt.close(figure)


def build_report(output):
    started = time.perf_counter()
    output = Path(output)
    rows = json.loads((output / "records.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    groups = aggregate(rows)
    write_json(report / "summary.json", groups)
    columns = ["parent_id", "method", "matrix_count", "direction_repeat", "training_repeat", "status",
               "parameter_success", "output_l1_error", "coefficient_l1_norm", "joint_parameter_error",
               "angular_gaussian_nmse", "logical_learning_queries", "minimum_hidden_weight", "run_id"]
    with (report / "records.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    good = [row for row in rows if row["status"] == "complete"]
    parents = sorted({row["parent_id"] for row in rows})
    primary = manifest["signature"]["specification"]["hessian"]["primary_count"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for index, parent in enumerate(parents):
        subset = [row for row in good if row["parent_id"] == parent and row["method"] == "random_64_gaussian_ols"]
        for axis, values in zip(axes, ([row["per_layer_errors"][0] for row in subset],
                                      [row["per_layer_errors"][1] for row in subset],
                                      [row["output_l1_error"] for row in subset])):
            if values:
                axis.scatter(index + np.linspace(-0.12, 0.12, len(values)), values, s=20, alpha=0.6)
                axis.plot(index, np.median(values), "k_", markersize=16, markeredgewidth=2)
    for axis, label in zip(axes, ("First-layer operator error", "Second-layer operator error", "Output coefficient L1 error")):
        axis.set_title(label)
        axis.set_xticks(range(len(parents)), parents, rotation=25, ha="right", fontsize=8)
        axis.axhline(0.1, color="#b34536", linestyle="--", label="Target: 0.1")
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.15)
    figure.suptitle("Multi-Hessian + Gaussian OLS: all crossed repetitions", y=1.04)
    save_figure(figure, report, "parameter_errors")
    figure, axis = plt.subplots(figsize=(9, 4.6))
    methods = [("random_64", "Anchor formula", "#888888"),
               ("random_64_gaussian_ols", "Gaussian OLS", "#087c78"),
               ("random_64_gaussian_ols_abs", "Absolute normalization + OLS", "#c08036")]
    for offset, (method, label, color) in zip((-0.2, 0, 0.2), methods):
        for index, parent in enumerate(parents):
            values = [row["output_l1_error"] for row in good if row["parent_id"] == parent and
                      row["method"] == method and row["matrix_count"] == primary]
            if values:
                middle = np.median(values)
                axis.errorbar(index + offset, middle, yerr=[[middle - min(values)], [max(values) - middle]],
                              fmt="o", color=color, capsize=4, label=label if index == 0 else None)
    axis.axhline(0.1, color="#b34536", linestyle="--")
    axis.set_xticks(range(len(parents)), parents, rotation=15, ha="right")
    axis.set_ylabel("Output coefficient L1 error")
    axis.set_yscale("log")
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=8)
    axis.set_title("Median and full range; fixed prefixes are shared")
    save_figure(figure, report, "output_comparison")
    figure, axis = plt.subplots(figsize=(9, 4.5))
    physical = manifest["physical_queries_this_invocation"]
    names, values = list(physical), list(physical.values())
    axis.bar(range(len(names)), values, color=["#36576b", "#087c78", "#98b7ad", "#c08036", "#a5aeb5"])
    axis.set_xticks(range(len(names)), [name.replace("_", " ") for name in names], rotation=15, ha="right")
    if any(value > 0 for value in values):
        axis.set_yscale("symlog", linthresh=1)
    for index, value in enumerate(values):
        axis.text(index, max(value, 1), f"{value:,}", ha="center", va="bottom", fontsize=8)
    axis.set_ylabel("Physical real-value queries")
    axis.set_title("This invocation only; shared data is charged once")
    save_figure(figure, report, "query_cost")
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    colors = {parent: plt.get_cmap("tab10")(index) for index, parent in enumerate(parents)}
    styles = {"two_hessian": ":", "best_single": "--", "random_64": "-"}
    for parent in parents:
        for method in ("two_hessian", "best_single", "random_64"):
            selected = sorted([group for group in groups if group["parent_id"] == parent and
                               group["method"] == method and group["completed"]], key=lambda group: group["matrix_count"])
            if selected:
                axes[0].plot([group["matrix_count"] for group in selected],
                             [group["second_layer_error"]["median"] for group in selected], marker="o", markersize=3,
                             color=colors[parent], linestyle=styles[method],
                             label=f"{parent} / {method}")
        selected = [row for row in good if row["parent_id"] == parent and
                    "hessian_diagnostics" in row and row["method"] == "random_64"]
        if selected:
            axes[1].scatter([row["hessian_diagnostics"]["joint_residual"] for row in selected],
                            [row["per_layer_errors"][1] for row in selected],
                            s=18, alpha=0.7, color=colors[parent], label=parent)
    axes[0].set(xlabel="Probe Hessians", ylabel="Second-layer operator error", yscale="log")
    if axes[0].get_legend_handles_labels()[0]:
        handles = [Line2D([0], [0], color=colors[parent], label=parent) for parent in parents]
        handles += [Line2D([0], [0], color="black", linestyle=style, label=method)
                    for method, style in styles.items()]
        axes[0].legend(handles=handles, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    axes[1].set(xlabel="Training joint residual", ylabel="Second-layer operator error", xscale="log", yscale="log")
    figure.suptitle("More probes and smaller residuals do not guarantee better recovery")
    save_figure(figure, report, "hessian_diagnostics")
    figures = ["parameter_errors", "output_comparison", "query_cost", "hessian_diagnostics"]
    write_json(report / "figures.sources.json", {"records": "../records.json", "manifest": "../manifest.json", "figures": figures})
    compact = [{key: row.get(key) for key in columns + ["per_layer_errors"]} for row in rows]
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASPIRE reproducible numerical experiments</title><style>
body{font:16px/1.6 system-ui;background:#f5f8f9;color:#243e47;margin:0}main{max-width:1300px;margin:auto;padding:32px}
h1{font-size:32px;line-height:1.2}h2{margin-top:35px}a{color:#087c78}.notice{background:#e5f0ed;border-left:4px solid #087c78;padding:18px}
img{width:100%;background:white;border-radius:8px;margin:12px 0}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:13px}
th,td{padding:9px;border-bottom:1px solid #dce4e6;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}
select{padding:8px;margin:0 18px 16px 5px}code{background:#e6edef;padding:2px 5px}.muted{color:#52666d}
</style><main><h1>ASPIRE numerical experiments</h1>
<p class="notice">Column recovery and ASPIRE moments for the first layer; randomized multi-Hessian recovery for the second; standard Gaussian ordinary least squares for the output.</p>
<p>RUN_SUMMARY</p><p class="muted">Twenty-five crossed OLS fits share one first-layer estimate: they are not 25 independent network recoveries. Signed normalization can produce negative hidden weights. Passing the numerical threshold does not certify all assumptions of the original theorem.</p>
<p><a href="summary.json">Group summaries</a> · <a href="records.csv">Download CSV</a> · <a href="../manifest.json">Configuration, seeds and query accounting</a></p>
<h2>Recovery accuracy</h2><img src="parameter_errors.png" alt="First layer, second layer and output errors for all OLS repetitions">
<h2>Output regression</h2><img src="output_comparison.png" alt="Anchor formula and two Gaussian OLS normalization variants">
<h2>Individual runs</h2><label>Prefix<select id="parent"></select></label><label>Method<select id="method"></select></label><span id="count"></span>
<div class="scroll"><table><thead><tr><th>Prefix</th><th>Method</th><th>M</th><th>Direction</th><th>OLS</th><th>W1 error</th><th>W2 error</th><th>Output L1 error</th><th>Coefficient L1 norm</th><th>Pass</th><th>Record</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Hessian diagnostics</h2><img src="hessian_diagnostics.png" alt="Probe count and diagonalization residual versus parameter error">
<h2>Query accounting</h2><img src="query_cost.png" alt="Physical function queries in the current invocation">
<p class="muted">Per-run logical costs include inherited first-layer costs. Do not sum these repeated logical counts across shared runs. Physical invocation costs are recorded separately in the manifest.</p>
<script>const data=RECORD_DATA;const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>x==null?'—':Number(x).toPrecision(6);for(const [id,key] of [['parent','parent_id'],['method','method']]){document.getElementById(id).innerHTML='<option value="">All</option>'+[...new Set(data.map(r=>r[key]))].sort().map(x=>'<option>'+escape(x)+'</option>').join('');document.getElementById(id).onchange=render}
function render(){const parent=document.getElementById('parent').value,method=document.getElementById('method').value;const selected=data.filter(r=>(!parent||r.parent_id===parent)&&(!method||r.method===method));document.getElementById('count').textContent=selected.length+' records';document.getElementById('rows').innerHTML=selected.map(r=>'<tr>'+[escape(r.parent_id),escape(r.method),r.matrix_count,r.direction_repeat,r.training_repeat??'—',fmt(r.per_layer_errors?.[0]),fmt(r.per_layer_errors?.[1]),fmt(r.output_l1_error),fmt(r.coefficient_l1_norm),r.parameter_success?'Yes':'No','<a href="../runs/'+encodeURIComponent(r.run_id)+'/metrics.json">JSON</a>'].map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')}render();</script></main></html>'''
    description = (f"Mode: {manifest['signature']['mode']}; preset: {manifest['signature']['preset']}; "
                   f"{len(rows)} records; computation time {manifest['seconds']:.2f} seconds. "
                   f"Fresh physical queries: {manifest['physical_total_queries_this_invocation']:,}.")
    page = page.replace("RUN_SUMMARY", html.escape(description)).replace("RECORD_DATA", json.dumps(compact).replace("</", "<\\/"))
    (report / "index.html").write_text(page, encoding="utf-8")
    lines = ["# ASPIRE reproduction results", "", description, "",
             "[Offline report](index.html) | [CSV records](records.csv) | [Group summaries](summary.json)", "",
             "## Main output regression results", ""]
    for group in groups:
        if group["method"] == "random_64_gaussian_ols" and group["completed"]:
            error = group["output_l1_error"]
            lines.append(f"- **{group['parent_id']}**: {group['successes']}/{group['runs']} passed; "
                         f"output L1 error median {error['median']:.8f}, range [{error['min']:.8f}, {error['max']:.8f}].")
    lines += ["", "Crossed repeats share a first-layer estimate. The best prefix is one locally successful setting; "
              "the other historical prefixes do not meet the same full-network threshold. Negative signed weights are reported explicitly.", ""]
    for name in figures:
        lines += [f"![{name.replace('_', ' ').capitalize()}]({name}.png)", ""]
    (report / "results.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(report / "build.json", {"seconds": time.perf_counter() - started, "figure_files": len(figures) * 3,
                                      "new_oracle_queries": 0})
    print(f"REPORT {report / 'index.html'}", flush=True)
