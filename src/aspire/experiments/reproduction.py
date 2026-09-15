"""Self-contained first-layer, multi-Hessian, and Gaussian OLS experiments.

Target parameters belong to the experiment evaluator. Recovery receives only
the counted real-value interface, public bounds, and previously learned layers.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import gc
import hashlib
import json
import time

import numpy as np
import yaml

from .. import ROOT
from ..architecture import Architecture
from ..config import resolve
from ..evaluation.metrics import parameter_metrics, risk_batch
from ..io import inside, write_json, save_npz
from ..oracles.suffix import SuffixOracle
from ..query_ledger import QueryLedger, CountedRealOracle
from ..recovery.generalized_joint import recover_generalized, random_coefficients, joint_residuals
from ..recovery.hessian_bank import collect_hessians, orthogonal_directions
from ..recovery.moments import normalize_directions
from ..recovery.network import recover_network
from ..recovery.output import hidden_features, ordinary_least_squares
from ..status import NumericalFailure
from ..teacher import Teacher
from .runner import seed_map, environment, implementation_snapshot, learner_settings


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_seed(entropy, *parts):
    return int(np.random.SeedSequence([entropy, *parts]).generate_state(1)[0])


def first_layer_config(specification, parent):
    settings = specification["first_layer"]
    return resolve({
        "architecture": specification["architecture"], "seed": parent["seed"],
        "oracle": {"mode": "strict_real", "cache_size": 0,
                   "max_real_learning_queries": settings["query_budget"],
                   "wall_seconds": settings["wall_seconds"]},
        "recovery": {"moment_samples_by_layer": [parent["samples"]],
                     "first_span_points": settings["first_span_points"],
                     "original_direction_step": settings["original_direction_step"],
                     "reduced_gradient_step_factor": settings["reduced_gradient_step_factor"]},
        "sampler": {"mode": "independent_endpoints", "endpoint_steps": parent["steps"],
                    "batch_size": settings["batch_size"],
                    "bisection_absolute_tolerance": settings["bisection_absolute_tolerance"],
                    "max_bisection_iterations": settings["max_bisection_iterations"]},
    })


def load_target(parent, reference):
    path = ROOT / "reference/teachers" / f"seed_{parent['seed']}.npz"
    expected = reference["teacher_files"][path.name]
    if file_hash(path) != expected:
        raise ValueError("Frozen target checksum mismatch")
    with np.load(path, allow_pickle=False) as saved:
        return Teacher([saved["W1"].copy(), saved["W2"].copy()], saved["a"].copy(), 4)


def obtain_prefix(output, parent, teacher, specification, mode, reference):
    folder = output / "prefixes" / parent["id"]
    folder.mkdir(parents=True, exist_ok=True)
    config = first_layer_config(specification, parent)
    metadata_path, parameter_path = folder / "state.json", folder / "parameters.npz"
    if metadata_path.exists():
        state = read_json(metadata_path)
        if file_hash(parameter_path) != state["parameter_sha256"]:
            raise ValueError("Computed prefix checksum mismatch")
        with np.load(parameter_path, allow_pickle=False) as saved:
            return saved["W1"].copy(), config, state, 0
    started = time.perf_counter()
    if mode == "checkpoint":
        source = ROOT / "reference/prefixes" / f"{parent['id']}.npz"
        state = deepcopy(reference["prefixes"][parent["id"]])
        if file_hash(source) != state["parameter_sha256"]:
            raise ValueError("Reference prefix checksum mismatch")
        if (state["samples"], state["steps"]) != (parent["samples"], parent["steps"]):
            raise ValueError("Reference prefix settings differ from requested settings")
        with np.load(source, allow_pickle=False) as saved:
            weights = saved["W1"].copy()
        state.update(mode="inherited_checkpoint", source=source.relative_to(ROOT).as_posix())
        new_queries = 0
    else:
        architecture = Architecture(**specification["architecture"])
        ledger = QueryLedger(config["oracle"]["max_real_learning_queries"], 0,
                             config["oracle"]["wall_seconds"])
        real_oracle = CountedRealOracle(teacher.forward, architecture.d, ledger,
                                       batch_value=teacher.forward)
        print(f"FIRST_LAYER {parent['id']} samples={parent['samples']} steps={parent['steps']}", flush=True)
        result = recover_network(real_oracle, architecture=architecture,
            public_bounds=specification["public_bounds"], config=learner_settings(config),
            rng=np.random.default_rng(seed_map(config)["algorithm"]), stop_after_layer=1)
        if result["status"] != "prefix_complete":
            write_json(folder / "failure.json", {key: value for key, value in result.items()
                                                  if key not in ("artifacts", "hidden_weights")})
            raise RuntimeError(f"First-layer recovery failed: {result['failure_reason']}")
        weights = result["hidden_weights"][0].copy()
        state = {"mode": "fresh_real_queries", "samples": parent["samples"], "steps": parent["steps"],
                 "counts": ledger.snapshot(), "stages": result["stages"],
                 "seed_manifest": seed_map(config), "wall_seconds": time.perf_counter() - started}
        new_queries = ledger.counts["n_real_calls_total"]
        del result
        gc.collect()
    save_npz(parameter_path, W1=weights)
    state.update(parameter_sha256=file_hash(parameter_path), parent_id=parent["id"],
                 current_invocation_seconds=time.perf_counter() - started)
    write_json(metadata_path, state)
    write_json(folder / "configuration.json", config)
    print(f"PREFIX_READY {parent['id']} queries={state['counts']['n_real_calls_total']}", flush=True)
    return weights, config, state, new_queries


def score(teacher, prefix, hidden, coefficients, inputs, labels, delta):
    result = {"hidden_weights": [prefix, hidden], "output_weights_raw": coefficients, "status": "complete"}
    metrics = parameter_metrics(teacher, result, delta)
    metrics["joint_parameter_error"] = max(metrics["max_weight_operator_error"], metrics["output_l1_error"])
    metrics["coefficient_l1_norm"] = float(np.linalg.norm(coefficients, 1))
    metrics["coefficients_raw"] = coefficients.tolist()
    metrics["minimum_hidden_weight"] = float(hidden.min())
    prediction = Teacher([prefix, hidden], coefficients, 4).forward(inputs)
    metrics["angular_gaussian_nmse"] = risk_batch(prediction, labels)["nmse"]
    return metrics


def evaluation_data(teacher, prefix, specification, parent):
    basis, singular, _ = np.linalg.svd(np.concatenate([teacher.weights[0], prefix], axis=1), full_matrices=False)
    rank = int(np.count_nonzero(singular > 1e-10 * singular[0]))
    random = np.random.default_rng(stream_seed(specification["randomness"]["evaluation_entropy"], parent["stream_index"]))
    directions = random.normal(size=(specification["evaluation"]["directions"], rank))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    inputs = directions @ basis[:, :rank].T
    ledger = QueryLedger(len(inputs), 0)
    oracle = CountedRealOracle(teacher.forward, 8, ledger, batch_value=teacher.forward)
    with ledger.scope("evaluation"):
        labels = oracle.batch(inputs)
    return inputs, labels, ledger.snapshot()


def compare_reference(rows, reference_path, tolerance=1e-7):
    reference_rows = read_json(reference_path)
    fields = ("parent_id", "direction_repeat", "matrix_count", "method", "training_repeat")
    lookup = {tuple(row.get(key) for key in fields): row for row in reference_rows}
    maximum = 0.0
    matched = 0
    failures = []
    seen = set()
    for row in rows:
        if row["method"].endswith("_abs"):
            continue
        key = tuple(row.get(field) for field in fields)
        if row["status"] != "complete" or key not in lookup or key in seen:
            failures.append({"key": list(key), "reason": "Incomplete, unknown, or duplicate main record"})
            continue
        seen.add(key)
        expected = lookup[key]
        actual_metrics = np.asarray([row["output_l1_error"], *row["per_layer_errors"]], dtype=float)
        expected_metrics = np.asarray([expected["output_l1_error"], *expected["per_layer_errors"]], dtype=float)
        if actual_metrics.shape != expected_metrics.shape or not np.all(np.isfinite(actual_metrics)):
            failures.append({"key": list(key), "reason": "Invalid or non-finite numerical metrics"})
            continue
        error = float(np.max(np.abs(actual_metrics - expected_metrics)))
        maximum = max(maximum, error)
        matched += 1
    return {"matched_records": matched, "maximum_metric_difference": maximum,
            "absolute_tolerance": tolerance, "failures": failures,
            "passed": matched > 0 and not failures and maximum <= tolerance}


def run(specification, *, mode, preset, output, verify_reference=False):
    specification = deepcopy(specification)
    if preset == "smoke":
        mode = "full"
        smoke = specification["smoke"]
        specification["regression"].update(samples=smoke["gaussian_samples"], repeats=[0])
        specification["hessian"]["direction_repeats"] = [0]
        specification["evaluation"]["directions"] = smoke["evaluation_directions"]
    parents = [parent for parent in specification["parents"]
               if preset == "study" or parent["id"] == specification["best_parent"]]
    if preset == "smoke":
        parents[0].update(samples=smoke["first_layer_samples"], steps=smoke["endpoint_steps"])
    started = time.perf_counter()
    version, snapshot = implementation_snapshot()
    output = inside(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = {"specification": specification, "mode": mode, "preset": preset,
                 "implementation_hash": version}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous["signature"] != signature:
            raise ValueError("Output belongs to a different protocol; choose a new --output directory")
        if (output / "records.json").exists():
            print("Saved result found; use a new --output directory for fresh computation.", flush=True)
            rows = read_json(output / "records.json")
            if verify_reference:
                verification = compare_reference(rows, ROOT / "reference/expected_metrics.json")
                previous["reference_verification"] = verification
                write_json(manifest_path, previous)
                if not verification["passed"]:
                    raise RuntimeError(f"Historical reference verification failed: {verification}")
                print(f"REFERENCE verified {verification['matched_records']} saved main records", flush=True)
            return rows
    reference = read_json(ROOT / "reference/manifest.json")
    manifest = {"signature": signature, "environment": environment(),
                "implementation_snapshot": snapshot.relative_to(ROOT).as_posix(),
                "reference_manifest_sha256": file_hash(ROOT / "reference/manifest.json"),
                "guarantee_status": "empirical_only", "status": "running"}
    write_json(manifest_path, manifest)
    rows, physical = [], {"first_layer": 0, "hessian_training": 0, "hessian_validation": 0,
                          "gaussian_labels": 0, "prediction_evaluation": 0}
    options, random_settings = specification["hessian"], specification["randomness"]
    counts = options["matrix_counts"] if preset == "study" else [options["primary_count"]]
    regression, evaluation = specification["regression"], specification["evaluation"]
    for parent in parents:
        teacher = load_target(parent, reference)
        prefix, config, prefix_state, new_queries = obtain_prefix(
            output, parent, teacher, specification, mode, reference)
        physical["first_layer"] += new_queries
        eval_inputs, eval_labels, eval_counts = evaluation_data(teacher, prefix, specification, parent)
        physical["prediction_evaluation"] += eval_counts["n_real_calls_total"]
        save_npz(output / "data" / parent["id"] / "evaluation.npz", x=eval_inputs, y=eval_labels)
        for repeat in options["direction_repeats"]:
            directions = orthogonal_directions(3, max(counts), np.random.default_rng(
                stream_seed(random_settings["direction_entropy"], parent["stream_index"], repeat, 0)))
            heldout_directions = orthogonal_directions(3, options["heldout_directions"], np.random.default_rng(
                stream_seed(random_settings["direction_entropy"], parent["stream_index"], repeat, 1)))
            ledger = QueryLedger(100000, 0)
            oracle = CountedRealOracle(teacher.forward, 8, ledger)
            suffix = SuffixOracle(oracle, Architecture(**specification["architecture"]), 2, [prefix], config["oracle"], ledger)
            matrices, centers, train_info = collect_hessians(suffix, directions, degree=4,
                tau=options["tau"], step=options["step"], ledger=ledger)
            heldout, heldout_centers, validation_info = collect_hessians(suffix, heldout_directions, degree=4,
                tau=options["tau"], step=options["step"], ledger=ledger, include_anchor=False, scope="validation")
            physical["hessian_training"] += ledger.counts["n_real_calls_final_hessian"]
            physical["hessian_validation"] += ledger.counts["n_real_calls_validation"]
            bank_folder = output / "data" / parent["id"] / f"repeat_{repeat}"
            save_npz(bank_folder / "hessians.npz", bank=matrices, centers=centers, heldout=heldout,
                     heldout_centers=heldout_centers, directions=directions, W1=prefix)
            write_json(bank_folder / "bank.json", {"training": train_info, "validation": validation_info,
                       "counts": ledger.snapshot(), "sha256": file_hash(bank_folder / "hessians.npz")})
            for count in counts:
                coefficients = random_coefficients(options["combinations"], count, np.random.default_rng(
                    stream_seed(random_settings["combination_entropy"], parent["stream_index"], repeat, count)))
                methods = [("random_64", coefficients, "residual")]
                if preset == "study":
                    methods = [("two_hessian", np.eye(count)[:1], "first"),
                               ("best_single", np.eye(count), "gap"),
                               ("random_1", coefficients[:1], "first"), *methods]
                    if count == options["primary_count"]:
                        methods += [(f"random_{number}", coefficients[:number], "residual")
                                    for number in options["candidate_ablation"]]
                for method, combinations, selection in methods:
                    run_id = f"{parent['id']}_repeat{repeat}_M{count}_{method}"
                    folder = output / "runs" / run_id
                    row = {"parent_id": parent["id"], "direction_repeat": repeat, "matrix_count": count,
                           "method": method, "run_id": run_id, "status": "complete",
                           "parameter_success": False, "prefix_source_mode": prefix_state["mode"],
                           "inherited_prefix_queries": prefix_state["counts"]["n_real_calls_total"],
                           "logical_hessian_queries": 60 if method == "two_hessian" else 30 * (count + 1)}
                    try:
                        hidden, anchor_output, diagnostics, arrays = recover_generalized(matrices[0], matrices[1:count + 1],
                            3, 4, combinations, selection=selection, gap_floor=options["gap_floor"],
                            rank_floor=options["rank_floor"], tie_tolerance=options["tie_tolerance"])
                        row.update(score(teacher, prefix, hidden, anchor_output, eval_inputs, eval_labels, evaluation["parameter_delta"]))
                        row["hessian_diagnostics"] = diagnostics
                        row["heldout_diagnostics"] = joint_residuals(matrices[0], heldout, arrays["C"])
                        save_npz(folder / "parameters.npz", W1=prefix, W2=hidden, a_raw=anchor_output)
                        save_npz(folder / "decomposition.npz", **arrays)
                        if count == options["primary_count"] and method == "random_64":
                            absolute_hidden = normalize_directions(arrays["V"], layer=2)
                            inverse = np.linalg.pinv(absolute_hidden)
                            absolute_output = np.diag(inverse @ matrices[0] @ inverse.T) / 12
                            save_npz(folder / "absolute_parameters.npz", W1=prefix, W2=absolute_hidden, a_raw=absolute_output)
                    except NumericalFailure as failure:
                        row.update(status="failed", failure_reason=failure.reason)
                    row["logical_learning_queries"] = row["inherited_prefix_queries"] + row["logical_hessian_queries"]
                    write_json(folder / "metrics.json", row)
                    rows.append(row)
            print(f"HESSIAN {parent['id']} repeat={repeat}", flush=True)
        primary_rows = [row for row in rows if row["parent_id"] == parent["id"] and
                        row["matrix_count"] == options["primary_count"] and row["method"] == "random_64" and row["status"] == "complete"]
        for repeat in regression["repeats"]:
            random = np.random.default_rng(stream_seed(random_settings["gaussian_entropy"], repeat))
            inputs = random.normal(size=(regression["samples"], 8))
            input_path = output / "data/gaussian" / f"inputs_repeat_{repeat}.npz"
            if not input_path.exists():
                save_npz(input_path, x=inputs)
            label_path = output / "data/gaussian" / f"labels_seed_{parent['seed']}_repeat_{repeat}.npz"
            if label_path.exists():
                with np.load(label_path, allow_pickle=False) as saved:
                    labels = saved["y"].copy()
            else:
                ledger = QueryLedger(len(inputs), 0)
                oracle = CountedRealOracle(teacher.forward, 8, ledger, batch_value=teacher.forward)
                labels = np.empty(len(inputs))
                with ledger.scope("training"):
                    for start in range(0, len(inputs), regression["query_batch_size"]):
                        stop = start + regression["query_batch_size"]
                        labels[start:stop] = oracle.batch(inputs[start:stop])
                physical["gaussian_labels"] += ledger.counts["n_real_calls_total"]
                save_npz(label_path, y=labels)
            for original in primary_rows:
                folder = output / "runs" / original["run_id"]
                normalizations = [("signed", "parameters.npz"), ("absolute", "absolute_parameters.npz")]
                for normalization, filename in normalizations:
                    with np.load(folder / filename, allow_pickle=False) as saved:
                        hidden = saved["W2"].copy()
                    fit_started = time.perf_counter()
                    fitted, numerical = ordinary_least_squares(hidden_features(inputs, [prefix, hidden], 4), labels)
                    fit_seconds = time.perf_counter() - fit_started
                    suffix = "" if normalization == "signed" else "_abs"
                    run_id = f"{original['run_id']}_OLS{repeat}{suffix}"
                    row = {key: original[key] for key in ("parent_id", "direction_repeat", "matrix_count",
                           "inherited_prefix_queries", "logical_hessian_queries", "prefix_source_mode")}
                    row.update(run_id=run_id, method="random_64_gaussian_ols" + suffix, status="complete",
                        training_repeat=repeat, training_samples=len(inputs), normalization=normalization,
                        wall_seconds_fit=fit_seconds, regression_diagnostics=numerical,
                        logical_learning_queries=original["logical_learning_queries"] + len(inputs))
                    row.update(score(teacher, prefix, hidden, fitted, eval_inputs, eval_labels, evaluation["parameter_delta"]))
                    save_npz(output / "runs" / run_id / "parameters.npz", W1=prefix, W2=hidden, a_raw=fitted)
                    write_json(output / "runs" / run_id / "metrics.json", row)
                    rows.append(row)
            print(f"OLS {parent['id']} repeat={repeat}", flush=True)
        write_json(output / "partial_records.json", rows)
    verification = compare_reference(rows, ROOT / "reference/expected_metrics.json") if verify_reference else None
    elapsed = time.perf_counter() - started
    manifest.update(status="complete", physical_queries_this_invocation=physical,
                    physical_learning_queries_this_invocation=sum(physical[key] for key in ("first_layer", "hessian_training", "gaussian_labels")),
                    physical_total_queries_this_invocation=sum(physical.values()), seconds=elapsed,
                    records=len(rows), reference_verification=verification)
    write_json(output / "records.json", rows)
    write_json(manifest_path, manifest)
    print(f"COMPLETE records={len(rows)} seconds={elapsed:.2f} output={output.relative_to(ROOT).as_posix()}", flush=True)
    if verification is not None and not verification["passed"]:
        raise RuntimeError(f"Historical reference verification failed: {verification}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reproduction.yaml")
    parser.add_argument("--mode", choices=("checkpoint", "full"), default="checkpoint")
    parser.add_argument("--preset", choices=("best", "study", "smoke"), default="best")
    parser.add_argument("--output", help="Output directory relative to the project root")
    parser.add_argument("--verify-reference", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    if args.preset == "smoke" and args.verify_reference:
        parser.error("Smoke uses a smaller sample budget; it is not a reference accuracy run")
    mode = "full" if args.preset == "smoke" else args.mode
    output = inside(args.output or f"results/reproduction_{args.preset}_{mode}")
    if not args.report_only:
        specification = yaml.safe_load(inside(args.config).read_text(encoding="utf-8"))
        run(specification, mode=mode, preset=args.preset, output=output, verify_reference=args.verify_reference)
    if not args.no_report:
        from ..reporting.reproduction import build_report
        build_report(output)


if __name__ == "__main__":
    main()
