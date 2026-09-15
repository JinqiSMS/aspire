"""Experiment 1: column recovery, ASPIRE moments, multi-Hessian recovery, and OLS."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import platform
import time

import numpy as np
import yaml

from .. import ROOT
from ..architecture import Architecture
from ..config import resolve
from ..evaluation.metrics import parameter_metrics, risk_batch
from ..io import inside, save_npz, write_json
from ..oracles.suffix import SuffixOracle
from ..query_ledger import CountedRealOracle, QueryLedger
from ..recovery.generalized_joint import recover_generalized, joint_residuals, random_coefficients
from ..recovery.hessian_bank import collect_hessians, orthogonal_directions
from ..recovery.network import recover_network
from ..recovery.output import hidden_features, ordinary_least_squares
from ..teacher import Teacher
from ..status import NumericalFailure


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_hash():
    files = sorted((ROOT / "src").rglob("*.py"))
    return hashlib.sha256(b"".join(path.relative_to(ROOT).as_posix().encode() +
        path.read_bytes().replace(b"\r\n", b"\n") for path in files)).hexdigest()


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "mpmath", "matplotlib", "PyYAML")}}


def first_layer_config(specification):
    settings = specification["first_layer"]
    return resolve({
        "architecture": specification["architecture"],
        "oracle": {"mode": "strict_real", "cache_size": 0,
                   "max_real_learning_queries": settings["query_budget"],
                   "wall_seconds": settings["wall_seconds"]},
        "recovery": {"moment_samples_by_layer": [settings["samples"]],
                     "first_span_points": settings["first_span_points"],
                     "original_direction_step": settings["original_direction_step"],
                     "reduced_gradient_step_factor": settings["reduced_gradient_step_factor"]},
        "sampler": {"mode": "independent_endpoints", "endpoint_steps": settings["steps"],
                    "batch_size": settings["batch_size"],
                    "bisection_absolute_tolerance": settings["bisection_absolute_tolerance"],
                    "max_bisection_iterations": settings["max_bisection_iterations"]},
    })


def load_target(k=4):
    folder = ROOT / "data/experiment_01"
    manifest = read_json(folder / "manifest.json")
    path = folder / "ground_truth.npz"
    if file_hash(path) != manifest["files"][path.name]:
        raise ValueError("Ground-truth file checksum mismatch")
    with np.load(path, allow_pickle=False) as saved:
        target = Teacher([saved["W1"].copy(), saved["W2"].copy()], saved["a"].copy(), k)
    return target, manifest


def load_first_layer_from(source, architecture, specification, reference):
    """Validate a completed first layer before reusing it across source versions."""
    source = inside(source)
    manifest = read_json(source / "run.json")
    signature = manifest["signature"]
    for section in ("architecture", "public_bounds", "first_layer"):
        if signature["configuration"][section] != specification[section]:
            raise ValueError(f"First-layer source has different {section} settings")
    if signature["target_sha256"] != reference["files"]["ground_truth.npz"]:
        raise ValueError("First-layer source belongs to a different target")
    state = read_json(source / "first_layer.json")
    parameters = source / "first_layer.npz"
    if file_hash(parameters) != state["parameter_sha256"]:
        raise ValueError("First-layer source checksum mismatch")
    if not state.get("stages") or state["stages"][0].get("status") != "complete":
        raise ValueError("First-layer source is incomplete")
    with np.load(parameters, allow_pickle=False) as saved:
        weights = saved["W1"].copy()
    expected_shape = (architecture.d, architecture.hidden_widths[0])
    if (weights.shape != expected_shape or not np.isfinite(weights).all() or
            np.linalg.matrix_rank(weights) != expected_shape[1]):
        raise ValueError("First-layer source has invalid weights")
    state["reuse_source"] = {"directory": source.relative_to(ROOT).as_posix(),
        "source_hash": signature["source_hash"], "parameter_sha256": state["parameter_sha256"],
        "run_sha256": file_hash(source / "run.json"), "origin": state.get("origin")}
    return weights, state


def obtain_first_layer(real_oracle, architecture, specification, output, use_checkpoint, reference,
                       first_layer_from=None):
    """Only the real oracle, public architecture/bounds, and fixed settings reach recovery."""
    started = time.perf_counter()
    parameters = output / "first_layer.npz"
    state_path = output / "first_layer.json"
    if use_checkpoint and first_layer_from is not None:
        raise ValueError("Choose only one first-layer source")
    imported = (load_first_layer_from(first_layer_from, architecture, specification, reference)
                if first_layer_from is not None else None)
    if state_path.exists():
        state = read_json(state_path)
        if file_hash(parameters) != state["parameter_sha256"]:
            raise ValueError("Saved first-layer checksum mismatch")
        with np.load(parameters, allow_pickle=False) as saved:
            if imported is not None and not np.array_equal(saved["W1"], imported[0]):
                raise ValueError("Output already contains a different first layer; choose a new --output")
            return saved["W1"].copy(), state, "saved_first_layer", 0, time.perf_counter() - started
    if imported is not None:
        weights, state = imported
        origin, new_queries = "imported_first_layer", 0
    elif use_checkpoint:
        if architecture.k != 4:
            raise ValueError("The provided first-layer checkpoint uses k=4; compute a fresh first layer")
        if (specification["first_layer"] != reference["first_layer_configuration"] or
                specification["public_bounds"] != reference["public_bounds"]):
            raise ValueError("The first-layer checkpoint has different settings")
        path = ROOT / "data/experiment_01/first_layer.npz"
        if file_hash(path) != reference["files"][path.name]:
            raise ValueError("First-layer checkpoint checksum mismatch")
        with np.load(path, allow_pickle=False) as saved:
            weights = saved["W1"].copy()
        state = deepcopy(reference["first_layer"])
        origin, new_queries = "provided_first_layer", 0
    else:
        config = first_layer_config(specification)
        learner = {key: deepcopy(config[key]) for key in ("oracle", "recovery", "sampler", "final_layer")}
        print(f"FIRST LAYER: {specification['first_layer']['samples']:,} chains, "
              f"{specification['first_layer']['steps']} steps per chain", flush=True)
        recovered = recover_network(real_oracle, architecture=architecture,
            public_bounds=specification["public_bounds"], config=learner,
            rng=np.random.default_rng(specification["first_layer"]["algorithm_seed"]), stop_after_layer=1)
        if recovered["status"] != "prefix_complete":
            write_json(output / "failure.json", {"stage": "first_layer", "query_counts": real_oracle.ledger.snapshot(),
                       **{key: recovered[key] for key in ("status", "failure_reason", "stages")}})
            raise NumericalFailure(recovered["failure_reason"])
        weights = recovered["hidden_weights"][0].copy()
        state = {"counts": real_oracle.ledger.snapshot(), "stages": recovered["stages"],
                 "wall_seconds": time.perf_counter() - started}
        origin, new_queries = "fresh_real_queries", state["counts"]["n_real_calls_total"]
        del recovered
        gc.collect()
    save_npz(parameters, W1=weights)
    state.update(parameter_sha256=file_hash(parameters), origin=origin)
    write_json(state_path, state)
    return weights, state, origin, new_queries, time.perf_counter() - started


def recover_second_layer(suffix, settings, dimension, degree, ledger):
    """Measure Hessians and select a mixture using its observed joint residual."""
    directions = orthogonal_directions(dimension, settings["directions"],
                                       np.random.default_rng(settings["direction_seed"]))
    matrices, centers, information = collect_hessians(suffix, directions, degree=degree,
        tau=settings["tau"], step=settings["step"], ledger=ledger)
    combinations = random_coefficients(settings["combinations"], settings["directions"],
                                        np.random.default_rng(settings["combination_seed"]))
    weights, _, diagnostics, arrays = recover_generalized(matrices[0], matrices[1:], dimension,
        degree, combinations, selection="residual", gap_floor=settings["gap_floor"],
        rank_floor=settings["rank_floor"], tie_tolerance=settings["tie_tolerance"])
    heldout_directions = orthogonal_directions(dimension, settings["heldout_directions"],
                                              np.random.default_rng(settings["heldout_seed"]))
    heldout, _, _ = collect_hessians(suffix, heldout_directions, degree=degree,
        tau=settings["tau"], step=settings["step"], ledger=ledger,
        include_anchor=False, scope="validation")
    diagnostics["heldout"] = joint_residuals(matrices[0], heldout, arrays["C"])
    diagnostics["measurements"] = information
    return weights, diagnostics


def fit_output(real_oracle, weights, settings, dimension, degree):
    inputs = np.random.default_rng(settings["seed"]).normal(size=(settings["samples"], dimension))
    labels = np.empty(len(inputs))
    with real_oracle.ledger.scope("training"):
        for start in range(0, len(inputs), settings["batch_size"]):
            stop = start + settings["batch_size"]
            labels[start:stop] = real_oracle.batch(inputs[start:stop])
    return ordinary_least_squares(hidden_features(inputs, weights, degree), labels)


def evaluate(target, weights, coefficients, settings):
    """Ground truth is used here only after parameter estimation has finished."""
    metrics = parameter_metrics(target, {"hidden_weights": weights,
        "output_weights_raw": coefficients, "status": "complete"}, settings["parameter_delta"])
    metrics.update(coefficient_l1_norm=float(np.linalg.norm(coefficients, 1)),
                   joint_parameter_error=max(*metrics["per_layer_errors"], metrics["output_l1_error"]))
    basis, singular, _ = np.linalg.svd(np.concatenate([target.weights[0], weights[0]], axis=1), full_matrices=False)
    rank = int(np.count_nonzero(singular > 1e-10 * singular[0]))
    directions = np.random.default_rng(settings["seed"]).normal(size=(settings["directions"], rank))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    inputs = directions @ basis[:, :rank].T
    ledger = QueryLedger(len(inputs), 0)
    oracle = CountedRealOracle(target.forward, inputs.shape[1], ledger, batch_value=target.forward)
    with ledger.scope("evaluation"):
        labels = oracle.batch(inputs)
    metrics["angular_gaussian_nmse"] = risk_batch(Teacher(weights, coefficients, target.k).forward(inputs), labels)["nmse"]
    return metrics, ledger.counts["n_real_calls_total"]


def _run(specification, output, use_checkpoint=False, first_layer_from=None):
    started = time.perf_counter()
    output = inside(output)
    output.mkdir(parents=True, exist_ok=True)
    architecture = Architecture(**specification["architecture"])
    if architecture.d != 8 or list(architecture.hidden_widths) != [3, 3]:
        raise ValueError("Experiment 1 uses architecture 8 -> 3 -> 3 -> 1")
    if use_checkpoint and architecture.k != 4:
        raise ValueError("The provided first-layer checkpoint uses k=4; compute a fresh first layer")
    if use_checkpoint and first_layer_from is not None:
        raise ValueError("Choose only one first-layer source")
    target, reference = load_target(architecture.k)
    signature = {"configuration": specification, "source_hash": implementation_hash(),
                 "target_sha256": reference["files"]["ground_truth.npz"]}
    manifest_path = output / "run.json"
    if manifest_path.exists() and read_json(manifest_path)["signature"] != signature:
        raise ValueError("Output contains a different configuration or source version; choose a new --output")
    manifest = {"experiment": "experiment_01", "signature": signature, "environment": environment(),
                "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                "activation_k": architecture.k, "active_stage": "first_layer"}
    write_json(manifest_path, manifest)
    settings = specification["first_layer"]
    first_ledger = QueryLedger(settings["query_budget"], 0, settings["wall_seconds"])
    real = CountedRealOracle(target.forward, architecture.d, first_ledger, batch_value=target.forward)
    first, state, origin, new_first_queries, first_seconds = obtain_first_layer(
        real, architecture, specification, output, use_checkpoint, reference, first_layer_from)
    manifest.update(active_stage="second_layer", first_layer_origin=origin,
                    first_layer_queries_this_execution=new_first_queries,
                    first_layer_reuse_source=state.get("reuse_source"))
    write_json(manifest_path, manifest)
    print(f"FIRST LAYER READY: {origin}", flush=True)
    hessian_started = time.perf_counter()
    hessian_ledger = QueryLedger(100000, 0)
    oracle = CountedRealOracle(target.forward, architecture.d, hessian_ledger)
    suffix = SuffixOracle(oracle, architecture, 2, [first], first_layer_config(specification)["oracle"], hessian_ledger)
    try:
        second, hessian_diagnostics = recover_second_layer(suffix, specification["hessian"],
            architecture.hidden_widths[-1], architecture.k, hessian_ledger)
    except (NumericalFailure, np.linalg.LinAlgError) as error:
        write_json(output / "failure.json", {"stage": "second_layer", "failure_reason": str(error),
                   "query_counts": hessian_ledger.snapshot(), "details": getattr(error, "details", {})})
        raise NumericalFailure(str(error)) from error
    save_npz(output / "hidden_layers.npz", W1=first, W2=second)
    hessian_seconds = time.perf_counter() - hessian_started
    print("SECOND LAYER READY: multi-Hessian generalized eigendecomposition", flush=True)
    manifest.update(active_stage="output_layer", hessian_counts=hessian_ledger.snapshot())
    write_json(manifest_path, manifest)
    regression_started = time.perf_counter()
    regression_ledger = QueryLedger(specification["regression"]["samples"], 0)
    oracle = CountedRealOracle(target.forward, architecture.d, regression_ledger, batch_value=target.forward)
    try:
        coefficients, regression_diagnostics = fit_output(oracle, [first, second], specification["regression"],
                                                          architecture.d, architecture.k)
    except (AssertionError, np.linalg.LinAlgError, NumericalFailure) as error:
        write_json(output / "failure.json", {"stage": "output_layer", "failure_reason": str(error),
                   "query_counts": regression_ledger.snapshot()})
        raise NumericalFailure("output_regression_failed", detail=str(error)) from error
    regression_seconds = time.perf_counter() - regression_started
    metrics, evaluation_queries = evaluate(target, [first, second], coefficients, specification["evaluation"])
    metrics.update(status="complete", activation_k=architecture.k,
                   hessian_diagnostics=hessian_diagnostics, regression_diagnostics=regression_diagnostics)
    current = {"first_layer": new_first_queries,
               "hessian_training": hessian_ledger.counts["n_real_calls_final_hessian"],
               "hessian_validation": hessian_ledger.counts["n_real_calls_validation"],
               "gaussian_labels": regression_ledger.counts["n_real_calls_total"],
               "prediction_evaluation": evaluation_queries}
    learning = state["counts"]["n_real_calls_total"] + current["hessian_training"] + current["gaussian_labels"]
    queries = {"current_execution": current, "current_execution_total": sum(current.values()),
               "complete_learning_total": learning, "first_layer_origin": origin,
               "first_layer_breakdown": state["counts"],
               "complete_total_with_evaluation": learning + current["hessian_validation"] + evaluation_queries}
    write_json(output / "metrics.json", metrics)
    write_json(output / "queries.json", queries)
    from ..reporting.parameters import save_comparison, create_figures
    save_comparison(output, target.weights, target.a, [first, second], coefficients, activation_k=architecture.k)
    manifest.update(status="complete", active_stage="complete", first_layer_origin=origin, computation_seconds=time.perf_counter() - started,
                    first_layer_seconds_this_execution=first_seconds,
                    hessian_seconds=hessian_seconds, regression_seconds=regression_seconds)
    write_json(manifest_path, manifest)
    create_figures(output)
    print((output / "weights.txt").read_text(encoding="utf-8"))
    print(f"Layer errors: {metrics['per_layer_errors']}; output L1 error: {metrics['output_l1_error']:.12g}")
    print(f"OUTPUT: {output.relative_to(ROOT).as_posix()}", flush=True)
    return metrics


def run(specification, output, use_checkpoint=False, first_layer_from=None):
    started = time.perf_counter()
    try:
        return _run(specification, output, use_checkpoint, first_layer_from)
    except NumericalFailure as error:
        output = inside(output)
        manifest = read_json(output / "run.json")
        failure = read_json(output / "failure.json")
        target, _ = load_target(specification["architecture"]["k"])
        weights = []
        if (output / "hidden_layers.npz").exists():
            with np.load(output / "hidden_layers.npz") as saved:
                weights = [saved["W1"].copy(), saved["W2"].copy()]
        elif (output / "first_layer.npz").exists():
            with np.load(output / "first_layer.npz") as saved:
                weights = [saved["W1"].copy()]
        metrics = parameter_metrics(target, {"hidden_weights": weights, "output_weights_raw": None,
            "status": "failed"}, specification["evaluation"]["parameter_delta"])
        metrics.update(status="failed", activation_k=target.k, failure_stage=failure["stage"], failure_reason=error.reason)
        manifest.update(status="failed", failure_reason=error.reason, computation_seconds=time.perf_counter() - started)
        write_json(output / "run.json", manifest)
        write_json(output / "metrics.json", metrics)
        write_json(output / "queries.json", {"first_layer_queries_this_execution": manifest.get("first_layer_queries_this_execution", 0),
                   "hessian_counts": manifest.get("hessian_counts", {}), "failed_stage": failure["stage"],
                   "failed_stage_counts": failure["query_counts"]})
        print(f"FAILED k={target.k} stage={failure['stage']}: {error.reason}", flush=True)
        return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_01.yaml")
    parser.add_argument("--activation", type=int, choices=(4, 6, 8), help="Override the activation exponent")
    parser.add_argument("--output", help="Output directory inside the project")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--use-first-layer-checkpoint", action="store_true",
                        help="Load the supplied k=4 first-layer estimate and run the later stages")
    source.add_argument("--first-layer-from", help="Reuse a validated first layer from an experiment directory")
    parser.add_argument("--figures-only", action="store_true", help="Plot existing weight files without oracle queries")
    args = parser.parse_args()
    specification = yaml.safe_load(inside(args.config).read_text(encoding="utf-8"))
    if args.activation is not None:
        specification["architecture"]["k"] = args.activation
    k = specification["architecture"]["k"]
    stem = "results/experiment_01" + (f"_k{k}" if k != 4 else "")
    output = inside(args.output or (stem + "_checkpoint" if args.use_first_layer_checkpoint else stem))
    if args.figures_only:
        from ..reporting.parameters import create_figures
        create_figures(output)
        return
    result = run(specification, output, args.use_first_layer_checkpoint, args.first_layer_from)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
