"""Test experiment input boundaries and raw/aligned parameter exports."""
import json
import numpy as np
import pytest
import yaml

from aspire import ROOT
from aspire.architecture import Architecture
from aspire.config import resolve
from aspire.experiments.experiment import (load_target, first_layer_config, obtain_first_layer,
                                           load_first_layer_from, file_hash)
from aspire.instances import generate
from aspire.io import inside, save_npz, write_json
from aspire.query_ledger import CountedRealOracle, QueryLedger
from aspire.reporting.parameters import save_comparison
from aspire.teacher import Teacher
from aspire.oracles.suffix import SuffixOracle
from aspire.oracles.interpolation import hessian
from aspire.numeric import even_power


def test_fixed_inputs_and_first_layer_settings():
    target, reference = load_target()
    specification = yaml.safe_load((ROOT / "configs/experiment_k4.yaml").read_text(encoding="utf-8"))
    assert target.weights[0].shape == (8, 3)
    assert specification["first_layer"] == reference["first_layer_configuration"]
    assert specification["public_bounds"] == reference["public_bounds"]
    configuration = first_layer_config(specification)
    assert configuration["sampler"]["endpoint_steps"] == 32
    assert configuration["recovery"]["moment_samples_by_layer"] == [16777216]


def test_first_layer_stops_before_hessian_measurements(tmp_path, monkeypatch):
    monkeypatch.setattr("aspire.io.ROOT", tmp_path)
    specification = yaml.safe_load((ROOT / "configs/experiment_k4.yaml").read_text(encoding="utf-8"))
    specification["architecture"] = {"d": 2, "hidden_widths": [2, 2], "k": 4}
    specification["public_bounds"] = {"kappa0": 10, "mu": 0.01}
    specification["first_layer"].update(samples=32, steps=4, batch_size=16, query_budget=100000)
    architecture = Architecture(**specification["architecture"])
    target, _ = generate(architecture, resolve({})["teacher"], np.random.default_rng(12))
    ledger = QueryLedger(100000, 0)
    oracle = CountedRealOracle(target.forward, 2, ledger, batch_value=target.forward)
    weights, state, origin, count, _ = obtain_first_layer(oracle, architecture, specification, tmp_path, False, {})
    assert weights.shape == (2, 2) and origin == "fresh_real_queries"
    assert count == ledger.counts["n_real_calls_total"] > 0
    assert ledger.counts["n_hessian_calls"] == 0
    before = ledger.snapshot()
    saved, _, origin, count, _ = obtain_first_layer(oracle, architecture, specification, tmp_path, False, {})
    np.testing.assert_array_equal(saved, weights)
    assert count == 0 and origin == "saved_first_layer" and ledger.snapshot() == before


def test_checkpoint_rejects_changed_sampling_settings(tmp_path, monkeypatch):
    monkeypatch.setattr("aspire.io.ROOT", tmp_path)
    target, reference = load_target()
    specification = yaml.safe_load((ROOT / "configs/experiment_k4.yaml").read_text(encoding="utf-8"))
    specification["first_layer"]["algorithm_seed"] += 1
    ledger = QueryLedger(100, 0)
    with pytest.raises(ValueError, match="different settings"):
        obtain_first_layer(CountedRealOracle(target.forward, 8, ledger),
            Architecture(**specification["architecture"]), specification, tmp_path, True, reference)
    assert ledger.counts["n_real_calls_total"] == 0


def test_parameter_exports_preserve_permutation_and_sign_symmetry(tmp_path, monkeypatch):
    monkeypatch.setattr("aspire.io.ROOT", tmp_path)
    random = np.random.default_rng(19)
    truth = [random.normal(size=(5, 3)), random.uniform(size=(3, 3))]
    coefficients = np.array([0.2, 0.3, 0.5])
    first_order, second_order = np.array([2, 0, 1]), np.array([1, 2, 0])
    estimates = [truth[0][:, first_order] * np.array([-1., 1., -1.]),
                 truth[1][first_order][:, second_order]]
    estimated_output = coefficients[second_order]
    saved = save_comparison(tmp_path, truth, coefficients, estimates, estimated_output)
    with np.load(tmp_path / "weights.npz") as arrays:
        for name in ("W1", "W2", "a"):
            np.testing.assert_allclose(arrays[f"{name}_aligned"], arrays[f"{name}_true"], atol=1e-14)
        np.testing.assert_array_equal(arrays["W1_raw"], estimates[0])
        aligned = Teacher([arrays["W1_aligned"], arrays["W2_aligned"]], arrays["a_aligned"], 4)
    inputs = random.normal(size=(20, 5))
    np.testing.assert_allclose(Teacher(estimates, estimated_output, 4).forward(inputs), aligned.forward(inputs), rtol=1e-12)
    assert len((tmp_path / "weights.csv").read_text().splitlines()) == 28
    serialized = json.loads((tmp_path / "weights.json").read_text())
    assert serialized["permutations"] == saved["permutations"]


def test_output_paths_are_bounded():
    assert inside("data\\experiment\\ground_truth.npz") == ROOT / "data/experiment/ground_truth.npz"
    with pytest.raises(ValueError):
        inside("../outside.json")


@pytest.mark.parametrize("k", [6, 8])
def test_higher_activation_real_suffix_hessian_and_checkpoint_guard(tmp_path, monkeypatch, k):
    monkeypatch.setattr("aspire.io.ROOT", tmp_path)
    target, reference = load_target(k)
    baseline, _ = load_target(4)
    for actual, expected in zip(target.weights, baseline.weights):
        np.testing.assert_array_equal(actual, expected)
    specification = yaml.safe_load((ROOT / "configs/experiment_k4.yaml").read_text(encoding="utf-8"))
    specification["architecture"]["k"] = k
    architecture = Architecture(**specification["architecture"])
    ledger = QueryLedger(10000, 0)
    oracle = CountedRealOracle(target.forward, 8, ledger)
    with pytest.raises(ValueError, match="checkpoint uses k=4"):
        obtain_first_layer(oracle, architecture, specification, tmp_path, True, reference)
    assert ledger.counts["n_real_calls_total"] == 0
    suffix = SuffixOracle(oracle, architecture, 2, [target.weights[0]], first_layer_config(specification)["oracle"], ledger)
    point = np.array([0.9, 1.0, 1.1])
    measured = hessian(suffix, point, degree=k, step=0.1, ledger=ledger)
    exact_suffix = Teacher([target.weights[1]], target.a, k)
    np.testing.assert_allclose(measured, exact_suffix.hessian(point), rtol=1e-8, atol=1e-8)
    assert ledger.counts["n_real_calls_total"] == 6 * (k + 1)


@pytest.mark.parametrize("k", [4, 6, 8])
def test_even_power_preserves_batch_values(k):
    values = np.concatenate([-np.logspace(-12, 12, 100), [0.0], np.logspace(-12, 12, 100)])
    np.testing.assert_allclose(even_power(values, k), values ** k, rtol=1e-15, atol=0)
    target, _ = load_target(k)
    inputs = np.random.default_rng(42).normal(size=(256, 8))
    scalar = np.array([target.forward(row) for row in inputs])
    np.testing.assert_allclose(target.forward(inputs), scalar, rtol=1e-12, atol=1e-14)


@pytest.fixture
def stored_first_layer(tmp_path, monkeypatch):
    target, reference = load_target(6)
    specification = yaml.safe_load((ROOT / "configs/experiment_k6.yaml").read_text(encoding="utf-8"))
    monkeypatch.setattr("aspire.io.ROOT", tmp_path)
    monkeypatch.setattr("aspire.experiments.experiment.ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    save_npz(source / "first_layer.npz", W1=target.weights[0])
    write_json(source / "first_layer.json", {"parameter_sha256": file_hash(source / "first_layer.npz"),
        "stages": [{"layer": 1, "status": "complete"}], "counts": {"n_real_calls_total": 123},
        "wall_seconds": 1.0, "origin": "fresh_real_queries"})
    write_json(source / "run.json", {"signature": {"configuration": specification,
        "source_hash": "earlier-implementation", "target_sha256": reference["files"]["ground_truth.npz"]}})
    return source, specification, target, reference


def test_explicit_first_layer_reuse_has_zero_queries(stored_first_layer, tmp_path, monkeypatch):
    source, specification, target, reference = stored_first_layer
    def reject_sampling(*args, **kwargs):
        pytest.fail("Reusing a first layer must not start sampling")
    monkeypatch.setattr("aspire.experiments.experiment.recover_network", reject_sampling)
    architecture = Architecture(**specification["architecture"])
    ledger = QueryLedger(100, 0)
    oracle = CountedRealOracle(target.forward, 8, ledger)
    output = tmp_path / "output"
    output.mkdir()
    weights, state, origin, new_queries, _ = obtain_first_layer(
        oracle, architecture, specification, output, False, reference, first_layer_from=source)
    np.testing.assert_array_equal(weights, target.weights[0])
    assert origin == "imported_first_layer" and new_queries == 0
    assert ledger.counts["n_real_calls_total"] == 0
    assert state["counts"]["n_real_calls_total"] == 123
    assert state["reuse_source"]["source_hash"] == "earlier-implementation"
    assert state["reuse_source"]["parameter_sha256"] == file_hash(source / "first_layer.npz")


@pytest.mark.parametrize("section", ["architecture", "first_layer", "public_bounds", "target", "checksum"])
def test_explicit_first_layer_reuse_rejects_mismatches(stored_first_layer, section):
    source, specification, target, reference = stored_first_layer
    if section == "architecture":
        specification[section]["k"] = 8
    elif section == "first_layer":
        specification[section]["algorithm_seed"] += 1
    elif section == "public_bounds":
        specification[section]["mu"] *= 2
    elif section == "target":
        reference["files"]["ground_truth.npz"] = "different-target"
    else:
        with (source / "first_layer.npz").open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError, match="different|checksum"):
        load_first_layer_from(source, Architecture(**specification["architecture"]), specification, reference)
