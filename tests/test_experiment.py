"""Test experiment input boundaries and raw/aligned parameter exports."""
import json
import numpy as np
import pytest
import yaml

from aspire import ROOT
from aspire.architecture import Architecture
from aspire.config import resolve
from aspire.experiments.experiment_01 import load_target, first_layer_config, obtain_first_layer
from aspire.instances import generate
from aspire.io import inside
from aspire.query_ledger import CountedRealOracle, QueryLedger
from aspire.reporting.parameters import save_comparison
from aspire.teacher import Teacher


def test_fixed_inputs_and_first_layer_settings():
    target, reference = load_target()
    specification = yaml.safe_load((ROOT / "configs/experiment_01.yaml").read_text(encoding="utf-8"))
    assert target.weights[0].shape == (8, 3)
    assert specification["first_layer"] == reference["first_layer_configuration"]
    assert specification["public_bounds"] == reference["public_bounds"]
    configuration = first_layer_config(specification)
    assert configuration["sampler"]["endpoint_steps"] == 32
    assert configuration["recovery"]["moment_samples_by_layer"] == [16777216]


def test_first_layer_stops_before_hessian_measurements(tmp_path, monkeypatch):
    monkeypatch.setattr("aspire.io.ROOT", tmp_path)
    specification = yaml.safe_load((ROOT / "configs/experiment_01.yaml").read_text(encoding="utf-8"))
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
    specification = yaml.safe_load((ROOT / "configs/experiment_01.yaml").read_text(encoding="utf-8"))
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
    assert inside("data\\experiment_01\\ground_truth.npz") == ROOT / "data/experiment_01/ground_truth.npz"
    with pytest.raises(ValueError):
        inside("../outside.json")
