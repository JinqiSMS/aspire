"""Check stage boundaries, portable paths, and frozen reproduction inputs."""
import json
import numpy as np
import pytest
import yaml

from aspire import ROOT
from aspire.architecture import Architecture
from aspire.config import resolve
from aspire.experiments.reproduction import file_hash, stream_seed, first_layer_config, load_target
from aspire.instances import generate
from aspire.io import inside
from aspire.query_ledger import QueryLedger, CountedRealOracle
from aspire.recovery.network import recover_network


def test_prefix_boundary_does_not_query_final_hessians():
    config = resolve({"architecture": {"d": 2, "hidden_widths": [2, 2]},
                      "recovery": {"moment_samples_by_layer": [16]},
                      "sampler": {"burn_in": 4, "thin": 1}})
    architecture = Architecture(**config["architecture"])
    teacher, _ = generate(architecture, config["teacher"], np.random.default_rng(12))
    ledger = QueryLedger(20000, 0)
    result = recover_network(CountedRealOracle(teacher.forward, 2, ledger),
        architecture=architecture, public_bounds={"kappa0": 10, "mu": 0.01},
        config=config, rng=np.random.default_rng(4), stop_after_layer=1)
    assert result["status"] == "prefix_complete"
    assert len(result["hidden_weights"]) == 1
    assert ledger.snapshot()["n_hessian_calls"] == 0
    assert result["output_weights_raw"] is None
    assert ledger.snapshot()["n_real_calls_total"] > 0
    before = ledger.snapshot()
    restored = recover_network(CountedRealOracle(teacher.forward, 2, ledger),
        architecture=architecture, public_bounds={"kappa0": 10, "mu": 0.01},
        config=config, rng=np.random.default_rng(4), stop_after_layer=1,
        restored={"weights": result["hidden_weights"], "stages": result["stages"]})
    assert restored["status"] == "prefix_complete" and ledger.snapshot() == before


def test_reference_files_and_explicit_stream_indices():
    specification = yaml.safe_load((ROOT / "configs/reproduction.yaml").read_text(encoding="utf-8"))
    reference = json.loads((ROOT / "reference/manifest.json").read_text(encoding="utf-8"))
    for parent in specification["parents"]:
        teacher = load_target(parent, reference)
        assert teacher.weights[0].shape == (8, 3)
        item = reference["prefixes"][parent["id"]]
        assert file_hash(ROOT / "reference/prefixes" / f"{parent['id']}.npz") == item["parameter_sha256"]
        config = first_layer_config(specification, parent)
        assert config["sampler"]["endpoint_steps"] == item["steps"]
        assert config["recovery"]["moment_samples_by_layer"] == [item["samples"]]
    best = next(parent for parent in specification["parents"] if parent["id"] == specification["best_parent"])
    assert best["stream_index"] == 2
    assert stream_seed(2026091521, best["stream_index"], 1, 0) == stream_seed(2026091521, 2, 1, 0)


def test_historical_windows_paths_are_portable():
    assert inside("reference\\prefixes\\T32_large_seed0.npz") == ROOT / "reference/prefixes/T32_large_seed0.npz"
    with pytest.raises(ValueError):
        inside("..\\outside.json")
