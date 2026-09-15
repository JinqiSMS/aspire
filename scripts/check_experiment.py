"""Check parameter files, evaluation identities, and Experiment 1 metrics."""
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aspire.teacher import Teacher
from aspire.io import inside


def main():
    folder = inside(sys.argv[1] if len(sys.argv) > 1 else "results/experiment_01")
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    k = manifest["signature"]["configuration"]["architecture"]["k"]
    with np.load(folder / "weights.npz", allow_pickle=False) as weights:
        raw = Teacher([weights["W1_raw"], weights["W2_raw"]], weights["a_raw"], k)
        aligned = Teacher([weights["W1_aligned"], weights["W2_aligned"]], weights["a_aligned"], k)
        assert np.all(weights["W2_raw"] >= 0)
        np.testing.assert_allclose(weights["W2_raw"].sum(axis=0), 1., atol=1e-14)
        inputs = np.random.default_rng(107).normal(size=(100, 8))
        np.testing.assert_allclose(raw.forward(inputs), aligned.forward(inputs), rtol=1e-10, atol=1e-10)
        for name in ("W1", "W2", "a"):
            np.testing.assert_allclose(weights[f"{name}_difference"], weights[f"{name}_aligned"] - weights[f"{name}_true"])
        np.testing.assert_allclose(np.linalg.norm(weights["a_difference"], 1), metrics["output_l1_error"], atol=1e-14)
        for layer in (1, 2):
            np.testing.assert_allclose(np.linalg.norm(weights[f"W{layer}_difference"], 2),
                                       metrics["per_layer_errors"][layer-1], atol=1e-14)
    assert metrics["hessian_diagnostics"]["normalization"] == "coordinatewise_absolute_value_and_l1"
    queries = json.loads((folder / "queries.json").read_text(encoding="utf-8"))
    configuration = manifest["signature"]["configuration"]
    counts = queries["first_layer_breakdown"]
    assert counts["n_kept_samples"] == configuration["first_layer"]["samples"]
    assert counts["n_hr_transitions"] == configuration["first_layer"]["samples"] * configuration["first_layer"]["steps"]
    assert queries["current_execution_total"] == sum(queries["current_execution"].values())
    assert queries["current_execution"]["gaussian_labels"] == configuration["regression"]["samples"]
    if manifest["first_layer_origin"] == "fresh_real_queries":
        assert queries["current_execution"]["first_layer"] == counts["n_real_calls_total"] > 0
    else:
        assert queries["current_execution"]["first_layer"] == 0
    assert not list(folder.rglob("*.html"))
    for name in ("weight_comparison", "parameter_errors", "output_coefficients"):
        for extension in ("png", "pdf"):
            assert (folder / "figures" / f"{name}.{extension}").stat().st_size > 1000
    print("Experiment 1: metric and parameter checks passed")


if __name__ == "__main__":
    main()
