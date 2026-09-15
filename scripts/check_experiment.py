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
    folder = inside(sys.argv[1] if len(sys.argv) > 1 else "results/experiment_01_checkpoint")
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    measured = [*metrics["per_layer_errors"], metrics["output_l1_error"]]
    np.testing.assert_allclose(measured, [0.013037289900738625, 0.030543022613746373, 0.058729695626], rtol=0, atol=1e-7)
    with np.load(folder / "weights.npz", allow_pickle=False) as weights:
        raw = Teacher([weights["W1_raw"], weights["W2_raw"]], weights["a_raw"], 4)
        aligned = Teacher([weights["W1_aligned"], weights["W2_aligned"]], weights["a_aligned"], 4)
        inputs = np.random.default_rng(107).normal(size=(100, 8))
        np.testing.assert_allclose(raw.forward(inputs), aligned.forward(inputs), rtol=1e-10, atol=1e-10)
        for name in ("W1", "W2", "a"):
            np.testing.assert_allclose(weights[f"{name}_difference"], weights[f"{name}_aligned"] - weights[f"{name}_true"])
        np.testing.assert_allclose(np.linalg.norm(weights["a_difference"], 1), metrics["output_l1_error"], atol=1e-14)
    assert not list(folder.rglob("*.html"))
    for name in ("weight_comparison", "parameter_errors", "output_coefficients"):
        for extension in ("png", "pdf"):
            assert (folder / "figures" / f"{name}.{extension}").stat().st_size > 1000
    print("Experiment 1: metric and parameter checks passed")


if __name__ == "__main__":
    main()
