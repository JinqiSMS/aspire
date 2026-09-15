"""Evaluator-only sphere quadrature for population ASPIRE moment gaps."""
from pathlib import Path
import os
import sys

sys.dont_write_bytecode = True
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from scipy.special import roots_legendre
from aspire.experiments.experiment_01 import load_target
from aspire.io import write_json
from aspire.recovery.moments import moment_directions


def population_moments(target, order):
    # In the true orthonormal first-layer basis, the first weight matrix is I.
    np.testing.assert_allclose(target.weights[0].T @ target.weights[0], np.eye(3), atol=1e-12)
    cosine, weights = roots_legendre(order)
    longitude = 2 * np.pi * np.arange(2 * order) / (2 * order)
    radius_xy = np.sqrt(1 - cosine * cosine)
    unit = np.stack([radius_xy[:, None] * np.cos(longitude),
                     radius_xy[:, None] * np.sin(longitude),
                     np.broadcast_to(cosine[:, None], (order, 2 * order))], axis=-1).reshape(-1, 3)
    sphere_weights = np.repeat(weights / (4 * order), 2 * order)
    k, degree = target.k, target.k ** 2
    hidden = unit ** k @ target.weights[1]
    value = hidden ** k @ target.a
    gradient = k * k * unit ** (k - 1) * ((hidden ** (k - 1) * target.a) @ target.weights[1].T)
    radial = value ** (-1 / degree)
    boundary = radial[:, None] * unit
    boundary_gradient = radial[:, None] ** (degree - 1) * gradient
    mass = sphere_weights * radial ** 3
    mass /= mass.sum()
    sx = 3 / 5 * (boundary.T * mass) @ boundary
    sg = 3 / (2 * degree + 1) * (boundary_gradient.T * mass) @ boundary_gradient
    eigenvalues, _, diagnostics = moment_directions(sx, sg)
    return {"quadrature_order": order, "directions": len(unit), "Sx": sx, "Sg": sg,
            "eigenvalues": eigenvalues, "relative_gap": diagnostics["moment_gap"],
            "symmetry_offdiagonal_norm": float(np.linalg.norm(sx-np.diag(np.diag(sx))) + np.linalg.norm(sg-np.diag(np.diag(sg))))}


def main():
    rows = []
    for k in (4, 6, 8):
        target, _ = load_target(k)
        estimates = [population_moments(target, order) for order in (64, 128, 256, 512)]
        if abs(estimates[-1]["relative_gap"] - estimates[-2]["relative_gap"]) > 1e-8:
            estimates.append(population_moments(target, 1024))
        row = {"k": k, "estimates": estimates,
               "last_gap_difference": abs(estimates[-1]["relative_gap"] - estimates[-2]["relative_gap"])}
        rows.append(row)
        print(f"k={k}: gap={estimates[-1]['relative_gap']:.12g}; grid difference={row['last_gap_difference']:.3g}", flush=True)
    write_json(PROJECT_ROOT / "results/activation_comparison/population_moments.json", {
        "purpose": "Evaluator-only population diagnostic; no information passed to the learner",
        "method": "Gauss-Legendre latitude and equispaced longitude; analytic radial integration",
        "real_learning_queries": 0, "rows": rows})


if __name__ == "__main__":
    main()
