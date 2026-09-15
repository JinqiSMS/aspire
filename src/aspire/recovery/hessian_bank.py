"""Real-value Hessian observations; no teacher or derivative inputs."""
import numpy as np
from ..oracles.interpolation import hessian


def orthogonal_directions(dimension, count, rng):
    if dimension < 1 or count < 1 or count % dimension:
        raise ValueError("Use complete orthogonal bases")
    blocks = []
    for _ in range(count // dimension):
        q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        q *= np.where(np.diag(r) >= 0, 1., -1.)
        blocks.append(q.T)
    return np.concatenate(blocks)


def collect_hessians(suffix, directions, *, degree, tau, step, ledger,
                     dps=0, include_anchor=True, scope="final_hessian"):
    directions = np.asarray(directions, float)
    if directions.ndim != 2 or not np.isfinite(directions).all():
        raise ValueError("Expected finite direction rows")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-12):
        raise ValueError("Expected unit directions")
    if not (0 < tau < 1 and step > 0):
        raise ValueError("Invalid Hessian probe radii")
    centers = 1. + tau * directions
    if include_anchor:
        centers = np.vstack([np.ones(directions.shape[1]), centers])
    # All interpolation directions are normalized; nodes lie in [-1, 1].
    minimum = float(np.min(centers) - step)
    if minimum <= 0:
        raise ValueError("Hessian interpolation nodes must remain positive")
    before = ledger.snapshot()
    with ledger.scope(scope):
        matrices = np.asarray([
            hessian(suffix, y, degree=degree, step=step, dps=dps, ledger=ledger)
            for y in centers], float)
    after = ledger.snapshot()
    return matrices, centers, {
        "query_counts": {key: value - before.get(key, 0) for key, value in after.items()},
        "minimum_node_coordinate_bound": minimum,
        "directions": directions.tolist(), "include_anchor": include_anchor,
        "tau": tau, "step": step, "degree": degree,
    }
