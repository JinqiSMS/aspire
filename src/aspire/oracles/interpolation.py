"""S1 alg:directional-derivative; S3 lem:stable-gradient-from-suffix."""
from contextlib import nullcontext
import mpmath as mp
import numpy as np
from ..numeric import interpolation_rule, array_mp, is_mp, finite
from ..status import NumericalFailure

def directional_derivative(oracle, x, v, *, degree, order=1, step=.25, dps=0, ledger=None):
    if step <= 0: raise ValueError("Positive interpolation step required")
    hp = bool(dps or is_mp(x) or is_mp(v))
    precision = dps or (mp.mp.dps if hp else 0)
    if ledger: ledger.tick("n_directional_calls")
    with mp.workdps(precision) if hp else nullcontext():
        x = array_mp(x) if hp else np.asarray(x, float)
        v = array_mp(v) if hp else np.asarray(v, float)
        norm = mp.sqrt(mp.fsum(t*t for t in v)) if hp else np.linalg.norm(v)
        if norm == 0: return mp.mpf(0) if hp else 0.0
        h = mp.mpf(step) if hp else step
        nodes, weights = interpolation_rule(degree, order, precision)
        values = [oracle(x + h * s * v / norm) for s in nodes]
        total = mp.fsum(w*y for w, y in zip(weights, values)) if hp else np.dot(weights, values)
        result = total * (norm / h) ** order
        if not finite(result): raise NumericalFailure("interpolation_unstable", degree=degree)
        return result

def gradient(oracle, x, *, degree, step, dps=0, ledger=None):
    n = len(x)
    return np.array([directional_derivative(oracle, x, e, degree=degree, step=step,
                                          dps=dps, ledger=ledger) for e in np.eye(n)],
                    dtype=object if dps or is_mp(x) else float)


def gradient_batch(batch_oracle, points, *, degree, step, ledger=None, batch_size=4096):
    """The same coordinate Chebyshev rule, vectorized over independent points.

    No derivatives are supplied by the value oracle. Every interpolation node
    is evaluated, including repeated center nodes, and charged by that oracle.
    """
    if step <= 0 or batch_size <= 0: raise ValueError("Positive step and batch size required")
    points = np.asarray(points, float)
    nodes, weights = interpolation_rule(degree, 1, 0)
    result = np.empty_like(points)
    for start in range(0, len(points), batch_size):
        p = points[start:start+batch_size]
        for axis in range(points.shape[1]):
            queries = np.broadcast_to(p[:,None,:], (len(p), len(nodes), points.shape[1])).copy()
            queries[:,:,axis] += step * nodes
            if ledger: ledger.tick("n_directional_calls", len(p))
            values = np.asarray(batch_oracle(queries.reshape(-1, points.shape[1])))
            result[start:start+len(p),axis] = values.reshape(len(p),len(nodes)) @ weights / step
    if not np.isfinite(result).all(): raise NumericalFailure("interpolation_unstable")
    return result

def hessian(oracle, x, *, degree, step, dps=0, ledger=None):
    """Polarization includes the norm-squared of e_i+e_j."""
    if ledger: ledger.tick("n_hessian_calls")
    n = len(x); eye = np.eye(n)
    h = np.empty((n, n), dtype=object if dps else float)
    def d2(v):
        return directional_derivative(oracle, x, v, degree=degree, order=2, step=step, dps=dps, ledger=ledger)
    for i in range(n): h[i, i] = d2(eye[i])
    for i in range(n):
        for j in range(i): h[i, j] = h[j, i] = (d2(eye[i] + eye[j]) - h[i, i] - h[j, j]) / 2
    return h
