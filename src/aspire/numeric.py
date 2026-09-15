"""Backend primitives: high precision inputs must never pass through float64."""
from functools import lru_cache
import math
import mpmath as mp
import numpy as np
from .status import NumericalFailure

def is_mp(x):
    return any(isinstance(v, (mp.mpf, mp.mpc)) for v in np.asarray(x, dtype=object).flat)

def array_mp(x, complex_=False):
    convert = mp.mpc if complex_ else mp.mpf
    return np.array([convert(v.item() if isinstance(v, np.generic) else v)
                     for v in np.asarray(x).flat], dtype=object).reshape(np.shape(x))

def matmul(a, b):
    if is_mp(b): return array_mp(a) @ np.asarray(b, dtype=object)
    return np.asarray(a) @ b

def right_inverse_transpose(w, high_precision=False):
    """W (W.T W)^-1 via SVD (double) or mp QR, no explicit Gram inverse."""
    w = np.asarray(w)
    if high_precision:
        a = mp.matrix(array_mp(w).tolist())
        q, r = mp.qr(a)
        s = w.shape[1]
        r = r[:s, :s]
        q = q[:, :s]
        out = mp.matrix(w.shape[0], s)
        for j in range(s):
            e = mp.eye(s)[:, j]
            col = q * mp.lu_solve(r.T, e)
            out[:, j] = col
        return np.array(out.tolist(), dtype=object)
    u, s, vt = np.linalg.svd(np.asarray(w, float), full_matrices=False)
    if s[-1] <= 1e-12 * s[0]:
        raise NumericalFailure("prefix_rank_deficient", singular_values=s.tolist())
    return (u / s) @ vt

def finite(value):
    if isinstance(value, (mp.mpf, mp.mpc)): return bool(mp.isfinite(value))
    return bool(np.isfinite(value))

def checked_float(value, reason="nonfinite_oracle_value"):
    val = float(value)
    if not math.isfinite(val): raise NumericalFailure(reason)
    return val

@lru_cache(maxsize=256)
def interpolation_rule(degree, order, dps=0):
    """Exact integer Chebyshev coefficients + DCT-I derivative functional."""
    if not 0 <= order <= degree or degree < 1: raise ValueError("Invalid interpolation order")
    coefficients = [[1], [0, 1]]
    for m in range(2, degree + 1):
        c = [0] * (m + 1)
        for i, value in enumerate(coefficients[-1]): c[i + 1] += 2 * value
        for i, value in enumerate(coefficients[-2]): c[i] -= value
        coefficients.append(c)
    derivative = [c[order] * math.factorial(order) if len(c) > order else 0
                  for c in coefficients]
    if dps:
        with mp.workdps(dps):
            nodes = [mp.cos(mp.pi * j / degree) for j in range(degree + 1)]
            weights = []
            for j in range(degree + 1):
                v = mp.fsum(mp.mpf(derivative[m]) * mp.cos(mp.pi * m * j / degree)
                            / (2 if m in (0, degree) else 1) for m in range(degree + 1))
                weights.append(v * 2 / degree / (2 if j in (0, degree) else 1))
        return tuple(nodes), tuple(weights)
    j = np.arange(degree + 1)
    c = np.ones(degree + 1); c[[0, -1]] = 2
    nodes = np.cos(np.pi * j / degree)
    weights = 2 / degree * (np.cos(np.pi / degree * np.outer(j, j)) @ (np.array(derivative) / c)) / c
    return nodes, weights

