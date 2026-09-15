"""problem_setup_and_preliminaries.tex alg:complex-query (positive phase)."""
from functools import lru_cache
from contextlib import nullcontext
import mpmath as mp
import numpy as np
from ..numeric import finite
from ..status import NumericalFailure

@lru_cache(maxsize=64)
def fourier_rule(q, dps):
    n = 2 * q + 1
    if dps:
        with mp.workdps(dps):
            angles = [2 * mp.pi * j / n for j in range(n)]
            return tuple(mp.cos(t) for t in angles), tuple(mp.sin(t) for t in angles), tuple(mp.exp(1j*q*t) for t in angles)
    angles = 2 * np.pi * np.arange(n) / n
    return np.cos(angles), np.sin(angles), np.exp(1j * q * angles)

def complex_value_from_real(real_oracle, z, *, total_degree, dps=0, ledger=None):
    """Return (value, cancellation diagnostic), all inputs to base oracle real."""
    if ledger: ledger.tick("n_complex_calls")
    with mp.workdps(dps) if dps else nullcontext():
        u = np.array([mp.re(t) for t in z], dtype=object) if dps else np.asarray(z).real
        v = np.array([mp.im(t) for t in z], dtype=object) if dps else np.asarray(z).imag
        if all(t == 0 for t in u) and all(t == 0 for t in v): return (mp.mpc(0) if dps else 0j), 1.0
        if all(t == 0 for t in v): return real_oracle(u), 1.0
        cos, sin, phase = fourier_rule(total_degree, dps)
        values = [real_oracle(u * c + v * s) for c, s in zip(cos, sin)]
        total = mp.fsum(p*y for p,y in zip(phase, values)) if dps else np.dot(phase, values)
        numerator = mp.fsum(abs(y) for y in values) if dps else sum(abs(y) for y in values)
        chi = numerator / abs(total) if total != 0 else (mp.inf if dps else np.inf)
        scale = mp.mpf(2) ** total_degree / len(values) if dps else 2.0 ** total_degree / len(values)
        result = scale * total
        if not finite(result): raise NumericalFailure("complex_cancellation_unresolved")
        return result, chi

