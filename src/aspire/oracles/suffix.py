"""S1 alg:suffix-oracle; S5 lem:multiplicative-suffix-oracle.

The estimated suffix is generally not a polynomial. Its real part supplies
fixed-node interpolation observations, not an autodifferentiable exact suffix.
"""
from contextlib import nullcontext
import mpmath as mp
import numpy as np
from ..numeric import array_mp, right_inverse_transpose, is_mp, finite
from ..status import NumericalFailure
from .complex_value import complex_value_from_real

class SuffixOracle:
    def __init__(self, real_oracle, architecture, layer, recovered_prefix, settings, ledger=None, direct_complex=None):
        self.real_oracle, self.arch, self.layer = real_oracle, architecture, layer
        self.prefix = tuple(np.asarray(w, float).copy() for w in recovered_prefix)
        if len(self.prefix) != layer - 1: raise ValueError("Wrong prefix length")
        self.settings, self.ledger, self.direct_complex = settings, ledger, direct_complex
        self.inverses = {}
        self.diagnostics = {"queries": 0, "precision_retries": 0, "max_dps": 0,
                            "max_imaginary": 0.0, "max_cancellation": 1.0,
                            "max_inverse_residual": 0.0, "max_precision_difference": 0.0}

    def invert(self, y, dps=0):
        """Recompute roots AND right inverses at the requested precision."""
        with mp.workdps(dps) if dps else nullcontext():
            if dps not in self.inverses:
                self.inverses[dps] = [right_inverse_transpose(w, bool(dps)) for w in self.prefix]
            z = array_mp(y, complex_=True) if dps else np.asarray(y, complex)
            for w, inv in reversed(list(zip(self.prefix, self.inverses[dps]))):
                root = np.array([mp.root(t, self.arch.k) if t else mp.mpc(0) for t in z], dtype=object) if dps else z ** (1 / self.arch.k)
                z = inv @ root
                residual = (array_mp(w).T if dps else w.T) @ z - root
                self.diagnostics["max_inverse_residual"] = max(self.diagnostics["max_inverse_residual"], float(max(abs(t) for t in residual)))
            return z

    def _compute(self, y, dps):
        with mp.workdps(dps) if dps else nullcontext():
            z = self.invert(y, dps)
            if self.direct_complex is not None:
                if self.ledger: self.ledger.tick("n_debug_complex_calls")
                return self.direct_complex(z), 1.0, True
            value, chi = complex_value_from_real(self.real_oracle, z, total_degree=self.arch.Q, dps=dps, ledger=self.ledger)
            pure_real = all((mp.im(t) if dps else t.imag) == 0 for t in z)
            return value, chi, pure_real

    def __call__(self, y):
        if self.ledger: self.ledger.tick("n_suffix_calls")
        self.diagnostics["queries"] += 1
        if self.layer == 1: return self.real_oracle(y)
        hp_input = is_mp(y)
        base = max(mp.mp.dps if hp_input else 0, self.settings["dps"] if self.settings["backend"] == "mpmath" else 0)
        retry = self.settings["precision_retry"]
        value, chi, real = self._compute(y, base)
        chosen = base
        need_retry = (not real and self.direct_complex is None and
                      (base > 0 or not np.isfinite(float(chi)) or float(chi) * np.finfo(float).eps * 16 > retry["relative_tolerance"]))
        if need_retry and retry["enabled"]:
            previous = value if base else None
            matched = False
            precisions = sorted(set([p for p in retry["decimal_digits"] if p > base] + ([base + 30] if base else [])))
            for precision in precisions:
                with mp.workdps(precision):
                    updated, chi, _ = self._compute(y, precision)
                    self.diagnostics["precision_retries"] += 1
                    if previous is not None:
                        difference = abs(updated - previous)
                        self.diagnostics["max_precision_difference"] = max(self.diagnostics["max_precision_difference"], float(difference))
                        if difference <= retry["absolute_tolerance"] + retry["relative_tolerance"] * abs(updated):
                            value, chosen, matched = updated, precision, True
                            break
                    previous, value, chosen = updated, updated, precision
            if not matched: raise NumericalFailure("complex_cancellation_unresolved", last_dps=chosen)
        elif need_retry and not retry["enabled"]:
            # Explicit precision ablation: report the instability, never invent accuracy.
            self.diagnostics["unverified_cancellation"] = True
        self.diagnostics["max_dps"] = max(self.diagnostics["max_dps"], chosen)
        self.diagnostics["max_imaginary"] = max(self.diagnostics["max_imaginary"], float(abs(mp.im(value))))
        self.diagnostics["max_cancellation"] = max(self.diagnostics["max_cancellation"], float(chi))
        answer = mp.re(value) if isinstance(value, (mp.mpf, mp.mpc)) else float(np.real(value))
        if not finite(answer): raise NumericalFailure("nonfinite_oracle_value")
        return answer if hp_input else float(answer)

