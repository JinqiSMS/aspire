"""Only the lowest real value interface charges queries; scopes partition costs."""
from collections import OrderedDict, Counter
from contextlib import contextmanager
import time
import mpmath as mp
import numpy as np
from .numeric import is_mp, finite, array_mp
from .status import NumericalFailure

class QueryLedger:
    def __init__(self, budget=2_000_000, cache_size=50_000, wall_seconds=3600, restored=None):
        self.budget, self.cache_size, self.wall_seconds = budget, cache_size, wall_seconds
        self.counts = Counter(restored or {})
        self.cache = OrderedDict()
        self.purpose = "span"
        self.started = time.perf_counter()
        self.oracle_seconds = 0.0

    @contextmanager
    def scope(self, name):
        previous, self.purpose = self.purpose, name
        try: yield
        finally: self.purpose = previous

    def tick(self, name, amount=1): self.counts[name] += amount

    def snapshot(self):
        names = ["n_real_calls_total", "n_cache_hits", "n_complex_calls", "n_suffix_calls",
                 "n_directional_calls", "n_hessian_calls", "n_hr_transitions", "n_kept_samples"]
        names += ["n_real_calls_" + p for p in ("span", "membership", "moment_gradient", "final_hessian", "validation", "evaluation")]
        return {**dict.fromkeys(names, 0), **dict(self.counts)}

    def charge(self, amount=1):
        if amount < 0: raise ValueError("Negative query charge")
        if self.counts["n_real_calls_total"] + amount > self.budget:
            raise NumericalFailure("budget_exhausted", budget=self.budget)
        if time.perf_counter() - self.started > self.wall_seconds:
            raise NumericalFailure("wall_time_exhausted", seconds=self.wall_seconds)
        self.tick("n_real_calls_total", amount)
        self.tick("n_real_calls_" + self.purpose, amount)

class CountedRealOracle:
    """Opaque scalar callable; no teacher attributes or derivative methods exposed."""
    __slots__ = ("__value", "__batch_value", "ledger", "dimension")

    def __init__(self, value, dimension, ledger, batch_value=None):
        self.__value, self.dimension, self.ledger = value, dimension, ledger
        self.__batch_value = batch_value

    def __call__(self, x):
        x = np.asarray(x)
        if x.shape != (self.dimension,): raise ValueError("Expected one real input vector")
        if any(isinstance(v, (complex, mp.mpc)) for v in x.flat):
            raise TypeError("Real oracle rejects complex input types")
        if not all(finite(v) for v in x.flat): raise NumericalFailure("nonfinite_query")
        hp = is_mp(x)
        if hp:x=array_mp(x)
        key = (mp.mp.dps, tuple(v._mpf_ for v in x)) if hp else (0, x.astype(np.float64).tobytes())
        if key in self.ledger.cache:
            self.ledger.tick("n_cache_hits")
            self.ledger.cache.move_to_end(key)
            return self.ledger.cache[key]
        self.ledger.charge()
        start = time.perf_counter()
        try: value = self.__value(x)
        finally: self.ledger.oracle_seconds += time.perf_counter() - start
        if isinstance(value, (complex, mp.mpc)) or not finite(value):
            raise NumericalFailure("nonfinite_oracle_value")
        if self.ledger.cache_size:
            self.ledger.cache[key] = value
            if len(self.ledger.cache) > self.ledger.cache_size: self.ledger.cache.popitem(last=False)
        return value

    def batch(self, x):
        """Optional vector evaluation of the SAME real values, charged per row.

        Fast path deliberately requires disabled caching: repeated inputs are
        actually evaluated and charged, rather than given an estimated cost.
        Arbitrary scalar callables and high precision keep the original path.
        """
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError("Expected a matrix of real input vectors")
        if self.__batch_value is None or self.ledger.cache_size or x.dtype == object:
            return np.array([self(row) for row in x])
        if np.iscomplexobj(x): raise TypeError("Real oracle rejects complex input types")
        if not np.isfinite(x).all(): raise NumericalFailure("nonfinite_query")
        if not len(x): return np.empty(0)
        self.ledger.charge(len(x))
        start = time.perf_counter()
        try: values = np.asarray(self.__batch_value(x))
        finally: self.ledger.oracle_seconds += time.perf_counter() - start
        if values.shape != (len(x),): raise ValueError("Batch oracle returned wrong shape")
        if np.iscomplexobj(values) or not np.isfinite(values).all():
            raise NumericalFailure("nonfinite_oracle_value")
        return values
