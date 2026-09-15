"""Evaluator-owned model. Never imported by recovery or oracle modules."""
from dataclasses import dataclass
import numpy as np
from .numeric import matmul, is_mp, array_mp

@dataclass
class Teacher:
    weights: list
    a: np.ndarray
    k: int

    def forward(self, x):
        h = np.asarray(x)
        with np.errstate(over="ignore", invalid="ignore"):
            for w in self.weights:
                h = matmul(w.T, h) if h.ndim == 1 else h @ w
                if h.ndim == 2 and h.dtype.kind == 'f' and self.k == 4:
                    # Same value oracle; avoid general pow for billions of
                    # vectorized real queries. Scalar/high-precision unchanged.
                    h = h*h
                    h = h*h
                else:
                    h = h ** self.k
            return matmul(self.a[None, :], h)[0] if h.ndim == 1 else h @ self.a

    def suffix(self, layer, y):
        return Teacher(self.weights[layer - 1:], self.a, self.k).forward(y)

    def gradient(self, x, layer=1):
        """Analytic debug reference, not available through CountedRealOracle."""
        s = np.asarray(x, float)
        hs = []
        for w in self.weights[layer - 1:]:
            h = s @ w; hs.append(h); s = h ** self.k
        g = self.a.copy()
        for w, h in reversed(list(zip(self.weights[layer - 1:], hs))):
            g = w @ (g * self.k * h ** (self.k - 1))
        return g

    def hessian(self, x, layer=1):
        x = np.asarray(x, float); n = len(x)
        state, jac, hess = x, np.eye(n), np.zeros((n, n, n))
        for w in self.weights[layer - 1:]:
            h = state @ w; j = w.T @ jac
            linear_hess = np.einsum("ij,iab->jab", w, hess)
            hess = (self.k * h ** (self.k - 1))[:, None, None] * linear_hess
            hess += (self.k * (self.k - 1) * h ** (self.k - 2))[:, None, None] * np.einsum("ia,ib->iab", j, j)
            jac = (self.k * h ** (self.k - 1))[:, None] * j
            state = h ** self.k
        return np.einsum("i,iab->ab", self.a, hess)

    def latent(self, layer):
        n = self.weights[layer - 1].shape[1]
        return Teacher([np.eye(n)] + self.weights[layer:], self.a, self.k)
