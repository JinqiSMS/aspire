"""intermediate.tex prop:stable-column-space; only known prefix weights."""
import numpy as np

def prefix_value_and_jacobian(x, prefix, k):
    state = np.asarray(x, float)
    jac = None
    for w in prefix:
        h = state @ w
        local = k * h ** (k - 1)
        jac = local[:, None] * (w.T if jac is None else w.T @ jac)
        state = h ** k
    return state, np.eye(len(x)) if jac is None else jac

