"""S4 lem:first-layer-column-space; S6 prop:stable-column-space."""
import numpy as np
from ..numeric import right_inverse_transpose
from ..oracles.interpolation import gradient, directional_derivative
from ..oracles.prefix_jacobian import prefix_value_and_jacobian
from ..status import NumericalFailure

def top_span(vectors, rank):
    u, s, _ = np.linalg.svd(np.asarray(vectors, float).T, full_matrices=False)
    if len(s) < rank or s[rank - 1] <= s[0] * 1e-12:
        raise NumericalFailure("subspace_unresolved", singular_values=s.tolist())
    return u[:, :rank], {"span_singular_values": s.tolist(), "span_condition": float(s[0] / s[rank-1])}

def recover_subspace(real, architecture, layer, prefix, cfg, rng, ledger):
    n, rank = architecture.widths[layer - 1:layer + 1]
    if n == rank: return np.eye(n), {"span_method": "full_space_identity"}
    dps = cfg["oracle"]["dps"] if cfg["oracle"]["backend"] == "mpmath" else 0
    vectors = []
    with ledger.scope("span"):
        if layer == 1:
            count = cfg["recovery"]["first_span_points"]
            for x in rng.normal(size=(count, architecture.d)):
                vectors.append(gradient(real, x, degree=architecture.Q, step=cfg["recovery"]["original_direction_step"], dps=dps, ledger=ledger))
        else:
            count = cfg["recovery"]["intermediate_span_points"]
            first_inverse = right_inverse_transpose(prefix[0])
            for _ in range(count):
                z = rng.uniform(1, 2, architecture.hidden_widths[0])
                x = first_inverse @ z ** (1 / architecture.k)
                _, jac = prefix_value_and_jacobian(x, prefix, architecture.k)
                singular = np.linalg.svd(jac, compute_uv=False)
                if singular[-1] <= singular[0] * 1e-12: raise NumericalFailure("jacobian_rank_deficient")
                # Original f directional derivatives, degree Q, not suffix degree.
                b = [directional_derivative(real, x, row, degree=architecture.Q,
                       step=cfg["recovery"]["original_direction_step"], dps=dps, ledger=ledger) for row in jac]
                vectors.append(np.linalg.solve(jac @ jac.T, np.array(b, float)))
    basis, diag = top_span(vectors, rank)
    return basis, {**diag, "span_method": "original_directional_jacobian" if layer > 1 else "original_gradients", "span_points": count}

