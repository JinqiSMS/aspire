"""Evaluator-only deterministic integration and independent sampling controls.

These routines use teacher parameters. They must never supply moments, sample
bounds, stopping decisions, or directions to the strict learner.
"""
import numpy as np
from scipy.special import roots_legendre
from scipy.optimize import linear_sum_assignment


def value_gradient_batch(teacher, points):
    state = np.asarray(points, float)
    preactivations = []
    for weight in teacher.weights:
        h = state @ weight
        preactivations.append(h)
        state = h ** teacher.k
    values = state @ teacher.a
    grad = np.broadcast_to(teacher.a, state.shape).copy()
    for weight, h in reversed(list(zip(teacher.weights, preactivations))):
        grad = (grad * teacher.k * h ** (teacher.k - 1)) @ weight.T
    return values, grad


def sphere_rule_3d(order, rotation_seed=8128):
    """Gauss-Legendre in cos(theta), periodic trapezoid in longitude.

    Rotate the grid independently of the teacher, so sign symmetry in the
    quadrature nodes cannot artificially force correct eigendirections.
    """
    t, weights = roots_legendre(order)
    phi = np.arange(2 * order) * np.pi / order
    nodes = np.stack(np.broadcast_arrays(
        np.sqrt(1 - t * t)[:, None] * np.cos(phi),
        np.sqrt(1 - t * t)[:, None] * np.sin(phi), t[:, None]), axis=-1).reshape(-1, 3)
    rotation = np.linalg.qr(np.random.default_rng(rotation_seed).normal(size=(3, 3)))[0]
    return nodes @ rotation, np.repeat(weights, 2 * order) / (4 * order)


def integrate_body_3d(value_gradient, degree, order, rotation_seed=8128):
    """Integrate FULL position/gradient matrices using homogeneity.

    Numerical convergence is checked by callers; this is not an interval
    arithmetic certificate. No known diagonal structure is imposed.
    """
    omega, weights = sphere_rule_3d(order, rotation_seed)
    values, grads = value_gradient(omega)
    if np.any(values <= 0) or not np.isfinite(grads).all():
        raise ValueError('Nonpositive or nonfinite population integrand')
    radius = values ** (-1 / degree)
    denominator = weights @ radius ** 3
    sx = (omega.T * (weights * radius ** 5)) @ omega * (3 / 5) / denominator
    exponent = 3 + 2 * degree - 2
    sg = (grads.T * (weights * radius ** exponent)) @ grads * (3 / exponent) / denominator
    return (sx + sx.T) / 2, (sg + sg.T) / 2


def aligned_operator_error(truth, estimate):
    cost = np.minimum(np.sum((truth[:, :, None] - estimate[:, None, :]) ** 2, axis=0),
                      np.sum((truth[:, :, None] + estimate[:, None, :]) ** 2, axis=0))
    _, permutation = linear_sum_assignment(cost)
    aligned = estimate[:, permutation].copy()
    aligned *= np.where(np.sum(truth * aligned, axis=0) < 0, -1, 1)
    return float(np.linalg.norm(aligned - truth, 2))


def independent_body_moments(latent_teacher, coordinate_matrix, counts, rng, batch_size=32768):
    """Exact-value rejection in a truth-owned bounding box (debug only).

    Positivity implies h(u) >= h(u_i e_i); hence every body point satisfies
    |u_i| <= h(e_i)^(-1/q). Uniform box proposals conditioned on h(u)<=1
    are i.i.d. uniform body points, with no Markov-chain mixing error.
    Outputs are mapped from latent u to reduced z=C^-T u.
    """
    n = latent_teacher.weights[0].shape[0]
    degree = latent_teacher.k ** len(latent_teacher.weights)
    radii = latent_teacher.forward(np.eye(n)) ** (-1 / degree)
    inverse = np.linalg.inv(coordinate_matrix)
    kept = proposed = accepted_total = 0
    sx = np.zeros((n, n)); sg = np.zeros((n, n)); rows = []
    pending = np.empty((0, n))
    for target in sorted(counts):
        while kept < target:
            if not len(pending):
                candidates = rng.uniform(-1, 1, size=(batch_size, n)) * radii
                pending = candidates[latent_teacher.forward(candidates) <= 1]
                proposed += batch_size
                accepted_total += len(pending)
                if not len(pending):
                    continue
            used = min(len(pending), target - kept)
            u, pending = pending[:used], pending[used:]
            _, latent_grad = value_gradient_batch(latent_teacher, u)
            z = u @ inverse
            g = latent_grad @ coordinate_matrix.T
            sx += z.T @ z; sg += g.T @ g; kept += used
        rows.append({'count': target, 'Sx': sx.copy() / kept, 'Sg': sg.copy() / kept,
                     'proposed': proposed, 'accepted_total': accepted_total,
                     'acceptance_rate': accepted_total / proposed})
    return rows
