"""Unconstrained output regression using fixed recovered hidden weights."""
import numpy as np


def hidden_features(x, weights, k):
    state = np.asarray(x, float)
    for w in weights:
        state = state @ w
        if k == 4:
            state = state * state
            state = state * state
        else:
            state = state ** k
    return state


def ordinary_least_squares(phi, labels):
    """Algebraically unweighted, unconstrained OLS; no intercept or ridge."""
    scales = np.linalg.norm(phi, axis=0)
    target_scale = np.max(np.abs(labels))
    assert np.all(scales > 0) and target_scale > 0
    design, target = phi / scales, labels / target_scale
    u, singular, vh = np.linalg.svd(design, full_matrices=False)
    tolerance = np.finfo(float).eps * max(design.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    assert rank == phi.shape[1], 'Regression feature matrix is rank deficient'
    coefficients_scaled = vh.T @ ((u.T @ target) / singular)
    coefficients = coefficients_scaled * target_scale / scales
    residual = design @ coefficients_scaled - target
    leverage = np.sum(u * u, axis=1)
    energy = target * target
    top = max(1, len(labels) // 100)
    raw_singular = np.linalg.svd(phi, compute_uv=False)
    reference, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    assert np.allclose(coefficients_scaled, reference, rtol=1e-10, atol=1e-14)
    return coefficients, {'rank': rank, 'column_scales': scales,
        'target_scale': target_scale, 'scaled_singular_values': singular,
        'scaled_design_condition': singular[0] / singular[-1],
        'raw_design_condition': raw_singular[0] / raw_singular[-1],
        'training_nmse': np.dot(residual, residual) / np.dot(target, target),
        'normal_equation_relative_residual': np.linalg.norm(design.T @ residual) / np.linalg.norm(target),
        'max_leverage': np.max(leverage),
        'max_label_energy_share': np.max(energy) / np.sum(energy),
        'top_one_percent_label_energy_share': np.sum(np.partition(energy, -top)[-top:]) / np.sum(energy),
        'lstsq_crosscheck_relative_error': np.linalg.norm(reference - coefficients_scaled) / np.linalg.norm(reference)}
