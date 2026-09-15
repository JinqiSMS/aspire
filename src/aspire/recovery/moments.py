"""algo_and_main_results.tex alg:aspire; correct right eigendirections."""
import numpy as np
from scipy.linalg import solve_triangular
from .sampling import effective_sample_size
from ..status import NumericalFailure

def normalize_directions(v, layer):
    if not np.isfinite(v).all(): raise NumericalFailure("normalization_failed")
    if layer == 1:
        norms = np.linalg.norm(v, axis=0)
        if np.min(norms) <= 1e-14: raise NumericalFailure("normalization_failed")
        result = v / norms
        signs = np.sign(result[np.argmax(abs(result), axis=0), np.arange(result.shape[1])])
        result *= signs
    else:
        result = abs(v); norms = result.sum(axis=0)
        if np.min(norms) <= 1e-14: raise NumericalFailure("normalization_failed")
        result /= norms
    if np.linalg.cond(result) > 1e12: raise NumericalFailure("prefix_rank_deficient")
    return result

def relative_gap(values):
    if len(values)<2: return 1.
    return float(np.min(np.diff(np.sort(values))) / max(abs(values))) if max(abs(values)) else 0.

def moment_directions(sx, sg, ridge=0., gap_floor=1e-10):
    sx, sg = (sx+sx.T)/2, (sg+sg.T)/2
    if ridge: sx = sx+ridge*np.eye(len(sx))
    try: chol = np.linalg.cholesky(sx)
    except np.linalg.LinAlgError: raise NumericalFailure("moment_not_spd")
    symmetric = chol.T @ sg @ chol
    theta, u = np.linalg.eigh((symmetric+symmetric.T)/2)
    gap = relative_gap(theta)
    if gap <= gap_floor: raise NumericalFailure("moment_gap_unresolved", observed_gap=gap)
    v = solve_triangular(chol.T, u, lower=False)
    v /= np.linalg.norm(v, axis=0)
    return theta, v, {"moment_gap": gap, "moment_condition": float(np.linalg.cond(sx)), "moment_ridge": ridge}

def raw_moments(points, gradients):
    z, g = np.asarray(points, float), np.asarray(gradients, float)
    if not np.isfinite(g).all(): raise NumericalFailure("interpolation_unstable")
    return z.T@z/len(z), g.T@g/len(g)

def moment_ess(points, gradients, chain_ids, mode):
    i, j = np.triu_indices(points.shape[1])
    features = np.concatenate([points[:,i]*points[:,j], gradients[:,i]*gradients[:,j]], axis=1)
    if mode == "independent_endpoints": return {"moment_ess_min": len(points), "ess_method": "independent_endpoints_count_not_mixing_certificate"}
    estimates = [effective_sample_size(features[chain_ids==c]) for c in np.unique(chain_ids)]
    total = np.sum(estimates, axis=0)
    return {"moment_ess_min": float(np.min(total)), "moment_ess_by_component": total.tolist(), "ess_method": "positive_pair_autocorrelation_by_chain"}

