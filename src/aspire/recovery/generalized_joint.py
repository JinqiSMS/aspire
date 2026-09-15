"""Random symmetric Hessian pencils and shared-basis diagnostics.

Only measured matrices and public dimensions enter this module. Generalized
eigenvectors are dual directions: weights are read from H0 @ C, not C.
"""
import numpy as np
from scipy.linalg import eigh, solve_triangular
from ..status import NumericalFailure


def offdiag(a):
    result = np.array(a, copy=True)
    result[..., np.arange(a.shape[-1]), np.arange(a.shape[-1])] = 0.
    return result


def _symmetric(a):
    a = np.asarray(a, float)
    if a.ndim < 2 or a.shape[-1] != a.shape[-2] or not np.isfinite(a).all():
        raise ValueError("Expected finite square matrices")
    return (a + a.swapaxes(-1, -2)) / 2


def random_coefficients(count, matrices, rng):
    if count < 1 or matrices < 1:
        raise ValueError("Positive candidate and matrix counts required")
    beta = rng.normal(size=(count, matrices))
    return beta / np.linalg.norm(beta, axis=1)[:, None]


def joint_residuals(h0, matrices, c):
    """C must already be orthonormal in the measured anchor metric."""
    h0, matrices = _symmetric(h0), _symmetric(matrices)
    transformed = np.einsum("ip,tij,jq->tpq", c, matrices, c)
    numerator = float(np.linalg.norm(offdiag(transformed)))
    centered = transformed - np.trace(transformed, axis1=1, axis2=2)[:, None, None] / c.shape[1] * np.eye(c.shape[1])
    signal = float(np.linalg.norm(centered))
    denominator = float(np.sqrt(np.linalg.norm(transformed)**2 + np.linalg.norm(c.T @ h0 @ c)**2))
    return {"joint_residual": numerator / max(denominator, np.finfo(float).tiny),
            "signal_residual": numerator / signal if signal > 0 else None,
            "offdiagonal_norm": numerator, "signal_norm": signal,
            "metric_orthogonality_error": float(np.linalg.norm(c.T @ h0 @ c - np.eye(c.shape[1])))}


def hessian_fit_residual(matrices, centers, w, a, k):
    reconstructed = np.asarray([
        k*(k-1)*(w*(a*(w.T @ y)**(k-2))) @ w.T for y in centers])
    return float(np.linalg.norm(matrices - reconstructed) /
                 max(np.linalg.norm(matrices), np.finfo(float).tiny))


def recover_generalized(h0, matrices, rank, k, betas, *, selection="residual",
                        gap_floor=1e-10, rank_floor=1e-12, tie_tolerance=1e-12):
    """Return weights, anchor coefficients, diagnostics and replayable arrays.

    Selection is residual (main method), gap (single-probe comparator), or first.
    Failure records never depend on parameter error or held-out matrices.
    """
    if selection not in ("residual", "gap", "first"):
        raise ValueError("Unknown candidate selection")
    raw0, raw = np.asarray(h0, float), np.asarray(matrices, float)
    h0, matrices = _symmetric(raw0), _symmetric(raw)
    n = h0.shape[0]
    if matrices.ndim != 3 or matrices.shape[1:] != h0.shape or not len(matrices):
        raise ValueError("Invalid matrix bank")
    if not 2 <= rank <= n or k < 3 or min(gap_floor, rank_floor, tie_tolerance) < 0:
        raise ValueError("Invalid recovery dimensions or tolerances")
    values0, eigenbasis = eigh(h0)
    if values0[-rank] <= max(np.max(np.abs(values0))*rank_floor, 0.):
        raise NumericalFailure("final_hessian_rank_unresolved", eigenvalues=values0.tolist())
    basis = np.eye(n) if rank == n else eigenbasis[:, -rank:]
    b0 = basis.T @ h0 @ basis
    bank = np.einsum("ip,tij,jq->tpq", basis, matrices, basis)
    chol = np.linalg.cholesky(b0)
    white = []
    for b in bank:
        left = solve_triangular(chol, b, lower=True)
        white.append(solve_triangular(chol, left.T, lower=True).T)
    white = _symmetric(np.asarray(white))
    centered = white - np.trace(white, axis1=1, axis2=2)[:, None, None] / rank * np.eye(rank)
    signal = float(np.linalg.norm(centered))
    if signal <= rank_floor * max(np.linalg.norm(white), np.finfo(float).tiny):
        raise NumericalFailure("final_joint_signal_unresolved", signal_norm=signal)
    commutators = np.zeros((len(bank), len(bank)))
    for i in range(len(bank)):
        for j in range(i):
            denom = np.linalg.norm(white[i])*np.linalg.norm(white[j])
            value = np.linalg.norm(white[i] @ white[j] - white[j] @ white[i]) / max(denom, np.finfo(float).tiny)
            commutators[i, j] = commutators[j, i] = value
    betas = np.asarray(betas, float)
    if betas.ndim != 2 or betas.shape[1] != len(bank) or not len(betas) or not np.isfinite(betas).all():
        raise ValueError("Invalid random coefficient matrix")
    lengths = np.linalg.norm(betas, axis=1)
    if np.any(lengths == 0):
        raise ValueError("Zero random combination")
    betas = betas / lengths[:, None]
    candidates, solutions = [], {}
    all_c = np.full((len(betas), n, rank), np.nan)
    for index, beta in enumerate(betas):
        pencil = _symmetric(np.einsum("t,tij->ij", beta, bank))
        eigenvalues, z = eigh(pencil, b0, type=1, check_finite=True)
        gap = float(np.min(np.diff(eigenvalues)))
        scale = max(1., float(np.max(abs(eigenvalues))))
        record = {"index": index, "eigenvalues": eigenvalues.tolist(),
                  "absolute_gap": gap, "relative_gap": gap/scale, "status": "complete"}
        if gap <= gap_floor*scale:
            record.update(status="failed", failure_reason="final_gap_unresolved")
            candidates.append(record)
            continue
        c = basis @ z
        diagnostics = joint_residuals(h0, matrices, c)
        equation = pencil @ z - (b0 @ z)*eigenvalues
        diagnostics["generalized_equation_residual"] = float(np.linalg.norm(equation) /
            max(np.linalg.norm(pencil)*np.linalg.norm(z) +
                np.linalg.norm(b0 @ z)*np.max(abs(eigenvalues)), np.finfo(float).tiny))
        record.update(diagnostics)
        candidates.append(record)
        solutions[index] = (c, z, eigenvalues)
        all_c[index] = c
    eligible = [r for r in candidates if r["status"] == "complete"]
    if not eligible:
        raise NumericalFailure("final_gap_unresolved", candidates=candidates)
    if selection == "first":
        best = candidates[0]
        if best["status"] != "complete":
            raise NumericalFailure("final_gap_unresolved", candidates=candidates)
    elif selection == "gap":
        best = max(eligible, key=lambda r: (r["relative_gap"], -r["index"]))
    else:
        minimum = min(r["signal_residual"] for r in eligible)
        tied = [r for r in eligible if r["signal_residual"] <= minimum + tie_tolerance]
        best = max(tied, key=lambda r: (r["absolute_gap"], -r["index"]))
    c, z, eigenvalues = solutions[best["index"]]
    v = basis @ b0 @ z
    # The paper normalizes lifted directions coordinatewise, not just by a
    # whole-column sign. This enforces nonnegativity even with noisy Hessians.
    magnitudes = np.abs(v)
    sums = magnitudes.sum(axis=0)
    if not np.isfinite(sums).all() or np.any(sums <= 0):
        raise NumericalFailure("final_column_sum_unresolved", column_sums=sums.tolist())
    w = magnitudes/sums
    oriented = v * np.where(v.sum(axis=0) < 0, -1., 1.)
    singular = np.linalg.svd(w, compute_uv=False)
    if singular[-1] <= rank_floor*singular[0]:
        raise NumericalFailure("final_weight_rank_unresolved", singular_values=singular.tolist())
    pinv = np.linalg.pinv(w)
    coefficient_matrix = pinv @ h0 @ pinv.T / (k*(k-1))
    a = np.diag(coefficient_matrix).copy()
    projected = np.einsum("ip,tpq,jq->tij", basis, bank, basis)
    probe_eigenvalues = np.linalg.eigvalsh(bank)
    nonpositive = probe_eigenvalues[:, 0] <= 0
    diagnostics = {
        **best, "method": "random_combination", "selection": selection,
        "selected_index": best["index"], "candidates": candidates,
        "candidate_count": len(candidates), "failed_candidates": len(candidates)-len(eligible),
        "final_gap": best["relative_gap"], "final_condition": float(np.linalg.cond(b0)),
        "hessian2_eigenvalues": values0.tolist(), "final_eigenvalues": eigenvalues.tolist(),
        "probe_eigenvalues": probe_eigenvalues.tolist(),
        "nonpositive_probe_count": int(np.count_nonzero(nonpositive)),
        "max_commutator": float(np.max(commutators)),
        "matrix_asymmetry": float(np.linalg.norm(raw-raw.swapaxes(-1, -2))),
        "anchor_asymmetry": float(np.linalg.norm(raw0-raw0.T)),
        "outside_subspace_residual": float(np.linalg.norm(matrices-projected)/np.linalg.norm(matrices)),
        "minimum_weight_entry": float(np.min(w)),
        "negative_weight_entries": int(np.count_nonzero(w < -1e-12)),
        "normalization": "coordinatewise_absolute_value_and_l1",
        "negative_direction_entries_before_normalization": int(np.count_nonzero(oriented < 0)),
        "normalization_direction_change": float(np.linalg.norm(magnitudes-oriented) / np.linalg.norm(v)),
        "coefficient_offdiagonal_residual": float(np.linalg.norm(offdiag(coefficient_matrix))),
        # This discrepancy vanishes for exact positive directions. After
        # entrywise absolute values it need not vanish for noisy directions.
        "coefficient_direction_scale_discrepancy": float(np.linalg.norm(a-sums*sums/(k*(k-1)))),
        "hessian_reconstruction_residual": float(np.linalg.norm(h0-k*(k-1)*(w*a)@w.T)/np.linalg.norm(h0)),
    }
    artifacts = {"C": c, "V": v, "basis": basis, "whitened_hessians": white,
                 "commutators": commutators, "betas": betas, "candidate_C": all_c,
                 "H0": h0, "hessian_bank": matrices}
    return w, a, diagnostics, artifacts
