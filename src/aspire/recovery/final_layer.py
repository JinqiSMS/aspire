"""S2 alg:final-hessian-recovery; S7 prop:stable-final-layer."""
import numpy as np
from scipy.linalg import solve_triangular
from ..oracles.interpolation import hessian
from .moments import normalize_directions, relative_gap
from ..status import NumericalFailure

def simplex_projection(v):
    u = np.sort(v)[::-1]; c = np.cumsum(u)-1
    rho = np.flatnonzero(u-c/np.arange(1,len(v)+1)>0)[-1]
    return np.maximum(v-c[rho]/(rho+1), 0)

def recover_from_hessians(h1, h2, rank, k, gap_floor=1e-10):
    h1, h2 = np.asarray(h1,float), np.asarray(h2,float)
    h1, h2 = (h1+h1.T)/2, (h2+h2.T)/2
    eig, u = np.linalg.eigh(h2)
    if eig[-rank] <= max(abs(eig))*1e-12 or eig[-rank] <= 0:
        raise NumericalFailure("final_hessian_rank_unresolved", eigenvalues=eig.tolist())
    basis = u[:, -rank:]
    m1, m2 = basis.T@h1@basis, basis.T@h2@basis
    chol = np.linalg.cholesky(m2)
    left = solve_triangular(chol, m1, lower=True)
    s = solve_triangular(chol, left.T, lower=True).T
    values, q = np.linalg.eigh((s+s.T)/2)
    gap = relative_gap(values)
    if gap <= gap_floor: raise NumericalFailure("final_gap_unresolved", observed_gap=gap)
    # This is L_h U, NOT L_h^-T U (the moment pencil uses the latter).
    w = normalize_directions(basis @ chol @ q, layer=2)
    pinv = np.linalg.pinv(w)
    coef = pinv @ h2 @ pinv.T / (k*(k-1))
    a = np.diag(coef).copy()
    residual = np.linalg.norm(h2-k*(k-1)*(w*a)@w.T)/np.linalg.norm(h2)
    return w, a, {"final_gap": gap, "hessian2_eigenvalues": eig.tolist(),
        "final_condition": float(np.linalg.cond(m2)), "coefficient_offdiagonal_residual": float(np.linalg.norm(coef-np.diag(a))),
        "hessian_reconstruction_residual": float(residual), "final_eigenvalues": values.tolist()}

def recover_final(suffix, architecture, cfg, rng, ledger):
    n = architecture.widths[-2]; rank = architecture.widths[-1]
    settings = cfg["final_layer"]
    dps = cfg["oracle"]["dps"] if cfg["oracle"]["backend"] == "mpmath" else 0
    if settings.get("method", "two_hessian") == "random_combination":
        from .hessian_bank import orthogonal_directions, collect_hessians
        from .generalized_joint import random_coefficients, recover_generalized
        directions = orthogonal_directions(n, settings["hessian_directions"], rng)
        bank, centers, bank_diag = collect_hessians(
            suffix, directions, degree=architecture.k, tau=settings["tau_fin"],
            step=cfg["recovery"]["final_hessian_step"], dps=dps, ledger=ledger)
        betas = random_coefficients(settings["combination_candidates"], len(directions), rng)
        w, a, diag, arrays = recover_generalized(
            bank[0], bank[1:], rank, architecture.k, betas,
            gap_floor=cfg["recovery"]["gap_floor"])
        diag["hessian_bank_diagnostics"] = bank_diag
        arrays.update(hessian_centers=centers, hessian_directions=directions)
        return w, a, diag, arrays
    with ledger.scope("final_hessian"):
        h2 = hessian(suffix, np.ones(n), degree=architecture.k, step=cfg["recovery"]["final_hessian_step"], dps=dps, ledger=ledger)
        candidates, failures = [], []
        for _ in range(settings["perturbation_candidates"]):
            direction = rng.normal(size=n); direction /= np.linalg.norm(direction)
            y1 = np.ones(n) + settings["tau_fin"]*direction
            h1 = hessian(suffix, y1, degree=architecture.k, step=cfg["recovery"]["final_hessian_step"], dps=dps, ledger=ledger)
            try:
                w, a, diag = recover_from_hessians(h1,h2,rank,architecture.k,cfg["recovery"]["gap_floor"])
                candidates.append((w,a,diag,{"H1":np.array(h1,float), "H2":np.array(h2,float)}))
            except NumericalFailure as error: failures.append(error.reason)
        if not candidates: raise NumericalFailure(failures[0] if len(set(failures))==1 else "final_gap_unresolved", candidate_failures=failures)
    best = max(candidates, key=lambda c:c[2]["final_gap"])
    best[2]["candidate_failures"] = failures
    return best
