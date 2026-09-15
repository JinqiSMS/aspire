import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from aspire.architecture import Architecture
from aspire.config import resolve
from aspire.teacher import Teacher
from aspire.query_ledger import QueryLedger, CountedRealOracle
from aspire.oracles.suffix import SuffixOracle
from aspire.recovery.hessian_bank import orthogonal_directions, collect_hessians
from aspire.recovery.generalized_joint import (
    recover_generalized, random_coefficients, joint_residuals)
from aspire.recovery.final_layer import recover_final
from aspire.status import NumericalFailure


def fixture(n=3, s=3, k=4):
    rng = np.random.default_rng(121)
    w = rng.uniform(.05, .15, (n, s))
    w[:s] += np.eye(s)
    w /= w.sum(axis=0)
    a = np.arange(1, s+1, dtype=float)
    a /= a.sum()
    teacher = Teacher([w], a, k)
    directions = orthogonal_directions(n, 2*n, rng)
    hs = np.array([teacher.hessian(1+.2*v) for v in directions])
    return rng, w, a, teacher.hessian(np.ones(n)), hs


@pytest.mark.parametrize("n,s", [(3, 3), (5, 3)])
@pytest.mark.parametrize("k", [4, 6, 8])
def test_exact_recovery_and_dual_direction_readout(n, s, k):
    rng, w, a, h0, hs = fixture(n, s, k)
    estimated, coef, diag, arrays = recover_generalized(
        h0, hs, s, k, random_coefficients(16, len(hs), rng))
    _, order = linear_sum_assignment(np.linalg.norm(w[:, :, None]-estimated[:, None, :], axis=0))
    np.testing.assert_allclose(estimated[:, order], w, atol=1e-11)
    np.testing.assert_allclose(coef[order], a, atol=1e-11)
    np.testing.assert_allclose(arrays["V"], h0 @ arrays["C"], atol=1e-12)
    assert diag["signal_residual"] < 1e-10
    assert diag["metric_orthogonality_error"] < 1e-11
    assert diag["generalized_equation_residual"] < 1e-12
    assert diag["max_commutator"] < 1e-12
    assert diag["coefficient_direction_scale_discrepancy"] < 1e-11
    assert diag["normalization"] == "coordinatewise_absolute_value_and_l1"
    assert np.all(estimated >= 0)
    np.testing.assert_allclose(estimated.sum(axis=0), 1., atol=1e-14)
    wrong = arrays["C"]/arrays["C"].sum(axis=0)
    assert np.linalg.norm(wrong[:, order]-w) > .1


def test_degenerate_probe_resolved_by_combination():
    _, w, _, h0, _ = fixture()
    v = w @ np.diag(np.sqrt(12*np.array([1., 2., 3.])/6))
    bank = np.array([v @ np.diag([1., 1., 2.]) @ v.T,
                     v @ np.diag([1., 2., 1.]) @ v.T])
    _, _, diag, _ = recover_generalized(h0, bank, 3, 4, [[1., 0.], [.4, .9]])
    assert diag["candidates"][0]["status"] == "failed"
    assert diag["selected_index"] == 1
    with pytest.raises(NumericalFailure, match="final_gap_unresolved"):
        recover_generalized(h0, bank, 3, 4, [[1., 0.]])


def test_mixed_sign_directions_are_normalized_coordinatewise():
    # An SPD matrix bank alone does not ensure positive eigendirections.
    # Model a small perturbation of a direction whose true coordinate is small.
    directions = np.array([[.92, .005, .075], [-.005, .845, .15], [.085, .15, .775]])
    coefficients = np.array([.2, .3, .5])
    h0 = 12 * (directions * coefficients) @ directions.T
    bank = np.array([12 * (directions * (coefficients * scales)) @ directions.T
                     for scales in ([.8, 1., 1.2], [1.1, .7, 1.4])])
    assert np.linalg.eigvalsh(bank).min() > 0
    estimated, coef, diag, arrays = recover_generalized(h0, bank, 3, 4, [[.4, .9]])
    assert diag["negative_direction_entries_before_normalization"] == 1
    assert diag["negative_weight_entries"] == 0
    assert diag["normalization_direction_change"] > 0
    assert np.all(estimated >= 0)
    np.testing.assert_allclose(estimated.sum(axis=0), 1., atol=1e-14)
    expected = np.abs(directions) / np.abs(directions).sum(axis=0)
    _, order = linear_sum_assignment(np.linalg.norm(expected[:, :, None]-estimated[:, None, :], axis=0))
    np.testing.assert_allclose(estimated[:, order], expected, atol=1e-12)
    # Anchor diagnostics must reflect the weights after normalization.
    fitted = 12 * (estimated * coef) @ estimated.T
    np.testing.assert_allclose(diag["hessian_reconstruction_residual"],
                               np.linalg.norm(h0-fitted)/np.linalg.norm(h0))
    assert diag["hessian_reconstruction_residual"] > 1e-5


def test_unidentifiable_bank_is_not_success():
    with pytest.raises(NumericalFailure, match="final_joint_signal_unresolved"):
        recover_generalized(np.eye(3), np.array([np.eye(3), 2*np.eye(3)]),
                            3, 4, [[.3, .7]])


def test_anchor_must_have_positive_effective_rank():
    _, _, _, h0, hs = fixture()
    with pytest.raises(NumericalFailure, match="final_hessian_rank_unresolved"):
        recover_generalized(-h0, hs, 3, 4, [[1.]*len(hs)])


def test_noncommuting_noise_and_candidate_selection():
    rng, _, _, h0, hs = fixture()
    hs[0] += .03*np.diag([1., -1., .5])
    _, _, diag, arrays = recover_generalized(h0, hs, 3, 4, random_coefficients(64, len(hs), rng))
    residuals = [r["signal_residual"] for r in diag["candidates"] if r["status"] == "complete"]
    assert diag["signal_residual"] <= min(residuals) + 1e-12
    assert diag["max_commutator"] > 1e-5
    assert joint_residuals(h0, hs, arrays["C"])["signal_residual"] > 1e-5


def test_real_value_bank_and_integrated_final_branch():
    rng, w, a, _, _ = fixture()
    w1 = np.linalg.qr(rng.normal(size=(6, 3)))[0]
    teacher = Teacher([w1, w], a, 4)
    cfg = resolve({"architecture": {"d": 6, "hidden_widths": [3, 3]},
                   "oracle": {"cache_size": 0},
                   "final_layer": {"method": "random_combination", "hessian_directions": 3}})
    arch = Architecture(**cfg["architecture"])
    ledger = QueryLedger(cache_size=0)
    real = CountedRealOracle(teacher.forward, 6, ledger)
    suffix = SuffixOracle(real, arch, 2, [w1], cfg["oracle"], ledger)
    estimate, coef, diag, _ = recover_final(suffix, arch, cfg, rng, ledger)
    _, order = linear_sum_assignment(np.linalg.norm(w[:, :, None]-estimate[:, None, :], axis=0))
    np.testing.assert_allclose(estimate[:, order], w, atol=1e-8)
    np.testing.assert_allclose(coef[order], a, atol=1e-8)
    assert ledger.counts["n_real_calls_total"] == 120
    assert ledger.counts["n_hessian_calls"] == 4
    assert diag["hessian_bank_diagnostics"]["query_counts"]["n_real_calls_total"] == 120
    assert not ledger.counts["n_debug_complex_calls"]


def test_direction_design_and_invalid_nodes():
    directions = orthogonal_directions(3, 12, np.random.default_rng(90))
    np.testing.assert_allclose(directions.T @ directions/12, np.eye(3)/3, atol=1e-15)
    ledger = QueryLedger(cache_size=0)
    with pytest.raises(ValueError, match="positive"):
        collect_hessians(lambda _: 1., -np.eye(3), degree=4,
                         tau=.95, step=.1, ledger=ledger)
    assert ledger.counts["n_real_calls_total"] == 0
    with pytest.raises(ValueError, match="orthogonal"):
        orthogonal_directions(3, 5, np.random.default_rng(1))


def test_invalid_configuration():
    with pytest.raises(ValueError, match="orthogonal"):
        resolve({"final_layer": {"method": "random_combination", "hessian_directions": 5}})
    with pytest.raises(ValueError, match="method"):
        resolve({"final_layer": {"method": "unknown"}})
