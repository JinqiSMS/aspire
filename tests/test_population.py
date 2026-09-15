import numpy as np
from aspire.teacher import Teacher
from aspire.diagnostics.population import value_gradient_batch, integrate_body_3d
from aspire.recovery.moments import moment_directions
from aspire.diagnostics.population import aligned_operator_error


def test_batch_reference_matches_scalar_and_finite_differences():
    rng = np.random.default_rng(871)
    teacher = Teacher([rng.normal(size=(5, 3)), rng.uniform(.1, .8, (3, 2))], np.array([.4, .6]), 4)
    points = rng.normal(size=(7, 5)) * .3
    values, gradients = value_gradient_batch(teacher, points)
    assert np.allclose(values, [teacher.forward(p) for p in points])
    assert np.allclose(gradients, [teacher.gradient(p) for p in points])
    step = 1e-6
    finite = np.stack([(teacher.forward(points+step*e)-teacher.forward(points-step*e))/(2*step)
                       for e in np.eye(5)], axis=1)
    assert np.allclose(gradients, finite, rtol=1e-7, atol=1e-9)


def test_radial_integral_against_exact_ball_moments():
    def radial(points):
        squared = np.sum(points**2, axis=1)
        return squared**2, 4*squared[:, None]*points
    sx, sg = integrate_body_3d(radial, 4, 16)
    assert np.allclose(sx, np.eye(3)/5, atol=1e-14)
    assert np.allclose(sg, np.eye(3)*16/9, atol=1e-13)


def test_full_rotated_quadrature_recovers_nonorthogonal_directions():
    first = np.array([[1., .15, -.1], [.1, 1., .2], [.08, -.1, 1.]])
    first /= np.linalg.norm(first, axis=0)
    second = np.array([[.7, .2, .1], [.1, .5, .2], [.2, .3, .7]])
    teacher = Teacher([first, second], np.array([.2, .3, .5]), 4)
    sx, sg = integrate_body_3d(lambda x: value_gradient_batch(teacher, x), 16, 160)
    _, directions, _ = moment_directions(sx, sg)
    assert aligned_operator_error(first, directions) < 1e-5
