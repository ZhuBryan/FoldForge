"""Tests for the M2 differentiable kinematics.

The core promise of M2 is correct gradients, so these check the analytic
Jacobian and the apex-height gradient against finite differences, plus the
rigidity (isometry) the whole approach relies on.
"""

import numpy as np

from foldforge.diff.kinematics import (
    fold_chain, spine_jacobian, apex_height, apex_height_grad,
)


def test_flat_chain_is_straight():
    res = fold_chain(np.zeros(6), seg=1.0)
    assert np.allclose(res.spine[:, 1], 0.0)              # no z when unfolded
    assert np.isclose(res.spine[-1, 0], 6.0)              # length preserved


def test_chain_is_isometric():
    res = fold_chain(np.array([0.3, -0.2, 0.5, 0.1]), seg=1.3)
    seglen = np.linalg.norm(np.diff(res.spine, axis=0), axis=1)
    assert np.allclose(seglen, 1.3)                       # rigid panels


def test_spine_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    worst = 0.0
    eps = 1e-6
    for _ in range(30):
        a = rng.uniform(-1, 1, 6)
        J = spine_jacobian(a)
        for j in range(6):
            ap = a.copy(); ap[j] += eps
            am = a.copy(); am[j] -= eps
            fd = (fold_chain(ap).spine - fold_chain(am).spine) / (2 * eps)
            worst = max(worst, np.abs(fd - J[:, :, j]).max())
    assert worst < 1e-6


def test_apex_height_gradient_matches_fd():
    rng = np.random.default_rng(1)
    a = rng.uniform(-0.5, 0.5, 7)
    g = apex_height_grad(a)
    eps = 1e-6
    for j in range(7):
        ap = a.copy(); ap[j] += eps
        am = a.copy(); am[j] -= eps
        fd = (apex_height(ap) - apex_height(am)) / (2 * eps)
        assert abs(fd - g[j]) < 1e-6
