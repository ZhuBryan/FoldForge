"""Tests for M3 differentiable inverse design (foldforge.diff.inverse).

These check the three capabilities: fold-angle recovery converges to the true
angles on a seeded well-conditioned chain (and is deterministic); the Miura
footprint fit reduces its loss by a large factor; and refine lowers a starting
objective. All small/fast.
"""

import numpy as np

from foldforge.diff.kinematics import fold_chain
from foldforge.diff.miura import footprint
from foldforge.diff.inverse import (
    recover_fold_angles, inverse_fit, refine,
    miura_footprint_objective, miura_param_project,
)


def test_recover_fold_angles_converges():
    """Fold a chain with known angles, recover them from near-flat init."""
    rng = np.random.default_rng(42)
    true = rng.uniform(-0.6, 0.6, 5)
    target = fold_chain(true, seg=1.0).spine
    res = recover_fold_angles(target, seg=1.0, iters=3000, lr=0.05, seed=0)
    assert np.max(np.abs(res.params - true)) < 1e-3     # headline precision
    assert res.final_loss < 1e-8


def test_recover_fold_angles_deterministic():
    rng = np.random.default_rng(7)
    true = rng.uniform(-0.5, 0.5, 4)
    target = fold_chain(true).spine
    a = recover_fold_angles(target, iters=1500, seed=0)
    b = recover_fold_angles(target, iters=1500, seed=0)
    assert np.array_equal(a.params, b.params)
    assert np.array_equal(a.history, b.history)


def test_inverse_fit_reduces_loss():
    """Recover a Miura's own footprint: loss must drop by a huge factor."""
    target = footprint(6, 6, 1.2, 0.9, np.radians(55), 0.5)
    res, achieved = inverse_fit(target, iters=400, lr=0.05)
    assert res.history[0] / max(res.final_loss, 1e-30) > 1e3
    assert np.allclose(achieved, target, atol=1e-3)


def test_refine_lowers_objective():
    """Starting from a default Miura, refine toward a target footprint."""
    target = footprint(6, 6, 1.2, 0.9, np.radians(55), 0.5)
    obj = miura_footprint_objective(target)
    p0 = np.array([1.0, 1.0, np.radians(60), 0.4])
    res = refine(p0, obj, iters=300, lr=0.02, project=miura_param_project)
    assert res.final_loss < res.history[0]
    assert res.final_loss < 1e-3 * res.history[0]       # substantially lower
