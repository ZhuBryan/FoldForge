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
    inverse_design_shape, kawasaki_penalty, _kawasaki_neighbors,
)
from foldforge.origamize.miura_fit import _flat_miura


def _bumpy_target(R, C):
    """A seeded, asymmetric, non-Miura height field (a developable sheet cannot
    match it exactly)."""
    xs = np.linspace(-1, 1, C); ys = np.linspace(-1, 1, R)
    X, Y = np.meshgrid(xs, ys)
    return (np.exp(-((X - 0.3) ** 2 + (Y + 0.2) ** 2) * 3)
            + 0.6 * np.exp(-((X + 0.5) ** 2 + (Y - 0.4) ** 2) * 6) + 0.2 * X)


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


# --- v2: arbitrary-target fit with the foldability penalty -------------------

def test_v2_fidelity_decreases():
    """The joint Q/P fit drives its total loss down by a large factor on a
    seeded non-Miura height field."""
    R, C = 9, 9
    Z = _bumpy_target(R, C)
    res = inverse_design_shape(Z, (R, C), target_kind="height",
                               w_fit=1.0, w_iso=4.0, w_kawasaki=0.0,
                               iters=500, lr=0.02)
    assert res.history[-1] < 1e-2 * res.history[0]      # loss collapses
    assert res.fidelity < 0.02                           # normalised RMSE small


def test_v2_pointcloud_fidelity_decreases():
    """Chamfer fidelity to a point cloud decreases substantially."""
    us, vs = np.meshgrid(np.linspace(0, 7, 15), np.linspace(0, 7, 15))
    Z = 0.15 * ((us - 3.5) ** 2 - (vs - 3.5) ** 2) / 3.0      # saddle
    T = np.stack([us.ravel(), vs.ravel(), Z.ravel()], axis=1)
    res = inverse_design_shape(T, (8, 8), target_kind="pointcloud",
                               w_fit=1.0, w_iso=2.0, iters=300, lr=0.03)
    assert res.history[-1] < 0.5 * res.history[0]


def test_kawasaki_gradient_matches_fd():
    """Analytic Kawasaki-penalty gradient agrees with finite differences."""
    rng = np.random.default_rng(1)
    R, C = 5, 5
    Q = _flat_miura(R, C, 1.0, 1.0, np.radians(60.0))
    Q = Q + rng.normal(0, 0.08, Q.shape); Q[:, 2] = 0.0    # warp off foldable
    centers, nbrs = _kawasaki_neighbors(R, C)
    _, grad, _ = kawasaki_penalty(Q, centers, nbrs)
    eps = 1e-6
    fd = np.zeros_like(Q)
    for i in range(Q.shape[0]):
        for c in range(2):
            Qp = Q.copy(); Qp[i, c] += eps
            Qm = Q.copy(); Qm[i, c] -= eps
            fd[i, c] = (kawasaki_penalty(Qp, centers, nbrs)[0]
                        - kawasaki_penalty(Qm, centers, nbrs)[0]) / (2 * eps)
    assert np.max(np.abs(grad[:, :2] - fd[:, :2])) < 1e-4


def test_kawasaki_zero_on_uniform_miura():
    """A uniform Miura is flat-foldable, so the penalty vanishes there."""
    R, C = 6, 6
    Q = _flat_miura(R, C, 1.0, 1.0, np.radians(60.0))
    centers, nbrs = _kawasaki_neighbors(R, C)
    pen, _, s = kawasaki_penalty(Q, centers, nbrs)
    assert pen < 1e-20
    assert np.max(np.abs(s)) < 1e-9


def test_v2_penalty_improves_foldability():
    """w_kawasaki > 0 yields a lower Kawasaki residual than w_kawasaki = 0 on
    the same target - the foldability penalty works (with a fidelity cost)."""
    R, C = 9, 9
    Z = _bumpy_target(R, C)
    off = inverse_design_shape(Z, (R, C), w_kawasaki=0.0, iters=500, lr=0.02)
    on = inverse_design_shape(Z, (R, C), w_kawasaki=0.5, iters=500, lr=0.02)
    assert on.kawasaki_residual < 0.5 * off.kawasaki_residual   # more foldable
    assert on.fidelity >= off.fidelity                          # fidelity cost
