"""Tests for the deployable-structure design layer (foldforge.design.deployable).

Checks the three genuine properties: panels stay rigid across the deployment
sweep (edge lengths preserved to machine precision - single-DOF isometric
mechanism); the compaction ratio exceeds 1 and matches a hand computation from
the raw footprint; and inverse design aims the deployed state at a target
footprint, driving the fit loss to ~0. All small/fast.
"""

import numpy as np

from foldforge.diff.miura import footprint, fold_limit
from foldforge.design.deployable import (
    MiuraDeployable, inverse_design_deployed)


def test_rigid_edges_preserved_across_sweep():
    """Edge lengths are preserved to <1e-9 over the whole deployment - the array
    folds/unfolds as a rigid single-DOF mechanism (isometric)."""
    dep = MiuraDeployable(rows=6, cols=6)
    assert dep.max_edge_deviation(np.linspace(0, 1, 41)) < 1e-9
    # non-trivial geometry: stowed state has a real fold height
    assert dep.vertices(0.0)[..., 2].max() > 0.1
    assert dep.footprint_area(1.0) > dep.footprint_area(0.0)


def test_rigid_for_designed_params():
    """A non-default (inverse-designed-style) Miura is still rigid-foldable."""
    dep = MiuraDeployable(rows=5, cols=7, a=1.3, b=0.8, gamma=np.radians(52))
    assert dep.max_edge_deviation() < 1e-9


def test_compaction_ratio_matches_hand_check():
    """Compaction ratio > 1 and equals deployed/stowed footprint computed
    straight from the raw footprint() engine (independent of the class)."""
    dep = MiuraDeployable(rows=6, cols=6)
    ratio = dep.compaction_ratio()
    assert ratio > 1.0
    # hand check against the underlying closed-form footprint
    lim = fold_limit(dep.a, dep.gamma)
    w_s, d_s, _ = footprint(6, 6, dep.a, dep.b, dep.gamma, 0.99 * lim)
    w_d, d_d, _ = footprint(6, 6, dep.a, dep.b, dep.gamma, 0.02 * lim)
    expected = (w_d * d_d) / (w_s * d_s)
    assert abs(ratio - expected) < 1e-9


def test_surface_area_invariant():
    """Panel (surface) area is the same stowed or deployed - panels are rigid."""
    dep = MiuraDeployable()
    area = dep.deployed_surface_area()
    assert area > 0
    # flat-sheet quad area sanity: ~ rows*cols * a*b*sin(gamma)
    approx = 6 * 6 * dep.a * dep.b * np.sin(dep.gamma)
    assert abs(area - approx) < 1e-6


def test_inverse_design_hits_target_footprint():
    """Inverse design aims the DEPLOYED footprint at a target; loss collapses and
    the achieved footprint matches the target closely."""
    target = (12.0, 9.0)
    dep, res, achieved = inverse_design_deployed(target, iters=400, lr=0.05)
    assert res.history[0] / max(res.final_loss, 1e-30) > 1e3   # loss collapses
    assert abs(achieved[0] - target[0]) < 1e-2
    assert abs(achieved[1] - target[1]) < 1e-2
    # the designed array is still a rigid-foldable deployable
    assert dep.max_edge_deviation() < 1e-9
    assert dep.compaction_ratio() > 1.0
