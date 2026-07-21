"""Tests for self-intersection detection and the optional repulsion."""

import numpy as np

from foldforge.geometry import examples
from foldforge.sim import (
    FoldMesh, fold, creases_along_x, intersection_count, self_intersections,
)


def test_flat_and_folded_miura_have_no_self_intersection():
    m = FoldMesh.from_pattern(examples.miura(4, 4))
    assert intersection_count(m, m.vertices) == 0
    r = fold(m, fold_fraction=0.6, actuate=creases_along_x(m))
    assert intersection_count(m, r.vertices) == 0


def test_detects_a_constructed_crossing():
    class Fake:
        pass
    fm = Fake()
    fm.triangles = np.array([[0, 1, 2], [3, 4, 5]])
    P = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0],
                  [0.5, 0.5, -1], [0.5, 0.5, 1], [1.5, 0.5, 0.0]])
    assert len(self_intersections(fm, P)) == 1
    P2 = P.copy(); P2[3:, 2] += 5
    assert len(self_intersections(fm, P2)) == 0


def test_avoid_intersection_flag_runs():
    m = FoldMesh.from_pattern(examples.miura(3, 3))
    r = fold(m, fold_fraction=0.6, avoid_intersection=True, stages=12, relax_iters=20)
    assert r.max_strain < 0.05
