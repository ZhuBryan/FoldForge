"""Origamizer tuck-folding: the folded crease pattern reconstructs the target.

Scope under test (see foldforge/design/origamizer.py): a square pyramid and a
tetrahedron, folded from a flat net-with-gaps whose gaps are closed by
flat-fold parallel-pleat tucks. We check the three things that make it a real
Origamizer result and not a fake:

  * reconstruction -- every surface vertex lands on its target, dihedrals exact
  * isometry       -- every crease-pattern edge keeps its length when folded
  * genuine tucks  -- excess paper exists, is hidden behind, and is accounted for
"""

import numpy as np
import pytest

from foldforge.design import (
    square_pyramid,
    tetrahedron,
    origamize_polyhedron,
)
from foldforge.geometry.crease_graph import CreasePattern


@pytest.fixture(params=["pyramid", "tetra"])
def result(request):
    target = square_pyramid() if request.param == "pyramid" else tetrahedron()
    return origamize_polyhedron(target)


def test_reconstructs_target(result):
    """Folded surface vertices land on the target polyhedron (positions + apex)."""
    assert result.report.face_position_error < 1e-9


def test_dihedrals_exact(result):
    """Every tucked edge reaches its target dihedral angle."""
    assert result.report.dihedral_error_deg < 1e-6


def test_isometry(result):
    """Folding preserves every edge length -- the paper does not stretch."""
    assert result.report.isometry_error < 1e-9


def test_tucks_are_genuine(result):
    """There is real excess paper, it is hidden behind, and it is accounted for."""
    r = result.report
    # the flat sheet is strictly larger than the target surface
    assert r.area_flat > r.area_target + 1e-6
    # the excess equals the tucked material
    assert abs((r.area_flat - r.area_target) - r.area_tucked) < 1e-9
    # every pleat ends up behind the surface
    assert r.tuck_behind is True
    assert r.n_tucks >= 3


def test_face_areas_preserved(result):
    """Each flat face area is reproduced in 3D (no stretch, per face)."""
    V, folded = result.target_vertices, result.folded
    pat = result.pattern

    def area2d(idx):
        p = pat.vertices[idx]
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def area3d(idx):
        p = folded[idx]
        o, tot = p[0], 0.0
        for i in range(1, len(p) - 1):
            tot += 0.5 * np.linalg.norm(np.cross(p[i] - o, p[i + 1] - o))
        return tot

    for f in pat.faces:
        assert abs(area2d(f) - area3d(f)) < 1e-9


def test_pattern_is_valid(result):
    """The produced object is a well-formed CreasePattern with M/V/B edges."""
    pat = result.pattern
    assert isinstance(pat, CreasePattern)
    assert set(pat.assignment) <= {"M", "V", "B"}
    assert "M" in pat.assignment and "V" in pat.assignment and "B" in pat.assignment
    # fold angles present for every edge
    assert pat.fold_angle is not None and len(pat.fold_angle) == pat.n_edges


def test_gap_controls_excess():
    """A wider gap tucks strictly more paper; zero error either way."""
    small = origamize_polyhedron(square_pyramid(), gap=0.2)
    large = origamize_polyhedron(square_pyramid(), gap=0.6)
    assert large.report.area_tucked > small.report.area_tucked
    assert small.report.face_position_error < 1e-9
    assert large.report.face_position_error < 1e-9


def test_scope_rejects_unsupported_target():
    """A face sharing no edge with the base is honestly rejected, not faked."""
    from foldforge.design.origamizer import PolyTarget

    # a 'floating' face disconnected from the base face
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 1], [6, 5, 1], [5, 6, 1]], float)
    bad = PolyTarget("bad", verts, [[0, 1, 2], [3, 4, 5]], base=0)
    with pytest.raises(ValueError):
        origamize_polyhedron(bad)
