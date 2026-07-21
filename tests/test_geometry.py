"""Tests for the M0 geometry core.

These target the things that are easy to get silently wrong: the sector-angle
geometry, the two foldability theorems, and FOLD round-tripping. If any of
these break, a later milestone built on top would fail in a confusing way, so
we pin them down now.

Run with:  pytest -q
"""

import numpy as np
import pytest

from foldforge.geometry import examples
from foldforge.geometry.crease_graph import CreasePattern
from foldforge.geometry.fold_io import read_fold, write_fold
from foldforge.geometry.foldability import (
    check_kawasaki,
    check_maekawa,
    foldability_report,
)


# --- sector-angle geometry --------------------------------------------------

def test_sector_angles_sum_to_360():
    """Whatever the pattern, the wedges around any vertex must close the circle."""
    for gen in examples.GENERATORS.values():
        p = gen()
        for v in p.interior_vertices():
            assert abs(sum(p.sector_angles(v)) - 360.0) < 1e-9


def test_single_vertex_known_angles():
    """A vertex built from known wedge angles reports those same angles back."""
    p = examples.single_vertex(angles_deg=(90, 90, 90, 90))
    sectors = sorted(p.sector_angles(0))
    assert np.allclose(sectors, [90, 90, 90, 90])


# --- Kawasaki ---------------------------------------------------------------

def test_kawasaki_passes_on_flat_foldable():
    p = examples.single_vertex(angles_deg=(60, 120, 120, 60))
    assert check_kawasaki(p, 0) is True


def test_kawasaki_fails_when_angles_unbalanced():
    # 80+100 = 180 but the alternating split is 80+80 vs 100+100 -> not equal.
    p = examples.single_vertex(angles_deg=(80, 100, 80, 100),
                               assignment=("M", "V", "M", "M"))
    assert check_kawasaki(p, 0) is False


def test_kawasaki_odd_degree_is_false():
    """Three creases can't split into two equal alternating groups."""
    p = examples.single_vertex(angles_deg=(120, 120, 120),
                               assignment=("M", "V", "M"))
    assert check_kawasaki(p, 0) is False


def test_kawasaki_not_applicable_on_border():
    p = examples.single_vertex()
    # vertex 1 is a ray endpoint -> sits on the border -> theorem N/A.
    assert check_kawasaki(p, 1) is None


# --- Maekawa ----------------------------------------------------------------

def test_maekawa_passes_on_three_minus_one():
    p = examples.single_vertex(assignment=("M", "V", "M", "M"))  # 3M, 1V
    assert check_maekawa(p, 0) is True


def test_maekawa_fails_on_balanced_assignment():
    p = examples.single_vertex(assignment=("M", "V", "M", "V"))  # 2M, 2V
    assert check_maekawa(p, 0) is False


def test_maekawa_undecidable_with_unassigned_crease():
    p = examples.single_vertex(assignment=("M", "V", "M", "U"))
    assert check_maekawa(p, 0) is None


# --- whole-pattern reports --------------------------------------------------

def test_miura_is_flat_foldable():
    assert foldability_report(examples.miura()).flat_foldable is True


def test_waterbomb_base_is_not_flat_foldable():
    """The waterbomb base is a 3D base; it must fail the flat-fold screen."""
    assert foldability_report(examples.waterbomb_base()).flat_foldable is False


def test_corrupting_a_miura_vertex_breaks_foldability():
    """Acceptance criterion: nudge one interior vertex -> validators catch it."""
    p = examples.miura()
    assert foldability_report(p).flat_foldable is True
    p.vertices[p.n_vertices // 2] += np.array([0.25, 0.18])
    assert foldability_report(p).flat_foldable is False


# --- FOLD I/O ---------------------------------------------------------------

def test_fold_roundtrip(tmp_path):
    orig = examples.miura()
    path = tmp_path / "m.fold"
    write_fold(orig, path)
    back = read_fold(path)
    assert np.allclose(orig.vertices, back.vertices)
    assert np.array_equal(orig.edges, back.edges)
    assert orig.assignment == back.assignment
    assert orig.faces == back.faces


def test_fold_missing_assignment_defaults_to_unassigned(tmp_path):
    path = tmp_path / "bare.fold"
    path.write_text('{"vertices_coords": [[0,0],[1,0]], "edges_vertices": [[0,1]]}')
    p = read_fold(path)
    assert p.assignment == ["U"]


# --- data-model guards ------------------------------------------------------

def test_assignment_length_must_match_edges():
    with pytest.raises(ValueError):
        CreasePattern(
            vertices=np.zeros((2, 2)), edges=np.array([[0, 1]]),
            assignment=["M", "V"],  # one too many
        )
