"""The canonical bird base (the crane's starting point) must be the *real*,
flat-foldable pattern: every interior vertex passes both Kawasaki and Maekawa,
and the FOLD export round-trips. Constructed directly (not circle-packed), so
these are exact, not approximate."""

import numpy as np
import pytest

from foldforge.design import bird_base
from foldforge.geometry.fold_io import write_fold, read_fold
from foldforge.geometry.foldability import (
    foldability_report, check_kawasaki, check_maekawa,
)


def test_bird_base_counts():
    """13 vertices, 28 edges, 16 faces, 5 interior vertices (4 petals + centre)."""
    p = bird_base()
    assert p.n_vertices == 13
    assert p.n_edges == 28
    assert len(p.faces) == 16
    assert len(p.interior_vertices()) == 5


def test_bird_base_passes_kawasaki_at_every_interior_vertex():
    p = bird_base()
    for v in p.interior_vertices():
        assert check_kawasaki(p, v) is True, f"vertex {v} fails Kawasaki"


def test_bird_base_passes_maekawa_at_every_interior_vertex():
    p = bird_base()
    for v in p.interior_vertices():
        assert check_maekawa(p, v) is True, f"vertex {v} fails Maekawa"


def test_bird_base_is_flat_foldable():
    """The whole-pattern report must be flat-foldable (both theorems, all vertices)."""
    p = bird_base()
    report = foldability_report(p)
    assert report.flat_foldable
    assert all(v.kawasaki and v.maekawa for v in report.vertices)
    assert len(report.vertices) == 5


def test_bird_base_centre_is_5m_3v():
    """The centre is a single 8-crease vertex; Maekawa forces |M - V| == 2."""
    p = bird_base()
    # centre is the last vertex (index 12); count its incident assignments.
    creases = p.vertex_creases(12)
    kinds = [a for _, a, _ in creases]
    assert len(kinds) == 8
    assert kinds.count("M") == 5 and kinds.count("V") == 3


def test_bird_base_fold_export_round_trips(tmp_path):
    p = bird_base()
    path = tmp_path / "bird_base.fold"
    write_fold(p, path)
    back = read_fold(path)
    assert back.n_vertices == p.n_vertices
    assert back.n_edges == p.n_edges
    assert list(back.assignment) == list(p.assignment)
    assert [list(f) for f in back.faces] == [list(f) for f in p.faces]


def test_bird_base_is_on_the_unit_square():
    p = bird_base()
    assert np.isclose(p.vertices.min(), 0.0)
    assert np.isclose(p.vertices.max(), 1.0)
