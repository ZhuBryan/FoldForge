"""Tests for the TreeMaker-lite figurative design path.

Covers the packing (non-overlap, determinism, on-paper), the FOLD export round
trip, and flat-foldability of the trees that fully fold flat - the 3-flap
triangle *and* the 4-flap tangential quad, each of which now has a single
interior vertex (the incircle-molecule incentre) passing Kawasaki + Maekawa with
exact flap lengths. The 5-flap pentagon and the river tree are exercised as
honest *partial* bases (see the module docstring).
"""

import numpy as np
import pytest

from foldforge.design import (
    get_tree, pack_tree, crease_pattern, design_base, flap_length_errors,
    BUILTIN_TREES,
)
from foldforge.geometry.fold_io import write_fold, read_fold
from foldforge.geometry.foldability import foldability_report


def _fully_folds(name, flap_tol=0.02):
    """True iff every interior vertex passes both theorems and flaps are exact."""
    packing, pattern = design_base(get_tree(name))
    report = foldability_report(pattern)
    err = flap_length_errors(packing, pattern)
    passes = all(v.kawasaki and v.maekawa for v in report.vertices)
    return passes and err.max() < flap_tol, report, err


def test_packing_non_overlap_and_on_paper():
    packing = pack_tree(get_tree("three-flap"))
    c, r = packing.centers, packing.radii
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            assert np.linalg.norm(c[i] - c[j]) >= r[i] + r[j] - 1e-5
    assert np.all(c - r[:, None] >= -1e-5)
    assert np.all(c + r[:, None] <= 1 + 1e-5)


def test_packing_is_deterministic():
    a = pack_tree(get_tree("four-flap"))
    b = pack_tree(get_tree("four-flap"))
    assert np.allclose(a.centers, b.centers)
    assert a.scale == pytest.approx(b.scale)


def test_fold_export_round_trips(tmp_path):
    _, pattern = design_base(get_tree("three-flap"))
    path = tmp_path / "base.fold"
    write_fold(pattern, path)
    back = read_fold(path)
    assert back.n_vertices == pattern.n_vertices
    assert back.n_edges == pattern.n_edges
    assert list(back.assignment) == list(pattern.assignment)


def test_three_flap_is_flat_foldable():
    """The single-triangle base folds flat: every interior vertex passes."""
    packing, pattern = design_base(get_tree("three-flap"))
    report = foldability_report(pattern)
    assert len(report.vertices) == 1              # one incentre, all else on border
    assert report.flat_foldable
    for v in report.vertices:
        assert v.kawasaki is True
        assert v.maekawa is True


def test_four_flap_is_flat_foldable():
    """The 4-flap tangential quad folds flat end to end: one incircle molecule.

    This is the multi-flap case the old per-triangle assembly could not fold.
    """
    packing, pattern = design_base(get_tree("four-flap"))
    report = foldability_report(pattern)
    assert len(report.vertices) == 1              # single incentre, rest on border
    assert report.flat_foldable
    for v in report.vertices:
        assert v.kawasaki is True
        assert v.maekawa is True


@pytest.mark.parametrize("name", ["three-flap", "four-flap"])
def test_working_trees_fold_flat_with_exact_flaps(name):
    """The trees we claim work pass both theorems everywhere; flaps are exact."""
    ok, report, err = _fully_folds(name)
    assert ok, f"{name}: not fully flat-foldable"
    assert err.max() < 1e-3, f"{name}: flap error {err.max():.4f}"


@pytest.mark.parametrize("name", ["five-flap", "river-four"])
def test_partial_trees_are_honestly_partial(name):
    """Documented limits: the pentagon and the river tree do NOT fully fold.

    They still pack and export; this pins the honest ceiling so a regression that
    silently 'fixed' them (or broke the packing) would fail loudly.
    """
    ok, report, err = _fully_folds(name)
    assert not ok, f"{name}: unexpectedly fully foldable - update the docs/claims"
    assert len(report.vertices) > 0                 # it still builds a real pattern
