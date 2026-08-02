"""Tests for the TreeMaker-lite figurative design path.

Covers the packing (non-overlap, determinism, on-paper), the FOLD export round
trip, and flat-foldability of the flagship 3-flap tree (Kawasaki + Maekawa at
its single interior vertex, the molecule incentre).
"""

import numpy as np
import pytest

from foldforge.design import (
    get_tree, pack_tree, crease_pattern, design_base, flap_length_errors,
    BUILTIN_TREES,
)
from foldforge.geometry.fold_io import write_fold, read_fold
from foldforge.geometry.foldability import foldability_report


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


def test_three_flap_flap_lengths_match_tree():
    """Tangent packing -> folded flap lengths equal the tree edges (~0 error)."""
    packing, pattern = design_base(get_tree("three-flap"))
    err = flap_length_errors(packing, pattern)
    assert err.max() < 1e-3


def test_every_incentre_passes_kawasaki():
    """Kawasaki holds at every molecule incentre for all built-in trees."""
    for name in BUILTIN_TREES:
        packing, pattern = design_base(get_tree(name))
        report = foldability_report(pattern)
        # The incentres are the degree-6 interior vertices; each must pass.
        deg6 = [v.vertex for v in report.vertices
                if len(pattern.incident_edges(v.vertex)) == 6]
        assert deg6, f"{name}: expected at least one incentre"
        for v in report.vertices:
            if v.vertex in deg6:
                assert v.kawasaki is True, f"{name}: incentre {v.vertex} fails Kawasaki"
