"""Tests for the standalone Lang universal molecule (foldforge.design.lang_molecule)."""

import numpy as np
import pytest

from foldforge.design import get_tree
from foldforge.design.lang_molecule import flat_fold_image, lang_base
from foldforge.design.treemaker import MetricTree, flap_length_errors
from foldforge.geometry.foldability import foldability_report

EXACT_TREES = ["three-flap", "four-flap", "five-flap", "river-four",
               "river-mirror", "river-asym", "river-caterpillar"]


def _check(tree):
    packing, pattern = lang_base(tree)
    report = foldability_report(pattern)
    assert report.vertices
    assert all(v.kawasaki and v.maekawa for v in report.vertices), report.summary()
    meta = pattern.metadata["lang"]
    assert meta["faces_connected"] and meta["fold_closure"] < 1e-6
    assert meta["hub_spread"] < 1e-6
    # Exactness is measured on the INDEPENDENTLY folded geometry
    # (folded_edge_lengths, read off the flat-folded image) against the external
    # target scale*edge - every tree edge, spines included. This is the sound
    # check; flap_length_errors re-reads lang's own metadata and is circular.
    edges = meta["folded_edge_lengths"]
    for u, v, w in packing.tree.edges:
        assert edges[(min(u, v), max(u, v))] == pytest.approx(packing.scale * w, rel=1e-6)
    return packing, pattern, flap_length_errors(packing, pattern)


@pytest.mark.parametrize("name", EXACT_TREES)
def test_known_trees_fold_flat_and_exact(name):
    _, _, err = _check(get_tree(name))
    assert np.all(np.isfinite(err)) and err.max() < 1e-3


def test_caterpillar_is_exact_on_every_flap():
    packing, pattern, err = _check(get_tree("river-caterpillar"))
    assert err.max() < 1e-6
    edges = pattern.metadata["lang"]["folded_edge_lengths"]
    for u, v, w in packing.tree.edges:                 # every tree edge, spines included
        assert edges[(min(u, v), max(u, v))] == pytest.approx(packing.scale * w, rel=1e-6)


def test_asymmetric_caterpillar_is_exact():
    tree = MetricTree(
        edges=[(0, 1, 0.4), (1, 2, 0.6), (0, 3, 1.0), (0, 4, 0.8),
               (1, 5, 0.9), (2, 6, 0.7), (2, 7, 1.1)], root=1, name="cat-asym")
    _, _, err = _check(tree)
    assert err.max() < 1e-3


def test_broken_pattern_is_detected():
    """Negative control: nudging an interior vertex must break the folding map."""
    _, pattern, _ = _check(get_tree("river-caterpillar"))
    bad = pattern.vertices.copy()
    v = pattern.interior_vertices()[0]
    bad[v] += 0.01
    pattern.vertices = bad
    _, closure, _ = flat_fold_image(pattern)
    assert closure > 1e-3
