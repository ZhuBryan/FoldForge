"""Tests for the photo -> stick-figure metric tree bridge (skeleton module).

Synthetic masks with a *known* number of arms exercise the medial-axis ->
graph -> prune pipeline without needing a photo or OpenCV. The leaf-count tests
need scikit-image's thinning (the scipy fallback is coarser and can leave stubs
that change the count), so they are gated with ``importorskip``; the
determinism / positive-length invariants hold for either backend.
"""

import numpy as np
import pytest

from foldforge.design.skeleton import tree_from_mask
from foldforge.design.treemaker import MetricTree


def _star_mask(n_arms, R=90, r=13, size=240):
    """A thick n-armed star: a filled hub with n spokes of radius ``r``."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size // 2
    m = np.zeros((size, size), bool)
    for a in np.linspace(0, 2 * np.pi, n_arms, endpoint=False):
        for t in np.linspace(0, 1, 400):
            y = cy + R * np.sin(a) * t
            x = cx + R * np.cos(a) * t
            m[(yy - y) ** 2 + (xx - x) ** 2 <= r ** 2] = True
    return m


def test_five_arm_star_has_five_leaves():
    pytest.importorskip("skimage")
    ext = tree_from_mask(_star_mask(5), smooth_frac=0.0)
    assert ext.n_leaves == 5
    assert isinstance(ext.tree, MetricTree)


def _stubby_star_mask(n_arms=5, R=95, r=11, body=48, size=260):
    """A 5-armed star with a large central *body*: the arms are short relative to
    the hub, so an over-aggressive spur pruner would eat them. Guards the count."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size // 2
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= body ** 2
    for a in np.linspace(0, 2 * np.pi, n_arms, endpoint=False):
        for t in np.linspace(0, 1, 400):
            y = cy + R * np.sin(a) * t
            x = cx + R * np.cos(a) * t
            m[(yy - y) ** 2 + (xx - x) ** 2 <= r ** 2] = True
    return m


def test_stubby_five_arm_star_keeps_five_leaves():
    """Short arms on a thick body must not be pruned away as spurs (regression:
    the bridge under-counted stubby limbs). Non-degenerate 5-leaf tree expected."""
    pytest.importorskip("skimage")
    ext = tree_from_mask(_stubby_star_mask(), smooth_frac=0.0)
    assert ext.n_leaves == 5
    assert not ext.degenerate


def test_plus_shape_has_four_leaves():
    pytest.importorskip("skimage")
    ext = tree_from_mask(_star_mask(4), smooth_frac=0.0)
    assert ext.n_leaves == 4


def test_blob_is_trivial_tree():
    """A featureless disc has no limbs: a degenerate 2-node stub, not a crash."""
    size = 200
    yy, xx = np.mgrid[0:size, 0:size]
    blob = (yy - 100) ** 2 + (xx - 100) ** 2 <= 55 ** 2
    ext = tree_from_mask(blob)
    assert ext.degenerate
    assert ext.n_leaves <= 2


def test_edges_have_positive_length():
    ext = tree_from_mask(_star_mask(5), smooth_frac=0.0)
    lengths = [L for _, _, L in ext.tree.edges]
    assert lengths and all(L > 0 for L in lengths)


def test_deterministic():
    m = _star_mask(5)
    a = tree_from_mask(m, smooth_frac=0.0)
    b = tree_from_mask(m, smooth_frac=0.0)
    assert a.n_leaves == b.n_leaves
    assert a.tree.edges == b.tree.edges


def test_expected_leaves_note():
    ext = tree_from_mask(_star_mask(5), smooth_frac=0.0, expected_leaves=5)
    assert "matches expected 5" in ext.note
