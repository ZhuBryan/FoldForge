"""Regression tests for the 2026-07 audit (boundary / degenerate-input bugs).

Each test pins a real bug found while hunting for siblings of the studio
``resampleProfile`` boundary overrun: an index computed from a float walking off
its array, or a degenerate input crashing deep instead of raising a clear error.
"""

import numpy as np
import pytest

from foldforge.origamize.surface import origamize_heightfield, close_relief
from foldforge.fabricate import to_gltf


def _nonmanifold_edges(T):
    from collections import defaultdict
    ec = defaultdict(int)
    for a, b, c in T:
        for u, v in ((a, b), (b, c), (c, a)):
            ec[frozenset((int(u), int(v)))] += 1
    return [e for e, c in ec.items() if c != 2]


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1)])
def test_heightfield_rejects_degenerate_grid(shape):
    # A single-row/column height field has no foldable panel: raise a clear
    # ValueError instead of crashing deep in CreasePattern construction.
    with pytest.raises(ValueError, match="at least 2x2"):
        origamize_heightfield(np.ones(shape))


def test_heightfield_rejects_non_2d():
    with pytest.raises(ValueError, match="must be 2D"):
        origamize_heightfield(np.ones((3, 3, 3)))


def test_heightfield_smallest_valid_grid_ok():
    r = origamize_heightfield(np.random.default_rng(0).random((2, 2)) * 6.0)
    assert np.isfinite(r.folded).all()


def test_to_gltf_rejects_empty_mesh():
    with pytest.raises(ValueError, match="empty mesh"):
        to_gltf(np.zeros((0, 3)), np.zeros((0, 3), int), "/tmp/_audit_empty.glb")


@pytest.mark.parametrize("g", [2, 3, 4, 5, 10, 20])
def test_close_relief_watertight_across_sizes(g):
    # close_relief must stay watertight (every edge shared by exactly 2 tris) and
    # finite on grids down to the 2x2 minimum, incl. all-zero (flat) fields.
    r = origamize_heightfield(np.random.default_rng(1).random((g, g)) * 6.0)
    V, T = close_relief(r)
    assert np.isfinite(V).all()
    assert not _nonmanifold_edges(T)


@pytest.mark.parametrize("mode", ["mirror", "flat"])
def test_close_relief_watertight_zero_field(mode):
    r = origamize_heightfield(np.zeros((6, 6)))
    V, T = close_relief(r, mode=mode)
    assert not _nonmanifold_edges(T)
