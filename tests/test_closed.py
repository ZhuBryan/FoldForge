"""Closed / double-sided relief: mirror the folded relief into a watertight solid.

The photo -> origami pipeline yields a one-sided open sheet with a hollow back;
:func:`foldforge.origamize.surface.close_relief` mirrors it into a printable,
watertight two-sheet solid. These tests use a tiny synthetic relief so they run
fast (no image, no network).
"""

from collections import defaultdict

import numpy as np
import pytest

from foldforge.origamize import close_relief, heightfield_dome
from foldforge.origamize.surface import origamize_heightfield, _signed_volume
from foldforge.fabricate import to_stl, to_gltf


def _small_relief(nx=8, ny=7, height=5.0):
    """A tiny dome relief (mixed rim: flat corners, raised edge-midpoints)."""
    return origamize_heightfield(heightfield_dome(nx=nx, ny=ny) * height)


def _edge_counts(triangles):
    """Map undirected edge -> number of triangles using it."""
    counts = defaultdict(int)
    for a, b, c in triangles:
        for e in ((a, b), (b, c), (c, a)):
            counts[frozenset((int(e[0]), int(e[1])))] += 1
    return counts


def _directed_ok(triangles):
    """True if every directed half-edge appears exactly once (consistent, closed)."""
    d = defaultdict(int)
    for a, b, c in triangles:
        for e in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            d[e] += 1
    return all(k == 1 and d.get((e[1], e[0]), 0) == 1 for e, k in d.items())


@pytest.mark.parametrize("mode", ["mirror", "flat"])
def test_watertight(mode):
    res = _small_relief()
    V, T = close_relief(res, mode=mode, base=-2.0)
    assert not np.isnan(V).any()
    counts = _edge_counts(T)
    # every edge shared by exactly two triangles -> a closed 2-manifold
    assert all(c == 2 for c in counts.values()), \
        f"non-manifold edges: {sorted(set(counts.values()))}"
    # consistent outward orientation and positive enclosed volume
    assert _directed_ok(T), "inconsistent triangle winding"
    assert _signed_volume(V, T) > 0.0


def test_mirror_symmetric():
    res = _small_relief()
    V, T = close_relief(res, mode="mirror")
    z = V[:, 2]
    # mirror about z=0: for every z there is a -z, and the set is symmetric
    assert abs(z.min() + z.max()) < 1e-9
    # the z multiset is symmetric about 0: sort(z) == sort(-z)
    assert np.allclose(np.sort(z), np.sort(-z))
    # the closed solid is thicker (more verts/tris) than the open front sheet
    assert len(V) > len(res.folded)
    assert len(T) > len(res.triangles)


def test_flat_back_is_planar():
    res = _small_relief()
    base = -3.0
    V, T = close_relief(res, mode="flat", base=base)
    # the appended back vertices all sit on the z=base plane
    back = V[len(res.folded):]
    assert back.shape[0] > 0
    assert np.allclose(back[:, 2], base)


def test_closed_solid_is_roughly_double():
    """A mirrored solid roughly doubles the surface vs. the one-sided sheet."""
    res = _small_relief()
    V, T = close_relief(res, mode="mirror")
    assert len(T) >= 2 * len(res.triangles)          # front + back (+ wall)


def test_exports_accept_closed_mesh(tmp_path):
    res = _small_relief()
    V, T = close_relief(res, mode="mirror")
    stl = tmp_path / "solid.stl"
    glb = tmp_path / "solid.glb"
    to_stl(V, T, stl)
    to_gltf(V, T, glb)
    assert stl.stat().st_size > 0 and glb.stat().st_size > 0
    assert stl.read_text().startswith("solid")


def test_bad_mode_raises():
    res = _small_relief()
    with pytest.raises(ValueError):
        close_relief(res, mode="nonsense")
