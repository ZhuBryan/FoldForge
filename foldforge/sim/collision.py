"""Self-intersection detection (does the folded paper pass through itself?).

Broad phase + narrow phase, the standard recipe:

  * broad phase - a spatial hash buckets each triangle by the grid cells its
    bounding box covers, so we only ever test triangle pairs that share a cell
    (avoids the O(T^2) all-pairs blow-up);
  * narrow phase - two triangles intersect iff an edge of one pierces the other,
    tested with Moller-Trumbore segment/triangle intersection.

Adjacent triangles (sharing a vertex) are skipped - they meet by design.
``separation_penalty`` returns a soft repulsion the solver can optionally use to
push penetrating panels apart.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _seg_tri_hit(o, d, a, b, c):
    """Moller-Trumbore: does segment ``o -> o+d`` cross triangle ``abc``?"""
    e1 = b - a; e2 = c - a
    pv = np.cross(d, e2); det = np.dot(e1, pv)
    if abs(det) < _EPS:
        return False
    inv = 1.0 / det
    tv = o - a; u = np.dot(tv, pv) * inv
    if u < -_EPS or u > 1 + _EPS:
        return False
    qv = np.cross(tv, e1); v = np.dot(d, qv) * inv
    if v < -_EPS or u + v > 1 + _EPS:
        return False
    t = np.dot(e2, qv) * inv
    return _EPS < t < 1 - _EPS          # strictly inside the segment


def _tris_intersect(A, B):
    """True if triangles A and B (each 3x3) intersect."""
    for (o, q) in ((A[0], A[1]), (A[1], A[2]), (A[2], A[0])):
        if _seg_tri_hit(o, q - o, B[0], B[1], B[2]):
            return True
    for (o, q) in ((B[0], B[1]), (B[1], B[2]), (B[2], B[0])):
        if _seg_tri_hit(o, q - o, A[0], A[1], A[2]):
            return True
    return False


def self_intersections(mesh, positions):
    """Return a list of intersecting, non-adjacent triangle-index pairs."""
    tris = mesh.triangles
    P = positions
    corners = [P[t] for t in tris]
    aabb_lo = np.array([c.min(0) for c in corners])
    aabb_hi = np.array([c.max(0) for c in corners])
    cell = float(np.mean(aabb_hi - aabb_lo)) + _EPS      # ~ a few triangles per cell
    buckets: dict = {}
    for i in range(len(tris)):
        lo = np.floor(aabb_lo[i] / cell).astype(int)
        hi = np.floor(aabb_hi[i] / cell).astype(int)
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    buckets.setdefault((x, y, z), []).append(i)
    seen = set()
    hits = []
    for members in buckets.values():
        for a_idx in range(len(members)):
            for b_idx in range(a_idx + 1, len(members)):
                i, j = members[a_idx], members[b_idx]
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                if set(tris[i]) & set(tris[j]):          # adjacent -> skip
                    continue
                if _tris_intersect(corners[i], corners[j]):
                    hits.append((i, j))
    return hits


def intersection_count(mesh, positions) -> int:
    """How many non-adjacent triangle pairs intersect (0 = clean fold)."""
    return len(self_intersections(mesh, positions))


def separation_penalty(mesh, positions, strength: float = 1.0):
    """Soft repulsion (per vertex) pushing intersecting panels apart.

    For each intersecting pair, repel their centroids along the line joining
    them. Continuous and cheap - an optional nudge inside the solver, not a hard
    constraint.
    """
    F = np.zeros_like(positions)
    tris = mesh.triangles
    for i, j in self_intersections(mesh, positions):
        d = positions[tris[i]].mean(0) - positions[tris[j]].mean(0)
        n = np.linalg.norm(d)
        dir_ = d / n if n > _EPS else np.array([0.0, 0.0, 1.0])
        F[tris[i]] += strength * dir_
        F[tris[j]] -= strength * dir_
    return F
