"""Classic, named origami bases built *directly* from their known crease pattern.

The circle-packing designer in :mod:`foldforge.design.treemaker` *derives* a base
from a metric tree. Some bases, though, are not approximations to discover - they
are canonical, exact, centuries-old patterns. The **bird base** is the one every
paper crane starts from, and this module constructs it straight from its
well-documented crease lines rather than packing circles at it (that is both the
lazier route and the correct one: the geometry is known to the last 22.5 degrees).

The bird base on a unit square (traditional crease pattern, matching the Origami
Simulator ``birdBase`` reference, with the petal points placed at the exact
22.5-degree reference ``q = (sqrt(2) - 1) / 2`` instead of a rounded pixel value):

    * 4 corners, 4 edge midpoints, 4 petal points, 1 centre -> 13 vertices.
    * both diagonals and the horizontal+vertical "plus" all meet at the centre,
      so the centre is a single 8-crease vertex (all sectors 45 degrees).
    * every edge is split where another crease crosses it (diagonals and the plus
      at the centre; the border at the edge midpoints), so the result is a valid
      planar FOLD graph with 28 edges and 16 triangular faces.

Mountain/valley assignment (mountain = the eight corner->petal creases and the
four petal->centre "plus" halves; valley = the four petal->midpoint creases and
three of the four diagonal halves; **one** diagonal half is a mountain). That
lone flip is what makes the centre satisfy Maekawa: without it the centre would
be a symmetric 4M/4V vertex, which cannot fold flat. With it the centre is 5M/3V
(|M - V| = 2) and every one of the five interior vertices passes both Kawasaki
and Maekawa - see ``tests/test_classic.py``.

    from foldforge.design.classic import bird_base
    pattern = bird_base()            # a CreasePattern, ready for write_fold / to_svg

Honest scope: this is the *base*, not a finished crane. The paper crane needs a
few more shaping folds beyond the bird base (narrowing the wings, and two reverse
folds for the head and tail); those are not part of the base's crease pattern.
"""

from __future__ import annotations

import math

import numpy as np

from foldforge.geometry.crease_graph import CreasePattern

# Petal-point inset from the centre of an edge: the exact 22.5-degree reference
# point. tan(22.5 deg) = sqrt(2) - 1, and the petal crease drops from the corner
# at 22.5 degrees to the vertical/horizontal midline, so q = 0.5 * tan(22.5).
_Q = 0.5 * math.tan(math.radians(22.5))          # = (sqrt(2) - 1) / 2 ~ 0.2071


def bird_base() -> CreasePattern:
    """The traditional bird base crease pattern on the unit square.

    Returns a :class:`CreasePattern` whose 5 interior vertices all pass Kawasaki
    and Maekawa (a genuine flat-foldable base, not an approximation).
    """
    q = _Q
    # --- vertices (index order is load-bearing: faces/edges reference it) ------
    A, B, C, D = 0, 1, 2, 3                       # corners, CCW from origin
    mAB, mBC, mCD, mDA = 4, 5, 6, 7               # edge midpoints
    pAB, pBC, pCD, pDA = 8, 9, 10, 11             # petal points (inset from mids)
    O = 12                                        # centre
    verts = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],   # A B C D
        [0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5],   # midpoints
        [0.5, q], [1.0 - q, 0.5], [0.5, 1.0 - q], [q, 0.5],  # petal points
        [0.5, 0.5],                                        # centre
    ], dtype=float)

    # Each square side as (corner1, corner2, midpoint, petal), CCW.
    sides = [
        (A, B, mAB, pAB),
        (B, C, mBC, pBC),
        (C, D, mCD, pCD),
        (D, A, mDA, pDA),
    ]

    edges: list[tuple[int, int]] = []
    assign: list[str] = []

    def add(u, v, kind):
        edges.append((u, v))
        assign.append(kind)

    # Border, split at each edge midpoint (mountain/valley never applies: "B").
    for c1, c2, m, _p in sides:
        add(c1, m, "B")
        add(m, c2, "B")

    # Petal creases: corner -> petal (mountain) and petal -> midpoint (valley).
    for c1, c2, m, p in sides:
        add(c1, p, "M")
        add(c2, p, "M")
        add(p, m, "V")

    # The "plus": petal -> centre halves of the two book folds (mountain).
    for *_cm, p in sides:
        add(p, O, "M")

    # Diagonals, split at the centre. Three halves are valleys; the A->O half is
    # flipped to a mountain so the centre is 5M/3V and satisfies Maekawa.
    add(A, O, "M")
    add(O, C, "V")
    add(B, O, "V")
    add(O, D, "V")

    # --- faces: each side's quadrant triangle A-B-O split by its m and p -------
    faces: list[list[int]] = []
    for c1, c2, m, p in sides:
        faces.append([c1, m, p])
        faces.append([m, c2, p])
        faces.append([c2, O, p])
        faces.append([O, c1, p])

    return CreasePattern(
        vertices=verts,
        edges=np.array(edges, dtype=int),
        assignment=assign,
        faces=faces,
        metadata={"name": "FoldForge bird base"},
    )
