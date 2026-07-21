"""Generators for classic crease patterns.

Hand-typing vertex coordinates is error-prone, so we build the example patterns
from their geometry. Each generator returns a :class:`CreasePattern`. The first
two (Miura, single vertex) are flat-foldable and pass both theorems; the
waterbomb *base* is included precisely because it does **not** fold flat, which
makes it a good test that our validators can also say "no".
"""

from __future__ import annotations

import numpy as np

from foldforge.geometry.crease_graph import CreasePattern


def miura(rows: int = 4, cols: int = 4, a: float = 1.0, b: float = 0.8,
          offset: float = 0.35) -> CreasePattern:
    """A Miura-ori tessellation: the famous rigid, flat-foldable fold.

    Geometry: a grid of vertices where the horizontal lines stay straight but
    the vertical lines zig-zag left/right by ``offset`` every row. That single
    zig-zag is what makes each interior vertex a degree-4 flat-foldable vertex
    (wedge angles theta, 180-theta, 180-theta, theta -> Kawasaki holds).

    Assignment (the classic Miura look): the horizontal "major" folds alternate
    mountain/valley by row, and the zig-zag folds flip the opposite way each
    row, giving the 3-mountain / 1-valley vertices Maekawa wants.

    Args:
        rows, cols: number of grid cells.
        a:          horizontal cell width.
        b:          vertical cell height.
        offset:     horizontal zig-zag shift per row (0 -> degenerate grid).
    """
    nr, nc = rows + 1, cols + 1

    def index(i: int, j: int) -> int:
        return i * nc + j

    coords = np.array(
        [[j * a + (i % 2) * offset, i * b] for i in range(nr) for j in range(nc)],
        dtype=float,
    )

    edges: list[tuple[int, int]] = []
    assignment: list[str] = []

    # Horizontal edges: straight major folds, alternating M/V by row.
    for i in range(nr):
        major = "M" if i % 2 == 0 else "V"
        for j in range(cols):
            edges.append((index(i, j), index(i, j + 1)))
            assignment.append("B" if i in (0, rows) else major)

    # Vertical edges: the zig-zag folds, flipping the opposite way each row.
    for i in range(rows):
        zig = "V" if i % 2 == 0 else "M"
        for j in range(nc):
            edges.append((index(i, j), index(i + 1, j)))
            assignment.append("B" if j in (0, cols) else zig)

    faces = [
        [index(i, j), index(i, j + 1), index(i + 1, j + 1), index(i + 1, j)]
        for i in range(rows)
        for j in range(cols)
    ]

    return CreasePattern(
        vertices=coords, edges=np.array(edges), assignment=assignment,
        faces=faces, metadata={"name": f"miura_{rows}x{cols}"},
    )


def single_vertex(angles_deg=(60, 120, 120, 60),
                  assignment=("M", "V", "M", "M")) -> CreasePattern:
    """The smallest interesting pattern: one interior vertex with four creases.

    Four creases radiate from a centre point to four border corners, with the
    given wedge ``angles_deg`` between them. The default angles (60,120,120,60)
    satisfy Kawasaki (odd wedges 60+120 = even wedges 120+60 = 180), and the
    default 3M/1V assignment satisfies Maekawa. Great for reasoning by hand.
    """
    if abs(sum(angles_deg) - 360.0) > 1e-9:
        raise ValueError("sector angles must sum to 360 degrees")

    pts = [[0.0, 0.0]]                                # 0 = centre
    bearing = 0.0
    for ang in angles_deg:
        rad = np.radians(bearing)
        pts.append([np.cos(rad), np.sin(rad)])
        bearing += ang

    n = len(angles_deg)
    edges = [(0, k + 1) for k in range(n)]           # centre -> each ray (creases)
    edges += [(1 + k, 1 + (k + 1) % n) for k in range(n)]  # border ring of rays
    assignment = list(assignment) + ["B"] * n
    faces = [[0, 1 + k, 1 + (k + 1) % n] for k in range(n)]  # one wedge per sector

    return CreasePattern(
        vertices=np.array(pts, dtype=float), edges=np.array(edges),
        assignment=assignment, faces=faces, metadata={"name": "single_vertex"},
    )


def waterbomb_base() -> CreasePattern:
    """The waterbomb base: a square, both diagonals, and both mid-lines.

    This is a *3D* base, not a flat fold, so it deliberately FAILS Maekawa at
    its degree-8 centre (its mountains and valleys don't balance to |M-V|=2).
    We keep it as the honest negative example: a real, recognisable pattern
    that our validators correctly flag as not flat-foldable.
    """
    # Corners, edge midpoints, centre of a unit square.
    coords = np.array([
        [-1, -1], [1, -1], [1, 1], [-1, 1],   # 0..3 corners
        [0, -1], [1, 0], [0, 1], [-1, 0],      # 4..7 edge midpoints
        [0, 0],                                # 8 centre
    ], dtype=float)

    edges = [
        # border (square through the midpoints)
        (0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0),
        # diagonals through the centre -> mountains
        (0, 8), (8, 2), (1, 8), (8, 3),
        # mid-lines through the centre -> valleys
        (4, 8), (8, 6), (7, 8), (8, 5),
    ]
    assignment = (
        ["B"] * 8
        + ["M", "M", "M", "M"]   # diagonals
        + ["V", "V", "V", "V"]   # mid-lines
    )

    # Eight triangular panels, each = centre + two consecutive border points
    # (going around: corner, midpoint, corner, ...).
    ring = [0, 4, 1, 5, 2, 6, 3, 7]
    faces = [[8, ring[i], ring[(i + 1) % 8]] for i in range(8)]

    return CreasePattern(
        vertices=coords, edges=np.array(edges), assignment=assignment,
        faces=faces, metadata={"name": "waterbomb_base"},
    )


#: Name -> zero-argument generator, used by the CLI and the example writer.
GENERATORS = {
    "miura": miura,
    "single_vertex": single_vertex,
    "waterbomb_base": waterbomb_base,
}
