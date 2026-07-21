"""From a flat crease pattern to a foldable 3D mesh: bars and hinges.

The solver needs three things, all built here:

    bars     every triangle edge, with its flat rest length. Stiff springs on
             these keep the faces rigid (paper doesn't stretch).
    hinges   every interior edge shared by two triangles, with the fold angle it
             wants. Real creases (M/V) want to fold; "facet" creases (the
             diagonals we add when triangulating a quad) want to stay flat (0).
    masses   one per vertex, for the dynamics.

We triangulate every face by a simple fan from its first vertex. Origami faces
here are convex (parallelograms, triangles), so a fan is valid and the added
diagonals become facet hinges that hold the original face flat.

This module also holds the dihedral-angle geometry the hinge forces are built
on. The gradient ``dihedral_grad`` is verified against finite differences in
the tests — getting it right is what makes the fold behave.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from foldforge.geometry.crease_graph import CreasePattern

# How far a real crease wants to fold at fold_fraction = 1.0, in radians.
# Not the flat-folded limit (pi): we stop short to stay clear of the
# self-touching, singular fully-flat state. ~143 degrees is a strong, clean fold.
FULL_FOLD_ANGLE = 2.5


# --- dihedral-angle geometry (the heart of the crease force) ----------------

def _hinge_frame(p1, p2, p3, p4):
    """Shared math for the dihedral angle and its gradient at one hinge.

    Nodes: ``p1``-``p2`` is the hinge edge; ``p3`` and ``p4`` are the opposite
    ("wing") corners of the two triangles sharing it. Returns the pieces both
    the angle and the gradient need.
    """
    edge = p2 - p1
    edge_len = np.linalg.norm(edge)
    edge_hat = edge / edge_len
    cross1 = np.cross(p2 - p1, p3 - p1)     # 2 * area1 * normal1
    cross2 = np.cross(p4 - p1, p2 - p1)     # 2 * area2 * normal2
    norm1 = np.linalg.norm(cross1)
    norm2 = np.linalg.norm(cross2)
    n1 = cross1 / norm1
    n2 = cross2 / norm2
    return edge, edge_len, edge_hat, n1, n2, norm1, norm2


def dihedral_angle(p1, p2, p3, p4) -> float:
    """Signed dihedral (fold) angle across the hinge edge ``p1``-``p2``.

    Zero when the two triangles are coplanar; positive/negative for the two fold
    directions (so mountains and valleys come out with opposite sign).
    """
    _, _, edge_hat, n1, n2, _, _ = _hinge_frame(p1, p2, p3, p4)
    sin_t = np.dot(np.cross(n1, n2), edge_hat)
    cos_t = np.dot(n1, n2)
    return float(np.arctan2(sin_t, cos_t))


def dihedral_grad(p1, p2, p3, p4):
    """Gradient of the dihedral angle w.r.t. each of the four nodes.

    Returns ``(g1, g2, g3, g4)``, each a length-3 vector. ``g3``/``g4`` (the
    wing corners) point along their triangle's normal, scaled by 1 / (height of
    that corner above the edge); the edge nodes ``g1``/``g2`` get the leftover,
    split by where each wing's perpendicular foot lands along the edge. This is
    the standard discrete bending gradient (Bridson et al.); the sign is set to
    match the ``atan2`` convention above and is checked against finite
    differences in the tests.
    """
    edge, edge_len, _, n1, n2, norm1, norm2 = _hinge_frame(p1, p2, p3, p4)
    h1 = norm1 / edge_len                    # height of p3 above the edge
    h2 = norm2 / edge_len                    # height of p4 above the edge
    # Fraction along the edge (measured from p1) of each wing's perpendicular foot.
    f3 = np.dot(p3 - p1, edge) / edge_len ** 2
    f4 = np.dot(p4 - p1, edge) / edge_len ** 2
    g3 = n1 / h1
    g4 = n2 / h2
    g1 = -(1 - f3) * g3 - (1 - f4) * g4
    g2 = -f3 * g3 - f4 * g4
    return -g1, -g2, -g3, -g4


# --- the foldable mesh ------------------------------------------------------

@dataclass
class Hinge:
    """An interior edge shared by two triangles, that can fold.

    Attributes:
        edge:        the two vertex indices of the hinge line (p1, p2).
        wings:       the opposite corners of the two triangles (p3, p4).
        target:      fold angle (radians) wanted at full fold. 0 for facet
                     hinges (hold a face flat); +/-FULL_FOLD_ANGLE for creases.
        is_crease:   True for real M/V creases, False for triangulation facets.
    """

    edge: tuple[int, int]
    wings: tuple[int, int]
    target: float
    is_crease: bool


@dataclass
class FoldMesh:
    """A triangulated, foldable version of a crease pattern."""

    vertices: np.ndarray            # (V, 3) flat starting positions (z = 0)
    triangles: np.ndarray           # (T, 3) vertex indices
    bars: np.ndarray                # (B, 2) vertex-index pairs
    rest_lengths: np.ndarray        # (B,) flat lengths of the bars
    hinges: list[Hinge]
    masses: np.ndarray              # (V,)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_pattern(cls, pattern: CreasePattern,
                     full_fold_angle: float = FULL_FOLD_ANGLE) -> "FoldMesh":
        """Build a foldable mesh from a flat :class:`CreasePattern`.

        ``pattern.faces`` must be filled in (the generators do this). Each face
        is fan-triangulated; original M/V edges become crease hinges and the
        diagonals added by triangulation become facet hinges.
        """
        if not pattern.faces:
            raise ValueError(
                "FoldMesh needs faces; this pattern has none. "
                "Use a pattern whose generator defines faces_vertices."
            )

        verts = np.zeros((pattern.n_vertices, 3))
        verts[:, :2] = pattern.vertices[:, :2]

        # Look up each original edge's assignment by its vertex pair.
        edge_kind: dict[frozenset[int], str] = {
            frozenset((int(a), int(b))): k
            for (a, b), k in zip(pattern.edges, pattern.assignment)
        }

        triangles: list[tuple[int, int, int]] = []
        # edge (as frozenset) -> list of (triangle_index, apex_vertex)
        edge_tris: dict[frozenset[int], list[tuple[int, int]]] = {}

        def add_edge(a: int, b: int, tri_idx: int, apex: int) -> None:
            edge_tris.setdefault(frozenset((a, b)), []).append((tri_idx, apex))

        for face in pattern.faces:
            v0 = face[0]
            for k in range(1, len(face) - 1):
                a, b = face[k], face[k + 1]
                t = len(triangles)
                triangles.append((v0, a, b))
                add_edge(v0, a, t, b)
                add_edge(a, b, t, v0)
                add_edge(b, v0, t, a)

        # Bars: one per unique triangle edge.
        bar_set: set[frozenset[int]] = set(edge_tris.keys())
        bars = np.array([tuple(e) for e in (sorted(map(sorted, bar_set)))])
        rest_lengths = np.linalg.norm(
            verts[bars[:, 1]] - verts[bars[:, 0]], axis=1
        )

        # Hinges: every interior edge shared by exactly two triangles.
        hinges: list[Hinge] = []
        for edge, tris in edge_tris.items():
            if len(tris) != 2:
                continue  # border edge (one triangle) -> not a hinge
            (i, j) = tuple(edge)
            apex_a, apex_b = tris[0][1], tris[1][1]
            kind = edge_kind.get(edge)  # M / V / B / None(=interior diagonal)
            if kind in ("M", "V"):
                # Mountains and valleys fold opposite ways.
                sign = -1.0 if kind == "M" else 1.0
                hinges.append(
                    Hinge((i, j), (apex_a, apex_b), sign * full_fold_angle, True)
                )
            elif kind in (None, "F"):
                # Facet diagonal (or explicit flat edge): keep the face planar.
                hinges.append(Hinge((i, j), (apex_a, apex_b), 0.0, False))
            # "B" border edges shared by two triangles (rare) get no hinge.

        masses = np.ones(pattern.n_vertices)

        return cls(
            vertices=verts,
            triangles=np.array(triangles),
            bars=bars,
            rest_lengths=rest_lengths,
            hinges=hinges,
            masses=masses,
            metadata={**pattern.metadata, "n_creases":
                      sum(h.is_crease for h in hinges)},
        )

    def max_strain(self, positions: np.ndarray) -> float:
        """Largest relative stretch of any bar at ``positions`` (0 = rigid).

        This is how we measure "faces didn't stretch": a rigid fold keeps every
        bar at its rest length, so the max strain stays near zero.
        """
        lengths = np.linalg.norm(
            positions[self.bars[:, 1]] - positions[self.bars[:, 0]], axis=1
        )
        return float(np.max(np.abs(lengths - self.rest_lengths) / self.rest_lengths))
