"""The CreasePattern: how FoldForge represents a flat sheet with fold lines.

A crease pattern is a planar graph drawn on a flat piece of paper:

    * vertices  -> points where creases meet (or where a crease hits the border)
    * edges     -> the crease lines (and the paper's border)
    * faces     -> the flat polygons of paper between the creases

Every edge carries an *assignment* telling us what kind of line it is:

    "M"  mountain fold  (folds away from you, like a tent ridge)
    "V"  valley fold    (folds toward you, like a ditch)
    "B"  border         (the boundary of the paper, not a fold)
    "F"  flat / unfolded (a line that stays flat)
    "U"  unassigned     (we don't know yet)

This module only deals with the *flat* pattern. Folding it into 3D is M1.
The one thing we compute here that the rest of M0 leans on is the list of
**sector angles** around a vertex: the wedge angles of paper between
consecutive creases. Both foldability theorems are statements about those
angles, so we get them right once, here, and reuse them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Edge assignments that are actual folds (as opposed to border / flat / unknown).
FOLD_ASSIGNMENTS = ("M", "V")


@dataclass
class CreasePattern:
    """A flat crease pattern: vertices, creased edges, and (optional) faces.

    Attributes:
        vertices:   (V, 2) float array of flat xy coordinates, one row per vertex.
        edges:      (E, 2) int array; each row is the two vertex indices of an edge.
        assignment: length-E list of "M"/"V"/"B"/"F"/"U" strings.
        faces:      list of vertex-index lists, one per face (may be empty).
        fold_angle: optional length-E array of target fold angles in degrees
                    (sign convention: +valley / -mountain, matching the FOLD spec).
        metadata:   free-form dict carried through from the FOLD file (name, etc.).

    The arrays are plain numpy so the geometry stays easy to read and, later,
    easy to swap for autodiff tensors in the differentiable simulator (M2).
    """

    vertices: np.ndarray
    edges: np.ndarray
    assignment: list[str]
    faces: list[list[int]] = field(default_factory=list)
    fold_angle: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float)
        self.edges = np.asarray(self.edges, dtype=int)
        if self.vertices.ndim != 2 or self.vertices.shape[1] < 2:
            raise ValueError("vertices must be an (V, 2+) array of coordinates")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges must be an (E, 2) array of vertex indices")
        if len(self.assignment) != len(self.edges):
            raise ValueError(
                f"assignment has {len(self.assignment)} entries "
                f"but there are {len(self.edges)} edges"
            )
        if self.edges.size and self.edges.max() >= len(self.vertices):
            raise ValueError("an edge references a vertex index that does not exist")

    # --- basic counts -------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    # --- adjacency ----------------------------------------------------------

    def incident_edges(self, v: int) -> list[int]:
        """Return the indices of every edge that touches vertex ``v``."""
        # ponytail: linear scan, so a full report is O(V*E). Fine for the
        # hand-sized patterns of M0; build a vertex->edges adjacency list if a
        # later milestone runs this on big tessellations in a hot loop.
        return [e for e, (a, b) in enumerate(self.edges) if a == v or b == v]

    def is_boundary_vertex(self, v: int) -> bool:
        """True if ``v`` sits on the paper's border.

        The two foldability theorems only constrain *interior* vertices, so we
        need to tell them apart. A vertex is on the border if any edge touching
        it is a border ("B") edge.
        """
        return any(self.assignment[e] == "B" for e in self.incident_edges(v))

    def interior_vertices(self) -> list[int]:
        """All vertices that are not on the border (the ones theorems apply to)."""
        return [v for v in range(self.n_vertices) if not self.is_boundary_vertex(v)]

    # --- the geometric query everything else uses ---------------------------

    def vertex_creases(self, v: int) -> list[tuple[float, str, int]]:
        """Creases around ``v`` as ``(angle_deg, assignment, edge_index)``, sorted CCW.

        For each edge touching ``v`` we measure the direction it leaves the
        vertex as a compass-style angle in [0, 360). Sorting by that angle puts
        the creases in the order you'd meet them sweeping counter-clockwise,
        which is exactly what we need to read off the wedges between them.
        """
        out: list[tuple[float, str, int]] = []
        vx = self.vertices[v]
        for e in self.incident_edges(v):
            a, b = self.edges[e]
            other = b if a == v else a
            d = self.vertices[other] - vx
            angle = math.degrees(math.atan2(d[1], d[0])) % 360.0
            out.append((angle, self.assignment[e], int(e)))
        out.sort(key=lambda t: t[0])
        return out

    def sector_angles(self, v: int) -> list[float]:
        """The wedge angles (degrees) of paper between consecutive creases at ``v``.

        Walking counter-clockwise around the vertex, each gap between one crease
        and the next is a sector. By construction they always sum to 360 degrees.
        Kawasaki's theorem is a statement about the *alternating* sum of these.
        """
        creases = self.vertex_creases(v)
        angles = [a for a, _, _ in creases]
        n = len(angles)
        if n == 0:
            return []
        return [(angles[(i + 1) % n] - angles[i]) % 360.0 for i in range(n)]

    def __repr__(self) -> str:
        name = self.metadata.get("name", "unnamed")
        return (
            f"CreasePattern({name!r}: {self.n_vertices} vertices, "
            f"{self.n_edges} edges, {len(self.faces)} faces)"
        )
