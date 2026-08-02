"""Figurative origami design by circle packing - a small TreeMaker-lite.

TreeMaker (Robert Lang) turns a *stick figure* of a subject into a foldable
uniaxial base: every stick (edge) of the tree becomes a flap of matching length.
The engine has three moving parts, and this module implements a clean, honest
subset of each:

1.  **Metric tree**  -- nodes and edge lengths. A leaf is a flap tip; the single
    internal node is the body. We ship *star* (single-hub) trees, which is the
    subset the pure circle method handles exactly.

2.  **Circle packing**  -- one disc per leaf in the unit square. The disc radius
    is ``scale * flap_length``; we push ``scale`` as large as possible with
    scipy's SLSQP subject to two families of constraints:

        - non-overlap:  ||c_i - c_j|| >= scale * pathlen(leaf_i, leaf_j)
        - stay on paper: each disc fits inside the unit square.

    At the optimum the binding pairs are *tangent* - and a tangent packing is
    exactly what makes the molecule flap lengths come out right (below).
    Circles only: no rivers, gussets or sub-trees.

3.  **Rabbit-ear molecules**  -- Delaunay-triangulate the packed centres and fill
    each triangle with the universal *rabbit-ear* molecule: the incentre, the
    three angle bisectors (corner -> incentre) and the three perpendiculars
    (incentre -> the foot on each edge). Kawasaki holds *exactly* at every
    incentre (the six wedges are ``90 - alpha/2`` etc.; both alternating sums are
    180), and for a tangent packing the tangent length from a corner equals its
    disc radius = ``scale * flap_length``, so the folded flap matches the tree.

Honest scope: the molecule assembly is flat-foldable at the incentres always,
and end-to-end (every interior vertex passes) for a **single-triangle** tree -
the classic 3-flap "rabbit ear" base. Trees with more leaves pack and export
fine and every incentre still passes, but the *shared* triangle edges introduce
degree-4 tangency vertices that a proper river/gusset would resolve; some of
those can fail the necessary checks. See ``design_base`` and the tests.

    from foldforge.design.treemaker import get_tree, design_base
    packing, pattern = design_base(get_tree("three-flap"))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

from foldforge.geometry.crease_graph import CreasePattern


# --- metric tree ------------------------------------------------------------

@dataclass
class MetricTree:
    """A metric tree: integer nodes 0..n-1 and weighted edges.

    ``edges`` are ``(u, v, length)`` triples. ``root`` is the hub node used to
    measure a leaf's flap length (path from the leaf to the root). For the star
    trees we ship, the root is the single internal node and a leaf's flap length
    is just its own edge.
    """

    edges: list[tuple[int, int, float]]
    root: int = 0
    name: str = "tree"

    def __post_init__(self) -> None:
        self._adj: dict[int, list[tuple[int, float]]] = {}
        for u, v, w in self.edges:
            self._adj.setdefault(u, []).append((v, float(w)))
            self._adj.setdefault(v, []).append((u, float(w)))

    @property
    def nodes(self) -> list[int]:
        return sorted(self._adj)

    def leaves(self) -> list[int]:
        """Nodes of degree 1 - the flap tips."""
        return [n for n in self.nodes if len(self._adj[n]) == 1]

    def path_length(self, a: int, b: int) -> float:
        """Total edge length along the unique tree path from ``a`` to ``b``."""
        # DFS from a; tree so the first time we reach b is the only path.
        stack = [(a, -1, 0.0)]
        while stack:
            node, parent, dist = stack.pop()
            if node == b:
                return dist
            for nxt, w in self._adj[node]:
                if nxt != parent:
                    stack.append((nxt, node, dist + w))
        raise ValueError(f"no path between {a} and {b}")

    def flap_length(self, leaf: int) -> float:
        """Target flap length = path from the leaf to the root hub."""
        return self.path_length(leaf, self.root)


def star_tree(lengths: list[float], name: str = "star") -> MetricTree:
    """A single-hub tree: node 0 is the body, nodes 1..k the flaps."""
    edges = [(0, i + 1, L) for i, L in enumerate(lengths)]
    return MetricTree(edges=edges, root=0, name=name)


# Built-in trees. Star topology (one hub) -> the circle method is exact.
BUILTIN_TREES = {
    # Three equal flaps -> a single Delaunay triangle -> the clean rabbit-ear
    # base that folds flat end to end. This is the flagship working example.
    "three-flap": star_tree([1.0, 1.0, 1.0], "three-flap"),
    # A four-limb figure (bird-base-like silhouette): two long flaps, two short.
    "four-flap": star_tree([1.0, 1.0, 0.6, 0.6], "four-flap"),
    # A five-limb figure (head, two arms, two legs) with a longer "head".
    "five-flap": star_tree([1.2, 0.9, 0.9, 0.8, 0.8], "five-flap"),
}


def get_tree(name: str) -> MetricTree:
    if name not in BUILTIN_TREES:
        raise KeyError(
            f"unknown tree {name!r}; choose from {', '.join(BUILTIN_TREES)}"
        )
    return BUILTIN_TREES[name]


# --- circle packing ---------------------------------------------------------

@dataclass
class Packing:
    """Result of packing a tree's leaf discs into the unit square."""

    tree: MetricTree
    leaves: list[int]            # node id for each disc, in row order
    centers: np.ndarray          # (N, 2) disc centres in [0, 1]^2
    radii: np.ndarray            # (N,) disc radii = scale * flap_length
    scale: float                 # the packing scale that was maximised

    @property
    def flap_targets(self) -> np.ndarray:
        """Scaled target flap length for each disc (== radius for a star tree)."""
        return np.array([self.scale * self.tree.flap_length(l) for l in self.leaves])


def pack_tree(tree: MetricTree) -> Packing:
    """Pack one disc per leaf into the unit square, maximising the scale.

    Deterministic (fixed initial layout, no randomness). Uses scipy SLSQP with
    the non-overlap + on-paper constraints described in the module docstring.
    """
    leaves = tree.leaves()
    n = len(leaves)
    if n < 3:
        raise ValueError("need at least 3 leaves to triangulate a base")

    rho = np.array([tree.flap_length(l) for l in leaves])          # flap lengths
    d = np.array([[tree.path_length(a, b) for b in leaves] for a in leaves])

    # Deterministic start: leaves evenly on a circle, a modest scale.
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0 = np.column_stack([0.5 + 0.28 * np.cos(ang), 0.5 + 0.28 * np.sin(ang)])
    z0 = np.concatenate([x0.ravel(), [0.15]])

    def unpack(z):
        return z[:-1].reshape(n, 2), z[-1]

    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": (lambda z, i=i, j=j: (
                    (c := unpack(z)[0])[i] - c[j]).dot(c[i] - c[j])
                    - (z[-1] * d[i, j]) ** 2),
            })
    for i in range(n):
        # disc must sit inside [0,1]^2: centre at least (scale*rho_i) from walls.
        for axis in (0, 1):
            cons.append({"type": "ineq",
                         "fun": lambda z, i=i, a=axis: unpack(z)[0][i, a] - z[-1] * rho[i]})
            cons.append({"type": "ineq",
                         "fun": lambda z, i=i, a=axis: 1.0 - unpack(z)[0][i, a] - z[-1] * rho[i]})

    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-4, 2.0)]
    res = minimize(lambda z: -z[-1], z0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 400, "ftol": 1e-9})

    centers, scale = unpack(res.x)
    radii = scale * rho
    return Packing(tree=tree, leaves=leaves, centers=centers, radii=radii,
                   scale=float(scale))


# --- rabbit-ear molecules ---------------------------------------------------

def _incenter(p0, p1, p2):
    a = np.linalg.norm(p1 - p2)       # side opposite p0
    b = np.linalg.norm(p2 - p0)
    c = np.linalg.norm(p0 - p1)
    return (a * p0 + b * p1 + c * p2) / (a + b + c)


def _foot(point, e0, e1):
    """Perpendicular foot of ``point`` on the segment (e0, e1)."""
    d = e1 - e0
    t = np.dot(point - e0, d) / np.dot(d, d)
    return e0 + t * d


class _VertexBag:
    """Collects unique vertices with a rounding tolerance (tangent feet coincide)."""

    def __init__(self, tol=1e-6):
        self.coords: list[np.ndarray] = []
        self._index: dict[tuple, int] = {}
        self.tol = tol

    def add(self, xy) -> int:
        key = (round(float(xy[0]) / self.tol), round(float(xy[1]) / self.tol))
        if key in self._index:
            return self._index[key]
        idx = len(self.coords)
        self._index[key] = idx
        self.coords.append(np.asarray(xy, dtype=float))
        return idx


def crease_pattern(packing: Packing) -> CreasePattern:
    """Assemble a rabbit-ear molecule per Delaunay triangle into one pattern.

    Vertices: the packed leaf centres, plus each triangle's incentre and the
    perpendicular feet on its edges. Creases: three bisectors (corner->incentre)
    and three perpendiculars (incentre->foot) per triangle, plus the triangle
    edges (split at the feet) as hinges/border. Kawasaki holds at every incentre.
    """
    pts = packing.centers
    tri = Delaunay(pts)
    triangles = tri.simplices

    # Which undirected triangle edges are shared (interior hinge) vs boundary.
    edge_count: dict[tuple[int, int], int] = {}
    for t in triangles:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_count[tuple(sorted((int(a), int(b))))] = \
                edge_count.get(tuple(sorted((int(a), int(b)))), 0) + 1

    bag = _VertexBag()
    corner_idx = [bag.add(p) for p in pts]     # keep leaf centres first
    edges: list[tuple[int, int]] = []
    assign: list[str] = []
    edge_seen: dict[tuple[int, int], str] = {}

    def add_edge(u, v, kind):
        key = tuple(sorted((u, v)))
        if key in edge_seen:                    # shared edge: keep first, prefer hinge
            return
        edge_seen[key] = kind
        edges.append((u, v))
        assign.append(kind)

    for t in triangles:
        p = [pts[t[0]], pts[t[1]], pts[t[2]]]
        ic = _incenter(*p)
        ic_i = bag.add(ic)
        # three edges of this triangle, as (local corner a, local corner b)
        local = ((0, 1), (1, 2), (2, 0))
        for k, (la, lb) in enumerate(local):
            ga, gb = int(t[la]), int(t[lb])
            foot = _foot(ic, p[la], p[lb])
            fi = bag.add(foot)
            # triangle edge, split at its foot -> two collinear halves
            boundary = edge_count[tuple(sorted((ga, gb)))] == 1
            edge_kind = "B" if boundary else "V"
            add_edge(corner_idx[ga], fi, edge_kind)
            add_edge(fi, corner_idx[gb], edge_kind)
            # perpendicular incentre->foot. One foot per triangle is V, rest M,
            # so each incentre gets 4 V + 2 M -> Maekawa |M-V| == 2.
            add_edge(ic_i, fi, "V" if k == 0 else "M")
        # three bisectors incentre->corner (valley ridges)
        for la in (0, 1, 2):
            add_edge(ic_i, corner_idx[int(t[la])], "V")

    verts = np.array(bag.coords)
    pattern = CreasePattern(
        vertices=verts,
        edges=np.array(edges, dtype=int),
        assignment=assign,
        metadata={"name": f"treemaker:{packing.tree.name}"},
    )
    return pattern


# --- one-call design + reporting -------------------------------------------

@dataclass
class DesignReport:
    """End-to-end summary a caller (CLI, test, proof script) can print."""

    packing: Packing
    pattern: CreasePattern
    flap_errors: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def max_flap_error(self) -> float:
        return float(self.flap_errors.max()) if len(self.flap_errors) else 0.0


def flap_length_errors(packing: Packing, pattern: CreasePattern) -> np.ndarray:
    """Relative error between each folded flap length and its tree target.

    The folded flap length of a leaf is the tangent length from its corner - the
    distance from the centre to the foot on an incident triangle edge - which we
    read straight back off the assembled geometry.
    """
    pts = packing.centers
    tri = Delaunay(pts)
    measured = np.full(len(packing.leaves), np.nan)
    for t in tri.simplices:
        p = [pts[t[0]], pts[t[1]], pts[t[2]]]
        ic = _incenter(*p)
        for la, (a, b) in zip((0, 1, 2), ((0, 1), (1, 2), (2, 0))):
            foot = _foot(ic, p[a], p[b])
            tan = np.linalg.norm(pts[t[a]] - foot)      # tangent length from corner a
            gi = int(t[a])
            measured[gi] = tan if np.isnan(measured[gi]) else min(measured[gi], tan)
    target = packing.flap_targets
    return np.abs(measured - target) / np.maximum(target, 1e-9)


def design_base(tree: MetricTree) -> tuple[Packing, CreasePattern]:
    """Pack the tree and build its crease pattern in one call."""
    packing = pack_tree(tree)
    return packing, crease_pattern(packing)


def folded_schematic(packing: Packing) -> tuple[np.ndarray, np.ndarray]:
    """A 2-D schematic of the folded uniaxial base: flaps hanging off one axis.

    Returns ``(spine, tips)``: ``spine`` are the projected hinge points laid out
    along a horizontal axis, ``tips`` the flap tips at their (folded) length. A
    faithful-enough picture of the base for the proof image.
    """
    lengths = packing.radii
    n = len(lengths)
    xs = np.linspace(0.0, float(n - 1), n)
    spine = np.column_stack([xs, np.zeros(n)])
    tips = np.column_stack([xs, lengths])
    return spine, tips
