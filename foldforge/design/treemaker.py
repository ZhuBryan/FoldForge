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

3.  **Incircle universal molecule**  -- fill the packed base with a single
    *tangential-polygon* molecule instead of gluing independent Delaunay
    triangles. Take the convex hull of the packed centres; because every hull
    edge joins a tangent disc pair its length is ``r_i + r_j``, so by Pitot's
    theorem a triangle or a quadrilateral hull is always a *tangential* polygon:
    it has one inscribed circle that touches all sides. The molecule is then the
    incentre, one angle bisector per corner (corner -> incentre, a ridge) and one
    perpendicular per side (incentre -> the incircle touch point, a hinge). The
    touch point on side ``(i, j)`` sits at distance ``r_i`` from corner ``i`` -
    exactly the disc-disc tangent point - so the folded flap length equals the
    tree edge with ~0 error, and Kawasaki holds exactly at the incentre (opposite
    wedges are ``90 - alpha_i/2`` complementary; both alternating sums are 180).

    A tangential polygon has the *whole* base in one molecule with a single
    interior vertex (the incentre) - no shared-edge feet to reconcile - so it
    folds flat end to end. This is the well-defined case the construction can
    guarantee: it covers the 3-flap triangle **and every 4-flap quad**.

Honest scope (see ``crease_pattern`` and the tests):

    * ``three-flap`` (triangle), ``four-flap`` (tangential quad) and
      ``five-flap`` (a pentagon) all fold flat end to end. They share ONE
      construction - the universal (gusset) molecule below - which fills any
      convex tight tangent-disc polygon: every interior vertex passes Kawasaki
      *and* Maekawa and every flap length is exact (< ~1e-8 relative error). The
      triangle rabbit-ear and the single-incircle quad are just its base cases;
      pentagons and larger add the gusset creases that reconcile the parts.
    * **Rivers** (trees with an internal spine edge - a body with limbs). The
      packing is exact for every river: a disc radius is the leaf's *terminal*
      edge (not its full path to the root), so two leaves sharing a hub stay
      tangent and the spine shows up only as the extra path-distance that spaces
      cross-river leaves apart - a clean open river band, no overlap. The
      molecule handles the **single-spine** river (``river-four``: two flaps off
      each of two sub-hubs) end to end: it insets to two hub nodes joined by one
      spine crease, and every interior vertex passes Kawasaki + Maekawa with exact
      flap lengths (root-side ``r_i``, far-side ``r_i + scale*spine``). It is built
      behind an honest self-check, so *asymmetric*, *many-leaf* and *multi-junction*
      rivers - which need the extra gusset nodes this two-node construction does
      not emit - fail the check and fall back to the partial rabbit-ear base
      rather than claim a fold they do not achieve.

    from foldforge.design.treemaker import get_tree, design_base
    packing, pattern = design_base(get_tree("four-flap"))
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay

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

    def terminal_length(self, leaf: int) -> float:
        """Length of the leaf's own (single) incident edge - its *terminal* edge.

        For a star tree this equals ``flap_length`` (the root *is* the leaf's
        parent). For a river tree a spine-side leaf's terminal edge is shorter
        than its flap length by the spine it must still cross to reach the root;
        that difference is exactly the river band the packing must leave open.
        """
        return self._adj[leaf][0][1]

    def parent_hub(self, leaf: int) -> int:
        """The internal node the leaf hangs off (the other end of its edge)."""
        return self._adj[leaf][0][0]


def star_tree(lengths: list[float], name: str = "star") -> MetricTree:
    """A single-hub tree: node 0 is the body, nodes 1..k the flaps."""
    edges = [(0, i + 1, L) for i, L in enumerate(lengths)]
    return MetricTree(edges=edges, root=0, name=name)


def river_tree(spine: float, left: list[float], right: list[float],
               name: str = "river") -> MetricTree:
    """A two-hub tree with an internal *spine* edge (the thing rivers model).

    Node 0 and node 1 are hubs joined by an edge of length ``spine``; ``left``
    hangs off hub 0 and ``right`` off hub 1. Leaves on opposite hubs are
    separated by the spine, which the packing honours by leaving an open river
    band between them. A *single-spine, symmetric* river (e.g. two equal flaps
    off each hub) folds flat end to end; asymmetric or many-leaf variants pack
    cleanly but fall back to the honest partial base (see the module docstring).
    """
    edges = [(0, 1, spine)]
    nid = 2
    for L in left:
        edges.append((0, nid, L)); nid += 1
    for L in right:
        edges.append((1, nid, L)); nid += 1
    return MetricTree(edges=edges, root=0, name=name)


BUILTIN_TREES = {
    # Three equal flaps -> a triangle -> the clean rabbit-ear base. Folds flat
    # end to end (single incentre passes both theorems, flap error ~0).
    "three-flap": star_tree([1.0, 1.0, 1.0], "three-flap"),
    # A four-limb figure -> a tangential quad -> one incircle molecule. Also
    # folds flat end to end: this is the flagship *multi*-flap working example.
    "four-flap": star_tree([1.0, 1.0, 0.6, 0.6], "four-flap"),
    # A five-limb figure -> a pentagon (generally non-tangential) filled by the
    # universal (gusset) molecule: folds flat end to end (see the module docstring).
    "five-flap": star_tree([1.2, 0.9, 0.9, 0.8, 0.8], "five-flap"),
    # A body with limbs: an internal spine edge with two flaps off each sub-hub.
    # Packs with a clean open river band between the hubs and folds flat end to
    # end - the single-spine river molecule (two hub nodes + one spine crease).
    "river-four": river_tree(0.5, [1.0, 1.0], [1.0, 1.0], "river-four"),
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
    radii: np.ndarray            # (N,) disc radii = scale * terminal edge length
    scale: float                 # the packing scale that was maximised

    @property
    def flap_targets(self) -> np.ndarray:
        """Scaled target flap length for each disc = scale * path-to-root.

        This is the *full* flap length (a spine-side leaf's target includes the
        river it crosses to reach the root), which equals the disc radius only
        for a star tree (where terminal edge == path to root).
        """
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

    # Disc radius = the leaf's own TERMINAL edge (leaf-to-parent), NOT its full
    # path to the root. For a star tree these coincide; for a river tree a
    # spine-side leaf's terminal edge is shorter, so two leaves sharing a hub
    # stay tangent (radii sum == their path distance) instead of overlapping by
    # the spine. The spine shows up purely in the path-distance constraint below,
    # which spaces cross-river leaves an extra river band apart.
    rho = np.array([tree.terminal_length(l) for l in leaves])      # terminal edges
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


def _incircle(poly: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Inscribed-circle centre and radius of a convex polygon (CCW ``poly``).

    Solves, in a least-squares sense, ``n_e . x + rho = c_e`` for every edge
    (``n_e`` the outward unit normal, ``c_e = n_e . a``). For a *tangential*
    polygon the residual is ~0 and the perpendicular foot on each side is that
    side's incircle touch point. Returns ``(centre, rho, max_residual)`` - the
    residual being how far from equidistant the best circle is (0 <=> tangential).
    """
    m = len(poly)
    A = np.empty((m, 3))
    b = np.empty(m)
    for i in range(m):
        p0, p1 = poly[i], poly[(i + 1) % m]
        d = p1 - p0
        d = d / np.linalg.norm(d)
        nrm = np.array([d[1], -d[0]])          # outward normal for a CCW polygon
        A[i] = (nrm[0], nrm[1], 1.0)
        b[i] = nrm.dot(p0)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    centre, rho = sol[:2], float(sol[2])
    resid = float(np.max(np.abs(A @ sol - b)))
    return centre, rho, resid


def _is_tangential(poly: np.ndarray, tol: float = 1e-6) -> bool:
    """True if ``poly`` admits one inscribed circle touching every side."""
    _, _, resid = _incircle(poly)
    return resid < tol


# --- the universal (gusset) molecule ---------------------------------------
#
# Fills ANY convex active polygon whose vertices are packed disc centres (each
# edge (i, j) is tight: ``|P_i - P_j| = r_i + r_j``). One algorithm subsumes the
# triangle rabbit-ear and the tangential-quad incircle as base cases, and adds
# the gusset creases that the per-molecule code was missing for pentagons+.
#
# It is Lang's universal molecule, computed by insetting the polygon: every edge
# moves inward perpendicular at unit speed, so each vertex slides along its
# interior-angle bisector at speed ``1 / sin(theta_i / 2)`` and its reduced
# radius shrinks at rate ``cot(theta_i / 2)`` (keeping every edge tight). Two
# things can happen:
#
#   * the polygon reaches a state where its incircle's tangent lengths equal the
#     reduced radii (a *triangle*, always; or a symmetric tangential polygon) -
#     the base case: one incentre, ``m`` bisector ridges, ``m`` perpendicular
#     hinges to the tangent points (== the old rabbit-ear / incircle molecule);
#   * a non-adjacent vertex pair ``(i, j)`` becomes tight
#     (``|Q_i - Q_j| = rho_i + rho_j``) before that - a **gusset event**: add the
#     gusset crease ``Q_i-Q_j`` and recurse on the two sub-polygons it splits.
#
# The bisector traces are the ridge creases, the perpendiculars from the border
# tangent points inward are the hinge creases, and the split diagonals are the
# gussets. Because every reduced radius is tracked exactly, each border edge is
# cut at distance ``r_i`` from its corner, so the folded flap length equals the
# tree edge with ~0 error. Kawasaki holds at every interior vertex by the
# bisector/perpendicular geometry; the mountain/valley labels are chosen by a
# global search so ``|M - V| = 2`` (Maekawa) holds everywhere at once.


def _poly_geometry(Q: np.ndarray):
    """Per-vertex inward bisector *velocity* and ``cot(theta_i / 2)`` for CCW Q."""
    m = len(Q)
    vel = np.zeros((m, 2))
    cot = np.zeros(m)
    for k in range(m):
        a, b, c = Q[(k - 1) % m], Q[k], Q[(k + 1) % m]
        u = (a - b) / np.linalg.norm(a - b)
        v = (c - b) / np.linalg.norm(c - b)
        th = math.acos(max(-1.0, min(1.0, float(u.dot(v)))))
        bis = u + v
        bis = bis / np.linalg.norm(bis)
        vel[k] = bis / math.sin(th / 2.0)          # unit perpendicular inset speed
        cot[k] = 1.0 / math.tan(th / 2.0)          # reduced-radius shrink rate
    return vel, cot


def _first_gusset_event(Q: np.ndarray, rho: np.ndarray, offs: np.ndarray | None = None):
    """Smallest inset time at which a non-adjacent vertex pair becomes tight.

    Returns ``(t, i, j)`` or ``None``. ``Q_i(t) = Q_i + t * vel_i`` and
    ``rho_i(t) = rho_i - t * cot_i``; the event solves
    ``|Q_i(t) - Q_j(t)| = rho_i(t) + rho_j(t)`` (a quadratic in ``t``), then a
    few Newton steps polish it to machine precision so the split is exact.
    """
    m = len(Q)
    vel, cot = _poly_geometry(Q)
    best_t = None
    best_pair = None
    for i in range(m):
        for j in range(i + 1, m):
            if j == (i + 1) % m or i == (j + 1) % m:
                continue                            # adjacent: that's a tight edge already
            dv = vel[i] - vel[j]
            d0 = Q[i] - Q[j]
            off = 0.0 if offs is None else float(offs[i, j])
            rs = rho[i] + rho[j] + off       # river band widens the tight length
            rc = -(cot[i] + cot[j])
            A = float(dv.dot(dv) - rc * rc)
            B = float(2 * d0.dot(dv) - 2 * rs * rc)
            C = float(d0.dot(d0) - rs * rs)
            roots: list[float] = []
            if abs(A) < 1e-12:
                if abs(B) > 1e-12:
                    roots = [-C / B]
            else:
                disc = B * B - 4 * A * C
                if disc >= 0:
                    sq = math.sqrt(disc)
                    roots = [(-B - sq) / (2 * A), (-B + sq) / (2 * A)]
            for t in roots:
                if t > 1e-9 and (best_t is None or t < best_t):
                    best_t, best_pair = t, (i, j)
    if best_t is None:
        return None
    i, j = best_pair
    t = best_t
    for _ in range(80):                             # Newton polish of the tight pair
        Qn = Q + t * vel
        d = float(np.linalg.norm(Qn[i] - Qn[j]))
        off = 0.0 if offs is None else float(offs[i, j])
        rr = float(rho[i] - t * cot[i] + rho[j] - t * cot[j] + off)
        f = d - rr
        dd = float((Qn[i] - Qn[j]).dot(vel[i] - vel[j]) / d)
        t = t - f / (dd + cot[i] + cot[j])
    return t, i, j


def _incircle_flaps_ok(Q: np.ndarray, rho: np.ndarray, tol: float = 1e-7):
    """True (with the incentre) iff Q's incircle preserves every flap length.

    A triangle always qualifies; a general tangential polygon does *not* - its
    incircle tangent lengths need not equal the reduced radii, which is exactly
    why pentagons must be split by gussets rather than filled with one incircle.
    """
    ic, _, resid = _incircle(Q)
    if resid > tol:
        return False, ic
    for k in range(len(Q)):
        foot = _foot(ic, Q[k], Q[(k + 1) % len(Q)])
        if abs(np.linalg.norm(Q[k] - foot) - rho[k]) > tol:
            return False, ic
    return True, ic


def _river_split(Q: np.ndarray, offs: np.ndarray):
    """Detect a river-contraction (spine) event on the current active polygon.

    A *river edge* is a polygon edge ``k = (k, k+1)`` whose band ``offs`` is
    positive: the two leaf discs on it are held apart by more than ``r_i + r_j``
    (a spine crosses between them), so the edge is cut at ``r_i`` / ``r_j`` with a
    band left open in the middle. A single-spine river has exactly two such edges
    - the two sides of one band. Insetting the polygon at unit speed slides those
    two river lines inward until they meet: the band pinches to the **spine**, a
    segment joining the two sub-hub incentres. Each sub-hub is the incentre of the
    lines bounding its side *including both river lines* (so it lies on the band's
    mid-line and every wedge at it is a bisector pair - Kawasaki holds).

    Returns ``(t, e0, e1, grpA, grpB, C0, C1)`` where ``t`` is the inset time the
    band closes (== the sub-hub inradius, since an edge insets at unit speed), or
    ``None`` when there is no clean single-spine river (0 or >2 river edges, the
    two sub-hubs disagree, or the incentre fit is not tangent). A ``None`` keeps
    the caller on the ordinary gusset path, so stars are untouched.
    """
    m = len(Q)
    river_edges = [k for k in range(m) if offs[k, (k + 1) % m] > 1e-9]
    if len(river_edges) != 2:
        return None                                 # not a single-spine river
    e0, e1 = river_edges

    def arc(a, b):                                  # vertices strictly between edges
        out, k = [], (a + 1) % m
        while True:
            out.append(k)
            if k == b:
                break
            k = (k + 1) % m
        return out

    grpA, grpB = arc(e0, e1), arc(e1, e0)
    centroid = Q.mean(axis=0)
    rl0 = _line_normal(Q[e0], Q[(e0 + 1) % m])
    rl1 = _line_normal(Q[e1], Q[(e1 + 1) % m])

    def hub(grp):
        lines = [rl0, rl1]                          # both river lines constrain each side
        for k in grp:
            nk = (k + 1) % m
            if offs[k, nk] <= 1e-9 and nk in grp:   # that side's own tight border edges
                lines.append(_line_normal(Q[k], Q[nk]))
        return _incenter_of_lines(lines, centroid)

    C0, R0, res0 = hub(grpA)
    C1, R1, res1 = hub(grpB)
    if max(res0, res1) > 1e-6 or abs(R0 - R1) > 1e-6:
        return None                                 # not a clean tangent single spine
    return 0.5 * (R0 + R1), e0, e1, grpA, grpB, C0, C1


def _um_fill(Q, rho, outer, creases, leaf_ids, packing, river_info=None):
    """Recursively emit the universal-molecule creases of one active polygon.

    ``outer[k]`` is the paper-border tangent point that edge ``k``'s hinge must
    reach (``None`` for a gusset edge, whose hinge stops at its own tangent
    point). Appends ``(p0, p1, kind)`` segments; the arrangement is planarised by
    the caller.

    ``leaf_ids[k]`` is the disc (leaf) row index that active-polygon vertex ``k``
    came from - a parallel array carried alongside ``Q``/``rho`` so any point in
    the recursion knows which leaf each corner is. It is purely bookkeeping here
    (Stage 1): it does not touch the geometry. A gusset split reuses the two
    existing corner vertices as the diagonal's endpoints, so each sub-polygon
    inherits its corners' leaf ids directly via ``leaf_ids[idx]``.
    """
    m = len(Q)
    # Reduced (river-aware) tight lengths: for each active-polygon vertex pair the
    # extra band width ``scale*treepath(leaf_i, leaf_j) - (r_i + r_j)`` the tree
    # path opens between them. Zero for a star (reduced_dist == r_i + r_j on every
    # pair), positive where the path crosses a spine - so every tightness test
    # below is river-aware while staying byte-identical on stars.
    offs = np.zeros((m, m))
    for a in range(m):
        for b in range(a + 1, m):
            la, lb = int(leaf_ids[a]), int(leaf_ids[b])
            w = (_reduced_dist(packing, la, lb)
                 - (packing.radii[la] + packing.radii[lb]))
            offs[a, b] = offs[b, a] = w
    # Base-case river-awareness: a river crossing the polygon *interior* (a
    # non-adjacent pair held an extra band apart) can never be one incircle fan -
    # it must be split, so it never reaches the single-incentre base case.
    interior_river = any(
        offs[a, b] > 1e-9
        for a in range(m) for b in range(a + 1, m)
        if b != (a + 1) % m and a != (b + 1) % m)
    if m == 3:
        ic = _incenter(Q[0], Q[1], Q[2])
        base = True
    else:
        base, ic = _incircle_flaps_ok(Q, rho)
    if base and interior_river:
        base = False                                # force the river split
    if base:
        for k in range(m):
            creases.append((Q[k], ic, "ridge"))
            tgt = outer[k] if outer[k] is not None else _foot(ic, Q[k], Q[(k + 1) % m])
            creases.append((ic, tgt, "hinge"))
            creases.append((Q[k], Q[(k + 1) % m], "edge"))
        return

    # River-contraction (spine) event: a band bounded by two river edges pinches
    # to a spine before any transversal pair tightens. In the symmetric single
    # spine this fires where the ordinary gusset quadratic only grazes (a double
    # root), so it must be checked explicitly - otherwise the polygon wrongly
    # falls through to one incircle. Fire it when it is the earliest event.
    event = _first_gusset_event(Q, rho, offs)
    river = _river_split(Q, offs)
    if river is not None and (event is None or river[0] <= event[0] + 1e-9):
        t, e0, e1, grpA, grpB, C0, C1 = river
        setA = set(grpA)
        hub_of = lambda v: C0 if v in setA else C1
        for k in grpA:
            creases.append((Q[k], C0, "ridge"))     # leaf ridges to its sub-hub
        for k in grpB:
            creases.append((Q[k], C1, "ridge"))
        creases.append((C0, C1, "ridge"))           # the spine crease (river core)
        for k in range(m):
            i, j = k, (k + 1) % m
            u = Q[j] - Q[i]
            u = u / np.linalg.norm(u)
            fi, fj = Q[i] + rho[i] * u, Q[j] - rho[j] * u  # feet cut at r_i / r_j
            creases.append((hub_of(i), fi, "hinge"))       # feet hinge to their hub
            creases.append((hub_of(j), fj, "hinge"))
            creases.append((Q[i], Q[j], "edge"))
        if river_info is not None:
            # Record the geometric spine so the caller can read the honest flap
            # length of every far-side leaf (its terminal cut plus this spine it
            # crosses to reach the root). Root-side leaves cross no spine.
            sl = float(np.linalg.norm(C0 - C1))
            far = [int(leaf_ids[k]) for grp in (grpA, grpB) for k in grp
                   if packing.tree.parent_hub(packing.leaves[int(leaf_ids[k])])
                   != packing.tree.root]
            river_info.append((sl, far))
        return

    if event is None:                               # degenerate guard: fill as incircle
        ic, _, _ = _incircle(Q)
        for k in range(m):
            creases.append((Q[k], ic, "ridge"))
            tgt = outer[k] if outer[k] is not None else _foot(ic, Q[k], Q[(k + 1) % m])
            creases.append((ic, tgt, "hinge"))
            creases.append((Q[k], Q[(k + 1) % m], "edge"))
        return

    t, i, j = event
    vel, cot = _poly_geometry(Q)
    Qn = Q + t * vel
    rhon = rho - t * cot
    for k in range(m):
        creases.append((Q[k], Qn[k], "ridge"))      # collar bisector trace
        creases.append((Q[k], Q[(k + 1) % m], "edge"))
    # gusset, split at its tangent point (shared exactly by both sub-molecules);
    # the river band offset shifts the split so each side still keeps its flap.
    off = float(offs[i, j])
    tg = Qn[i] + (Qn[j] - Qn[i]) * (rhon[i] / (rhon[i] + rhon[j] + off))
    creases.append((Qn[i], tg, "gusset"))
    creases.append((tg, Qn[j], "gusset"))

    def arc(a, b):
        out = []
        k = a
        while True:
            out.append(k)
            if k == b:
                break
            k = (k + 1) % m
        return out

    for idx in (arc(i, j), arc(j, i)):
        L = len(idx)
        sub_outer = []
        for s in range(L):
            a, b = idx[s], idx[(s + 1) % L]
            if (a == i and b == j) or (a == j and b == i):
                sub_outer.append(tg)                # gusset edge: hinge to gusset tangent
            else:
                sub_outer.append(outer[a])          # inherit the border tangent
        _um_fill(Qn[idx], rhon[idx], sub_outer, creases, leaf_ids[idx], packing,
                 river_info)


def _seg_split_params(a, b, c, d):
    """Parameters along segment ``ab`` where it meets segment ``cd`` (incl. overlap)."""
    r = b - a
    s = d - c
    rxs = r[0] * s[1] - r[1] * s[0]
    qp = c - a
    out = []
    if abs(rxs) > 1e-12:
        t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
        u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
        if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
            out.append(min(max(t, 0.0), 1.0))
    elif abs(qp[0] * r[1] - qp[1] * r[0]) < 1e-8 * max(1.0, float(np.linalg.norm(r))):
        rr = float(r.dot(r))
        for p in (c, d):
            tt = float((p - a).dot(r)) / rr
            if -1e-9 <= tt <= 1 + 1e-9:
                out.append(min(max(tt, 0.0), 1.0))
    return out


def _extract_faces(verts: np.ndarray, edges: np.ndarray) -> list[list[int]]:
    """Interior faces of the planar crease graph (for the FOLD ``faces`` array).

    Standard half-edge walk: at each vertex the neighbours are sorted CCW, and a
    face is traced by always taking the next-clockwise edge; the CCW-oriented
    (positive-area) loops are the interior faces, the one CW loop is the paper's
    outer boundary and is dropped.
    """
    from collections import defaultdict
    adj: dict[int, list[int]] = defaultdict(list)
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    for u in adj:
        cx, cy = verts[u]
        adj[u].sort(key=lambda w: math.atan2(verts[w][1] - cy, verts[w][0] - cx))
    pos = {u: {w: k for k, w in enumerate(adj[u])} for u in adj}

    def nxt(u, v):                                   # half-edge u->v, turn clockwise at v
        nb = adj[v]
        return nb[(pos[v][u] - 1) % len(nb)]

    used: set = set()
    faces: list[list[int]] = []
    for u, v in edges:
        for start in ((int(u), int(v)), (int(v), int(u))):
            if start in used:
                continue
            face = []
            cur = start
            while cur not in used:
                used.add(cur)
                face.append(cur[0])
                cur = (cur[1], nxt(cur[0], cur[1]))
                if len(face) > len(edges) * 2 + 5:
                    break
            if len(face) >= 3:
                area = 0.0
                for p, q in zip(face, face[1:] + face[:1]):
                    area += verts[p][0] * verts[q][1] - verts[q][0] * verts[p][1]
                if area > 0:
                    faces.append([int(x) for x in face])
    return faces


def _assign_mountain_valley(edges, assign, interior, seed=104729):
    """Choose M/V on the fold creases so ``|M - V| = 2`` at every interior vertex.

    Kawasaki is already guaranteed by the geometry; this is the Maekawa half. The
    default ridge=M / hinge=valley labels leave every vertex balanced
    (``|M - V| = 0``), so we search for a globally consistent flip pattern that
    hits ``|M - V| = 2`` everywhere - a small degree-constrained problem solved
    by seeded local search (deterministic across the fixed seed sweep). Returns
    ``(assignment, residual)``; residual 0 means every vertex satisfies Maekawa.
    """
    fold = [e for e, a in enumerate(assign) if a in ("M", "V")]
    foldset = set(fold)
    inc = {v: [] for v in interior}
    for e, (u, v) in enumerate(edges):
        if e in foldset:
            if u in inc:
                inc[int(u)].append(e)
            if v in inc:
                inc[int(v)].append(e)

    def run(sd):
        rng = random.Random(sd)
        s = {e: (1 if assign[e] == "M" else -1) for e in fold}

        def violation():
            return sum(abs(abs(sum(s[e] for e in inc[v])) - 2) for v in interior)

        cur = violation()
        for _ in range(30000):
            if cur == 0:
                break
            bad = [v for v in interior if abs(sum(s[e] for e in inc[v])) != 2]
            if not bad:
                break
            v = rng.choice(bad)
            e = rng.choice(inc[v])
            s[e] = -s[e]
            new = violation()
            if new <= cur or rng.random() < 0.05:
                cur = new
            else:
                s[e] = -s[e]
        return s, cur

    best = None
    for sd in range(seed, seed + 24):
        s, resid = run(sd)
        if best is None or resid < best[1]:
            best = (s, resid)
        if resid == 0:
            break
    s, resid = best
    out = list(assign)
    for e in fold:
        out[e] = "M" if s[e] > 0 else "V"
    return out, resid


def _reduced_dist(packing: Packing, i: int, j: int) -> float:
    """The *reduced* target distance between two discs = ``scale * treepath``.

    This is the general "tight length" the molecule must reproduce: for two
    leaves sharing a hub it equals ``r_i + r_j`` (tangent, no river); across a
    spine it is larger by the river band ``scale * spine``. Star trees hit the
    first case on every edge, so everything below reduces to the old
    ``r_i + r_j`` test with zero change for them.
    """
    return packing.scale * packing.tree.path_length(
        packing.leaves[i], packing.leaves[j])


def reduced_dist(packing: Packing, i: int, j: int) -> float:
    """Reduced target distance between disc rows ``i`` and ``j``.

    Public alias for :func:`_reduced_dist`: ``scale * treepath(leaf_i, leaf_j)``.
    For hull-adjacent leaves of a *star* tree this equals ``r_i + r_j`` (the
    tangent-disc length); Stage 2 relies on that invariant. Defined and tested
    here but not yet wired into any tightness decision (Stage 1 = no behaviour
    change).
    """
    return _reduced_dist(packing, i, j)


def _hull_river_widths(packing: Packing, hidx: list[int]) -> list[float]:
    """River band width left open on each convex-hull edge (0 == tangent pair).

    ``W_k = scale*treepath(i, j) - (r_i + r_j)`` for hull edge ``k = (i, j)``.
    A star's hull has ``W_k == 0`` everywhere; a single-spine river has exactly
    two positive entries (the two sides of the band between the sub-hubs).
    """
    m = len(hidx)
    out = []
    for k in range(m):
        i, j = hidx[k], hidx[(k + 1) % m]
        r = packing.radii[i] + packing.radii[j]
        out.append(_reduced_dist(packing, i, j) - r)
    return out


def _is_tight_hull(packing: Packing, hidx: list[int], tol: float = 1e-4) -> bool:
    """True iff every hull edge is *reduced-tight* (``len == scale*treepath``).

    Generalises the old tangent-disc test (``len == r_i + r_j``): a hull edge is
    tight when its length equals the reduced tree-path distance between the two
    leaves it joins. For a star every hull edge is a tangent disc pair, so this
    is unchanged; for a (correctly packed) river the river edges are longer than
    ``r_i + r_j`` yet still tight in this reduced sense.
    """
    pts = packing.centers
    m = len(hidx)
    for k in range(m):
        i, j = hidx[k], hidx[(k + 1) % m]
        edge = float(np.linalg.norm(pts[i] - pts[j]))
        if abs(edge - _reduced_dist(packing, i, j)) > tol:
            return False
    return True


def _universal_pattern(packing: Packing, hidx: list[int]) -> CreasePattern:
    """Fill the (tight, convex) active hull with the universal molecule."""
    pts = packing.centers
    Q0 = pts[hidx].copy()
    r0 = packing.radii[hidx].copy()
    m = len(Q0)

    outer = []
    borders = []
    for k in range(m):
        d = Q0[(k + 1) % m] - Q0[k]
        d = d / np.linalg.norm(d)
        fi = Q0[k] + d * r0[k]                      # near corner's tangent point
        fj = Q0[(k + 1) % m] - d * r0[(k + 1) % m]  # far corner's tangent point
        outer.append(fi)
        borders.append((Q0[k], fi, "B"))
        borders.append((fi, fj, "B"))               # river band (zero-length on a star)
        borders.append((fj, Q0[(k + 1) % m], "B"))

    # Per-vertex leaf (disc) row index for every active-polygon corner: each hull
    # vertex is exactly the disc `hidx[k]`. Threaded through the recursion so any
    # sub-polygon corner still knows its leaf (Stage 1: bookkeeping only, no
    # geometry change).
    leaf_ids = np.asarray(hidx)
    creases: list = []
    river_info: list = []
    _um_fill(Q0, r0, outer, creases, leaf_ids, packing, river_info)
    creases += borders

    # Planarise: leaf centres are added first so their row indices are 0..N-1.
    bag = _VertexBag()
    for centre in pts:
        bag.add(centre)
    segs = [(np.asarray(a, float), np.asarray(b, float), kind)
            for a, b, kind in creases
            if np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)) > 1e-9]
    n = len(segs)
    params = [[0.0, 1.0] for _ in range(n)]
    for i in range(n):
        a, b, _ = segs[i]
        for jj in range(n):
            if i != jj:
                params[i] += _seg_split_params(a, b, segs[jj][0], segs[jj][1])

    prio = {"B": 3, "gusset": 2, "ridge": 2, "hinge": 1, "edge": 1}
    edgemap: dict[tuple[int, int], str] = {}
    for i in range(n):
        a, b, kind = segs[i]
        ts = sorted({round(t, 11) for t in params[i]})
        pts_i = [a + t * (b - a) for t in ts]
        for p, q in zip(pts_i[:-1], pts_i[1:]):
            ia, ib = bag.add(p), bag.add(q)
            if ia == ib:
                continue
            key = (ia, ib) if ia < ib else (ib, ia)
            if key in edgemap:
                if prio[kind] > prio[edgemap[key]]:
                    edgemap[key] = kind
            else:
                edgemap[key] = kind

    verts = np.array(bag.coords)
    kind_to_assign = {"B": "B", "ridge": "M", "hinge": "V", "gusset": "V", "edge": "V"}
    edges = np.array(list(edgemap.keys()), dtype=int)
    assign = [kind_to_assign[edgemap[tuple(e)]] for e in map(tuple, edges)]

    scratch = CreasePattern(vertices=verts, edges=edges, assignment=list(assign),
                            faces=[], metadata={})
    interior = scratch.interior_vertices()
    assign, _resid = _assign_mountain_valley(edges, assign, interior)
    faces = _extract_faces(verts, edges)
    metadata = {"name": f"treemaker:{packing.tree.name}"}
    if river_info:
        # Honest geometric flap lengths for a river base: each leaf's border cut
        # plus every spine it crosses to reach the root. Read straight off the
        # built geometry (spine == |C0 - C1|), so this verifies the fold, it does
        # not assume it. Stars emit no river_info and keep the recompute path.
        pat0 = CreasePattern(vertices=verts, edges=edges, assignment=list(assign),
                             faces=faces, metadata={})
        flap = np.full(len(packing.leaves), np.nan)
        for gi in range(len(packing.leaves)):
            b = [e for e in pat0.incident_edges(gi) if pat0.assignment[e] == "B"]
            if not b:
                continue
            base = min(float(np.linalg.norm(verts[edges[e][0]] - verts[edges[e][1]]))
                       for e in b)
            flap[gi] = base + sum(sl for sl, far in river_info if gi in far)
        metadata["flap_lengths"] = flap
    return CreasePattern(
        vertices=verts,
        edges=edges,
        assignment=assign,
        faces=faces,
        metadata=metadata,
    )


def _line_normal(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit normal ``n`` and offset ``c`` of the line through ``p0, p1`` (``n.x = c``)."""
    d = p1 - p0
    d = d / np.linalg.norm(d)
    n = np.array([d[1], -d[0]])
    return n, float(n.dot(p0))


def _incenter_of_lines(lines, interior_pt):
    """Point equidistant (inward) from a set of lines, least squares.

    Solves ``n_e . x + r = c_e`` with every normal oriented to face
    ``interior_pt``. Returns ``(centre, r, max_residual)``; residual ~0 means the
    lines share one inscribed circle (the sub-hub is a clean tangent fan).
    """
    A, b = [], []
    for n, c in lines:
        s = 1.0 if (n.dot(interior_pt) - c) < 0 else -1.0
        A.append([s * n[0], s * n[1], 1.0])
        b.append(s * c)
    A = np.array(A); b = np.array(b)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:2], float(sol[2]), float(np.max(np.abs(A @ sol - b)))


def _planarise(packing: Packing, creases: list) -> CreasePattern:
    """Shared arrangement builder: split every crease at its crossings, dedupe,
    assign M/V by the global Maekawa search, and extract faces.

    Leaf centres are inserted first so their row indices stay ``0..N-1`` (the
    flap-length reader relies on that).
    """
    pts = packing.centers
    bag = _VertexBag()
    for centre in pts:
        bag.add(centre)
    segs = [(np.asarray(a, float), np.asarray(b, float), kind)
            for a, b, kind in creases
            if np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)) > 1e-9]
    n = len(segs)
    params = [[0.0, 1.0] for _ in range(n)]
    for i in range(n):
        a, b, _ = segs[i]
        for jj in range(n):
            if i != jj:
                params[i] += _seg_split_params(a, b, segs[jj][0], segs[jj][1])

    prio = {"B": 3, "gusset": 2, "ridge": 2, "hinge": 1, "edge": 1}
    edgemap: dict[tuple[int, int], str] = {}
    for i in range(n):
        a, b, kind = segs[i]
        ts = sorted({round(t, 11) for t in params[i]})
        pts_i = [a + t * (b - a) for t in ts]
        for p, q in zip(pts_i[:-1], pts_i[1:]):
            ia, ib = bag.add(p), bag.add(q)
            if ia == ib:
                continue
            key = (ia, ib) if ia < ib else (ib, ia)
            if key in edgemap:
                if prio[kind] > prio[edgemap[key]]:
                    edgemap[key] = kind
            else:
                edgemap[key] = kind

    verts = np.array(bag.coords)
    kind_to_assign = {"B": "B", "ridge": "M", "hinge": "V", "gusset": "V", "edge": "V"}
    edges = np.array(list(edgemap.keys()), dtype=int)
    assign = [kind_to_assign[edgemap[tuple(e)]] for e in map(tuple, edges)]
    scratch = CreasePattern(vertices=verts, edges=edges, assignment=list(assign),
                            faces=[], metadata={})
    interior = scratch.interior_vertices()
    assign, _resid = _assign_mountain_valley(edges, assign, interior)
    faces = _extract_faces(verts, edges)
    return CreasePattern(
        vertices=verts, edges=edges, assignment=assign, faces=faces,
        metadata={"name": f"treemaker:{packing.tree.name}"},
    )


def crease_pattern(packing: Packing) -> CreasePattern:
    """Build the crease pattern for a packed base.

    Routing (all three share the one reduced-distance tight test):

    * **Star** - the hull is reduced-tight with *no* river edges: fill it with
      the universal (gusset) molecule (triangle rabbit-ear / tangential-quad
      incircle / pentagon+ gussets), which folds flat end to end.
    * **Single-spine river** - the hull is reduced-tight with exactly two river
      edges: build the two-node river molecule and, *only if it self-verifies*
      flat-foldable with exact flap lengths, return it. This is what makes a body
      with limbs fold, not just a star.
    * **Otherwise** (river molecule fails its self-check, or the hull is not
      reduced-tight) fall back to the per-Delaunay-triangle rabbit-ear assembly -
      an honest *partial* base whose non-tangent diagonals are unresolved.
    """
    pts = packing.centers
    hull = ConvexHull(pts)
    hidx = [int(i) for i in hull.vertices]     # CCW hull corners (row indices)
    if len(hidx) == len(pts) and _is_tight_hull(packing, hidx):
        widths = _hull_river_widths(packing, hidx)
        n_river = sum(1 for w in widths if w > 1e-6)
        if n_river == 0:
            return _universal_pattern(packing, hidx)
        if n_river == 2:
            # The unified molecule's river-contraction event emits the spine for a
            # single-spine river, so route it through _um_fill like a star. Keep
            # the honest self-check: an asymmetric river the event cannot close
            # falls back to the partial base rather than faking a flat fold.
            from foldforge.geometry.foldability import foldability_report
            pattern = _universal_pattern(packing, hidx)
            report = foldability_report(pattern)
            ok = bool(report.vertices) and all(
                v.kawasaki and v.maekawa for v in report.vertices)
            ok = ok and float(np.nanmax(flap_length_errors(packing, pattern))) < 0.02
            if ok:
                return pattern
    return _rabbit_ear_pattern(packing)


def _rabbit_ear_pattern(packing: Packing) -> CreasePattern:
    """Fallback: a rabbit-ear molecule per Delaunay triangle glued into one map.

    Vertices: the packed leaf centres, plus each triangle's incentre and the
    perpendicular feet on its edges. Kawasaki holds at every incentre, but at a
    *non-tangent* shared triangle edge the two incentres' feet do not coincide,
    so those interior vertices can fail the checks - an honest partial base.
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
    faces: list[list[int]] = []
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
            # two flat sub-triangles of this wedge
            faces.append([corner_idx[ga], fi, ic_i])
            faces.append([fi, corner_idx[gb], ic_i])
        # three bisectors incentre->corner (valley ridges)
        for la in (0, 1, 2):
            add_edge(ic_i, corner_idx[int(t[la])], "V")

    verts = np.array(bag.coords)
    pattern = CreasePattern(
        vertices=verts,
        edges=np.array(edges, dtype=int),
        assignment=assign,
        faces=faces,
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
    distance from the centre to the foot on an incident edge - read straight back
    off the assembled geometry, matching whichever molecule ``crease_pattern``
    used (the universal molecule for a tight tangent hull, per-triangle otherwise).
    """
    # A river pattern caches its geometric flap lengths (border tangent + the
    # spine crease each far-side flap crosses) when it was built - use them.
    if "flap_lengths" in pattern.metadata:
        measured = np.asarray(pattern.metadata["flap_lengths"], dtype=float)
        target = packing.flap_targets
        return np.abs(measured - target) / np.maximum(target, 1e-9)

    pts = packing.centers
    hull = ConvexHull(pts)
    hidx = [int(i) for i in hull.vertices]
    if len(hidx) == len(pts) and _is_tight_hull(packing, hidx):
        # Universal molecule: each border edge is cut at its disc tangent point,
        # so a leaf's flap length is the shorter of its two incident border (B)
        # crease halves - read directly off the built pattern.
        measured = np.full(len(packing.leaves), np.nan)
        for gi in range(len(packing.leaves)):
            b_edges = [e for e in pattern.incident_edges(gi)
                       if pattern.assignment[e] == "B"]
            if not b_edges:
                continue
            lengths = [float(np.linalg.norm(pattern.vertices[pattern.edges[e][0]]
                                            - pattern.vertices[pattern.edges[e][1]]))
                       for e in b_edges]
            measured[gi] = min(lengths)
        target = packing.flap_targets
        return np.abs(measured - target) / np.maximum(target, 1e-9)

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
