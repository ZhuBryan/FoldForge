"""Global layer-ordering / self-intersection validation for flat folds.

The local theorems (Kawasaki + Maekawa, see :mod:`foldforge.geometry.foldability`)
are *per-vertex*: they ask whether each vertex, in isolation, could fold flat.
Passing them does **not** prove the whole folded sheet avoids passing through
itself.  A flat-folded state puts every face at a 2D position (faces stack where
they overlap) and assigns the stack a *layer order*.  A valid order has to
respect three classic non-crossing conditions between faces that overlap:

* **taco-taco** - two creases whose folded images lie on top of each other act
  like two folded tacos; their two face-pairs may *nest* or be *disjoint* in the
  stack but may not *interleave* (that would make the two folds cross).
* **taco-tortilla** - a flat face (the tortilla) that lies across a folded
  crease (the taco) cannot slip *between* the taco's two faces; it is wholly
  above or wholly below both.
* **transitivity** - above/below is a consistent order: no cycles.

If no layer order satisfies all of these, the paper would have to pass through
itself: the pattern is **not** globally flat-foldable even though every vertex
is locally fine.  That gap is the whole point of this module.

Honesty / scope
---------------
Deciding flat-foldability of a general multi-vertex crease pattern is
**NP-hard** (Bern & Hayes 1996).  This solver is exact for the cases it is
meant for and explicit about its ceiling:

* it derives the taco-taco, taco-tortilla and transitivity constraints from the
  real folded geometry (no hand-waving);
* it finds an order by backtracking search over the overlap-connected
  components, which is exact and fast for single-vertex folds and for the
  uniaxial / Miura / bird-base patterns FoldForge produces;
* it is *not* a complete general solver: an adversarial pattern can make the
  search exceed its step budget, in which case the result says ``valid=None``
  rather than guessing.  No claim of a polynomial general algorithm.

Constraint types implemented: taco-taco, taco-tortilla, transitivity (via a
total-order search).  Not implemented: nothing of the three is skipped for the
convex-face patterns here; non-convex faces are out of scope (FoldForge faces
are triangles and convex quads).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from foldforge.geometry.crease_graph import CreasePattern
from foldforge.design.lang_molecule import flat_fold_image

# Geometry tolerances. Folded coordinates are O(1); generators carry ~1e-6 noise.
_AREA_TOL = 1e-7     # overlap must have at least this much shared area
_LINE_TOL = 1e-7     # point-on-line / collinearity slack
_LEN_TOL = 1e-6      # a shared segment must be at least this long to count
_BUDGET = 500_000    # backtracking node ceiling before we admit defeat honestly


@dataclass
class LayerOrderResult:
    """Outcome of the global layer-ordering check.

    ``valid`` is True if a consistent order exists, False if none does (the
    paper self-intersects), or ``None`` if the search hit its budget without
    deciding.  ``order`` is face indices listed bottom layer -> top layer when
    ``valid`` is True, else ``None``.  ``overlaps`` is the list of face pairs
    that share area in the plane.  ``note`` is a one-line human summary.
    """

    valid: bool | None
    order: list[int] | None
    overlaps: list[tuple[int, int]]
    note: str


# --- convex-polygon geometry --------------------------------------------------

def _ccw(poly: np.ndarray) -> np.ndarray:
    """Return ``poly`` oriented counter-clockwise (flip if the area is negative)."""
    x, y = poly[:, 0], poly[:, 1]
    area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return poly if area2 >= 0 else poly[::-1]


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of convex ``subject`` by convex CCW ``clipper``."""
    out = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        edge = b - a
        inp, out = out, []
        if not inp:
            break
        # Inside = left of the directed clip edge (CCW clipper -> interior on left).
        def side(p):
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
        s = [side(p) for p in inp]
        for j in range(len(inp)):
            cur, nxt = inp[j], inp[(j + 1) % len(inp)]
            sc, sn = s[j], s[(j + 1) % len(inp)]
            if sc >= -_LINE_TOL:
                out.append(cur)
            if (sc > _LINE_TOL) != (sn > _LINE_TOL):
                t = sc / (sc - sn)
                out.append(cur + t * (nxt - cur))
    return np.array(out) if out else np.empty((0, 2))


def _area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _overlap_area(p: np.ndarray, q: np.ndarray) -> float:
    return _area(_clip(_ccw(p), _ccw(q)))


def _seg_inside_interior(seg, poly: np.ndarray) -> bool:
    """True if segment ``seg`` crosses the *interior* of convex ``poly``.

    Used for taco-tortilla: the crease segment must pass through the face, not
    merely run along one of its edges (that would make the face a taco partner,
    not a tortilla).
    """
    poly = _ccw(poly)
    p0, p1 = np.asarray(seg[0], float), np.asarray(seg[1], float)
    t0, t1 = 0.0, 1.0
    d = p1 - p0
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        n = np.array([a[1] - b[1], b[0] - a[0]])      # inward normal (CCW)
        denom = n @ d
        num = n @ (a - p0)
        if abs(denom) < _LINE_TOL:
            if num > _LINE_TOL:                          # parallel & on exterior side
                return False
            continue
        t = num / denom
        if denom > 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 - t1 > _LINE_TOL:
            return False
    if t1 - t0 < _LEN_TOL:
        return False
    mid = p0 + 0.5 * (t0 + t1) * d
    # Midpoint strictly inside (not on an edge) => genuine tortilla crossing.
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        if (b[0] - a[0]) * (mid[1] - a[1]) - (b[1] - a[1]) * (mid[0] - a[0]) < _LINE_TOL:
            return False
    return True


def _segments_overlap(s1, s2) -> bool:
    """True if two segments are collinear and share a sub-segment of real length."""
    p, q = np.asarray(s1[0], float), np.asarray(s1[1], float)
    r, s = np.asarray(s2[0], float), np.asarray(s2[1], float)
    d = q - p
    L = np.linalg.norm(d)
    if L < _LEN_TOL:
        return False
    u = d / L
    # both endpoints of s2 must lie on the line through s1
    for pt in (r, s):
        w = pt - p
        if abs(w[0] * u[1] - w[1] * u[0]) > _LINE_TOL:
            return False
    a0, a1 = 0.0, L
    b0, b1 = float((r - p) @ u), float((s - p) @ u)
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    return hi - lo > _LEN_TOL


# --- constraint extraction ----------------------------------------------------

def _faces_of_edges(faces: list[list[int]]) -> dict[frozenset, list[int]]:
    """Map each undirected polygon edge to the faces that carry it."""
    m: dict[frozenset, list[int]] = {}
    for fi, f in enumerate(faces):
        for u, v in zip(f, f[1:] + f[:1]):
            m.setdefault(frozenset((int(u), int(v))), []).append(fi)
    return m


def _build_constraints(pattern: CreasePattern, img: np.ndarray):
    """Return ``(polys, overlaps, tortilla, taco)`` from the folded geometry.

    ``tortilla`` items are ``(a, b, t)``: face ``t`` must not sit strictly
    between faces ``a`` and ``b``.  ``taco`` items are ``(a, b, c, d)``: the
    pairs ``{a,b}`` and ``{c,d}`` must not interleave.
    """
    faces = [list(map(int, f)) for f in pattern.faces]
    polys = [np.array([img[v] for v in f], float) for f in faces]

    # Overlapping face pairs (shared area in the plane).
    overlaps: list[tuple[int, int]] = []
    ov = set()
    for i, j in itertools.combinations(range(len(faces)), 2):
        if _overlap_area(polys[i], polys[j]) > _AREA_TOL:
            overlaps.append((i, j))
            ov.add((i, j))

    def overlapping(a, b):
        return (min(a, b), max(a, b)) in ov

    edge_faces = _faces_of_edges(faces)
    # A crease is an interior M/V edge shared by exactly two faces.
    creases = []  # (faceA, faceB, segment)
    for e, (a, b) in enumerate(pattern.edges):
        if pattern.assignment[e] not in ("M", "V"):
            continue
        fs = edge_faces.get(frozenset((int(a), int(b))), [])
        if len(fs) == 2:
            seg = (img[int(a)], img[int(b)])
            creases.append((fs[0], fs[1], seg))

    # taco-tortilla: crease (a,b) vs a third face t whose image the crease
    # segment crosses through the interior of.
    tortilla = []
    for a, b, seg in creases:
        for t in range(len(faces)):
            if t in (a, b):
                continue
            if overlapping(a, t) and overlapping(b, t) and _seg_inside_interior(seg, polys[t]):
                tortilla.append((a, b, t))

    # taco-taco: two creases whose folded segments overlap, with four distinct
    # faces that mutually overlap.
    taco = []
    for i, j in itertools.combinations(range(len(creases)), 2):
        a, b, s1 = creases[i]
        c, d, s2 = creases[j]
        if len({a, b, c, d}) != 4:
            continue
        if not _segments_overlap(s1, s2):
            continue
        if all(overlapping(x, y) for x in (a, b) for y in (c, d)):
            taco.append((a, b, c, d))

    return polys, overlaps, tortilla, taco


# --- solving: find a total order consistent with every constraint -------------
#
# We keep a partial order as a reachability matrix R (R[i][j] = "face i is below
# face j"), add one overlapping-pair relation at a time while maintaining the
# transitive closure, and reject the moment a forbidden *chain* appears.  Every
# constraint is a set of forbidden chains in the final total order:
#   taco-tortilla (a,b,t): a below t below b, or b below t below a   (t between)
#   taco-taco (a,b,c,d):   the 8 orderings where the two pairs interleave
# Because the chains are checked on the transitive closure, a violation is caught
# as soon as it is *forced*, not only once all faces are placed.  Transitivity
# and acyclicity are structural (the closure refuses to add a back-edge), so any
# complete assignment topologically sorts into a valid stacking order.


def _taco_chains(a, b, c, d):
    """The 8 full orderings of the four faces in which the two pairs interleave.

    Pair P={a,b} has label 0, pair Q={c,d} label 1; interleaving is any order
    whose labels alternate (0,1,0,1 or 1,0,1,0).  Each such order is a forbidden
    bottom->top chain w<x<y<z.
    """
    lab = {a: 0, b: 0, c: 1, d: 1}
    chains = []
    for perm in itertools.permutations((a, b, c, d)):
        seq = [lab[f] for f in perm]
        if seq in ([0, 1, 0, 1], [1, 0, 1, 0]):
            chains.append(perm)
    return chains


def _order_component(comp, overlaps, tortilla, taco, budget: list[int]):
    """Backtracking search for a stacking order of the faces in ``comp``.

    Returns the order (bottom->top) or ``None`` (no valid order).  Sets
    ``budget[0] <= 0`` if the node budget ran out (caller treats as undecided).
    """
    cset = set(comp)
    N = len(comp)
    idx = {f: i for i, f in enumerate(comp)}
    R = [[False] * N for _ in range(N)]     # R[i][j]: i below j (transitive)

    # Forbidden chains in local indices.
    chains: list[tuple[int, ...]] = []
    for a, b, t in tortilla:
        if {a, b, t} <= cset:
            chains.append((idx[a], idx[t], idx[b]))
            chains.append((idx[b], idx[t], idx[a]))
    for a, b, c, d in taco:
        if {a, b, c, d} <= cset:
            for ch in _taco_chains(a, b, c, d):
                chains.append(tuple(idx[x] for x in ch))

    # Index chains by each node for a cheap "did this edge create a chain" check.
    def chain_broken() -> bool:
        for ch in chains:
            if all(R[ch[k]][ch[k + 1]] for k in range(len(ch) - 1)):
                return True
        return False

    def add_edge(i, j):
        """Add 'i below j' with transitive closure. Return added edges or None (cycle)."""
        if R[i][j]:
            return []
        if R[j][i]:
            return None
        ups = [i] + [u for u in range(N) if R[u][i]]
        downs = [j] + [v for v in range(N) if R[j][v]]
        added = []
        for u in ups:
            for v in downs:
                if not R[u][v]:
                    R[u][v] = True
                    added.append((u, v))
        return added

    def undo(added):
        for u, v in added:
            R[u][v] = False

    # Only overlapping pairs need orienting; non-overlapping faces never collide,
    # so the final topological sort may place them in any relative order.
    pairs = [(idx[a], idx[b]) for a, b in overlaps if a in cset and b in cset]

    def pick():
        for i, j in pairs:
            if not R[i][j] and not R[j][i]:
                return (i, j)
        return None

    def dfs() -> bool:
        if budget[0] <= 0:
            return False
        pair = pick()
        if pair is None:
            return True                      # every pair oriented, order fixed
        i, j = pair
        for a, b in ((i, j), (j, i)):
            budget[0] -= 1
            if budget[0] <= 0:
                return False
            added = add_edge(a, b)
            if added is not None:
                if not chain_broken() and dfs():
                    return True
                undo(added)
        return False

    if not dfs():
        return None
    # Topological sort of R gives the stacking order (bottom -> top).
    indeg = [sum(1 for u in range(N) if R[u][v]) for v in range(N)]
    ready = sorted(v for v in range(N) if indeg[v] == 0)
    order = []
    while ready:
        v = ready.pop(0)
        order.append(comp[v])
        for w in range(N):
            if R[v][w]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    ready.append(w)
        ready.sort()
    return order


def _components(n: int, overlaps) -> list[list[int]]:
    """Connected components of the overlap graph over faces 0..n-1."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in overlaps:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for f in range(n):
        groups.setdefault(find(f), []).append(f)
    return list(groups.values())


def layer_order(pattern: CreasePattern) -> LayerOrderResult:
    """Decide whether a valid global layer order exists for ``pattern``.

    Folds the pattern to the plane, derives the taco-taco / taco-tortilla /
    transitivity constraints from the overlapping faces and creases, and
    searches for a consistent stacking order (bottom layer first).
    """
    n = len(pattern.faces)
    if n == 0:
        return LayerOrderResult(True, [], [], "no faces: trivially orderable")

    img, closure, connected = flat_fold_image(pattern)
    polys, overlaps, tortilla, taco = _build_constraints(pattern, img)

    budget = [_BUDGET]
    full_order: list[int] = []
    for comp in _components(n, overlaps):
        if len(comp) == 1:
            full_order.append(comp[0])
            continue
        sub = _order_component(comp, overlaps, tortilla, taco, budget)
        if budget[0] <= 0:
            return LayerOrderResult(
                None, None, overlaps,
                f"undecided: search exceeded {_BUDGET} nodes on a "
                f"{len(comp)}-face component (NP-hard worst case)",
            )
        if sub is None:
            return LayerOrderResult(
                False, None, overlaps,
                f"no valid layer order: {len(comp)} overlapping faces cannot be "
                f"stacked without self-intersection "
                f"({len(tortilla)} taco-tortilla, {len(taco)} taco-taco constraints)",
            )
        full_order.extend(sub)

    # Final independent guard: never certify an order the arbiter rejects.
    # self_intersections_in_order uses a different primitive than the constraint
    # builder, so this catches a stale blind spot rather than rubber-stamping it.
    residual = self_intersections_in_order(full_order, pattern)
    if residual:
        return LayerOrderResult(
            False, None, overlaps,
            f"no valid layer order: search returned a stack but the independent "
            f"self-intersection check found {len(residual)} violation(s)",
        )

    note = (
        f"valid order found over {len(overlaps)} overlapping face pair(s); "
        f"{len(tortilla)} taco-tortilla, {len(taco)} taco-taco constraints"
    )
    if closure > 1e-4:
        note += f"; WARNING fold closure={closure:.2e} (pattern may not fold flat)"
    return LayerOrderResult(True, full_order, overlaps, note)


def order_violations(order: list[int], pattern: CreasePattern) -> list[str]:
    """Return a list of constraints the given ``order`` violates (empty = valid).

    Independent re-check used by the demo and tests: it rebuilds the
    constraints from geometry and verifies ``order`` against each one, so a
    returned order can be trusted without trusting the solver.
    """
    img, _, _ = flat_fold_image(pattern)
    _, _, tortilla, taco = _build_constraints(pattern, img)
    pos = {f: i for i, f in enumerate(order)}
    bad = []
    for a, b, t in tortilla:
        lo, hi = sorted((pos[a], pos[b]))
        if lo < pos[t] < hi:
            bad.append(f"taco-tortilla: face {t} wedged between {a} and {b}")
    for a, b, c, d in taco:
        lab = {a: 0, b: 0, c: 1, d: 1}
        seq = [lab[f] for f in sorted((a, b, c, d), key=pos.get)]
        if seq in ([0, 1, 0, 1], [1, 0, 1, 0]):
            bad.append(f"taco-taco: pairs ({a},{b}) and ({c},{d}) interleave")
    return bad


def self_intersections_in_order(order: list[int], pattern: CreasePattern) -> list[str]:
    """Independent physical self-intersection check for a stacking ``order``.

    This is the *arbiter* for a returned layer order and is deliberately built
    on a different primitive than :func:`_build_constraints`.  Instead of the
    analytic half-plane clip in :func:`_seg_inside_interior` (the routine that
    carried the sign bug), it folds the pattern to 2D and tests crease-piercing
    and face overlap by *sampling points* and querying
    :class:`matplotlib.path.Path.contains_points`.  So it cannot inherit the
    constraint builder's blind spot: if it reports ``[]`` the stack genuinely
    does not pass paper through itself.

    A fold crease joins two faces hinged along it.  The stack self-intersects
    when:

    * **taco-tortilla** - a crease (joining faces ``a`` and ``b``) passes through
      the *interior* of a third face ``t`` whose layer sits strictly between
      ``a`` and ``b``: the flat face ``t`` is wedged inside the fold.
    * **taco-taco** - two creases whose folded images lie on top of each other,
      whose four faces interleave in the stack: the two folds cross.
    """
    from matplotlib.path import Path

    img, _, _ = flat_fold_image(pattern)
    faces = [list(map(int, f)) for f in pattern.faces]
    polys = [np.array([img[v] for v in f], float) for f in faces]
    paths = [Path(p) for p in polys]
    pos = {f: i for i, f in enumerate(order)}

    # Creases: interior M/V edges shared by exactly two faces (structural).
    edge_faces = _faces_of_edges(faces)
    creases = []  # (faceA, faceB, (P0, P1))
    for e, (a, b) in enumerate(pattern.edges):
        if pattern.assignment[e] not in ("M", "V"):
            continue
        fs = edge_faces.get(frozenset((int(a), int(b))), [])
        if len(fs) == 2:
            creases.append((fs[0], fs[1],
                            (np.asarray(img[int(a)], float),
                             np.asarray(img[int(b)], float))))

    def pierces_interior(seg, path, poly) -> bool:
        """True if ``seg`` crosses the interior of the face ``poly`` as a tortilla.

        Two conditions, checked by a different route than the constraint
        builder's half-plane clip:

        * the face must *straddle* the crease line -- have area on both sides
          (some vertices with signed distance > +tol, some < -tol).  A crease
          that merely runs along one of the face's edges keeps the whole face
          on one side, so it is not a tortilla.
        * the crease segment must actually lie over the face -- a run of sampled
          points along it falls inside the polygon (matplotlib containment).
        """
        p0, p1 = seg
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < _LEN_TOL:
            return False
        u = d / L
        nrm = np.array([-u[1], u[0]])
        sd = (poly - p0) @ nrm
        if not (sd.max() > _LINE_TOL and sd.min() < -_LINE_TOL):
            return False                         # whole face on one side => edge, not tortilla
        ts = np.linspace(0.02, 0.98, 25)[:, None]
        inside = path.contains_points(p0 + ts * d)
        run = 0
        for flag in inside:
            run = run + 1 if flag else 0
            if run >= 2:
                return True
        return False

    def seg_coincident(s1, s2) -> bool:
        """True if two folded segments are collinear and share real length.

        Independent of :func:`_segments_overlap`: projects onto ``s1``'s unit
        direction and checks lateral distance + 1D range overlap directly.
        """
        p, q = s1
        r, s = s2
        d = q - p
        L = float(np.linalg.norm(d))
        if L < _LEN_TOL:
            return False
        u = d / L
        nrm = np.array([-u[1], u[0]])
        if abs(float((r - p) @ nrm)) > _LINE_TOL or abs(float((s - p) @ nrm)) > _LINE_TOL:
            return False
        b0, b1 = float((r - p) @ u), float((s - p) @ u)
        lo = max(0.0, min(b0, b1))
        hi = min(L, max(b0, b1))
        return hi - lo > _LEN_TOL

    bad: list[str] = []
    seen_tt = set()
    for a, b, seg in creases:
        lo, hi = sorted((pos[a], pos[b]))
        for t, path in enumerate(paths):
            if t in (a, b) or not (lo < pos[t] < hi):
                continue
            if pierces_interior(seg, path, polys[t]) and (a, b, t) not in seen_tt:
                seen_tt.add((a, b, t))
                bad.append(f"taco-tortilla: face {t} wedged between {a} and {b}")

    for i, j in itertools.combinations(range(len(creases)), 2):
        a, b, s1 = creases[i]
        c, d, s2 = creases[j]
        if len({a, b, c, d}) != 4:
            continue
        if not seg_coincident(s1, s2):
            continue
        lab = {a: 0, b: 0, c: 1, d: 1}
        seq = [lab[f] for f in sorted((a, b, c, d), key=pos.get)]
        if seq in ([0, 1, 0, 1], [1, 0, 1, 0]):
            bad.append(f"taco-taco: pairs ({a},{b}) and ({c},{d}) interleave")
    return bad


def globally_flat_foldable(pattern: CreasePattern) -> bool:
    """Local (Kawasaki + Maekawa everywhere) AND a valid global layer order.

    This is the honest, stronger flat-foldability verdict: it catches patterns
    that pass every per-vertex theorem yet cannot be folded flat because the
    layers would have to pass through each other.
    """
    from foldforge.geometry.foldability import foldability_report

    if not foldability_report(pattern).flat_foldable:
        return False
    return layer_order(pattern).valid is True
