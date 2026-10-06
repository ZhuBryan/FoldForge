"""Lang's universal molecule, written as an event-driven wavefront (research build).

This module is deliberately standalone: it imports helpers from ``treemaker``
but changes nothing there, and nothing in the live design path calls it.

Algorithm
---------
The active polygon is the convex hull of the packed disc centres.  Instead of
per-vertex flap radii it carries a *reduced-distance matrix* ``D`` (initially
``scale * treepath(i, j)``, rivers included).  Adjacent pairs are always tight
(``|Q_i Q_j| == D_ij``).  The polygon is inset: every edge moves inward at unit
speed, each vertex slides along its angle bisector (the trace is a *ridge*
crease) and every reduced distance shrinks by ``t (cot_i + cot_j)`` where
``cot_k = cot(theta_k / 2)`` -- the part of the tree path "used up" at each end.
Two kinds of event are processed in time order:

* **edge collapse** -- a wavefront edge reaches length 0.  Its vertices merge
  into one point (a tree node: a hub, or the final incentre) and *hinge* creases
  drop from that point perpendicular onto the source line of every edge that
  touched it.  Merging averages the rows of ``D``; the spread of those rows is
  the Gromov (hub-consistency) residual that tells us whether the packing really
  puts those leaves on one hub.
* **gusset (active-path) event** -- a non-adjacent pair becomes tight
  (``|Q_i Q_j| == D_ij``).  A *gusset* crease joins them, the polygon splits into
  two sub-polygons sharing that new edge, and each recurses.

A polygon that has merged to one point is finished (the incircle / hub fan); one
that has merged to two points leaves the segment between them as a final ridge
(a river *spine*).  There is no hard-coded river or caterpillar case: a
multi-junction tree simply makes several edges collapse together (each hub) and
the surviving polygon of hubs finish as a smaller polygon.

Verification
------------
``lang_base`` never fakes a result.  It computes the flat-folded image of the
pattern (unfolding across every crease, checking the images agree everywhere)
and reads the *folded tree*: leaf tip to hub foot, and hub foot to hub foot, in
the folded plane.  Those folded edge lengths summed along each leaf's path to the
root are the flap lengths stored in ``pattern.metadata["flap_lengths"]`` so
``treemaker.flap_length_errors`` consumes them unchanged.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.spatial import ConvexHull

from foldforge.design.treemaker import (
    MetricTree,
    Packing,
    _planarise,
    _poly_geometry,
    pack_tree,
    reduced_dist,
)
from foldforge.geometry.crease_graph import CreasePattern

_EPS_LEN = 1e-7      # an edge shorter than this has collapsed
_EPS_TIGHT = 1e-7    # a non-adjacent pair this close to tight is tight
_MAX_STEPS = 400


class LangError(RuntimeError):
    """The wavefront hit a configuration this build cannot continue from."""


# --- small geometry ----------------------------------------------------------

def _line(p0: np.ndarray, p1: np.ndarray):
    u = p1 - p0
    return np.asarray(p0, float), u / np.linalg.norm(u)


def _drop(x: np.ndarray, line) -> np.ndarray:
    """Orthogonal projection of ``x`` onto ``line = (point, unit direction)``."""
    p, u = line
    return p + u * float((x - p).dot(u))


# --- gusset event ------------------------------------------------------------

def _first_event(Q, D, vel, cot):
    """Earliest ``(t, kind, data)`` over edge collapses and non-adjacent pairs."""
    m = len(Q)
    best = None
    for k in range(m):
        j = (k + 1) % m
        rate = cot[k] + cot[j]
        if rate > 1e-14:
            t = float(np.linalg.norm(Q[j] - Q[k])) / rate
            if best is None or t < best[0]:
                best = (t, "collapse", k)
    for i in range(m):
        for j in range(i + 2, m):
            if i == 0 and j == m - 1:
                continue                                  # adjacent (wraps)
            dv = vel[i] - vel[j]
            d0 = Q[i] - Q[j]
            c = cot[i] + cot[j]
            A = float(dv.dot(dv) - c * c)
            B = float(2 * d0.dot(dv) + 2 * D[i, j] * c)
            C = float(d0.dot(d0) - D[i, j] ** 2)
            roots = []
            if abs(A) < 1e-12:
                if abs(B) > 1e-12:
                    roots = [-C / B]
            else:
                disc = B * B - 4 * A * C
                if disc >= 0:
                    sq = math.sqrt(disc)
                    roots = [(-B - sq) / (2 * A), (-B + sq) / (2 * A)]
            for t in roots:
                if t > 1e-10 and D[i, j] - t * c > -1e-9 and (best is None or t < best[0]):
                    for _ in range(30):                   # Newton polish to machine zero
                        d = float(np.linalg.norm(d0 + t * dv))
                        if d < 1e-14:
                            break
                        f = d - (D[i, j] - t * c)
                        t -= f / (float(d0.dot(dv) + t * dv.dot(dv)) / d + c)
                        if abs(f) < 1e-15:
                            break
                    if best is None or t < best[0]:
                        best = (t, "gusset", (i, j))
    return best


# --- the recursion -----------------------------------------------------------

class _Run:
    def __init__(self):
        self.creases: list = []
        self.events: list = []            # (kind, time) log, for reporting
        self.merge_resid = 0.0            # worst Gromov spread seen at a merge
        self.steps = 0


def _hinges(run: _Run, X, lines):
    for ln in lines:
        run.creases.append((X, _drop(X, ln), "hinge"))


def _merge_collapsed(run: _Run, Q, D, src):
    """Merge every run of collapsed edges into one vertex (hub / incentre)."""
    m = len(Q)
    col = [float(np.linalg.norm(Q[(k + 1) % m] - Q[k])) < _EPS_LEN for k in range(m)]
    if not any(col):
        return Q, D, src
    if all(col):
        X = Q.mean(axis=0)
        _hinges(run, X, src)
        run.merge_resid = max(run.merge_resid, float(np.ptp(D[np.triu_indices(m, 1)])) if m > 1 else 0)
        return X[None, :], np.zeros((1, 1)), []
    start = next(k for k in range(m) if not col[(k - 1) % m])
    groups, k = [], start
    for _ in range(m):
        grp = [k]
        while col[k]:
            k = (k + 1) % m
            grp.append(k)
        k = (k + 1) % m
        groups.append(grp)
        if sum(len(g) for g in groups) >= m:
            break
    g = len(groups)
    Qn = np.array([Q[grp].mean(axis=0) for grp in groups])
    Dn = np.zeros((g, g))
    for a in range(g):
        for b in range(g):
            if a != b:
                blk = D[np.ix_(groups[a], groups[b])]
                Dn[a, b] = blk.mean()
                run.merge_resid = max(run.merge_resid, float(np.ptp(blk)))
    sn = [src[grp[-1]] for grp in groups]                 # edge leaving each group
    for a, grp in enumerate(groups):
        if len(grp) > 1:
            lines = [src[x] for x in grp[:-1]]            # the collapsed edges
            lines += [sn[a - 1], sn[a]]                   # and both neighbours
            _hinges(run, Qn[a], lines)
    for a in range(g):                                    # keep adjacent pairs exactly tight
        b = (a + 1) % g
        Dn[a, b] = Dn[b, a] = float(np.linalg.norm(Qn[b] - Qn[a]))
    return Qn, Dn, sn


def _fill(run: _Run, Q, D, src):
    """Run the wavefront on one active polygon (recursing on gusset splits)."""
    while True:
        run.steps += 1
        if run.steps > _MAX_STEPS:
            raise LangError("wavefront did not terminate")
        Q, D, src = _merge_collapsed(run, Q, D, src)
        m = len(Q)
        if m <= 1:
            return
        if m == 2:                                        # river core: final spine ridge
            run.creases.append((Q[0], Q[1], "ridge"))
            for X in Q:
                _hinges(run, X, src)
            return
        # gusset: a non-adjacent pair that is already tight
        pair = None
        for i in range(m):
            for j in range(i + 2, m):
                if i == 0 and j == m - 1:
                    continue
                if float(np.linalg.norm(Q[i] - Q[j])) - D[i, j] <= _EPS_TIGHT:
                    pair = (i, j)
                    break
            if pair:
                break
        if pair:
            i, j = pair
            run.events.append(("gusset", None))
            run.creases.append((Q[i], Q[j], "gusset"))
            gl = _line(Q[i], Q[j])
            for a, b in ((i, j), (j, i)):
                idx = [a]
                k = a
                while k != b:
                    k = (k + 1) % m
                    idx.append(k)
                s = [src[x] for x in idx[:-1]] + [gl]
                _fill(run, Q[idx], D[np.ix_(idx, idx)], s)
            return
        vel, cot = _poly_geometry(Q)
        ev = _first_event(Q, D, vel, cot)
        if ev is None:
            raise LangError("no further event on an unfinished polygon")
        t, kind, _ = ev
        run.events.append((kind, t))
        Qn = Q + t * vel
        for k in range(m):
            run.creases.append((Q[k], Qn[k], "ridge"))
        c = cot[:, None] + cot[None, :]
        Dn = D - t * c
        np.fill_diagonal(Dn, 0.0)
        Q, D = Qn, Dn


def _um_creases(packing: Packing, hidx: list[int]):
    Q0 = packing.centers[hidx].copy()
    m = len(Q0)
    D = np.array([[reduced_dist(packing, hidx[a], hidx[b]) for b in range(m)] for a in range(m)])
    for a in range(m):
        b = (a + 1) % m
        D[a, b] = D[b, a] = float(np.linalg.norm(Q0[b] - Q0[a]))
    src = [_line(Q0[k], Q0[(k + 1) % m]) for k in range(m)]
    run = _Run()
    _fill(run, Q0, D, src)
    borders = [(Q0[k], Q0[(k + 1) % m], "B") for k in range(m)]
    return run, borders


# --- flat-folded image and folded-tree measurement ----------------------------

def flat_fold_image(pattern: CreasePattern):
    """Unfold the pattern across every crease and return ``(image, closure, connected)``.

    ``image`` is the (V, 2) flat-folded position of every vertex; ``closure`` is
    the largest disagreement between the images a vertex gets from different
    faces (0 for a pattern that genuinely folds flat); ``connected`` is False if
    the faces do not form one connected piece.  Mountain/valley and layer order
    are not involved: this is the pure-geometry folding map.
    """
    V, F = pattern.vertices, pattern.faces
    if not F:
        return V.copy(), float("inf"), False
    owner = {}
    for fi, f in enumerate(F):
        for a, b in zip(f, f[1:] + f[:1]):
            owner[(a, b)] = fi
    maps = {0: (np.eye(2), np.zeros(2))}
    q = deque([0])
    while q:
        fi = q.popleft()
        A, t = maps[fi]
        f = F[fi]
        for a, b in zip(f, f[1:] + f[:1]):
            nb = owner.get((b, a))
            if nb is None or nb in maps:
                continue
            pa, pb = A @ V[a] + t, A @ V[b] + t
            d = (pb - pa) / np.linalg.norm(pb - pa)
            R = 2 * np.outer(d, d) - np.eye(2)
            maps[nb] = (R @ A, R @ (t - pa) + pa)
            q.append(nb)
    acc = [[] for _ in V]
    for fi, f in enumerate(F):
        if fi in maps:
            A, t = maps[fi]
            for v in f:
                acc[v].append(A @ V[v] + t)
    img = np.zeros((len(V), 2))
    closure = 0.0
    for v, pts in enumerate(acc):
        if pts:
            pts = np.array(pts)
            img[v] = pts.mean(axis=0)
            closure = max(closure, float(np.abs(pts - pts[0]).max()))
        else:
            img[v] = V[v]
    return img, closure, len(maps) == len(F)


def _tree_path_edges(tree: MetricTree, a: int, b: int):
    stack = [(a, -1, [])]
    while stack:
        node, parent, path = stack.pop()
        if node == b:
            return path
        for nxt, _ in tree._adj[node]:
            if nxt != parent:
                stack.append((nxt, node, path + [(node, nxt)]))
    raise ValueError("no path")


def folded_tree_lengths(packing: Packing, pattern: CreasePattern, img: np.ndarray):
    """Read the tree back off the folded pattern.

    Each leaf's *foot* is the border vertex a distance ``r_i`` from its tip along
    each incident border side; a hub's folded position is the image of its leaves'
    feet.  Returns ``(flap_lengths, hub_spread, edge_lengths)`` where
    ``flap_lengths[i]`` is the folded tip-to-root path length, ``hub_spread`` the
    worst disagreement between feet that should be one hub, and ``edge_lengths``
    maps each measurable tree edge to its folded length.
    """
    tree, V = packing.tree, pattern.vertices
    bord = set()
    for e, (a, b) in enumerate(pattern.edges):
        if pattern.assignment[e] == "B":
            bord |= {int(a), int(b)}
    hub_pts: dict[int, list] = {}
    spread = 0.0
    for i, leaf in enumerate(packing.leaves):
        feet = []
        for e, (a, b) in enumerate(pattern.edges):
            if pattern.assignment[e] != "B" or i not in (int(a), int(b)):
                continue
            o = int(b) if int(a) == i else int(a)
            u = (V[o] - V[i]) / np.linalg.norm(V[o] - V[i])
            for w in bord:
                dw = V[w] - V[i]
                if (abs(float(dw.dot(u)) - packing.radii[i]) < 1e-6
                        and abs(float(dw[0] * u[1] - dw[1] * u[0])) < 1e-7):
                    feet.append(w)
                    break
        if len(feet) == 2:
            spread = max(spread, float(np.linalg.norm(img[feet[0]] - img[feet[1]])))
        if feet:
            hub_pts.setdefault(tree.parent_hub(leaf), []).extend(img[f] for f in feet)
    hub = {}
    for h, pts in hub_pts.items():
        pts = np.array(pts)
        hub[h] = pts.mean(axis=0)
        spread = max(spread, float(np.abs(pts - hub[h]).max()))
    lengths: dict[tuple[int, int], float] = {}

    def elen(u, v):
        key = (min(u, v), max(u, v))
        if key not in lengths:
            pu = img[packing.leaves.index(u)] if u in packing.leaves else hub.get(u)
            pv = img[packing.leaves.index(v)] if v in packing.leaves else hub.get(v)
            lengths[key] = (float(np.linalg.norm(pu - pv))
                            if pu is not None and pv is not None else float("nan"))
        return lengths[key]

    flaps = np.array([
        sum(elen(u, v) for u, v in _tree_path_edges(tree, leaf, tree.root)) * 1.0
        for leaf in packing.leaves])
    return flaps, spread, lengths


# --- public API ---------------------------------------------------------------

def lang_pattern(packing: Packing) -> CreasePattern:
    """Fill the packed hull with the universal molecule and attach the folded-tree check."""
    pts = packing.centers
    hidx = [int(i) for i in ConvexHull(pts).vertices]
    if len(hidx) != len(pts):
        raise LangError("a disc sits inside the hull; the active-path polygon is not the hull")
    run, borders = _um_creases(packing, hidx)
    pat = _planarise(packing, run.creases + borders)
    img, closure, connected = flat_fold_image(pat)
    flaps, spread, lengths = folded_tree_lengths(packing, pat, img)
    pat.metadata.update({
        "name": f"lang:{packing.tree.name}",
        "flap_lengths": flaps,
        "lang": {
            "events": run.events,
            "merge_residual": run.merge_resid,
            "fold_closure": closure,
            "faces_connected": connected,
            "hub_spread": spread,
            "folded_edge_lengths": lengths,
        },
    })
    return pat


def lang_base(tree: MetricTree) -> tuple[Packing, CreasePattern]:
    """Pack the tree and build its universal-molecule base (mirrors ``design_base``)."""
    packing = pack_tree(tree)
    return packing, lang_pattern(packing)
