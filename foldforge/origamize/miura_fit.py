"""True 2D origamization: fit a warped Miura tessellation to a height field.

The corrugation path (:mod:`foldforge.origamize.surface`) folds every row of a
height field as an *independent* 1D pleat chain. Each strip reproduces its own
cross-section, but the strips do not share a single flat sheet: the one crease
pattern that path can honestly return is an extruded profile (constant across
the sheet), so folding it reproduces a *ridge*, not a dome. That is the 1D
bound this module breaks.

Here the sheet is a genuine **2D Miura tessellation**. A Miura-ori has two
crease families (a near-vertical zigzag and horizontal folds) meeting at
degree-4 vertices, and a *uniform* Miura folds to a flat plane. Following the
freeform-origami idea (Tachi), we let the tessellation **warp**: the folded
vertices and the flat crease pattern are optimised together so that

  * the folded mid-surface tracks the target height field (fidelity), while
  * every folded edge keeps its flat length (isometry -> a near-rigid fold), and
  * every facet stays planar -- each Miura quad is split by a triangulating
    diagonal, and *those diagonals are exported as facet creases* so the cut
    pattern's faces are the triangles, planar by construction.

The optimiser is plain Adam with analytic gradients over the same
edge-length / vertex-position quantities the rest of the differentiable core
uses. The result is a real flat crease pattern (FOLD-exportable, with M/V creases
plus the triangulating facet diagonals for the cutter) whose *rigid* folded state
approximates the surface. Because a flat (developable) sheet has zero Gaussian
curvature, a curved target cannot be matched with *zero* strain; the residual
fold strain is measured and reported (mean is typically ~1%; the max is usually
below ~5% but runs higher at extreme aspect ratios or curvature peaks), rather
than hidden. This is still relief / tessellation origami, not Origamizer
arbitrary-3D tuck-folding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foldforge.geometry.crease_graph import CreasePattern
from foldforge.origamize.surface import OrigamiResult


# --- fidelity metric (computed identically for both engines) -----------------

def _align_scale_offset(z: np.ndarray, t: np.ndarray):
    """Best (scale, offset) mapping ``z`` onto ``t`` in the least-squares sense."""
    A = np.stack([z.ravel(), np.ones(z.size)], axis=1)
    s, o = np.linalg.lstsq(A, t.ravel(), rcond=None)[0]
    return float(s), float(o)


def midsurface(folded_grid: np.ndarray) -> np.ndarray:
    """Corrugation-free mid-surface height of a folded ``(R, C, 3)`` grid.

    A Miura relief carries its shape in the *average* of the pleats, so the
    mid-surface is the node heights with the along-column (i) ripple smoothed by
    a fixed 3-tap ``[1, 2, 1]/4`` kernel. The identical operation is applied to
    both engines when scoring, so it never favours one over the other (the
    corrugation extrusion is already smooth across columns, so it is unchanged).
    """
    z = np.asarray(folded_grid, dtype=float)[..., 2].copy()
    if z.shape[1] >= 3:
        z[:, 1:-1] = (z[:, :-2] + 2 * z[:, 1:-1] + z[:, 2:]) / 4.0
    return z


def surface_fit_error(folded_grid: np.ndarray, target: np.ndarray,
                      use_midsurface: bool = True) -> float:
    """Normalised RMSE of a folded surface's height vs a target height field.

    ``folded_grid`` is an ``(R, C, 3)`` grid of folded vertices whose node
    ``(row, col)`` corresponds to target cell ``(row, col)``; ``target`` is the
    ``(R, C)`` height field. The folded heights are aligned to the target by a
    best-fit scale + offset (origami is scale/height-free), then the RMSE is
    divided by the target's height range so the number is comparable across
    shapes and engines. With ``use_midsurface`` the corrugation ripple is
    removed first (see :func:`midsurface`). This is the single fidelity metric,
    run identically on the corrugation and the 2D engine.
    """
    Z = np.asarray(target, dtype=float)
    zf = midsurface(folded_grid) if use_midsurface else np.asarray(folded_grid)[..., 2]
    s, o = _align_scale_offset(zf, Z)
    rmse = float(np.sqrt(np.mean((s * zf + o - Z) ** 2)))
    rng = float(Z.max() - Z.min())
    return rmse / (rng + 1e-9)


def corrugation_surface(result: OrigamiResult, grid_shape) -> np.ndarray:
    """The coherent single-sheet folded surface of a corrugation result.

    The corrugation engine's ``result.folded`` folds every row independently, so
    it is not a single developable sheet; the one crease pattern it *returns*
    (``result.pattern``, one representative cross-section) folds to that profile
    **extruded** across the sheet. This reconstructs that honest single-sheet
    geometry - the representative row's folded spine broadcast over every row -
    so the fidelity metric compares like with like (a valid single sheet from
    each engine). Returns an ``(R, C, 3)`` grid.
    """
    R, C = grid_shape
    F = np.asarray(result.folded, dtype=float).reshape(R, C, 3)
    mid = R // 2
    out = np.zeros((R, C, 3))
    out[..., 0] = F[mid, :, 0][None, :]          # representative x-profile
    out[..., 2] = F[mid, :, 2][None, :]          # representative z-profile (extruded)
    out[..., 1] = F[:, :, 1]                      # keep the row (y) coordinate
    return out


# --- the warped-Miura fit ----------------------------------------------------

def _flat_miura(R, C, a, b, gamma):
    i = np.arange(C)[None, :]
    j = np.arange(R)[:, None]
    X = i * a + (j % 2) * b * np.cos(gamma)
    Y = (j * b * np.sin(gamma)) * np.ones_like(X)
    return np.stack([X, Y, np.zeros_like(X)], axis=-1).astype(float).reshape(-1, 3)


def _folded_miura(R, C, a, b, gamma, h):
    w = np.sqrt(a ** 2 - h ** 2)
    sx = a * b * np.cos(gamma) / w
    d = np.sqrt(max(b ** 2 - sx ** 2, 0.0))
    i = np.arange(C)[None, :]
    j = np.arange(R)[:, None]
    X = i * w + (j % 2) * sx
    Y = (j * d) * np.ones_like(X)
    Z = (i % 2) * h * np.ones_like(X)
    return np.stack([X, Y, Z], axis=-1).astype(float).reshape(-1, 3)


def _edge_list(R, C):
    """Miura crease + triangulation edges of the quad grid (unique)."""
    def idx(j, i):
        return j * C + i
    E = []
    for j in range(R):
        for i in range(C):
            if i + 1 < C:
                E.append((idx(j, i), idx(j, i + 1)))            # horizontal
            if j + 1 < R:
                E.append((idx(j, i), idx(j + 1, i)))            # vertical
            if i + 1 < C and j + 1 < R:
                E.append((idx(j, i), idx(j + 1, i + 1)))        # facet diagonal
    return np.array(E, dtype=int)


def _triangles(R, C):
    def idx(j, i):
        return j * C + i
    T = []
    for j in range(R - 1):
        for i in range(C - 1):
            a = idx(j, i); b = idx(j, i + 1)
            c = idx(j + 1, i + 1); d = idx(j + 1, i)
            T.append([a, b, c]); T.append([a, c, d])
    return np.array(T, dtype=int)


@dataclass
class Miura2DResult(OrigamiResult):
    """A corrugation :class:`OrigamiResult` plus 2D-fit diagnostics."""

    grid_shape: tuple = (0, 0)     # (rows, cols) of the fitted vertex grid
    flat: np.ndarray = None        # (R*C, 2) flat crease-pattern coordinates
    mean_strain: float = 0.0       # mean |folded edge - flat edge| / flat edge
    max_strain: float = 0.0        # worst-case edge strain (curvature hot-spot)


def _fit_warped_miura(Z, *, relief, iters, wiso, wfit, wanc, lr):
    """Jointly optimise flat pattern Q (2D) and folded surface P (3D)."""
    R, C = Z.shape
    a = b = 1.0
    gamma = np.radians(60.0)
    h = 0.45
    Q = _flat_miura(R, C, a, b, gamma)
    P = _folded_miura(R, C, a, b, gamma, h)
    # normalise the target to [0, 1] and lift it to `relief` cell-heights
    Zn = Z - Z.min()
    Zn = Zn / (Zn.max() + 1e-9)
    Zt = (Zn * relief).reshape(-1)
    anchor = P[:, :2].copy()                       # keep the footprint a height field
    P[:, 2] = P[:, 2] + Zt

    E = _edge_list(R, C)
    e0, e1 = E[:, 0], E[:, 1]
    mP = np.zeros_like(P); vP = np.zeros_like(P)
    mQ = np.zeros_like(Q); vQ = np.zeros_like(Q)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        gP = np.zeros_like(P); gQ = np.zeros_like(Q)
        gP[:, 2] += 2 * wfit * (P[:, 2] - Zt)               # fidelity: heights -> target
        gP[:, :2] += 2 * wanc * (P[:, :2] - anchor)          # footprint anchor
        dP = P[e0] - P[e1]; lp = np.linalg.norm(dP, axis=1) + 1e-12
        dQ = Q[e0] - Q[e1]; lq = np.linalg.norm(dQ, axis=1) + 1e-12
        r = lp - lq                                          # isometry residual
        cP = (2 * wiso * r / lp)[:, None] * dP
        np.add.at(gP, e0, cP); np.add.at(gP, e1, -cP)
        cQ = (2 * wiso * (-r) / lq)[:, None] * dQ
        np.add.at(gQ, e0, cQ); np.add.at(gQ, e1, -cQ)
        for g, m, v, X in ((gP, mP, vP, P), (gQ, mQ, vQ, Q)):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            X -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)

    lp = np.linalg.norm(P[e0] - P[e1], axis=1)
    lq = np.linalg.norm(Q[e0] - Q[e1], axis=1)
    strain = np.abs(lp - lq) / (lq + 1e-12)
    return Q, P, strain


def _signed_dihedral(na, nb, axis):
    """Signed dihedral angle (radians) between two unit facet normals about a
    shared edge. The sign is ``sign(dot(na x nb, edge_axis))``, so it is
    consistent for every crease measured with the same normal winding and axis
    orientation: mountain < 0, valley > 0. Returns 0.0 for a flat (coplanar)
    pair.
    """
    c = np.clip(np.dot(na, nb), -1.0, 1.0)
    cr = np.cross(na, nb)
    s = float(np.dot(cr, axis / (np.linalg.norm(axis) + 1e-12)))
    return float(np.arctan2(s, c))


def _quad_dihedral_signs(P, R, C):
    """Signed fold angle for each horizontal / vertical quad-grid crease.

    Returns a dict keyed by (v_a, v_b) -> signed dihedral angle in radians, for
    the interior edges shared by two quad facets (mountain < 0, valley > 0). The
    angle is the signed dihedral between the two averaged quad normals about the
    shared edge (see :func:`_signed_dihedral`); no midpoint/centroid heuristic is
    involved.
    """
    P = P.reshape(R, C, 3)

    def quad_normal(j, i):
        p = [P[j, i], P[j, i + 1], P[j + 1, i + 1], P[j + 1, i]]
        n = np.cross(p[1] - p[0], p[2] - p[0]) + np.cross(p[2] - p[0], p[3] - p[0])
        nn = np.linalg.norm(n)
        return n / nn if nn > 1e-12 else n

    signs = {}
    idx = lambda j, i: j * C + i
    for j in range(R - 1):
        for i in range(C):
            if 0 < i < C - 1:
                # vertical edge (j,i)-(j+1,i) shared by quads (j,i-1) and (j,i)
                na = quad_normal(j, i - 1); nb = quad_normal(j, i)
                axis = P[j + 1, i] - P[j, i]
                signs[(idx(j, i), idx(j + 1, i))] = _signed_dihedral(na, nb, axis)
    for j in range(1, R - 1):
        for i in range(C - 1):
            # horizontal edge (j,i)-(j,i+1) shared by quads (j-1,i) and (j,i)
            na = quad_normal(j - 1, i); nb = quad_normal(j, i)
            axis = P[j, i + 1] - P[j, i]
            signs[(idx(j, i), idx(j, i + 1))] = _signed_dihedral(na, nb, axis)
    return signs


def _diagonal_dihedral_signs(P, R, C):
    """Signed fold angle for each triangulating facet diagonal.

    Every Miura quad ``(j,i)`` is split by the diagonal ``(j,i)-(j+1,i+1)`` into
    triangles ``[(j,i),(j,i+1),(j+1,i+1)]`` and ``[(j,i),(j+1,i+1),(j+1,i)]``.
    The diagonal is the edge those two triangles share, so its fold angle is the
    signed dihedral between their normals (:func:`_signed_dihedral`, same
    mountain < 0 / valley > 0 convention as the quad creases). A near-planar quad
    gives ~0 -> the diagonal exports as a flat facet crease. Returns a dict keyed
    by (v_a, v_c) -> angle in radians.
    """
    P = P.reshape(R, C, 3)
    idx = lambda j, i: j * C + i
    signs = {}
    for j in range(R - 1):
        for i in range(C - 1):
            a = P[j, i]; b = P[j, i + 1]; c = P[j + 1, i + 1]; d = P[j + 1, i]
            n1 = np.cross(b - a, c - a)
            n2 = np.cross(c - a, d - a)
            for n in (n1, n2):
                nn = np.linalg.norm(n)
                if nn > 1e-12:
                    n /= nn
            signs[(idx(j, i), idx(j + 1, i + 1))] = _signed_dihedral(n1, n2, c - a)
    return signs


def _build_pattern(Q, P, R, C, length, width):
    """Assemble the flat CreasePattern.

    The exported creases are the horizontal / vertical Miura folds *and* the
    triangulating facet diagonals, each with a signed M/V/F assignment from the
    folded dihedral. The exported faces are the triangles those diagonals cut, so
    every face in the FOLD file is planar by construction (the fix for the old
    quad export, whose folded quads bowed out of plane). The diagonal edge
    indices are recorded in ``metadata["facet_edges"]`` so the SVG/DXF exporters
    can put them on a distinct facet layer.
    """
    Q = Q.reshape(R, C, 3)
    # normalise the flat layout to a clean rectangle-ish footprint sized to
    # (length, width) so SVG/FOLD output and the studio see sensible units
    xy = Q[..., :2].reshape(-1, 2).copy()
    lo = xy.min(0); span = xy.max(0) - lo
    scale = np.array([length, width]) / np.maximum(span, 1e-9)
    verts = (xy - lo) * scale

    idx = lambda j, i: j * C + i
    signs = _quad_dihedral_signs(P, R, C)
    diag_signs = _diagonal_dihedral_signs(P, R, C)
    edges, assign, fold_angle, facet_edges = [], [], [], []

    def add(va, vb, kind, ang_deg, facet=False):
        if facet:
            facet_edges.append(len(edges))
        edges.append((va, vb)); assign.append(kind); fold_angle.append(ang_deg)

    thresh = np.radians(3.0)                         # near-flat creases -> flat
    for j in range(R):
        for i in range(C):
            if i + 1 < C:                            # horizontal crease
                key = (idx(j, i), idx(j, i + 1))
                if j == 0 or j == R - 1:
                    add(*key, "B", 0.0)
                else:
                    ang = signs.get(key, 0.0)
                    kind = "F" if abs(ang) < thresh else ("M" if ang < 0 else "V")
                    add(*key, kind, float(np.degrees(ang)))
            if j + 1 < R:                            # vertical crease
                key = (idx(j, i), idx(j + 1, i))
                if i == 0 or i == C - 1:
                    add(*key, "B", 0.0)
                else:
                    ang = signs.get(key, 0.0)
                    kind = "F" if abs(ang) < thresh else ("M" if ang < 0 else "V")
                    add(*key, kind, float(np.degrees(ang)))
            if i + 1 < C and j + 1 < R:              # triangulating facet diagonal
                key = (idx(j, i), idx(j + 1, i + 1))
                ang = diag_signs.get(key, 0.0)
                kind = "F" if abs(ang) < thresh else ("M" if ang < 0 else "V")
                add(*key, kind, float(np.degrees(ang)), facet=True)
    # faces are the triangles cut by those diagonals (planar by construction)
    faces = _triangles(R, C).tolist()
    return CreasePattern(vertices=verts, edges=np.array(edges), assignment=assign,
                         faces=faces, fold_angle=np.array(fold_angle),
                         metadata={"name": "miura2d", "facet_edges": facet_edges})


def origamize_miura(Z, length: float = 24.0, width: float = 24.0,
                    height: float = 6.0, relief: float = 1.6, iters: int = 600,
                    wiso: float = 4.0, wfit: float = 1.0, wanc: float = 0.04,
                    lr: float = 0.02) -> Miura2DResult:
    """Origamize a height field ``Z`` (``(R, C)``) with a warped 2D Miura fit.

    Returns a :class:`Miura2DResult` (an :class:`OrigamiResult` plus the flat
    pattern, grid shape, and fold-strain diagnostics). ``relief`` sets the
    corrugation amplitude in cell-heights (more relief => more developable slack
    to absorb curvature, at the cost of bolder pleats); ``height`` scales the
    folded output's z for display/export. ``error`` is the mid-surface
    :func:`surface_fit_error` against ``Z``.

    The reported ``max_strain`` is typically below ~5%, but climbs higher at
    extreme aspect ratios or curvature peaks; a warning is emitted when it does.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or min(Z.shape) < 3:
        raise ValueError(
            f"height field must be at least 3x3 to fold a 2D Miura; got {Z.shape}")
    if not np.isfinite(Z).all():
        raise ValueError(
            "height field must be finite to fold a 2D Miura; got NaN or Inf values")
    R, C = Z.shape
    Q, P, strain = _fit_warped_miura(
        Z, relief=relief, iters=iters, wiso=wiso, wfit=wfit, wanc=wanc, lr=lr)

    # scale the folded surface to the requested physical footprint + height
    Pg = P.reshape(R, C, 3).copy()
    xy = Pg[..., :2]; lo = xy.min((0, 1)); span = xy.max((0, 1)) - lo
    Pg[..., 0] = (Pg[..., 0] - lo[0]) / (span[0] + 1e-9) * length
    Pg[..., 1] = (Pg[..., 1] - lo[1]) / (span[1] + 1e-9) * width
    z = Pg[..., 2]; Pg[..., 2] = (z - z.min()) / (z.max() - z.min() + 1e-9) * height

    pattern = _build_pattern(Q, P, R, C, length, width)
    folded = Pg.reshape(-1, 3)
    tris = _triangles(R, C)
    err = surface_fit_error(Pg, Z, use_midsurface=True)

    max_strain = float(strain.max())
    if max_strain > 0.05:
        import warnings
        warnings.warn(
            f"warped-Miura fit peaks at {max_strain * 100:.1f}% fold strain "
            "(>5%): expected at extreme aspect ratios or curvature peaks",
            stacklevel=2)

    target = np.zeros((R, C, 3))
    xs = np.linspace(0, length, C); ys = np.linspace(0, width, R)
    Zn = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9) * height
    target[..., 0] = xs[None, :]; target[..., 1] = ys[:, None]; target[..., 2] = Zn
    return Miura2DResult(
        pattern=pattern, folded=folded, angles=np.array([]),
        target=target.reshape(-1, 3), error=err, triangles=tris,
        grid_shape=(R, C), flat=Q.reshape(-1, 3)[:, :2].copy(),
        mean_strain=float(strain.mean()), max_strain=max_strain)


def compare_engines(Z, length: float = 24.0, width: float = 24.0,
                    height: float = 6.0, **miura_kw) -> dict:
    """Score corrugation vs the 2D Miura engine on one height field ``Z``.

    Returns a dict with the mid-surface :func:`surface_fit_error` of the
    corrugation engine's coherent single-sheet fold (its returned pattern is an
    extrusion) and of the 2D Miura fit, plus the 2D fold strain. Both errors use
    the identical metric, so the comparison is apples-to-apples.
    """
    from foldforge.origamize.surface import origamize_heightfield
    Z = np.asarray(Z, dtype=float)
    R, C = Z.shape
    corr = origamize_heightfield(Z, length=length, width=width)
    corr_grid = corrugation_surface(corr, (R, C))
    mir = origamize_miura(Z, length=length, width=width, height=height, **miura_kw)
    return {
        "corrugation_error": surface_fit_error(corr_grid, Z),
        "miura2d_error": mir.error,
        "miura2d_mean_strain": mir.mean_strain,
        "miura2d_max_strain": mir.max_strain,
    }
