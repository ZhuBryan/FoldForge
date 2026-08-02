"""Origamize: decompose a 3D target into a foldable crease pattern (capstone).

Model the sheet as parallel rigid fold-chains; each strip's fold angles are the
target cross-section's own turning angles (closed form, exact, instant). Two
modes: ``origamize_profile`` (one extruded cross-section - exactly developable)
and ``origamize_heightfield`` (per-row strips - an approximate corrugation with
the error reported). A corrugation approximation, not full Origamizer tuck-folding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foldforge.geometry.crease_graph import CreasePattern
from foldforge.design.inverse import (
    resample_arclength, chamfer_distance, angles_from_curve,
)
from foldforge.diff.kinematics import fold_chain


# --- hand-foldability: crease budget, count, and difficulty -----------------

# Fold-budget presets -> a `folds` value (cells across the subject's longer
# side) that drives the existing coarseness knob. "hard" keeps the caller's
# current (detailed) resolution. Easy/Medium give a genuinely hand-foldable
# corrugation of tens of creases; it is still a coarse relief/corrugation, NOT
# figurative origami (it will not fold into a crane).
_FOLD_BUDGET = {"easy": 14, "medium": 26, "hard": None}


def budget_folds(level, default=None):
    """Map a fold-budget preset (``"easy"``/``"medium"``/``"hard"``) to a
    ``folds`` value. ``"hard"`` (or ``None``) keeps ``default`` (current
    detail). Unknown levels raise ``ValueError``."""
    if level is None:
        return default
    key = str(level).strip().lower()
    if key not in _FOLD_BUDGET:
        raise ValueError(
            f"unknown fold budget {level!r}; choose easy, medium, or hard")
    f = _FOLD_BUDGET[key]
    return default if f is None else f


def difficulty_label(crease_count: int) -> str:
    """A hand-folding difficulty label from a crease (fold-line) count."""
    if crease_count <= 24:
        return "Easy"
    if crease_count <= 60:
        return "Medium"
    return "Hard (machine)"


def crease_stats(pattern):
    """``(crease_count, difficulty)`` for a crease pattern.

    ``crease_count`` is the number of *distinct* mountain/valley fold lines a
    human would actually crease: collinear M/V edges lying on one infinite line
    are merged (an extruded pleat is a single fold across the sheet, not one per
    panel), and flat/border edges are ignored. The label comes from
    :func:`difficulty_label`. Tens of folds read as hand-foldable; hundreds are
    machine-only. This measures a coarse corrugation's real fold workload; it is
    not a claim of figurative-origami feasibility.
    """
    V = np.asarray(pattern.vertices, dtype=float)[:, :2]
    lines = set()
    for (a, b), k in zip(np.asarray(pattern.edges), pattern.assignment):
        if k not in ("M", "V"):
            continue
        p, q = V[int(a)], V[int(b)]
        d = q - p
        n = float(np.hypot(d[0], d[1]))
        if n < 1e-9:
            continue
        theta = np.arctan2(d[1], d[0]) % np.pi          # line direction mod pi
        nx, ny = -np.sin(theta), np.cos(theta)          # unit normal for offset
        offset = nx * p[0] + ny * p[1]
        lines.add((k, round(theta, 2), round(offset, 1)))
    count = len(lines)
    return count, difficulty_label(count)


@dataclass
class OrigamiResult:
    """A decomposed target: the crease pattern, the folded shape, and the error."""

    pattern: CreasePattern        # flat crease pattern (M/V pleats) with fold_angle
    folded: np.ndarray            # (V, 3) folded vertex positions
    angles: np.ndarray            # fold angle per pleat (radians)
    target: np.ndarray            # the resampled target (profile or grid)
    error: float                  # Chamfer distance, folded vs target
    triangles: object = None      # (T, 3) triangles indexing `folded` (for 3D export)
    solid: object = None          # optional closed (vertices, triangles) watertight mesh

    @property
    def crease_count(self) -> int:
        """Distinct M/V fold lines in the crease pattern (see :func:`crease_stats`)."""
        return crease_stats(self.pattern)[0]

    @property
    def difficulty(self) -> str:
        """Hand-folding difficulty label for the crease pattern."""
        return crease_stats(self.pattern)[1]


def origamize_profile(profile: np.ndarray, n_pleats: int = 28, width: float = 6.0,
                      n_rows: int = 6, iters: int = 900, lr: float = 0.05) -> OrigamiResult:
    """Decompose a cross-section ``profile`` (ordered ``(*, 2)`` x-z points) into
    an extruded, exactly-developable pleated sheet.
    """
    # ponytail: iters/lr kept for API compatibility but unused - the fold angles
    # are the target's own turning angles (closed form, exact and instant).
    target = resample_arclength(np.asarray(profile, dtype=float), n_pleats)
    angles, seg = angles_from_curve(target)
    spine = fold_chain(angles, seg=seg).spine
    spine = spine + (target[0] - spine[0])              # anchor to the target's start
    n = len(spine) - 1

    ys = np.linspace(0, width, n_rows)
    folded = np.array([[spine[i, 0], y, spine[i, 1]] for i in range(n + 1) for y in ys])

    def idx(i, r):
        return i * n_rows + r
    verts = np.array([[i * seg, y] for i in range(n + 1) for y in ys])
    edges, assign, fold_angle = [], [], []
    for i in range(n + 1):                              # vertical lines (pleats / borders)
        for r in range(n_rows - 1):
            edges.append((idx(i, r), idx(i, r + 1)))
            if i in (0, n):
                assign.append("B"); fold_angle.append(0.0)
            else:
                ang = angles[i]
                assign.append("M" if ang < 0 else "V")
                fold_angle.append(float(np.degrees(ang)))
    for r in range(n_rows):                             # horizontal border lines
        for i in range(n):
            edges.append((idx(i, r), idx(i + 1, r)))
            assign.append("B"); fold_angle.append(0.0)
    faces = [[idx(i, r), idx(i + 1, r), idx(i + 1, r + 1), idx(i, r + 1)]
             for i in range(n) for r in range(n_rows - 1)]

    pattern = CreasePattern(vertices=verts, edges=np.array(edges), assignment=assign,
                            faces=faces, fold_angle=np.array(fold_angle),
                            metadata={"name": "origamized_profile"})
    tris = np.array([t for q in faces for t in ([q[0], q[1], q[2]], [q[0], q[2], q[3]])])
    err = chamfer_distance(spine, target)
    return OrigamiResult(pattern, folded, angles, target, err, triangles=tris)


def origamize_heightfield(Z: np.ndarray, length: float = 24.0, width: float = 24.0,
                          iters: int = 600, lr: float = 0.05) -> OrigamiResult:
    """Decompose a height field ``Z`` (shape ``(n_rows, n_cols)``) into an
    approximate corrugated surface, origamizing each row independently.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"height field must be 2D (rows, cols); got shape {Z.shape}")
    n_rows, n_cols = Z.shape
    if n_rows < 2 or n_cols < 2:
        raise ValueError(
            f"height field must be at least 2x2 to fold (got {n_rows}x{n_cols}); "
            "a single row or column has no foldable panel")
    xs = np.linspace(0, length, n_cols)
    ys = np.linspace(0, width, n_rows)
    n_pleats = n_cols - 1
    folded = np.zeros((n_rows, n_cols, 3))
    target3d = np.zeros((n_rows, n_cols, 3))
    all_ang = []
    for j in range(n_rows):
        prof = np.stack([xs, Z[j]], axis=1)
        tgt = resample_arclength(prof, n_pleats)
        angles, seg = angles_from_curve(tgt)
        spine = fold_chain(angles, seg=seg).spine
        spine = spine + (tgt[0] - spine[0])
        all_ang.append(angles)
        folded[j, :, 0] = spine[:, 0]; folded[j, :, 1] = ys[j]; folded[j, :, 2] = spine[:, 1]
        target3d[j, :, 0] = tgt[:, 0]; target3d[j, :, 1] = ys[j]; target3d[j, :, 2] = tgt[:, 1]
    fold = folded.reshape(-1, 3)
    tgt = target3d.reshape(-1, 3)
    err = float(np.sqrt(((fold - tgt) ** 2).sum(axis=1)).mean())
    rep = origamize_profile(np.stack([xs, Z[n_rows // 2]], axis=1), n_pleats=n_pleats,
                            width=width, n_rows=n_rows)
    tris = []
    for j in range(n_rows - 1):
        for i in range(n_cols - 1):
            a = j * n_cols + i; b = a + 1
            c = (j + 1) * n_cols + i + 1; d = (j + 1) * n_cols + i
            tris += [[a, b, c], [a, c, d]]
    return OrigamiResult(rep.pattern, fold, np.array(all_ang), tgt, err,
                         triangles=np.array(tris))


def _signed_volume(vertices, triangles) -> float:
    """Signed volume of a triangle mesh (positive when outward-facing)."""
    v = np.asarray(vertices, dtype=float)
    t = np.asarray(triangles, dtype=int)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def close_relief(result, mode: str = "mirror", base: float = 0.0,
                 weld_tol: float = 1e-9):
    """Close a one-sided folded relief into a watertight, printable solid.

    The photo -> origami pipeline yields an *open* relief: a single folded sheet
    with a hollow back (only the camera-facing side of the subject exists). This
    mirrors that front sheet into a back sheet and stitches the two along their
    shared boundary, producing a closed, two-sheet shell - a hollow "inflated"
    model you can actually 3D-print.

    Honesty note: a photo shows only one side, so the back is a *mirrored
    estimate*, not measured geometry. The result is therefore a two-sheet folded
    model, not single-sheet origami.

    Parameters
    ----------
    result : OrigamiResult
        A folded relief; uses ``result.folded`` (V, 3) and ``result.triangles``.
    mode : {"mirror", "flat"}
        ``"mirror"`` reflects the front through the ``z=0`` plane (a puffed,
        symmetric back); ``"flat"`` lays the back flat at ``z=base`` (a
        relief-on-a-slab / plaque back).
    base : float
        The z-plane of the back sheet in ``"flat"`` mode.
    weld_tol : float
        Boundary vertices whose front and back copies land within this distance
        are welded (shared) rather than duplicated, so a boundary already lying
        on the mirror plane closes seamlessly with no side wall there.

    Returns
    -------
    (vertices, triangles)
        ``vertices`` (Nv, 3) and ``triangles`` (Nt, 3) of a closed mesh with
        consistent outward winding (signed volume > 0): the front sheet keeps
        its upward-facing winding, the back sheet is flipped, and the boundary
        ring is stitched with quads split into triangles. Feed straight to
        :func:`foldforge.fabricate.to_stl` / :func:`~foldforge.fabricate.to_gltf`.
    """
    from collections import defaultdict

    V = np.asarray(result.folded, dtype=float)
    T = result.triangles
    if T is None or np.asarray(T).size == 0:
        raise ValueError("result has no triangles to close (need result.triangles)")
    T = np.asarray(T, dtype=int)
    n = len(V)

    # --- back sheet positions ---
    Vb = V.copy()
    if mode == "mirror":
        Vb[:, 2] = -Vb[:, 2]
    elif mode == "flat":
        Vb[:, 2] = base
    else:
        raise ValueError(f"unknown mode {mode!r}; choose 'mirror' or 'flat'")

    # --- boundary loop: directed half-edges present in exactly one triangle ---
    dir_count: dict = defaultdict(int)
    for a, b, c in T:
        a, b, c = int(a), int(b), int(c)
        dir_count[(a, b)] += 1
        dir_count[(b, c)] += 1
        dir_count[(c, a)] += 1
    boundary = [e for e in dir_count if dir_count.get((e[1], e[0]), 0) == 0]
    boundary_verts = {v for e in boundary for v in e}

    # --- back index map: weld only *boundary* verts coincident with the front.
    # Interior coincidences must NOT weld (an interior edge would then touch four
    # triangles); a flat interior patch instead becomes a harmless zero-thickness
    # double layer. This also handles degenerate all-zero-z boundary columns:
    # such rim verts weld, so a flat rim closes with no wasted side wall.
    weld = np.zeros(n, dtype=bool)
    for i in boundary_verts:
        if abs(V[i, 2] - Vb[i, 2]) <= weld_tol:
            weld[i] = True
    # A boundary vertex must NOT weld if it shares an *interior* front edge with
    # another welded vertex (corner cells join two rim verts by a diagonal): the
    # welded flat triangle would otherwise duplicate that edge onto 4 triangles.
    boundary_edge = {frozenset((int(u), int(v))) for u, v in boundary}
    conflict = set()
    for a, b, c in T:
        for x, y in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            if weld[x] and weld[y] and frozenset((x, y)) not in boundary_edge:
                conflict.add(x); conflict.add(y)
    for i in conflict:
        weld[i] = False
    used_back = [i for i in range(n) if not weld[i]]        # back copies we keep
    remap = {i: n + k for k, i in enumerate(used_back)}
    back_idx = np.array([i if weld[i] else remap[i] for i in range(n)], dtype=int)
    Vb_used = Vb[used_back] if used_back else np.empty((0, 3))
    vertices = np.vstack([V, Vb_used])

    # --- front sheet (as-is) + back sheet (reversed winding) ---
    tris = [[int(a), int(b), int(c)] for a, b, c in T]
    for a, b, c in T:
        tris.append([int(back_idx[c]), int(back_idx[b]), int(back_idx[a])])

    # --- stitch the boundary ring with quads (drop degenerate triangles).
    # Quad [v, u, bu, bv] traverses edge {u, v} as (v, u) - opposite the front
    # triangle's (u, v) - so the wall is orientation-consistent with the shells.
    for u, v in boundary:
        bu, bv = int(back_idx[u]), int(back_idx[v])
        for a, b, c in (((v, u, bu)), ((v, bu, bv))):
            if a != b and b != c and a != c:
                tris.append([int(a), int(b), int(c)])

    tris = np.asarray(tris, dtype=int)
    if _signed_volume(vertices, tris) < 0:                 # guarantee outward
        tris = tris[:, ::-1]
    return vertices, tris


def trim_background_triangles(triangles, keep_vertex):
    """Drop triangles lying entirely in a silhouette relief's background.

    ``keep_vertex`` is a boolean array over the relief's vertices (True on the
    subject). A triangle survives if it touches at least one subject vertex, so
    the mesh shrinks to the subject silhouette plus a one-cell rim. Feeding the
    trimmed relief to :func:`close_relief` gives a solid that hugs the subject
    instead of carrying a wide flat background skirt (the two sheets meet along
    the silhouette, leaving only a thin rim).
    """
    T = np.asarray(triangles, dtype=int)
    keep = np.asarray(keep_vertex, dtype=bool)
    survive = keep[T].any(axis=1)
    return T[survive]

# --- target shape library ---------------------------------------------------

def profile_dome(n: int = 200, length: float = 24.0, height: float = 6.0) -> np.ndarray:
    """A semicircular dome cross-section."""
    t = np.linspace(0, np.pi, n)
    r = length / np.pi
    return np.stack([r * (1 - np.cos(t)), height * np.sin(t)], axis=1)


def profile_ridge(n: int = 200, length: float = 24.0, height: float = 5.0) -> np.ndarray:
    """A triangular ridge cross-section."""
    x = np.linspace(0, length, n)
    z = height * (1 - np.abs(2 * x / length - 1))
    return np.stack([x, z], axis=1)


def heightfield_dome(nx: int = 22, ny: int = 18, height: float = 6.0) -> np.ndarray:
    x = np.linspace(-1, 1, nx); y = np.linspace(-1, 1, ny)
    Y, X = np.meshgrid(y, x, indexing="ij")
    return height * np.clip(1 - (X ** 2 + Y ** 2), 0, None) ** 0.5


def heightfield_saddle(nx: int = 22, ny: int = 18, height: float = 4.0) -> np.ndarray:
    x = np.linspace(-1, 1, nx); y = np.linspace(-1, 1, ny)
    Y, X = np.meshgrid(y, x, indexing="ij")
    return height * (X ** 2 - Y ** 2)


def heightfield_ripple(nx: int = 26, ny: int = 20, height: float = 2.2) -> np.ndarray:
    x = np.linspace(-1, 1, nx); y = np.linspace(-1, 1, ny)
    Y, X = np.meshgrid(y, x, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2)
    return height * np.cos(4 * np.pi * R) * np.exp(-1.5 * R)
