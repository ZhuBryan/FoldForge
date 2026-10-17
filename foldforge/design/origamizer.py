"""Origamizer-style tuck-folding: fold a flat sheet into a target polyhedron.

This is a *scoped* implementation of Tomohiro Tachi's Origamizer principle
(Tachi 2017, "Origamizer: A Practical Algorithm for Folding Any Polyhedron"):

    Lay the faces of a target polyhedral surface out on a flat sheet, each one
    congruent (isometric) to its 3D counterpart.  Adjacent faces in 3D are
    separated by *gaps* on the flat sheet.  The excess paper in each gap is
    folded into a TUCK -- a hidden pleat that disappears behind the surface
    while pulling the two faces together at the correct dihedral angle.  The
    folded result reconstructs the target exactly; the paper is used
    isometrically (no stretch); the tucks are flat, hidden material.

What THIS module actually does (read ``origamize_polyhedron`` and the honest
scope notes at the bottom of this file before trusting it):

  * Targets: a built-in ``square_pyramid`` and ``tetrahedron`` -- convex solids
    whose every side face shares exactly one edge with a chosen base face
    (fan / pyramid-type nets).  That restriction is the scope, not a bug.
  * Face layout: a *net with gaps*.  The base face stays flat; each side face
    is unfolded into the plane next to its base edge, then pushed outward by a
    gap of width ``gap``.  Every face is placed congruent to its 3D shape.
  * Tuck gadget: the gap strip between a base edge and a side face is a
    three-crease parallel pleat (valley / flat-fold / valley).  Folding it
    flattens the pleat behind the surface and swings the side face up to its
    exact target dihedral.  The pleat is genuinely isometric and achieves an
    *arbitrary* dihedral (see ``_tuck_fold_angles``).

The headline it demonstrates: a single flat crease pattern folds into a 3D
*polyhedron* (not a relief, not a uniaxial base) -- faces land at their target
positions with error ~0, dihedral angles are exact, the paper is isometric,
and the excess is tucked out of sight.

API:
    square_pyramid(a, h) -> PolyTarget
    tetrahedron(edge)    -> PolyTarget
    origamize_polyhedron(target, gap=..., tuck_dir_deg=...) -> OrigamizerResult
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from foldforge.geometry.crease_graph import CreasePattern

_Z = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
#  Target polyhedra
# --------------------------------------------------------------------------- #
@dataclass
class PolyTarget:
    """A target polyhedral surface for origamizing.

    vertices: (N, 3) float; faces: list of vertex-index loops (CCW seen from
    outside).  ``base`` is the index into ``faces`` of the face that stays flat
    on the sheet; every other face must share exactly one edge with it.
    """

    name: str
    vertices: np.ndarray
    faces: list[list[int]]
    base: int = 0

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float)


def square_pyramid(a: float = 2.0, h: float = 1.5) -> PolyTarget:
    """Square base (side ``a``, in z=0, centred) with apex at height ``h``."""
    s = a / 2.0
    verts = np.array(
        [[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0], [0, 0, h]], dtype=float
    )
    faces = [[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    return PolyTarget("square_pyramid", verts, faces, base=0)


def tetrahedron(edge: float = 2.0) -> PolyTarget:
    """Regular tetrahedron with its base triangle lying in z=0."""
    # equilateral base triangle centred at origin
    r = edge / np.sqrt(3.0)  # circumradius of base
    base = np.array(
        [[r * np.cos(t), r * np.sin(t), 0.0] for t in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3)]
    )
    h = np.sqrt(2.0 / 3.0) * edge  # regular-tetra height
    apex = np.array([0.0, 0.0, h])
    verts = np.vstack([base, apex])
    faces = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
    return PolyTarget("tetrahedron", verts, faces, base=0)


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def _rot_axis(axis: np.ndarray, ang: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    x, y, z = a
    c, s, C = np.cos(ang), np.sin(ang), 1 - np.cos(ang)
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _rot_about_line(p, c, axis, ang):
    """Rotate point(s) ``p`` about the line through ``c`` with direction ``axis``."""
    p = np.asarray(p, dtype=float)
    return c + (p - c) @ _rot_axis(axis, ang).T


def _dihedral(p_edge_a, p_edge_b, p_base_far, p_flap_far):
    """Dihedral angle (deg) between two faces sharing edge (a,b).

    Measured as the angle between the in-face directions perpendicular to the
    shared edge, each pointing toward the given far point.  Used identically for
    the target and the folded result so the comparison is apples-to-apples.
    """
    a = np.asarray(p_edge_a, float)
    u = np.asarray(p_edge_b, float) - a
    u /= np.linalg.norm(u)
    mid = (a + np.asarray(p_edge_b, float)) / 2.0   # anchor at the shared-edge midpoint

    def perp(far):
        w = np.asarray(far, float) - mid
        w = w - np.dot(w, u) * u
        return w / np.linalg.norm(w)

    wb, wf = perp(p_base_far), perp(p_flap_far)
    return np.degrees(np.arccos(np.clip(np.dot(wb, wf), -1.0, 1.0)))


def _norm(ang_deg: float) -> float:
    """Normalise a fold angle to (-180, 180]."""
    a = (ang_deg + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


# --------------------------------------------------------------------------- #
#  The tuck gadget
# --------------------------------------------------------------------------- #
def _tuck_fold_angles(psi: float, tuck_dir: float):
    """Fold angles (radians) for the three-crease pleat that spans one gap.

    The pleat lives in a 2D cross-section (outward-normal x up).  Flat, the
    panels extend in the +normal direction (angle 0).  The chain is

        base --c_A--> P1 --c_mid--> P2 --c_B--> side-face

    with cumulative panel angles theta1, theta1+theta2, theta1+theta2+theta3.

      * c_mid must flat-fold (theta2 = 180 deg) so the two gap panels return to
        the base edge -- that is what makes the tucked edge coincide with the
        base edge, so the side face hinges at the right line.
      * theta1 = tuck_dir chooses which way the hidden pleat points (270 deg =
        straight down, behind a convex surface).
      * theta3 is then fixed so the side face reaches its target rise ``psi``.

    The net turn theta1+theta2+theta3 = psi reproduces the dihedral exactly for
    ANY psi, which is why a flat pleat can hit an arbitrary dihedral.
    """
    th1 = tuck_dir
    th2 = np.pi
    th3 = psi - th1 - th2
    return th1, th2, th3


# --------------------------------------------------------------------------- #
#  Reports / results
# --------------------------------------------------------------------------- #
@dataclass
class OrigamizerReport:
    target: str
    n_faces: int
    n_tucks: int
    face_position_error: float       # max over all folded surface vertices vs target
    dihedral_error_deg: float        # max over tucked edges
    isometry_error: float            # max |folded - flat| edge length over the whole pattern
    area_flat: float                 # total paper area in the flat pattern
    area_target: float               # area of the target surface
    area_tucked: float               # excess paper hidden in the tucks
    tuck_behind: bool                # every tuck pleat ends up behind the surface

    def __str__(self) -> str:
        return (
            f"Origamizer[{self.target}]: {self.n_faces} faces, {self.n_tucks} tucks\n"
            f"  face-position error : {self.face_position_error:.2e}\n"
            f"  dihedral error      : {self.dihedral_error_deg:.2e} deg\n"
            f"  isometry error      : {self.isometry_error:.2e}\n"
            f"  paper area  flat={self.area_flat:.3f}  target={self.area_target:.3f}"
            f"  tucked={self.area_tucked:.3f}\n"
            f"  tucks hidden behind : {self.tuck_behind}"
        )


@dataclass
class OrigamizerResult:
    pattern: CreasePattern                 # flat crease pattern (M/V/B assigned)
    folded: np.ndarray                     # (V,3) folded coordinates
    target_vertices: np.ndarray            # (N,3) target polyhedron
    target_faces: list[list[int]]
    surface_vertex_map: dict               # pattern-vertex -> target-vertex (for folded faces)
    report: OrigamizerReport
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def origamize_polyhedron(
    target: PolyTarget,
    gap: float = 0.4,
    tuck_dir_deg: float = 270.0,
) -> OrigamizerResult:
    """Lay ``target`` out as a net-with-gaps and build + fold the tuck gadgets.

    Returns the flat crease pattern, its folded 3D coordinates, and a report of
    face-position / dihedral / isometry errors.
    """
    V = target.vertices
    base = target.faces[target.base]
    base_pts = V[base]
    if not np.allclose(base_pts[:, 2], 0.0):
        raise ValueError("base face must lie in the z=0 plane")
    centroid = base_pts.mean(axis=0)

    # map each side face to the base edge it shares
    base_edges = [(base[i], base[(i + 1) % len(base)]) for i in range(len(base))]
    side_faces = [f for k, f in enumerate(target.faces) if k != target.base]

    # ---- accumulate flat vertices with dedup --------------------------------
    verts: list[np.ndarray] = []
    vindex: dict[tuple, int] = {}

    def add(p2):
        key = (round(float(p2[0]), 9), round(float(p2[1]), 9))
        if key not in vindex:
            vindex[key] = len(verts)
            verts.append(np.array([p2[0], p2[1]], float))
        return vindex[key]

    edges: list[tuple[int, int]] = []
    assign: list[str] = []
    fold_angle: list[float] = []
    faces_out: list[list[int]] = []

    # folding bookkeeping: for each flat vertex, which rigid transform lands it
    fold_fn: dict[int, callable] = {}
    surface_vertex_map: dict[int, int] = {}  # pattern vert -> target vert index

    tuck_dir = np.radians(tuck_dir_deg)
    tuck_records = []  # (flap_far_pattern_idx, target_far_idx, psi, edge endpoints 3d, edge pattern idxs)

    # base corners stay put (identity transform)
    base_flat_idx = {}
    for vi in base:
        idx = add(V[vi][:2])
        base_flat_idx[vi] = idx
        fold_fn[idx] = (lambda p: np.array([p[0], p[1], 0.0]))
        surface_vertex_map[idx] = vi
    # base border edges
    for i in range(len(base)):
        a, b = base[i], base[(i + 1) % len(base)]
        # (will be re-tagged below if this edge carries a tuck; base edges are c_A creases)

    for face in side_faces:
        # find shared base edge
        fset = set(face)
        shared = None
        for (ea, eb) in base_edges:
            if ea in fset and eb in fset:
                shared = (ea, eb)
                break
        if shared is None:
            raise ValueError(
                f"face {face} shares no edge with base face -- target outside "
                "scope (needs a pyramid/fan-type net)"
            )
        ea, eb = shared
        far = [v for v in face if v not in shared][0]  # the lifted (apex-side) vertex
        A3, B3, F3 = V[ea], V[eb], V[far]

        # local frame: edge dir u, outward in-plane normal n, axis so axis x n = +z
        A2 = A3[:2].astype(float)
        B2 = B3[:2].astype(float)
        u = B3 - A3
        u = u / np.linalg.norm(u)
        n = ((A3 + B3) / 2.0) - centroid
        n[2] = 0.0
        n = n / np.linalg.norm(n)
        axis = u.copy()
        if np.dot(np.cross(axis, n), _Z) < 0:
            axis = -axis

        # unfold the side triangle into the plane (isometric): foot of F on edge,
        # then place F outward at its true height above the edge line.
        foot = A3 + np.dot(F3 - A3, u) * u
        h_tri = np.linalg.norm(F3 - foot)
        foot2 = foot[:2]
        n2 = n[:2]

        # target rise angle psi (local angle of the real side face)
        vec = F3 - (A3 + B3) / 2.0
        psi = np.arctan2(np.dot(vec, _Z), np.dot(vec, n))

        th1, th2, th3 = _tuck_fold_angles(psi, tuck_dir)

        # flat vertices of this strip + flap
        iA, iB = base_flat_idx[ea], base_flat_idx[eb]      # c_A (base edge)
        iA1 = add(A2 + (gap / 2.0) * n2)
        iB1 = add(B2 + (gap / 2.0) * n2)
        iA2 = add(A2 + gap * n2)                            # c_B (flap base edge)
        iB2 = add(B2 + gap * n2)
        iF = add(foot2 + (gap + h_tri) * n2)               # flap apex (flat)
        surface_vertex_map[iF] = far
        # side-face base corners also land on the target base corners after fold
        surface_vertex_map[iA2] = ea
        surface_vertex_map[iB2] = eb

        # fold transforms: folded = R1(R2(R3(p))), each about the flat crease line
        cA = np.array([A3[0], A3[1], 0.0])
        cM = np.array([A2[0] + gap / 2 * n2[0], A2[1] + gap / 2 * n2[1], 0.0])
        cB = np.array([A2[0] + gap * n2[0], A2[1] + gap * n2[1], 0.0])

        def make_fold(th1=th1, th2=th2, th3=th3, cA=cA, cM=cM, cB=cB, axis=axis):
            def f(p):
                q = np.array([p[0], p[1], 0.0])
                q = _rot_about_line(q, cB, axis, th3)
                q = _rot_about_line(q, cM, axis, th2)
                q = _rot_about_line(q, cA, axis, th1)
                return q
            return f

        tuck_fold = make_fold()
        tuck_only = (lambda p, cA=cA, cM=cM, axis=axis, th1=th1, th2=th2:
                     _rot_about_line(_rot_about_line(np.array([p[0], p[1], 0.0]), cM, axis, th2), cA, axis, th1))
        # panel P1 verts (iA1,iB1) only pass through R1,R2; P2 + flap through all three.
        for vi in (iA2, iB2, iF):
            fold_fn[vi] = tuck_fold
        for vi in (iA1, iB1):
            fold_fn[vi] = tuck_only

        # ---- edges & assignments for this strip ----
        def add_edge(p, q, a, ang=0.0):
            edges.append((p, q))
            assign.append(a)
            fold_angle.append(ang)

        add_edge(iA, iB, "V", _norm(np.degrees(th1)))          # c_A
        add_edge(iA1, iB1, "M", _norm(np.degrees(th2)))        # c_mid (flat fold)
        add_edge(iA2, iB2, "V", _norm(np.degrees(th3)))        # c_B
        # strip side borders
        add_edge(iA, iA1, "B"); add_edge(iA1, iA2, "B")
        add_edge(iB, iB1, "B"); add_edge(iB1, iB2, "B")
        # flap sides (free boundary)
        add_edge(iA2, iF, "B"); add_edge(iB2, iF, "B")

        faces_out.append([iA, iB, iB1, iA1])
        faces_out.append([iA1, iB1, iB2, iA2])
        faces_out.append([iA2, iB2, iF])

        tuck_records.append(
            dict(iF=iF, far=far, ea=ea, eb=eb, iA2=iA2, iB2=iB2, psi=psi,
                 fold=tuck_fold, tuck_only=tuck_only, iA1=iA1)
        )

    # add base face polygon
    faces_out.insert(0, [base_flat_idx[v] for v in base])

    varr = np.array(verts, float)
    pattern = CreasePattern(
        vertices=varr,
        edges=np.array(edges, int),
        assignment=assign,
        faces=faces_out,
        fold_angle=np.array(fold_angle, float),
        metadata={"name": f"origamizer_{target.name}", "gap": gap},
    )

    # ---- fold every vertex -------------------------------------------------
    folded = np.zeros((len(verts), 3))
    for i in range(len(verts)):
        folded[i] = fold_fn[i](verts[i])

    # ---- errors ------------------------------------------------------------
    # face-position: folded surface vertices vs target vertices
    pos_err = 0.0
    for pi, ti in surface_vertex_map.items():
        pos_err = max(pos_err, float(np.linalg.norm(folded[pi] - V[ti])))

    # dihedral error per tuck
    dih_err = 0.0
    tuck_behind = True
    for rec in tuck_records:
        # folded dihedral between base (z=0) and this flap
        d_folded = _dihedral(V[rec["ea"]], V[rec["eb"]], centroid, folded[rec["iF"]])
        d_target = _dihedral(V[rec["ea"]], V[rec["eb"]], centroid, V[rec["far"]])
        dih_err = max(dih_err, abs(d_folded - d_target))
        # tuck hidden? the pleat vertex (iA1) must be behind the surface (z<0 for convex)
        if folded[rec["iA1"]][2] > 1e-9:
            tuck_behind = False

    # isometry: every edge keeps its flat length
    iso_err = 0.0
    for (p, q) in edges:
        flat_len = np.linalg.norm(varr[p] - varr[q])
        fold_len = np.linalg.norm(folded[p] - folded[q])
        iso_err = max(iso_err, abs(flat_len - fold_len))

    # areas
    def poly_area_2d(idx):
        pts = varr[idx]
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def poly_area_3d(idx3):
        pts = V[idx3]
        tot = 0.0
        o = pts[0]
        for i in range(1, len(pts) - 1):
            tot += 0.5 * np.linalg.norm(np.cross(pts[i] - o, pts[i + 1] - o))
        return tot

    area_flat = sum(poly_area_2d(f) for f in faces_out)
    area_target = sum(poly_area_3d(f) for f in target.faces)
    # tucked excess = flat paper that is not target surface (the gap pleats)
    area_tucked = area_flat - area_target

    report = OrigamizerReport(
        target=target.name,
        n_faces=len(target.faces),
        n_tucks=len(tuck_records),
        face_position_error=pos_err,
        dihedral_error_deg=dih_err,
        isometry_error=iso_err,
        area_flat=area_flat,
        area_target=area_target,
        area_tucked=area_tucked,
        tuck_behind=tuck_behind,
    )

    return OrigamizerResult(
        pattern=pattern,
        folded=folded,
        target_vertices=V,
        target_faces=target.faces,
        surface_vertex_map=surface_vertex_map,
        report=report,
        meta={"gap": gap, "tuck_dir_deg": tuck_dir_deg},
    )


# ponytail: scope is two convex fan-nets (square_pyramid, tetrahedron) with a
# flat-fold parallel-pleat tuck per base edge. NOT the general Origamizer:
# no arbitrary meshes, no optimal face packing, no vertex tucking-molecule
# network, no concave solids, no guaranteed non-self-intersection. The general
# problem (Tachi 2017) is open-ended research; this proves the principle on a
# tractable target. Upgrade path noted in the module docstring.


if __name__ == "__main__":
    for tgt in (square_pyramid(), tetrahedron()):
        res = origamize_polyhedron(tgt)
        print(res.report)
        print()
