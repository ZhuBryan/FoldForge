"""Sheet-metal bending-sequence planner: find an order to make the bends.

A press brake forms a sheet-metal part one straight bend at a time. The *order*
matters: once a flap is bent up it becomes an obstacle, and a later bend may no
longer be reachable - its bend line may have been lifted off the table by an
earlier bend (so the die can no longer seat it), or an over-bent flap may now
hang over it (so the descending punch would crash into formed material). This is
the same "folding is ordered" problem :mod:`foldforge.sim.sequencing` solves for
origami creases, specialised to manufacturing: bends have fixed target angles
(usually +/-90 deg) and the extra constraint is *tool access*.

What this reuses from FoldForge:

  * the triangle-triangle intersection test from
    :mod:`foldforge.sim.collision` (``_tris_intersect`` / ``self_intersections``)
    is the collision backbone - we check the simplified press-brake tool against
    already-formed panels, and the final formed part against itself, with exactly
    the detector the origami simulator uses;
  * the planner is a collision-scored search over bend orders, modelled on
    ``fold_sequence``'s ordering search. Because a real part has only a handful
    of bends, we run it *exhaustively* over permutations (a dedicated search,
    deterministic) rather than a beam - correct and simple at this scale.

Honest about the model (this is a demonstration, not a production CAM tool):

  * **Rigid kinematics, not the relaxation solver.** A sheet-metal bend is an
    exact rigid rotation of the flap about its bend line, so we fold with exact
    Rodrigues rotations composed down the panel tree - simpler and more accurate
    here than the physics solver, whose ``FULL_FOLD_ANGLE`` deliberately stops
    short of tight folds.
  * **Simplified tool.** The press brake is modelled as a thin vertical slab
    (a box of triangles) rising from the bend line - the volume the punch/upper
    tool needs clear above the die. It is NOT real punch/die geometry, back-gauge
    reach, flange-length or bend-radius limits. A bend is deemed *reachable* when
    (a) its bend line still lies in the table plane (an earlier bend has not
    lifted it) and (b) that slab does not intersect any already-formed panel.

The headline: a part whose naive (bend-line-index) order hits tool collisions
that the planned order avoids entirely. See :func:`plan_bend_order`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from foldforge.sim.collision import _tris_intersect, self_intersections

_Z = np.array([0.0, 0.0, 1.0])
_EPS = 1e-9


# --- geometry helpers -------------------------------------------------------

def _rotate_about_axis(points, a0, axis_dir, angle):
    """Rodrigues rotation of ``points`` about the line through ``a0``."""
    k = axis_dir / (np.linalg.norm(axis_dir) + _EPS)
    p = points - a0
    c, s = np.cos(angle), np.sin(angle)
    rotated = (p * c
               + np.cross(k, p) * s
               + np.outer(p @ k, k) * (1.0 - c))
    return rotated + a0


def _rect_tris(bl, br, tr, tl):
    """Two triangles for a quad with corners (bottom-left, br, tr, tl)."""
    return [(bl, br, tr), (bl, tr, tl)]


# --- part model -------------------------------------------------------------

@dataclass
class Bend:
    """One straight bend line connecting a parent panel to a child flap.

    Attributes:
        axis:   the two vertex indices of the bend line (the fold axis).
        angle:  signed bend angle in radians (sign already resolved so the flap
                folds the intended way).
        child:  panel id of the moving flap on the far side of the line.
        name:   human label (e.g. ``"L1"``).
    """

    axis: tuple[int, int]
    angle: float
    child: int
    name: str


@dataclass
class SheetMetalPart:
    """A flat blank partitioned by bend lines into a tree of panels.

    Panel 0 is the base (held on the die, never moves). Each :class:`Bend`
    attaches a child panel to its parent; the panels form a tree rooted at the
    base. ``vertices`` are flat (z=0); folding rotates each child subtree about
    its bend line.
    """

    name: str
    vertices: np.ndarray            # (V, 3) flat blank, z = 0
    triangles: np.ndarray           # (T, 3) vertex indices
    tri_panel: np.ndarray           # (T,) panel id per triangle
    parent: list[int]               # panel -> parent panel (-1 for the base)
    bends: list[Bend]
    _subtree: dict = field(default_factory=dict, repr=False)
    _order: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float)
        self.triangles = np.asarray(self.triangles, dtype=int)
        self.tri_panel = np.asarray(self.tri_panel, dtype=int)
        n_panels = int(self.tri_panel.max()) + 1
        # Panel -> its descendant panels (inclusive); depth for fold order.
        children = {p: [] for p in range(n_panels)}
        for p in range(n_panels):
            if self.parent[p] >= 0:
                children[self.parent[p]].append(p)

        def descendants(p):
            out = {p}
            for c in children[p]:
                out |= descendants(c)
            return out

        self._subtree = {p: descendants(p) for p in range(n_panels)}
        depth = [0] * n_panels
        for p in range(n_panels):
            d, q = 0, p
            while self.parent[q] >= 0:
                q = self.parent[q]
                d += 1
            depth[p] = d
        # Fold the bends nearest the base first (so a child's axis is already
        # positioned by its parent's rotation when we come to it).
        self._order = sorted(range(len(self.bends)),
                             key=lambda b: depth[self.bends[b].child])

    @property
    def n_panels(self) -> int:
        return len(self._subtree)

    def subtree(self, panel: int) -> set:
        return self._subtree[panel]

    def fold_state(self, active) -> np.ndarray:
        """Flat-blank vertices folded with the bends in ``active`` at full angle."""
        active = set(active)
        x = self.vertices.copy()
        for bi in self._order:
            if bi not in active:
                continue
            b = self.bends[bi]
            a0, a1 = x[b.axis[0]], x[b.axis[1]]
            # Rotate every vertex owned by a panel in the child's subtree.
            sub = self.subtree(b.child)
            vmask = self._panel_vertex_mask(sub)
            x[vmask] = _rotate_about_axis(x[vmask], a0, a1 - a0, b.angle)
        return x

    def _panel_vertex_mask(self, panels):
        """Boolean mask of vertices that belong only-or-deepest to ``panels``.

        A vertex on a bend line is shared by parent and child; we assign it to
        the child (deepest) panel, so rotating a subtree never double-moves a
        shared axis vertex (and axis vertices lie on the rotation line anyway).
        """
        if not hasattr(self, "_vert_panel"):
            vp = np.full(len(self.vertices), -1)
            # deepest panel wins: process triangles, keep max panel id per vertex
            for ti, tri in enumerate(self.triangles):
                for v in tri:
                    vp[v] = max(vp[v], self.tri_panel[ti])
            self._vert_panel = vp
        return np.isin(self._vert_panel, list(panels))


# --- simplified press-brake tool model --------------------------------------

def _tool_slab(q0, q1, height, thick, seat, inset=0.0):
    """Triangles of a thin vertical slab rising above bend line ``q0``-``q1``.

    The slab is the clearance volume the punch/upper tool needs above the die:
    the width of the bend line, ``thick`` across it, from just above the line
    (``seat``) up to ``height``. The punch is modelled slightly narrower than
    the flange (``inset`` pulls each end in along the line), so two walls that
    merely share a corner are not flagged as a crash. Returns 3x3 triangles.
    """
    u = q1 - q0
    length = np.linalg.norm(u)
    u = u / (length + _EPS)
    q0 = q0 + u * inset * length
    q1 = q1 - u * inset * length
    w = np.cross(u, _Z)
    nw = np.linalg.norm(w)
    if nw < 1e-6:                       # bend line is vertical; pick any horizontal
        w = np.cross(u, np.array([1.0, 0.0, 0.0]))
        nw = np.linalg.norm(w)
    w = w / (nw + _EPS)
    lift = _Z * seat
    # bottom rectangle corners (just above the line), then the same up high
    b0 = q0 - w * thick / 2 + lift
    b1 = q1 - w * thick / 2 + lift
    b2 = q1 + w * thick / 2 + lift
    b3 = q0 + w * thick / 2 + lift
    top = _Z * height
    t0, t1, t2, t3 = b0 + top, b1 + top, b2 + top, b3 + top
    tris = []
    tris += _rect_tris(b0, b1, b2, b3)          # bottom
    tris += _rect_tris(t0, t1, t2, t3)          # top
    tris += _rect_tris(b0, b1, t1, t0)          # four sides
    tris += _rect_tris(b1, b2, t2, t1)
    tris += _rect_tris(b2, b3, t3, t2)
    tris += _rect_tris(b3, b0, t0, t3)
    return [np.array(t) for t in tris]


def _panel_tris(part, panels, positions):
    """3x3 triangle corner arrays for every triangle in ``panels``."""
    out = []
    for ti, tri in enumerate(part.triangles):
        if part.tri_panel[ti] in panels:
            out.append(positions[tri])
    return out


def bend_interference(part: SheetMetalPart, bend_idx: int, active,
                      *, tool_height=None, tool_thick=None, seat=None,
                      seat_tol=1e-6):
    """Tool-access interference count for bending ``bend_idx`` after ``active``.

    Folds the part with the already-formed bends ``active``, then measures the
    simplified tool against the rest of the workpiece:

      * +1 if the bend line has been lifted out of the table plane (an earlier
        bend moved it, so the die can no longer seat it);
      * +N for the number of tool-slab / formed-panel triangle pairs that
        intersect (the punch would crash into already-formed material).

    0 means the bend is reachable in this state.
    """
    active = set(active)
    size = float(np.linalg.norm(part.vertices.max(0) - part.vertices.min(0)))
    tool_height = tool_height or 1.2 * size
    tool_thick = tool_thick or 0.01 * size
    seat = seat if seat is not None else 0.01 * size

    pos = part.fold_state(active)
    b = part.bends[bend_idx]
    q0, q1 = pos[b.axis[0]], pos[b.axis[1]]

    seat_violation = int(abs(q0[2]) > seat_tol or abs(q1[2]) > seat_tol)

    moving = part.subtree(b.child)
    obstacles = set(range(part.n_panels)) - moving
    slab = _tool_slab(q0, q1, tool_height, tool_thick, seat, inset=0.03)
    obs = _panel_tris(part, obstacles, pos)
    hits = sum(1 for s in slab for o in obs if _tris_intersect(s, o))
    return seat_violation + hits


# --- the planner ------------------------------------------------------------

@dataclass
class BendPlan:
    """Result of :func:`plan_bend_order`.

    Attributes:
        part:            the planned :class:`SheetMetalPart`.
        order:           bend indices in the chosen bending order.
        reachable:       per-step reachability (tool clear when the bend is made).
        interference:    per-step tool-access interference count.
        final_collisions: self-intersection count of the fully formed part
                          (the repo's triangle detector; 0 = clean part).
        feasible:        True if every step was reachable and the part is clean.
    """

    part: SheetMetalPart
    order: list
    reachable: list
    interference: list
    final_collisions: int

    @property
    def total_interference(self) -> int:
        return int(sum(self.interference))

    @property
    def feasible(self) -> bool:
        return all(self.reachable) and self.final_collisions == 0

    def named_order(self):
        return [self.part.bends[i].name for i in self.order]


def evaluate_order(part: SheetMetalPart, order, **tool_kw):
    """Simulate bending ``part`` in ``order``; return per-step interference etc.

    Returns ``(interference_per_step, reachable_per_step, final_collisions)``.
    """
    active: set = set()
    inter, reach = [], []
    for bi in order:
        n = bend_interference(part, bi, active, **tool_kw)
        inter.append(n)
        reach.append(n == 0)
        active.add(bi)
    final = _final_collisions(part)
    return inter, reach, final


def _final_collisions(part: SheetMetalPart) -> int:
    pos = part.fold_state(set(range(len(part.bends))))
    mesh = SimpleNamespace(triangles=part.triangles)
    return len(self_intersections(mesh, pos))


def plan_bend_order(part: SheetMetalPart, **tool_kw) -> BendPlan:
    """Search bend orders for one where every bend is reachable when it is made.

    A real part has only a handful of bends, so we evaluate every permutation
    (deterministic: ties broken by the lexicographically smallest order) and
    keep the best by ``(total tool interference, final self-collisions, order)``.
    If a fully collision-free order exists it is returned; otherwise the
    least-colliding order is returned and :attr:`BendPlan.feasible` is False.
    """
    n = len(part.bends)
    best = None
    for order in itertools.permutations(range(n)):
        inter, reach, final = evaluate_order(part, order, **tool_kw)
        key = (sum(inter), final, order)
        if best is None or key < best[0]:
            best = (key, order, inter, reach, final)
    _, order, inter, reach, final = best
    return BendPlan(part=part, order=list(order), reachable=reach,
                    interference=inter, final_collisions=final)


def naive_plan(part: SheetMetalPart, **tool_kw) -> BendPlan:
    """Evaluate the naive bend-line-index order (0, 1, 2, ...)."""
    order = tuple(range(len(part.bends)))
    inter, reach, final = evaluate_order(part, order, **tool_kw)
    return BendPlan(part=part, order=list(order), reachable=reach,
                    interference=inter, final_collisions=final)


# --- built-in parts ---------------------------------------------------------

def _profile(name, boundaries, width, bend_specs, root=0):
    """Build an extruded profile: panels along x, parallel bend lines along y.

    ``boundaries`` are the x positions separating panels (len = n_panels+1);
    ``bend_specs`` is one ``(angle_deg, "up"/"down")`` per interior boundary.
    ``root`` is the stationary base panel: panels chain outward from it on each
    side, so ``root=0`` gives a chain (every inner bend lifts the outer ones)
    and a middle ``root`` gives independent flaps on each side. Bend sign is
    resolved so "up" lifts the flap toward +z.
    """
    nb = len(boundaries)
    n_panels = nb - 1
    verts = []
    for x in boundaries:
        verts.append([x, 0.0, 0.0])
        verts.append([x, width, 0.0])
    verts = np.array(verts)

    triangles, tri_panel = [], []
    for p in range(n_panels):           # panel p between boundary p and p+1
        bl, tl = 2 * p, 2 * p + 1
        br, tr = 2 * (p + 1), 2 * (p + 1) + 1
        triangles += [(bl, br, tr), (bl, tr, tl)]
        tri_panel += [p, p]

    # Parent points toward the root; bend at boundary k joins panels k-1 and k,
    # its child is whichever is farther from the root.
    parent = [-1] * n_panels
    for p in range(n_panels):
        if p < root:
            parent[p] = p + 1
        elif p > root:
            parent[p] = p - 1
    bends = []
    for k, (ang_deg, direction) in enumerate(bend_specs, start=1):
        child = k if k > root else k - 1
        bends.append(Bend(axis=(2 * k, 2 * k + 1), angle=np.radians(ang_deg),
                          child=child, name=f"L{k}"))
    part = SheetMetalPart(name=name, vertices=verts,
                          triangles=np.array(triangles),
                          tri_panel=np.array(tri_panel),
                          parent=parent, bends=bends)
    for bi, (ang_deg, direction) in enumerate(bend_specs):
        _fix_sign(part, bi, direction)
    return part


def _fix_sign(part, bend_idx, direction):
    """Flip a bend's sign in place so its flap folds the intended way (+z=up)."""
    b = part.bends[bend_idx]
    pos = part.fold_state({bend_idx})
    sub = list(part.subtree(b.child))
    vmask = part._panel_vertex_mask(sub)
    dz = pos[vmask][:, 2].mean() - part.vertices[vmask][:, 2].mean()
    if (dz > 0) != (direction == "up"):
        part.bends[bend_idx] = Bend(axis=b.axis, angle=-b.angle,
                                    child=b.child, name=b.name)


def _resolve_sign(verts, axis, sub_verts, mag, direction):
    """Pick the bend sign so the flap moves the intended way (+z = up)."""
    a0 = verts[axis[0]]
    a1 = verts[axis[1]]
    pts = verts[sub_verts]
    up = _rotate_about_axis(pts, a0, a1 - a0, mag)[:, 2].mean()
    down = _rotate_about_axis(pts, a0, a1 - a0, -mag)[:, 2].mean()
    want_up = direction == "up"
    pick_pos = (up > down) == want_up
    return mag if pick_pos else -mag


def tray(flap_angle=90.0):
    """A shallow tray: a rectangular base with four walls bent up 90 deg.

    The four bends are independent (no flap is above another's bend line), so
    every bending order is reachable - a part where order does *not* matter.
    """
    # base corners 0..3, then one outer flap rectangle per edge.
    W, H, fl = 3.0, 2.0, 1.0
    v = [[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]]          # base 0..3
    tris = [(0, 1, 2), (0, 2, 3)]
    tp = [0, 0]
    bends = []
    parent = [-1]
    panel = 1
    # (edge vertex pair, outward offset) per wall
    edges = [((0, 1), [0, -fl, 0], "bottom"),
             ((1, 2), [fl, 0, 0], "right"),
             ((2, 3), [0, fl, 0], "top"),
             ((3, 0), [-fl, 0, 0], "left")]
    for (a, b), off, lbl in edges:
        off = np.array(off, dtype=float)
        ia, ib = len(v), len(v) + 1
        v.append((np.array(v[a]) + off).tolist())
        v.append((np.array(v[b]) + off).tolist())
        tris += [(a, b, ib), (a, ib, ia)]
        tp += [panel, panel]
        parent.append(0)
        bends.append((a, b, ia, ib, panel, lbl))
        panel += 1
    v = np.array(v, dtype=float)
    real = []
    for (a, b, ia, ib, pnl, lbl) in bends:
        mag = np.radians(flap_angle)
        angle = _resolve_sign(v, (a, b), [ia, ib], mag, "up")
        real.append(Bend(axis=(a, b), angle=angle, child=pnl, name=lbl))
    return SheetMetalPart(name="tray", vertices=v, triangles=np.array(tris),
                          tri_panel=np.array(tp), parent=parent, bends=real)


def u_channel(flap_angle=90.0):
    """A U-channel: a base strip with the two end panels bent up 90 deg.

    Two independent bends - like the tray, any order works."""
    return _profile("u_channel", [0.0, 1.0, 3.0, 4.0], 2.0,
                    [(flap_angle, "up"), (flap_angle, "up")], root=1)


def coil(flap_angle=90.0):
    """A coiled strip: three parallel bends, all folding the same way.

    This is the headline part. The bends form a chain, so folding an *inner*
    bend lifts every *outer* bend line off the table - after it, the press brake
    can no longer seat those bends. The naive index order (L1, L2, L3) bends
    inner-first and strands L2/L3; the only valid order is outermost-first
    (L3, L2, L1), the classic press-brake rule.
    """
    return _profile("coil", [0.0, 1.0, 2.0, 3.0, 4.0], 2.0,
                    [(flap_angle, "up")] * 3)


def overhang(flap_angle=90.0):
    """An over-bent return flange that hangs over an inner bend line.

    A short wall (L1, at x=2) and an outer flap (L2) that is over-bent past
    vertical so it folds back and hovers above L1's bend line. Nothing is ever
    lifted off the die here, so the seating test alone sees no problem - the
    constraint is pure *punch clearance*: form L2 first and its folded flap sits
    directly above L1, so the descending punch crashes into it (caught by the
    slab/triangle test). Form L1 first (while the sheet beyond it is still flat)
    and both bends are clean. The planner finds the clean order; the reverse
    collides. The final part is collision-free (the overhang clears the wall).
    """
    return _profile("overhang", [1.5, 2.0, 4.0, 7.0], 2.0,
                    [(flap_angle, "up"), (160.0, "up")], root=1)


PARTS = {
    "tray": tray,
    "u_channel": u_channel,
    "coil": coil,
    "overhang": overhang,
}
