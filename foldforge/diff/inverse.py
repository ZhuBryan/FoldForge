"""Differentiable inverse design (Milestone 3): recover fold parameters from a
target folded shape by gradient descent through the differentiable fold engine.

The forward direction is already differentiable: fold *parameters* (fold angles
of a chain, or the four Miura design parameters) map to folded geometry with
analytic Jacobians (see :mod:`foldforge.diff.kinematics` and
:mod:`foldforge.diff.miura`). Inverse design runs that backwards:

    params --(forward fold)--> geometry --(loss vs target)--> scalar
           <--(analytic gradient)-- d loss / d params <--------'
    params <- Adam step

and iterates. Nothing here modifies the forward engine; it only *reuses* it.

Three capabilities, smallest-first:

1. :func:`recover_fold_angles` - the clean, machine-checkable headline. Given the
   folded spine of a fold chain made by UNKNOWN angles, recover those angles.
   Loss is the sum of squared point distances; the gradient is formed in closed
   form from :func:`~foldforge.diff.kinematics.spine_jacobian`. On a
   well-conditioned case this recovers the true angles to ~1e-6.

2. :func:`inverse_fit` - generalisation. Fit the four Miura parameters
   ``(a, b, gamma, h)`` to a target folded footprint ``(w, d, h)`` by reusing
   :func:`~foldforge.diff.miura.fit_miura`. Drives the footprint error to ~0
   and recovers ``h`` exactly, but a 3-number footprint underdetermines the 4
   parameters, so ``a``, ``b`` and ``gamma`` sit on a solution manifold (small
   residual). Fits a non-exact target as closely as the 4-DOF model allows.

3. :func:`refine` - "make it better". Starting from an existing design, descend a
   scalar objective (fitting error, or a fold-strain proxy) to push it below the
   starting value. A generic Adam loop over a user-supplied objective.

Honesty: origami inverse design is non-convex. Angle recovery on a
well-conditioned chain is effectively exact; arbitrary targets have local minima
and need a reasonable init. v1 proves the well-posed cases and is honest about
the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foldforge.diff.kinematics import fold_chain, spine_jacobian
from foldforge.diff.miura import fit_miura, footprint, fold_limit
from foldforge.origamize.miura_fit import (
    _flat_miura, _folded_miura, _edge_list, surface_fit_error)


@dataclass
class InverseResult:
    """Result of an inverse-design optimisation."""

    params: np.ndarray       # recovered / optimised parameters
    history: np.ndarray      # loss (objective) value at each iteration
    final_loss: float        # objective at the returned params


def _adam(params, grad_fn, *, iters, lr, project=None, b1=0.9, b2=0.999,
          eps=1e-8, loss_fn=None):
    """Minimal Adam loop shared by the inverse routines.

    ``grad_fn(p) -> (loss, grad)`` (loss may be None if ``loss_fn`` is given).
    ``project`` optionally clamps ``p`` back into a valid region after each step.
    Returns ``(params, history)``.
    """
    p = np.array(params, dtype=float)
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    history = []
    for t in range(1, iters + 1):
        loss, grad = grad_fn(p)
        if loss is None and loss_fn is not None:
            loss = loss_fn(p)
        history.append(float(loss))
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        p = p - lr * mhat / (np.sqrt(vhat) + eps)
        if project is not None:
            p = project(p)
    return p, np.array(history)


# --------------------------------------------------------------------------
# 1. Parameter recovery - the verifiable headline demo
# --------------------------------------------------------------------------

def recover_fold_angles(target_spine, n=None, *, seg=1.0, init=None,
                        iters=2000, lr=0.05, seed=0):
    """Recover fold-chain angles from a target folded ``spine``.

    Given ``target_spine`` (shape ``(n+1, 2)``) produced by a chain of ``n``
    unknown fold angles, recover the angles by gradient descent. The loss is the
    sum of squared point distances between the current and the target spine, and
    its gradient is assembled in closed form from the analytic spine Jacobian::

        r      = fold_chain(angles).spine - target_spine        # (n+1, 2)
        loss   = sum(r**2)
        grad_j = 2 * sum_{k,c} r[k,c] * d spine[k,c] / d angle_j

    ``init`` defaults to a small random perturbation of zero (near-flat), which
    is the natural starting guess. Returns an :class:`InverseResult`; on a
    well-conditioned chain the recovered angles match the true ones to ~1e-6.
    """
    target_spine = np.asarray(target_spine, dtype=float)
    if n is None:
        n = target_spine.shape[0] - 1
    if init is None:
        rng = np.random.default_rng(seed)
        init = rng.normal(0.0, 1e-2, n)
    init = np.asarray(init, dtype=float)

    def grad_fn(angles):
        res = fold_chain(angles, seg=seg)
        r = res.spine - target_spine                 # (n+1, 2)
        loss = float(np.sum(r ** 2))
        J = spine_jacobian(angles, seg=seg)          # (n+1, 2, n)
        grad = 2.0 * np.einsum("kc,kcj->j", r, J)
        return loss, grad

    params, history = _adam(init, grad_fn, iters=iters, lr=lr)
    return InverseResult(params=params, history=history,
                         final_loss=float(history[-1]))


# --------------------------------------------------------------------------
# 2. Fit an arbitrary target (generalisation)
# --------------------------------------------------------------------------

def inverse_fit(target_wdh, *, rows=6, cols=6, iters=400, lr=0.05, seed=0):
    """Fit Miura parameters ``(a, b, gamma, h)`` to a target folded footprint.

    Thin wrapper over :func:`foldforge.diff.miura.fit_miura` (the existing
    closed-form Miura inverse solver) packaged as an :class:`InverseResult`.
    ``target_wdh`` is the folded bounding box ``(width, depth, height)``. Works
    well when the target is reachable (e.g. an actual Miura's own footprint);
    for an unreachable target it returns the closest the 4-DOF model allows.
    """
    params, history, achieved = fit_miura(
        target_wdh, rows=rows, cols=cols, iters=iters, lr=lr, seed=seed)
    return InverseResult(params=np.asarray(params, float), history=history,
                         final_loss=float(history[-1])), np.asarray(achieved)


# --------------------------------------------------------------------------
# 3. Refinement - "make it better"
# --------------------------------------------------------------------------

def refine(initial_params, objective, *, iters=300, lr=0.02, project=None,
           fd_step=1e-5):
    """Refine an existing design by descending a scalar ``objective(params)``.

    Starts from ``initial_params`` and runs Adam with a finite-difference
    gradient of ``objective`` (cheap - designs here have a handful of
    parameters). ``project`` optionally clamps parameters into a valid region
    after each step. Returns an :class:`InverseResult`; the final objective is
    guaranteed lower than the start on a well-behaved objective (verified in the
    tests / demo).
    """
    p0 = np.array(initial_params, dtype=float)

    def grad_fn(p):
        loss = float(objective(p))
        grad = np.zeros_like(p)
        for k in range(len(p)):
            d = fd_step * (1 + abs(p[k]))
            pp = p.copy(); pp[k] += d
            pm = p.copy(); pm[k] -= d
            grad[k] = (float(objective(pp)) - float(objective(pm))) / (2 * d)
        return loss, grad

    params, history = _adam(p0, grad_fn, iters=iters, lr=lr, project=project)
    return InverseResult(params=params, history=history,
                         final_loss=float(history[-1]))


def miura_footprint_objective(target_wdh, rows=6, cols=6):
    """Objective factory: squared footprint error of Miura params vs a target.

    Returns ``objective(params)`` with ``params = (a, b, gamma, h)``, suitable
    for :func:`refine`. ``h`` is clamped into its valid fold range internally so
    the objective is always well-defined.
    """
    target = np.asarray(target_wdh, dtype=float)

    def objective(p):
        a, b, g, h = p
        h = np.clip(h, 1e-3, fold_limit(abs(a), g) - 1e-3)
        fp = np.array(footprint(rows, cols, abs(a), abs(b), g, h))
        return float(np.sum((fp - target) ** 2))

    return objective


def miura_param_project(p):
    """Clamp Miura params ``(a, b, gamma, h)`` into a sane, foldable region."""
    p = p.copy()
    p[0] = abs(p[0]); p[1] = abs(p[1])
    p[2] = np.clip(p[2], np.radians(20), np.radians(85))
    p[3] = np.clip(p[3], 1e-3, fold_limit(p[0], p[2]) - 1e-3)
    return p


# --------------------------------------------------------------------------
# 4. v2 - fit an ARBITRARY target with FOLDABILITY baked into the loss
# --------------------------------------------------------------------------
#
# v1 above fits parameter *vectors* (a fold chain's angles, the 4 Miura
# numbers). v2 optimises a whole crease *mesh*: a flat pattern ``Q`` (2D) and a
# folded surface ``P`` (3D) over a quad grid, exactly like the proven
# ``miura_fit._fit_warped_miura`` template, but generalised two ways:
#
#   * the target is ARBITRARY - a height field (RMSE) OR a point cloud
#     (chamfer distance), not just a reachable Miura footprint; and
#   * a differentiable KAWASAKI flat-foldability penalty is added to the loss,
#     so the recovered flat pattern is pulled toward being genuinely
#     flat-foldable, not merely a good-looking fit.
#
# Total loss = w_fit*fidelity + w_iso*isometry + w_kawasaki*kawasaki, minimised
# by Adam over (P, Q) with analytic gradients for every term.
#
# Honesty: this is non-convex and has local minima, and a developable sheet
# cannot match a non-developable target with zero strain (the isometry residual
# is the price of curvature). v2 does NOT solve arbitrary 3D shapes exactly. The
# win it demonstrates is real: the Kawasaki penalty works (its gradient is
# finite-difference-checked) and produces a genuine fidelity-vs-foldability
# tradeoff - turning it on lowers the Kawasaki residual (more foldable) at a
# quantified, modest fidelity cost.


# --- the differentiable Kawasaki flat-foldability penalty --------------------
#
# ``foldforge.geometry.foldability.check_kawasaki`` is a boolean screen: sweep
# the sector angles around an interior vertex and test that the alternating sum
# ``a1 - a2 + a3 - a4 + ...`` is ~0 (equivalently odd sectors total == even
# sectors total). Here we make a DIFFERENTIABLE version of exactly that quantity
# and penalise its square, summed over interior vertices.
#
# We apply it to the four genuine Miura fold creases at each interior vertex
# (right, down, left, up - the mountain/valley lines), the degree-4 vertices for
# which Kawasaki is the classic flat-foldability condition. A uniform Miura is
# flat-foldable, so this penalty is ~0 there; warping the pattern to chase a
# non-developable target drives it up, which is precisely what the penalty
# fights. (The triangulating facet diagonals are not real folds, so they are not
# included - putting them in would make a flat-foldable uniform Miura score a
# spurious nonzero residual.)

# fixed CCW order of the four Miura creases around an interior vertex
_KAWASAKI_OFFSETS = ((0, 1), (1, 0), (0, -1), (-1, 0))   # right, down, left, up
_KAWASAKI_SIGN = np.array([1.0, -1.0, 1.0, -1.0])        # alternating a1-a2+a3-a4


def _kawasaki_neighbors(R, C):
    """Interior-vertex centres and their 4 crease-neighbour indices (CCW).

    Returns ``(centers, nbrs)`` where ``centers`` is ``(V,)`` flat indices of
    interior vertices and ``nbrs`` is ``(V, 4)`` flat indices of the right /
    down / left / up neighbours, in the fixed CCW order the alternating sum
    expects.
    """
    idx = lambda j, i: j * C + i
    centers, nbrs = [], []
    for j in range(1, R - 1):
        for i in range(1, C - 1):
            centers.append(idx(j, i))
            nbrs.append([idx(j + dj, i + di) for dj, di in _KAWASAKI_OFFSETS])
    return np.array(centers, dtype=int), np.array(nbrs, dtype=int)


def kawasaki_penalty(Q, centers, nbrs):
    """Differentiable Kawasaki residual of a flat pattern ``Q`` and its gradient.

    For each interior vertex, the four crease directions give sector angles
    ``s_0..s_3`` (consecutive CCW differences, each in ``(0, 2*pi)``); the
    Kawasaki quantity is the alternating sum ``s = s_0 - s_1 + s_2 - s_3`` (zero
    iff the vertex can fold flat). Returns ``(penalty, grad, s)`` with::

        penalty = sum_v s_v**2
        grad    = d penalty / d Q        (shape of Q[:, :2], zero elsewhere)

    ``s`` is the per-vertex alternating sum (radians), used for reporting the
    Kawasaki residual. Gradient is analytic (atan2 derivatives, chain rule) and
    is finite-difference checked in the tests.
    """
    Q = np.asarray(Q, dtype=float)
    N = Q.shape[0]
    qc = Q[centers, :2]                        # (V, 2)
    qn = Q[nbrs, :2]                           # (V, 4, 2)
    d = qn - qc[:, None, :]                    # (V, 4, 2)   crease vectors
    ang = np.arctan2(d[..., 1], d[..., 0])     # (V, 4)      crease angles
    sec = np.mod(np.roll(ang, -1, axis=1) - ang, 2 * np.pi)   # (V, 4) sectors
    s = sec @ _KAWASAKI_SIGN                   # (V,)        alternating sum
    penalty = float(np.sum(s ** 2))

    # d penalty / d ang_m: ang_m enters sector_m (coeff -1) and sector_{m-1}
    # (coeff +1), so d penalty / d ang_m = 2 s (c_{m-1} - c_m).
    cprev = np.roll(_KAWASAKI_SIGN, 1)
    fac = 2.0 * s[:, None] * (cprev - _KAWASAKI_SIGN)[None, :]   # (V, 4)
    r2 = (d ** 2).sum(-1) + 1e-12                               # (V, 4)
    # d ang_m / d qn = (-dy, dx)/r^2 ; d ang_m / d qc = (dy, -dx)/r^2
    gqn = fac[..., None] * np.stack([-d[..., 1], d[..., 0]], -1) / r2[..., None]
    gqc = -gqn.sum(1)                          # (V, 2)

    gxy = np.zeros((N, 2))
    np.add.at(gxy, centers, gqc)
    for k in range(nbrs.shape[1]):
        np.add.at(gxy, nbrs[:, k], gqn[:, k, :])
    grad = np.zeros_like(Q)
    grad[:, :2] = gxy
    return penalty, grad, s


# --- chamfer fidelity for point-cloud targets --------------------------------

def _chamfer(P, T):
    """Symmetric chamfer distance between folded points ``P`` and target ``T``.

    ``sum_i min_j |P_i - T_j|^2 + sum_j min_i |T_j - P_i|^2`` (O(n*m), fine for
    the small point sets here). Returns ``(value, grad)`` with ``grad`` the
    derivative w.r.t. ``P``.
    """
    P = np.asarray(P, float); T = np.asarray(T, float)
    D = ((P[:, None, :] - T[None, :, :]) ** 2).sum(-1)     # (n, m)
    jmin = D.argmin(1)                                     # nearest T per P
    imin = D.argmin(0)                                     # nearest P per T
    val = float(D[np.arange(len(P)), jmin].sum()
                + D[imin, np.arange(len(T))].sum())
    g = 2.0 * (P - T[jmin])                                # forward term
    np.add.at(g, imin, 2.0 * (P[imin] - T))                # reverse term
    return val, g


@dataclass
class ShapeDesignResult:
    """Result of :func:`inverse_design_shape` - a fitted crease mesh + metrics."""

    Q: np.ndarray              # (N, 3) flat crease pattern (z == 0)
    P: np.ndarray              # (N, 3) folded surface
    grid_shape: tuple          # (R, C)
    history: np.ndarray        # total weighted loss per iteration
    fid_history: np.ndarray    # unweighted fidelity term per iteration
    kaw_history: np.ndarray    # unweighted Kawasaki penalty per iteration
    final_loss: float          # total weighted loss at the end
    fidelity: float            # reported fidelity (normalised RMSE or chamfer)
    kawasaki_residual: float   # RMS per-vertex alternating-sum deviation (rad)
    mean_strain: float         # mean fold strain |len_P - len_Q| / len_Q
    max_strain: float          # worst-case fold strain


def inverse_design_shape(target, grid, *, target_kind="height",
                         w_fit=1.0, w_iso=4.0, w_kawasaki=0.0, w_anchor=0.04,
                         iters=600, lr=0.02, relief=1.6, seed=0,
                         a=1.0, b=1.0, gamma=np.radians(60.0), h=0.45):
    """Fit a crease mesh to an ARBITRARY ``target`` with foldability in the loss.

    Jointly optimises a flat pattern ``Q`` (2D) and a folded surface ``P`` (3D)
    over an ``R x C`` quad grid (``grid = (R, C)``), by Adam on::

        loss = w_fit * fidelity + w_iso * isometry + w_kawasaki * kawasaki

    with analytic gradients for every term. ``target_kind``:

      * ``"height"`` - ``target`` is an ``(R, C)`` height field; fidelity is the
        squared error of the folded z against the (normalised, relief-scaled)
        target, with a light anchor keeping the footprint a height field.
      * ``"pointcloud"`` - ``target`` is an ``(m, 3)`` point cloud; fidelity is
        the symmetric :func:`_chamfer` distance of the folded vertices to it.

    ``isometry`` keeps every folded edge the length of its flat edge (near-rigid
    fold); ``kawasaki`` is the differentiable flat-foldability penalty
    (:func:`kawasaki_penalty`). Set ``w_kawasaki > 0`` to trade a little fidelity
    for a more flat-foldable pattern. Returns a :class:`ShapeDesignResult`.

    Honest limits: non-convex (local minima; depends on the Miura init), and a
    developable sheet cannot match a non-developable target exactly - the
    isometry/fidelity residual is the curvature it cannot absorb.
    """
    R, C = grid
    Q = _flat_miura(R, C, a, b, gamma)
    P = _folded_miura(R, C, a, b, gamma, h)
    anchor = P[:, :2].copy()
    centers, nbrs = _kawasaki_neighbors(R, C)
    E = _edge_list(R, C)
    e0, e1 = E[:, 0], E[:, 1]

    kind = target_kind
    if kind == "height":
        Z = np.asarray(target, dtype=float)
        if Z.shape != (R, C):
            raise ValueError(f"height target must be {grid}, got {Z.shape}")
        Zn = Z - Z.min(); Zn = Zn / (Zn.max() + 1e-9)
        Zt = (Zn * relief).reshape(-1)
        P[:, 2] = P[:, 2] + Zt
    elif kind == "pointcloud":
        T = np.asarray(target, dtype=float)
        if T.ndim != 2 or T.shape[1] != 3:
            raise ValueError(f"pointcloud target must be (m, 3), got {T.shape}")
    else:
        raise ValueError("target_kind must be 'height' or 'pointcloud'")

    mP = np.zeros_like(P); vP = np.zeros_like(P)
    mQ = np.zeros_like(Q); vQ = np.zeros_like(Q)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist, fid_hist, kaw_hist = [], [], []

    for t in range(1, iters + 1):
        gP = np.zeros_like(P); gQ = np.zeros_like(Q)

        # --- fidelity ---
        if kind == "height":
            r = P[:, 2] - Zt
            fid = float(np.sum(r ** 2))
            gP[:, 2] += 2 * w_fit * r
            gP[:, :2] += 2 * w_anchor * (P[:, :2] - anchor)   # keep a height field
        else:
            fid, gfid = _chamfer(P, T)
            gP += w_fit * gfid

        # --- isometry (folded edge length == flat edge length), on both P and Q
        dP = P[e0] - P[e1]; lp = np.linalg.norm(dP, axis=1) + 1e-12
        dQ = Q[e0] - Q[e1]; lq = np.linalg.norm(dQ, axis=1) + 1e-12
        res = lp - lq
        iso = float(np.sum(res ** 2))
        cP = (2 * w_iso * res / lp)[:, None] * dP
        np.add.at(gP, e0, cP); np.add.at(gP, e1, -cP)
        cQ = (2 * w_iso * (-res) / lq)[:, None] * dQ
        np.add.at(gQ, e0, cQ); np.add.at(gQ, e1, -cQ)

        # --- Kawasaki flat-foldability penalty (on the flat pattern Q) ---
        kaw, gkaw, _ = kawasaki_penalty(Q, centers, nbrs)
        gQ += w_kawasaki * gkaw

        total = w_fit * fid + w_iso * iso + w_kawasaki * kaw
        hist.append(total); fid_hist.append(fid); kaw_hist.append(kaw)

        for g, m, v, X in ((gP, mP, vP, P), (gQ, mQ, vQ, Q)):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            X -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)

    # --- final metrics ---
    lp = np.linalg.norm(P[e0] - P[e1], axis=1)
    lq = np.linalg.norm(Q[e0] - Q[e1], axis=1)
    strain = np.abs(lp - lq) / (lq + 1e-12)
    _, _, s = kawasaki_penalty(Q, centers, nbrs)
    kaw_residual = float(np.sqrt(np.mean(s ** 2))) if len(s) else 0.0

    if kind == "height":
        fidelity = surface_fit_error(P.reshape(R, C, 3), Z, use_midsurface=False)
    else:
        fidelity = _chamfer(P, T)[0] / max(len(P) + len(T), 1)

    return ShapeDesignResult(
        Q=Q, P=P, grid_shape=(R, C),
        history=np.array(hist), fid_history=np.array(fid_hist),
        kaw_history=np.array(kaw_hist), final_loss=float(hist[-1]),
        fidelity=float(fidelity), kawasaki_residual=kaw_residual,
        mean_strain=float(strain.mean()), max_strain=float(strain.max()))
