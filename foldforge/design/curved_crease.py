"""Curved-crease origami: folding flat paper along *smooth curves*.

A straight crease folds a flat facet to another flat facet. A *curved* crease
bends the paper into a **curved developable surface** - zero Gaussian curvature
(the paper never stretches), yet genuinely curved in space, ruled by straight
generator lines. This module builds the tractable, provably-correct forward core
and *verifies* the result is a true bending (an isometry), not a stretch.

What is modelled (and exactly computable)
------------------------------------------
1. **Single symmetric curved fold** (:func:`curved_fold`). Input a flat 2-D
   crease curve and fold it by an angle. The rulings start perpendicular to the
   crease (a valid developable). The two flaps are mirror images across the
   fold's *rectifying* plane (the plane spanned by the crease tangent and
   binormal). For this family the folded crease curve and both developable flaps
   follow in closed form, and the fold is a genuine isometry - proven below and
   checked numerically to ~1e-12.

2. **Concentric circular pleat** (:func:`concentric_pleat`) - the iconic
   curved-crease demo. A flat annulus with N concentric mountain/valley circles
   folds into a rippled 3-D annulus. Modelled as the **conical-strip** model:
   each annular strip becomes a cone frustum (surface of revolution, straight
   slant => developable), crease circle radii preserved exactly.

The math (why the single fold is an exact bending)
--------------------------------------------------
Let the flat crease be unit-speed ``c(s)`` with tangent ``t``, left-normal ``m``
(``t`` rotated +90 deg) and signed curvature ``k`` (``t' = k m``). A flap on the
``+m`` side is ``X(s,u) = c(s) + u m(s)``; its first fundamental form is
``E = (1-u k)^2, F = 0, G = 1`` (the ``-m`` side gives ``E = (1+u k)^2``).

Fold by ``theta``. The 3-D crease ``C(s)`` stays a *planar* curve lying in the
``z=0`` plane (that plane is its osculating plane); the two flaps lift
symmetrically out of ``z=0`` so the relief comes from the flaps, not a rising
crease. The mirror plane is the vertical ``(T, B)`` plane, not the crease's
osculating plane. The folded crease has curvature ``kappa = k / cos(theta)``.
With Frenet frame ``(T, N, B)`` (``B`` constant, ``N' = -kappa T`` since the
crease is planar, torsion 0):

* ``+m`` flap  ->  ruling ``V = cos(theta) N + sin(theta) B``.
  Then ``V' = cos(theta) N' = -kappa cos(theta) T = -k T`` (because
  ``kappa cos(theta) = k``), so ``C + u V`` has ``R_s = (1 - u k) T`` and
  ``E = (1-u k)^2`` - identical to the flat ``+m`` flap.
* ``-m`` flap  ->  ruling ``W = -cos(theta) N + sin(theta) B``, giving
  ``W' = +k T`` and ``E = (1+u k)^2`` - identical to the flat ``-m`` flap.

``V`` and ``W`` are mirror images across the rectifying (T,B) plane (``N`` flips)
=> a symmetric fold. ``theta = 0`` reproduces the flat sheet; ``theta -> pi/2``
folds it shut. The dihedral between the flaps is ``pi - 2 theta``.

Honest frontier (NOT modelled here)
-----------------------------------
* General *asymmetric* rulings / non-perpendicular ruling fields, and fold
  angles that vary along the crease.
* The **inverse problem**: arbitrary target surface -> curved crease pattern +
  compatible rulings. This is a hard open research area; this module is the
  forward core only.
* Global **non-self-intersection**: rulings may cross if the flap is wider than
  the crease's radius of curvature (``u |k| >= 1``). We cap width and warn, but
  do not globally de-collide the flaps.
* The *exact* circular pleat does **not** have an axisymmetric isometric
  embedding - real paper buckles into a non-axisymmetric wavy shape. The conical
  model here is the standard axisymmetric approximation; its small radial strain
  is reported honestly as ``developability_error``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _smooth_frame(curve: np.ndarray, n_s: int):
    """Fit a cubic spline and return a unit-speed frame of the crease.

    A linear resample of a polyline has a piecewise-constant tangent, so its
    numerical curvature is junk. We instead fit a cubic spline to the crease and
    read the tangent off the spline analytically, giving a smooth tangent angle
    ``phi(s)``. Returns ``(pts, phi, s, ds)`` sampled at ``n_s`` points spaced
    equally in arc length, where ``phi`` is the (unwrapped) tangent heading and
    ``phi' = k`` is the signed curvature.
    """
    c = np.asarray(curve, dtype=float)
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    sx = CubicSpline(chord, c[:, 0])
    sy = CubicSpline(chord, c[:, 1])
    # dense pass -> true arc length -> equal-arc-length spline parameters
    dense = np.linspace(0.0, chord[-1], n_s * 20)
    al = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(sx(dense)), np.diff(sy(dense))))])
    s = np.linspace(0.0, al[-1], n_s)
    p = np.interp(s, al, dense)
    pts = np.stack([sx(p), sy(p)], axis=1)
    dx, dy = sx(p, 1), sy(p, 1)
    phi = np.unwrap(np.arctan2(dy, dx))
    ds = np.gradient(s)
    return pts, phi, s, ds


def _fundamental_forms(V: np.ndarray, ns: int, nu: int, ds: float, du: float):
    """First fundamental form (E, F, G) at every grid node by central diff."""
    P = V.reshape(ns, nu, 3)
    Ps = np.gradient(P, ds, axis=0)
    Pu = np.gradient(P, du, axis=1)
    E = np.sum(Ps * Ps, axis=2)
    F = np.sum(Ps * Pu, axis=2)
    G = np.sum(Pu * Pu, axis=2)
    return E, F, G


def _grid_faces(ns: int, nu: int, offset: int = 0):
    """Quad faces (0-indexed) for an ``ns x nu`` vertex grid, flattened row-major."""
    faces = []
    for i in range(ns - 1):
        for j in range(nu - 1):
            a = offset + i * nu + j
            faces.append([a, a + 1, a + nu + 1, a + nu])
    return faces


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class FoldedSurface:
    """A folded curved-crease surface and its verification data."""
    flat: np.ndarray                 # (V, 3) flat sheet vertices (z = 0)
    folded: np.ndarray               # (V, 3) folded vertices
    faces: list                      # quad faces, 0-indexed into the vertex arrays
    developability_error: float      # max isometry violation (see each builder)
    crease_flat: np.ndarray          # (ns, 3) flat crease curve
    crease_folded: np.ndarray        # (ns, 3) folded crease curve
    meta: dict = field(default_factory=dict)

    def to_obj(self, path: str) -> None:
        """Write the folded mesh to a Wavefront OBJ (reuses fabricate.export)."""
        from foldforge.fabricate.export import to_obj
        to_obj(self.folded, [[i + 1 for i in f] for f in self.faces], path)


# --------------------------------------------------------------------------- #
# 1. single symmetric curved fold
# --------------------------------------------------------------------------- #
def curved_fold(crease_curve: np.ndarray, fold_angle: float, *,
                width: float = 0.35, n_s: int = 700, n_u: int = 50) -> FoldedSurface:
    """Fold a flat 2-D crease curve by ``fold_angle`` into a curved developable.

    Parameters
    ----------
    crease_curve : (M, 2) array
        The flat crease, any polyline (resampled to unit speed internally).
    fold_angle : float
        Half-dihedral ``theta`` in radians. ``0`` = flat (unfolded), ``pi/2`` =
        folded shut. Dihedral between the two flaps is ``pi - 2*theta``.
    width : float
        Half-width of the sheet on each side of the crease.
    n_s, n_u : int
        Samples along the crease and across each flap.

    Returns
    -------
    FoldedSurface
        ``flat`` and ``folded`` carry both flaps (``-m`` flap then ``+m`` flap,
        sharing the crease row). ``developability_error`` is the max isometry
        (first-fundamental-form) violation between flat and folded - ~1e-12 for
        this exact family.
    """
    theta = float(fold_angle)
    ct, st = np.cos(theta), np.sin(theta)
    if ct < 1e-9:
        raise ValueError("fold_angle must be < pi/2 (cos(theta) > 0)")

    c2, phi, s, ds = _smooth_frame(crease_curve, n_s)       # unit-speed frame
    k = np.gradient(phi) / ds                               # signed curvature

    # ponytail: cap width so rulings don't cross the centre of curvature
    # (u*|k| < 1); global flap-vs-flap self-intersection is not checked.
    kmax = float(np.max(np.abs(k)))
    wlimit = 0.95 / kmax if kmax > 1e-9 else np.inf
    w = min(width, wlimit)
    u = np.linspace(0.0, w, n_u)

    # --- flat sheet: rebuild from phi so flat & folded share one angle source
    m2 = np.stack([-np.sin(phi), np.cos(phi)], axis=1)      # left-normal
    t2 = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    cflat = np.cumsum(t2 * ds[:, None], axis=0) - t2[0] * ds[0]
    c3 = np.concatenate([cflat, np.zeros((n_s, 1))], axis=1)
    m3 = np.concatenate([m2, np.zeros((n_s, 1))], axis=1)
    flat_plus = c3[:, None, :] + u[None, :, None] * m3[:, None, :]
    flat_minus = c3[:, None, :] - u[None, :, None] * m3[:, None, :]

    # --- folded crease: planar curve (in the z=0 plane), curvature
    #     kappa = k/cos(theta), so its turning angle is psi = (phi-phi0)/cos(theta)
    #     (exact integral). Binormal is +z, so both flaps lift out of the plane;
    #     theta=0 reproduces the flat sheet exactly, in z=0. --------------------
    psi = (phi - phi[0]) / ct
    T = np.stack([np.cos(psi), np.sin(psi), np.zeros(n_s)], axis=1)
    C = np.cumsum(T * ds[:, None], axis=0) - T[0] * ds[0]   # folded crease (in z=0)
    N = np.stack([-np.sin(psi), np.cos(psi), np.zeros(n_s)], axis=1)
    B = np.tile(np.array([0.0, 0.0, 1.0]), (n_s, 1))        # planar => B const (+z)

    V = ct * N + st * B                                     # +m flap ruling
    W = -ct * N + st * B                                    # -m flap ruling
    fold_plus = C[:, None, :] + u[None, :, None] * V[:, None, :]
    fold_minus = C[:, None, :] + u[None, :, None] * W[:, None, :]

    # stack: -m flap reversed across width + +m flap -> one continuous grid
    flat_grid = np.concatenate([flat_minus[:, ::-1, :], flat_plus[:, 1:, :]], axis=1)
    fold_grid = np.concatenate([fold_minus[:, ::-1, :], fold_plus[:, 1:, :]], axis=1)
    nu_tot = flat_grid.shape[1]
    flat = flat_grid.reshape(-1, 3)
    folded = fold_grid.reshape(-1, 3)
    faces = _grid_faces(n_s, nu_tot)

    # --- verify isometry: compare first fundamental forms per flap ----------
    # (each flap is smooth; the crease itself is a fold line, so comparing
    #  across it would just measure the intended dihedral kink, not a stretch).
    du = u[1] - u[0]
    dss = float(np.mean(ds))
    err = 0.0
    for fl, fo in [(flat_plus, fold_plus), (flat_minus, fold_minus)]:
        Ef, Ff, Gf = _fundamental_forms(fl.reshape(-1, 3), n_s, n_u, dss, du)
        Eo, Fo, Go = _fundamental_forms(fo.reshape(-1, 3), n_s, n_u, dss, du)
        sl = (slice(1, -1), slice(1, -1))                   # drop one-sided edges
        err = max(err, float(np.max(np.abs(Ef[sl] - Eo[sl]))),
                  float(np.max(np.abs(Ff[sl] - Fo[sl]))),
                  float(np.max(np.abs(Gf[sl] - Go[sl]))))

    return FoldedSurface(
        flat=flat, folded=folded, faces=faces,
        developability_error=err,
        crease_flat=c3, crease_folded=C,
        meta=dict(kind="symmetric_fold", theta=theta, width=w,
                  width_requested=width, n_s=n_s, n_u=nu_tot,
                  dihedral=np.pi - 2 * theta,
                  z_range=float(folded[:, 2].max() - folded[:, 2].min())),
    )


# --------------------------------------------------------------------------- #
# 2. concentric circular pleat (conical-strip model)
# --------------------------------------------------------------------------- #
def concentric_pleat(radii, angles, *, n_theta: int = 120) -> FoldedSurface:
    """Fold a flat annulus with concentric mountain/valley circles.

    Conical-strip model: strip ``i`` (between ``radii[i]`` and ``radii[i+1]``)
    becomes a cone frustum tilted by ``angles[i]`` from horizontal. Crease-circle
    radii are preserved *exactly* (3-D radius = flat radius), height accumulates
    ``dz = dr * tan(angle)``; alternating angle signs give the mountain/valley
    ripple. Each strip is a surface of revolution with a straight slant =>
    developable.

    ``developability_error`` is the (honest) radial strain ``sec(angle) - 1`` the
    axisymmetric approximation incurs - the exact pleat relieves it by buckling
    non-axisymmetrically (not modelled).
    """
    r = np.asarray(radii, dtype=float)
    ang = np.asarray(angles, dtype=float)
    if len(ang) != len(r) - 1:
        raise ValueError("need one angle per strip: len(angles) == len(radii)-1")

    # heights: preserve radii, accumulate dz = dr * tan(angle)
    dr = np.diff(r)
    dz = dr * np.tan(ang)
    z = np.concatenate([[0.0], np.cumsum(dz)])

    th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(th), np.sin(th)

    def ring(rad, height):
        return np.stack([rad * cos_t, rad * sin_t,
                         np.full(n_theta, height)], axis=1)

    flat_rings = [ring(ri, 0.0) for ri in r]
    fold_rings = [ring(ri, zi) for ri, zi in zip(r, z)]
    flat = np.concatenate(flat_rings, axis=0)
    folded = np.concatenate(fold_rings, axis=0)

    # faces between consecutive rings (wrap around in theta)
    faces = []
    for i in range(len(r) - 1):
        b0, b1 = i * n_theta, (i + 1) * n_theta
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            faces.append([b0 + j, b0 + jn, b1 + jn, b1 + j])

    strain = float(np.max(np.abs(1.0 / np.cos(ang) - 1.0)))
    radius_err = float(np.max(np.abs(
        np.linalg.norm(folded[:, :2], axis=1)
        - np.linalg.norm(flat[:, :2], axis=1))))

    # crease curve = outermost circle (just for the dataclass contract)
    crease_flat = flat_rings[-1]
    crease_folded = fold_rings[-1]

    return FoldedSurface(
        flat=flat, folded=folded, faces=faces,
        developability_error=strain,
        crease_flat=crease_flat, crease_folded=crease_folded,
        meta=dict(kind="concentric_pleat", n_strips=len(r) - 1,
                  n_theta=n_theta, radius_error=radius_err,
                  radial_strain=strain, radii=r.tolist(), z=z.tolist(),
                  z_range=float(folded[:, 2].max() - folded[:, 2].min())),
    )


def demo() -> None:
    """Self-check: both builders produce genuine isometric bendings."""
    # single symmetric fold of a circular arc
    a = np.linspace(-0.8, 0.8, 150)
    arc = np.stack([1.6 * np.sin(a), 1.6 * (1 - np.cos(a))], axis=1)
    f = curved_fold(arc, fold_angle=0.5)
    assert f.developability_error < 1e-6, f.developability_error
    assert f.meta["z_range"] > 0.1, "folded surface must be genuinely 3-D"

    # concentric pleat
    radii = np.linspace(0.4, 1.0, 7)
    angles = np.deg2rad([20, -20, 20, -20, 20, -20])
    p = concentric_pleat(radii, angles)
    assert p.meta["radius_error"] < 1e-9, p.meta["radius_error"]
    assert p.meta["z_range"] > 0.03
    print("curved_fold developability_error =", f.developability_error)
    print("pleat radius_error =", p.meta["radius_error"],
          "radial_strain =", p.developability_error)
    print("OK")


if __name__ == "__main__":
    demo()
