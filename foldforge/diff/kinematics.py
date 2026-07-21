"""Differentiable rigid-origami kinematics (Milestone 2), pure numpy.

A *fold chain* is a strip of rigid panels hinged in sequence - the simplest
rigid-origami mechanism. Folding it bends the strip into an arch. Because the
panels are rigid, the whole shape is an exact function of the fold angles, and
that function is smooth, so we can differentiate it: given a scalar readout of
the folded shape (say its apex height), we get the gradient with respect to
every fold angle in closed form.

That gradient is what makes inverse design (M3) and simulator-in-the-loop
training (M4) possible. We verify it against finite differences in the tests.

Geometry: the chain folds in the x-z plane. Joint k sits at heading
``a_k = rho_0 + ... + rho_k`` from the start, so

    P_0 = (0, 0)
    P_{k+1} = P_k + seg * (cos a_k, sin a_k)

and each joint is extruded across the strip width in y to make the 3D panels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FoldChainResult:
    """Folded chain geometry: the 2D spine and the extruded 3D panel vertices."""

    spine: np.ndarray        # (n+1, 2) joint positions in the x-z plane
    vertices: np.ndarray     # (2*(n+1), 3) extruded 3D panel vertices


def fold_chain(angles: np.ndarray, seg: float = 1.0, width: float = 1.0) -> FoldChainResult:
    """Fold a strip of ``len(angles)`` panels by the given hinge ``angles`` (rad).

    ``angles[k]`` is the bend introduced at the start of panel k, so cumulative
    sums give each segment's heading. Returns a :class:`FoldChainResult`.
    """
    angles = np.asarray(angles, dtype=float)
    headings = np.cumsum(angles)                       # a_k
    steps = seg * np.stack([np.cos(headings), np.sin(headings)], axis=1)
    spine = np.zeros((len(angles) + 1, 2))
    spine[1:] = np.cumsum(steps, axis=0)               # P_{k+1}
    verts = []
    for (x, z) in spine:                               # extrude across width in y
        verts.append([x, 0.0, z])
        verts.append([x, width, z])
    return FoldChainResult(spine=spine, vertices=np.array(verts))


def spine_jacobian(angles: np.ndarray, seg: float = 1.0) -> np.ndarray:
    """Analytic Jacobian d(spine)/d(angles), shape ``(n+1, 2, n)``.

    From ``P_{k+1} = sum_{m<=k} seg*(cos a_m, sin a_m)`` with
    ``a_m = sum_{i<=m} rho_i``: ``dP_{k+1}/drho_j = sum_{m=j..k}
    seg*(-sin a_m, cos a_m)`` for ``j <= k`` (zero otherwise).
    """
    angles = np.asarray(angles, dtype=float)
    n = len(angles)
    headings = np.cumsum(angles)
    dseg = seg * np.stack([-np.sin(headings), np.cos(headings)], axis=1)  # (n, 2)
    jac = np.zeros((n + 1, 2, n))
    for k in range(1, n + 1):           # joint P_k depends on segments 0..k-1
        for j in range(k):              # rho_j affects segments m = j..k-1
            jac[k, :, j] = dseg[j:k].sum(axis=0)
    return jac


def apex_height(angles: np.ndarray, seg: float = 1.0) -> float:
    """Height (z) of the chain's midpoint joint - a smooth 'dome height' readout."""
    spine = fold_chain(angles, seg=seg).spine
    return float(spine[len(spine) // 2, 1])


def apex_height_grad(angles: np.ndarray, seg: float = 1.0) -> np.ndarray:
    """Analytic gradient of :func:`apex_height` w.r.t. each fold angle."""
    jac = spine_jacobian(angles, seg=seg)
    mid = jac.shape[0] // 2
    return jac[mid, 1, :]              # d z_mid / d angles
