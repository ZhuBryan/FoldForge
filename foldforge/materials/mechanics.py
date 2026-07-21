"""Origami metamaterials (Milestone 5): mechanical behaviour from fold state.

A Miura-ori is not just a shape - it is a mechanical metamaterial whose
properties are set by its geometry and how far it is folded. The headline
property is a **negative in-plane Poisson's ratio** (it is auxetic: pull it one
way and it widens the other way), which falls straight out of the cell
kinematics. We compute it analytically and reproduce the known trend, then
inverse-design the sector angle to match a target response curve.

All quantities here are *kinematic* (geometry-only). They reproduce the right
qualitative trends but are not a substitute for a validated stress analysis -
stated plainly so nobody over-claims engineering accuracy.

Cell dimensions follow the standard parametrisation (Schenk & Guest 2013; Wei
et al., PRL 2013): facet sides ``a``, ``b``, acute sector angle ``alpha``, and
fold angle ``theta`` in ``(0, pi/2)`` (0 = flat/unfolded, pi/2 = tightly folded).
"""

from __future__ import annotations

import numpy as np


def cell_dims(alpha: float, theta: float, a: float = 1.0, b: float = 1.0):
    """Miura unit-cell half-dimensions ``(S, L, H, V)`` at sector ``alpha``, fold ``theta``.

    S - half width (corrugation direction), L - half length (major-fold
    direction), H - height, V - the other in-plane projection.
    """
    ta, ct, st, sa = np.tan(alpha), np.cos(theta), np.sin(theta), np.sin(alpha)
    den = np.sqrt(1 + ct ** 2 * ta ** 2)
    S = b * ct * ta / den
    L = a * np.sqrt(1 - sa ** 2 * st ** 2)
    H = a * sa * st
    V = b / den
    return S, L, H, V


def poisson_ratio(alpha: float, theta: float, a: float = 1.0, b: float = 1.0,
                  h: float = 1e-4) -> float:
    """In-plane Poisson's ratio ``nu_WL`` of a Miura-ori (negative = auxetic).

    Defined as ``-(dW/W)/(dL/L)`` as the cell deploys; W scales with S and L
    with L, so the constants drop out. Computed via a tiny central difference in
    the fold angle.
    """
    S1, L1, _, _ = cell_dims(alpha, theta - h, a, b)
    S2, L2, _, _ = cell_dims(alpha, theta + h, a, b)
    return float(-(np.log(S2) - np.log(S1)) / (np.log(L2) - np.log(L1)))


def deployment_ratio(alpha: float, theta: float, a: float = 1.0, b: float = 1.0) -> float:
    """Footprint area at ``theta`` relative to the flat (theta->0) footprint.

    1.0 when flat, shrinking as it folds - how compactly the panel stows.
    """
    S, L, _, _ = cell_dims(alpha, theta, a, b)
    S0, L0, _, _ = cell_dims(alpha, 1e-4, a, b)
    return float((S * L) / (S0 * L0))


def stiffness_proxy(alpha: float, theta: float, a: float = 1.0, b: float = 1.0,
                    h: float = 1e-4) -> float:
    """Geometric stiffness proxy against in-plane stretch: ``1/|dL/dtheta|``.

    Large when a unit of force barely moves the structure, small when it deploys
    freely. Kinematic only - a relative trend, not an absolute modulus.
    """
    _, L1, _, _ = cell_dims(alpha, theta - h, a, b)
    _, L2, _, _ = cell_dims(alpha, theta + h, a, b)
    dL = (L2 - L1) / (2 * h)
    return float(1.0 / (abs(dL) + 1e-9))


def poisson_curve(alpha: float, thetas: np.ndarray) -> np.ndarray:
    """Poisson's ratio sampled over an array of fold angles."""
    return np.array([poisson_ratio(alpha, t) for t in thetas])


def fit_sector_angle(target_thetas: np.ndarray, target_nu: np.ndarray,
                     grid: np.ndarray | None = None) -> float:
    """Inverse design: find the sector angle whose auxetic curve best matches a
    target Poisson's-ratio curve (least squares over a sector-angle grid).
    """
    if grid is None:
        grid = np.radians(np.linspace(20, 85, 261))
    errs = [np.mean((poisson_curve(al, target_thetas) - target_nu) ** 2) for al in grid]
    return float(grid[int(np.argmin(errs))])
