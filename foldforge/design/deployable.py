"""Deployable-structure design: a flat sheet that stows compactly and deploys.

This is the aerospace/engineering face of origami - solar arrays, antenna
reflectors and booms that launch folded into a small volume and unfold on orbit.
The canonical pattern is the **Miura-ori**: a single-DOF rigidly-foldable
tessellation. One parameter folds or unfolds the *entire* array, so one actuator
deploys it, and the panels move as rigid plates (edge lengths preserved), so the
sheet never stretches or tears.

We reuse the existing differentiable Miura engine unchanged:

  * :func:`foldforge.diff.miura.folded_miura` is the exact rigid fold - vertex
    positions as a closed form of ``(a, b, gamma, h)``, where ``h`` is the one
    mechanism DOF (``0`` flat/deployed .. ``a*sin(gamma)`` flat-folded/stowed).
  * :func:`foldforge.diff.inverse.inverse_fit` designs ``(a, b, gamma, h)`` so a
    folded footprint hits a target - here we aim the *deployed* state.

:class:`MiuraDeployable` wraps that with a deployment parameter ``t`` in
``[0, 1]`` (0 = stowed/compact, 1 = deployed/flat-ish) and reports the headline
engineering numbers: the compaction ratio (deployed footprint / stowed
footprint) and the max edge-length deviation across the deployment sweep (the
rigid-foldability number - the array moves isometrically).

Honest scope: this is the GEOMETRIC / KINEMATIC design layer - compaction,
single-DOF rigid-foldability, and inverse design to a target. It models no
material thickness, hinge stiffness, panel mass or actuation force; it is the
front end of deployable-structure design, not a structural/FEA tool. A real
hinged array with thick panels deviates from the zero-thickness kinematics
(thickness accommodation is its own design problem). The rigid single-DOF motion
is the genuine property demonstrated, and the edge-length preservation is
reported as measured. A curved deployed dish is deferred: it needs the
generalized / tapered Miura (the warped-mesh path in ``inverse_design_shape``),
which trades exact single-DOF rigid folding for a small strain residual and so
doesn't belong in a clean rigid-deployment demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from foldforge.diff.miura import (
    flat_miura, folded_miura, fold_limit, footprint)


def _crease_edges(rows: int, cols: int) -> np.ndarray:
    """Flat-index pairs of the real Miura fold creases on an ``(rows+1,cols+1)``
    vertex grid: the horizontal and vertical grid lines (mountain/valley folds).

    Diagonals are deliberately excluded - they are facet interiors, not creases.
    Both crease families have lengths preserved exactly by the rigid fold, so
    these are the edges whose preservation certifies rigid-foldability.
    """
    R, C = rows + 1, cols + 1
    idx = lambda i, j: i * C + j
    E = []
    for i in range(R):
        for j in range(C):
            if j + 1 < C:
                E.append((idx(i, j), idx(i, j + 1)))
            if i + 1 < R:
                E.append((idx(i, j), idx(i + 1, j)))
    return np.array(E, dtype=int)


@dataclass
class MiuraDeployable:
    """A single-DOF deployable Miura array.

    The deployment parameter ``t`` in ``[0, 1]`` maps linearly onto the fold DOF
    ``h``: ``t=0`` is stowed (``h`` near the flat-folded limit, compact) and
    ``t=1`` is deployed (``h`` near 0, flat-ish). ``stow_frac`` / ``deploy_frac``
    are the fractions of the flat-folded limit used at the two ends (kept off the
    exact limit, where the fold is singular).
    """

    rows: int = 6
    cols: int = 6
    a: float = 1.0
    b: float = 1.0
    gamma: float = np.radians(60.0)
    stow_frac: float = 0.99
    deploy_frac: float = 0.02
    edges: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.edges is None:
            self.edges = _crease_edges(self.rows, self.cols)

    # --- deployment kinematics ------------------------------------------
    def height(self, t: float) -> float:
        """Fold DOF ``h`` at deployment ``t`` (t=0 stowed, t=1 deployed)."""
        lim = fold_limit(self.a, self.gamma)
        h_stow = self.stow_frac * lim
        h_deploy = self.deploy_frac * lim
        return float((1.0 - t) * h_stow + t * h_deploy)

    def vertices(self, t: float) -> np.ndarray:
        """Folded vertex grid at deployment ``t`` (shape ``(rows+1,cols+1,3)``)."""
        return folded_miura(self.rows, self.cols, self.a, self.b, self.gamma,
                            self.height(t))

    def footprint_area(self, t: float) -> float:
        """In-plane (x-y) bounding-box area of the array at deployment ``t``."""
        w, d, _ = footprint(self.rows, self.cols, self.a, self.b, self.gamma,
                            self.height(t))
        return float(w * d)

    def deployed_surface_area(self) -> float:
        """Total facet (panel) area - invariant, the rigid panels never stretch.

        Computed on the flat sheet (= every folded state, by rigidity): sum of
        the quad-facet areas.
        """
        V = flat_miura(self.rows, self.cols, self.a, self.b, self.gamma)
        area = 0.0
        for i in range(self.rows):
            for j in range(self.cols):
                p00 = V[i, j]; p10 = V[i + 1, j]
                p01 = V[i, j + 1]; p11 = V[i + 1, j + 1]
                area += 0.5 * np.linalg.norm(np.cross(p10 - p00, p01 - p00))
                area += 0.5 * np.linalg.norm(np.cross(p10 - p11, p01 - p11))
        return float(area)

    def compaction_ratio(self) -> float:
        """Headline number: deployed footprint / stowed footprint (>1)."""
        return self.footprint_area(1.0) / self.footprint_area(0.0)

    # --- rigid-foldability ----------------------------------------------
    def edge_lengths(self, t: float) -> np.ndarray:
        """Lengths of all crease edges at deployment ``t``."""
        V = self.vertices(t).reshape(-1, 3)
        d = V[self.edges[:, 0]] - V[self.edges[:, 1]]
        return np.linalg.norm(d, axis=1)

    def flat_edge_lengths(self) -> np.ndarray:
        """Crease-edge lengths of the undeployed flat sheet (the reference)."""
        V = flat_miura(self.rows, self.cols, self.a, self.b, self.gamma
                      ).reshape(-1, 3)
        d = V[self.edges[:, 0]] - V[self.edges[:, 1]]
        return np.linalg.norm(d, axis=1)

    def max_edge_deviation(self, ts=None) -> float:
        """Max |edge length - flat edge length| over the deployment sweep ``ts``.

        For an exact Miura this is ~machine precision: the whole array folds and
        unfolds as a rigid (isometric) mechanism driven by the single DOF. This
        is the rigid-foldability number.
        """
        if ts is None:
            ts = np.linspace(0.0, 1.0, 21)
        L0 = self.flat_edge_lengths()
        return float(max(np.max(np.abs(self.edge_lengths(t) - L0)) for t in ts))

    def summary(self, ts=None) -> dict:
        """Collect the headline metrics in one dict (for the demo / tests)."""
        return {
            "stowed_footprint": self.footprint_area(0.0),
            "deployed_footprint": self.footprint_area(1.0),
            "compaction_ratio": self.compaction_ratio(),
            "deployed_surface_area": self.deployed_surface_area(),
            "max_edge_deviation": self.max_edge_deviation(ts),
        }


def inverse_design_deployed(target_wd, *, rows=6, cols=6, stow_frac=0.99,
                            target_height=0.05, iters=400, lr=0.05, seed=0):
    """Design a :class:`MiuraDeployable` whose DEPLOYED state hits a target.

    Reuses :func:`foldforge.diff.inverse.inverse_fit` (the existing closed-form
    Miura inverse-design loop) to solve ``(a, b, gamma, h)`` so the *deployed*
    (near-flat) footprint matches the target in-plane bounding box
    ``target_wd = (width, depth)`` with a small deployed height. The solved
    ``(a, b, gamma)`` define the array; the solved ``h`` becomes the deployed end
    of the DOF (``deploy_frac``), and ``stow_frac`` sets the stowed end.

    Returns ``(deployable, result, achieved_wd)`` where ``result`` is the
    :class:`~foldforge.diff.inverse.InverseResult` from the loop and
    ``achieved_wd`` is the deployed footprint actually reached. The designed
    array is rigid-foldable by construction (it is a genuine Miura), so one
    actuator still deploys it and edge lengths are preserved across the sweep.
    """
    # ponytail: lazy import breaks the cycle origamize.surface -> design ->
    # deployable -> diff.inverse -> origamize.miura_fit -> origamize.surface.
    from foldforge.diff.inverse import inverse_fit

    W, D = float(target_wd[0]), float(target_wd[1])
    target = (W, D, float(target_height))
    result, achieved = inverse_fit(target, rows=rows, cols=cols,
                                   iters=iters, lr=lr, seed=seed)
    a, b, gamma, h = result.params
    a, b = abs(a), abs(b)
    lim = fold_limit(a, gamma)
    deploy_frac = float(np.clip(h / lim, 1e-3, 0.5))
    dep = MiuraDeployable(rows=rows, cols=cols, a=a, b=b, gamma=gamma,
                          stow_frac=stow_frac, deploy_frac=deploy_frac)
    w, d, _ = footprint(rows, cols, a, b, gamma, dep.height(1.0))
    return dep, result, (float(w), float(d))
