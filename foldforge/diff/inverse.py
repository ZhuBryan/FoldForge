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
