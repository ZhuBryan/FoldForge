"""Inverse design (Milestone 3): find the fold that makes a target shape.

Given a target curve, we search for the fold angles whose folded strip matches
it, by gradient descent. The gradient of the match w.r.t. every fold angle comes
in closed form from the differentiable kinematics (M2), so this is fast and
exact - no finite differences in the inner loop.

We match an *ordered* curve (joint k to target sample k), which is the right
correspondence for an open profile and converges cleanly. ``chamfer_distance``
is provided as a correspondence-free quality metric for reporting.

Because the panels are rigid, the chain has a fixed total length, so a target
must be sampled at equal arc-length spacing for the chain to reach it; the
``target_*`` helpers and :func:`resample_arclength` take care of that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foldforge.diff.kinematics import fold_chain, spine_jacobian


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between two 2D point sets."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean()) / 2.0


def angles_from_curve(target: np.ndarray):
    """Closed-form fold angles whose chain *exactly* traces ``target``.

    A fold chain's segment headings are the cumulative sum of its fold angles,
    so the angles that reproduce an (arc-length-sampled) curve are simply that
    curve's own turning angles: ``angle[k] = heading_k - heading_{k-1}``. This
    is exact and instant - no optimisation - whenever the target is sampled at
    the chain's segment length (which the origamizer ensures). Returns
    ``(angles, seg)``.
    """
    target = np.asarray(target, dtype=float)
    seg = np.linalg.norm(np.diff(target, axis=0), axis=1)
    dirs = np.diff(target, axis=0)
    headings = np.arctan2(dirs[:, 1], dirs[:, 0])
    angles = np.diff(headings, prepend=0.0)
    angles = (angles + np.pi) % (2 * np.pi) - np.pi      # wrap each turn to (-pi, pi]
    return angles, float(seg.mean())


def resample_arclength(curve: np.ndarray, n: int) -> np.ndarray:
    """Resample an ordered curve to ``n+1`` points at equal arc-length spacing."""
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    target_s = np.linspace(0, s[-1], n + 1)
    x = np.interp(target_s, s, curve[:, 0])
    z = np.interp(target_s, s, curve[:, 1])
    return np.stack([x, z], axis=1)


@dataclass
class FitResult:
    """Recovered fold + how well it matched."""

    angles: np.ndarray          # recovered fold angles
    spine: np.ndarray           # the folded curve they produce
    losses: np.ndarray          # loss per iteration (should decrease)
    chamfer: float              # final Chamfer distance to the target


def fit_chain(target: np.ndarray, *, seg: float | None = None, iters: int = 600,
              lr: float = 0.05, seed: int = 0) -> FitResult:
    """Recover fold angles so the chain's spine matches ``target``.

    ``target`` is an ordered ``(n+1, 2)`` curve in the x-z plane. Uses Adam on
    the mean-squared joint residual; the gradient is analytic (from the
    kinematics Jacobian). If ``seg`` is None it is set to the target's mean point
    spacing so the rigid, fixed-length chain can actually reach the curve.
    """
    n = len(target) - 1
    if seg is None:
        seg = float(np.linalg.norm(np.diff(target, axis=0), axis=1).mean())
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-0.05, 0.05, n)        # start nearly flat
    m = np.zeros(n); v = np.zeros(n)
    b1, b2, eps = 0.9, 0.999, 1e-8
    losses = []
    for t in range(1, iters + 1):
        spine = fold_chain(angles, seg=seg).spine
        resid = spine - target
        losses.append(float(np.mean(np.sum(resid ** 2, axis=1))))
        jac = spine_jacobian(angles, seg=seg)               # (n+1, 2, n)
        grad = (2.0 / (n + 1)) * np.einsum("kd,kdj->j", resid, jac)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        angles = angles - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    spine = fold_chain(angles, seg=seg).spine
    return FitResult(angles=angles, spine=spine, losses=np.array(losses),
                     chamfer=chamfer_distance(spine, target))


# --- target shapes (each arc-length consistent so a chain can reach them) ----

def target_arch(n: int, length: float = 24.0) -> np.ndarray:
    """Semicircular arch of total arc-length ``length`` (dome cross-section)."""
    r = length / np.pi
    t = np.linspace(0, np.pi, n + 1)
    return np.stack([r * (1 - np.cos(t)), r * np.sin(t)], axis=1)


def target_wave(n: int, length: float = 24.0, cycles: float = 1.0,
                amp: float = 2.5) -> np.ndarray:
    """A saddle-like S-wave, resampled to equal arc-length spacing."""
    x = np.linspace(0, length, 400)
    z = amp * np.sin(2 * np.pi * cycles * x / length)
    return resample_arclength(np.stack([x, z], axis=1), n)


def target_step(n: int, length: float = 24.0, rise: float = 4.0) -> np.ndarray:
    """A flat-rise-flat 'table' profile (a harder, cornered target)."""
    x = np.linspace(0, length, 400)
    z = (np.clip((x - length * 0.35) * 4, 0, rise)
         - np.clip((x - length * 0.65) * 4, 0, rise))
    return resample_arclength(np.stack([x, z], axis=1), n)
