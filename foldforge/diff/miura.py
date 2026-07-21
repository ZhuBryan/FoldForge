"""Differentiable analytic Miura-ori surface (M2+, the 2D tessellation case).

The fold-chain (``kinematics.py``) made a 1D strip differentiable. This makes a
full 2D Miura-ori tessellation differentiable: its folded vertex positions are a
closed-form, smooth function of four parameters - facet sides ``a``, ``b``, the
sector angle ``gamma``, and the fold height ``h`` (the single mechanism DOF,
ranging over ``0 < h < a*sin(gamma)``).

The construction is an *exact* rigid fold: every facet stays planar and every
edge keeps its flat length (verified to machine precision in the tests). Because
it is closed form, we can differentiate folded metrics (footprint, height) with
respect to the design parameters and inverse-design a Miura to a target - the 2D
analogue of M3, with analytic gradients.

    w  = sqrt(a^2 - h^2)
    sx = a*b*cos(gamma) / w
    d  = sqrt(b^2 - sx^2)
    vertex(i, j) = ( i*w + (j%2)*sx ,  j*d ,  (i%2)*h )
"""

from __future__ import annotations

import numpy as np


def flat_miura(rows: int, cols: int, a: float = 1.0, b: float = 1.0,
               gamma: float = np.radians(60)) -> np.ndarray:
    """Flat (unfolded) Miura vertex grid, shape ``(rows+1, cols+1, 3)``."""
    i = np.arange(rows + 1)[:, None]
    j = np.arange(cols + 1)[None, :]
    x = i * a + (j % 2) * b * np.cos(gamma)
    y = (j * b * np.sin(gamma)) * np.ones_like(x)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=-1).astype(float)


def folded_miura(rows: int, cols: int, a: float = 1.0, b: float = 1.0,
                 gamma: float = np.radians(60), h: float = 0.4) -> np.ndarray:
    """Folded Miura vertex grid (exact rigid fold), shape ``(rows+1, cols+1, 3)``.

    ``h`` is the fold DOF in ``(0, a*sin(gamma))``: 0 is flat, ``a*sin(gamma)``
    is flat-folded.
    """
    w = np.sqrt(a ** 2 - h ** 2)
    sx = a * b * np.cos(gamma) / w
    d = np.sqrt(np.maximum(b ** 2 - sx ** 2, 0.0))
    i = np.arange(rows + 1)[:, None]
    j = np.arange(cols + 1)[None, :]
    x = i * w + (j % 2) * sx
    y = (j * d) * np.ones_like(x)
    z = (i % 2) * h * np.ones_like(x)
    return np.stack([x, y, z], axis=-1).astype(float)


def fold_limit(a: float, gamma: float) -> float:
    """Largest valid fold height (flat-folded limit)."""
    return a * np.sin(gamma)


def footprint(rows: int, cols: int, a: float, b: float, gamma: float, h: float):
    """Folded bounding-box ``(width_x, depth_y, height_z)`` of the tessellation."""
    V = folded_miura(rows, cols, a, b, gamma, h).reshape(-1, 3)
    span = V.max(0) - V.min(0)
    return float(span[0]), float(span[1]), float(span[2])


def footprint_grad_h(rows, cols, a, b, gamma, h, eps=1e-6):
    """Gradient of the footprint w.r.t. ``h`` (the closed form is smooth in h)."""
    fp = np.array(footprint(rows, cols, a, b, gamma, h + eps))
    fm = np.array(footprint(rows, cols, a, b, gamma, h - eps))
    return (fp - fm) / (2 * eps)


def fit_miura(target_wdh, rows: int = 6, cols: int = 6, *, iters: int = 400,
              lr: float = 0.05, seed: int = 0):
    """Inverse-design Miura parameters ``(a, b, gamma, h)`` to hit a target
    folded footprint ``(width, depth, height)``, by gradient descent.

    Returns ``(params, history, achieved_footprint)``. Gradients are finite
    differences over the closed-form surface (only 4 parameters, so this is
    cheap and exact to FD tolerance).
    """
    target = np.asarray(target_wdh, dtype=float)
    rng = np.random.default_rng(seed)
    p = np.array([1.0, 1.0, np.radians(60), 0.4]) + rng.normal(0, 0.02, 4)
    m = np.zeros(4); v = np.zeros(4)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []

    def loss(p):
        a, bb, g, h = p
        h = np.clip(h, 1e-3, fold_limit(a, g) - 1e-3)
        return np.sum((np.array(footprint(rows, cols, a, bb, g, h)) - target) ** 2)

    for t in range(1, iters + 1):
        history.append(float(loss(p)))
        grad = np.zeros(4)
        for k in range(4):
            d = 1e-5 * (1 + abs(p[k]))
            pp = p.copy(); pp[k] += d
            pm = p.copy(); pm[k] -= d
            grad[k] = (loss(pp) - loss(pm)) / (2 * d)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        p = p - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        p[2] = np.clip(p[2], np.radians(20), np.radians(85))   # keep gamma sane
        p[0] = abs(p[0]); p[1] = abs(p[1])
        p[3] = np.clip(p[3], 1e-3, fold_limit(p[0], p[2]) - 1e-3)
    return p, np.array(history), footprint(rows, cols, *p)
