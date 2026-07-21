"""JAX port of the fold kinematics: automatic gradients, GPU-ready (the moonshot).

Because the core math was written in plain numpy, porting it to ``jax.numpy`` is
almost a find-and-replace - and then ``jax.grad`` gives exact forward/reverse-mode
derivatives for free, with no hand-derived Jacobians, and the same code runs on a
GPU/TPU. This module mirrors ``foldforge.diff`` (fold chain + folded Miura) in
JAX and exposes autodiff'd gradients. It's optional: import it only if JAX is
installed (``pip install jax``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)        # match numpy float64 precision


def fold_chain(angles, seg: float = 1.0):
    """Folded fold-chain spine (JAX), differentiable end to end."""
    headings = jnp.cumsum(angles)
    steps = seg * jnp.stack([jnp.cos(headings), jnp.sin(headings)], axis=1)
    return jnp.concatenate([jnp.zeros((1, 2)), jnp.cumsum(steps, axis=0)], axis=0)


def apex_height(angles, seg: float = 1.0):
    spine = fold_chain(angles, seg)
    return spine[spine.shape[0] // 2, 1]


def folded_miura(rows, cols, a, b, gamma, h):
    """Folded Miura vertex grid (JAX), differentiable in (a, b, gamma, h)."""
    w = jnp.sqrt(a ** 2 - h ** 2)
    sx = a * b * jnp.cos(gamma) / w
    d = jnp.sqrt(jnp.maximum(b ** 2 - sx ** 2, 0.0))
    i = jnp.arange(rows + 1)[:, None]
    j = jnp.arange(cols + 1)[None, :]
    x = i * w + (j % 2) * sx
    y = (j * d) * jnp.ones_like(x)
    z = (i % 2) * h * jnp.ones_like(x)
    return jnp.stack([x, y, z], axis=-1)


def miura_height(a, b, gamma, h, rows=4, cols=4):
    V = folded_miura(rows, cols, a, b, gamma, h).reshape(-1, 3)
    return V[:, 2].max() - V[:, 2].min()


# Free gradients - no hand-derived Jacobians:
apex_height_grad = jax.grad(apex_height)                          # d apex / d angles
miura_height_grad = jax.grad(miura_height, argnums=(0, 1, 2, 3))  # wrt (a,b,gamma,h)
