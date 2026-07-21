"""JAX backend: autodiff must reproduce the hand-derived gradients.

Skipped automatically if JAX isn't installed (it's an optional dependency).
"""

import numpy as np
import pytest

pytest.importorskip("jax")


def test_jax_apex_grad_matches_analytic():
    import jax.numpy as jnp
    from foldforge.jaxsim import apex_height_grad as jax_grad
    from foldforge.diff.kinematics import apex_height_grad as analytic

    a = np.array([0.2, -0.3, 0.4, 0.1, 0.25, -0.15])
    assert np.abs(np.array(jax_grad(jnp.array(a))) - analytic(a)).max() < 1e-9


def test_jax_miura_grad_matches_finite_differences():
    import math
    from foldforge.jaxsim import miura_height, miura_height_grad

    p = [1.1, 0.95, math.radians(58), 0.45]
    jg = [float(x) for x in miura_height_grad(*p)]
    fd = []
    for k in range(4):
        e = 1e-6
        pp = p.copy(); pp[k] += e
        pm = p.copy(); pm[k] -= e
        fd.append((float(miura_height(*pp)) - float(miura_height(*pm))) / (2 * e))
    assert max(abs(j - f) for j, f in zip(jg, fd)) < 1e-6
