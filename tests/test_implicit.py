"""Test implicit differentiation of the general fold solver vs finite differences."""

import numpy as np

from foldforge.geometry import examples
from foldforge.sim import FoldMesh
from foldforge.diff.implicit import equilibrium, energy_grad, implicit_grad, distance_output


def test_equilibrium_has_zero_force():
    m = FoldMesh.from_pattern(examples.miura(2, 2))
    rho = np.array([h.target * 0.4 if h.is_crease else 0.0 for h in m.hinges])
    x = equilibrium(m, rho)
    assert np.linalg.norm(energy_grad(m, x.reshape(-1), rho)) < 1e-6


def test_implicit_grad_matches_finite_differences():
    m = FoldMesh.from_pattern(examples.miura(2, 2))
    rho = np.array([h.target * 0.4 if h.is_crease else 0.0 for h in m.hinges])
    value, grad = distance_output(0, len(m.vertices) - 1)
    x = equilibrium(m, rho)
    g_imp = implicit_grad(m, rho, grad, x_star=x)

    eps = 1e-3
    creases = [i for i, h in enumerate(m.hinges) if h.is_crease]
    for hi in creases:
        rp = rho.copy(); rp[hi] += eps
        rm = rho.copy(); rm[hi] -= eps
        fd = (value(equilibrium(m, rp)) - value(equilibrium(m, rm))) / (2 * eps)
        assert abs(g_imp[hi] - fd) < 1e-4


def test_analytic_hessian_matches_numerical():
    from foldforge.diff.implicit import energy_hessian, _num_hessian
    m = FoldMesh.from_pattern(examples.miura(2, 2))
    rho = np.array([h.target * 0.4 if h.is_crease else 0.0 for h in m.hinges])
    x = equilibrium(m, rho).reshape(-1)
    Ha = energy_hessian(m, x, rho, sparse=False)
    Hn = _num_hessian(m, x, rho, 8.0)
    assert Ha.shape == Hn.shape
    assert np.allclose(Ha, Hn, rtol=1e-5, atol=1e-6)
    # analytic Hessian is symmetric (rigid-motion gauge aside)
    assert np.allclose(Ha, Ha.T, atol=1e-9)


def test_sparse_hessian_matches_dense():
    import pytest
    sp = pytest.importorskip("scipy.sparse")
    from foldforge.diff.implicit import energy_hessian
    m = FoldMesh.from_pattern(examples.miura(2, 2))
    rho = np.array([h.target * 0.4 if h.is_crease else 0.0 for h in m.hinges])
    x = equilibrium(m, rho).reshape(-1)
    Hs = energy_hessian(m, x, rho, sparse=True)
    Hd = energy_hessian(m, x, rho, sparse=False)
    assert sp.issparse(Hs)
    assert np.allclose(Hs.toarray(), Hd, atol=1e-9)


def test_implicit_grad_sparse_path_matches_dense():
    m = FoldMesh.from_pattern(examples.miura(2, 2))
    rho = np.array([h.target * 0.4 if h.is_crease else 0.0 for h in m.hinges])
    value, grad = distance_output(0, len(m.vertices) - 1)
    x = equilibrium(m, rho)
    g_dense = implicit_grad(m, rho, grad, x_star=x, sparse=False)
    g_sparse = implicit_grad(m, rho, grad, x_star=x, sparse=True)
    assert np.allclose(g_dense, g_sparse, atol=1e-6)
