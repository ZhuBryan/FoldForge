"""Implicit differentiation of the general fold solver (the roadmap's holy grail).

The forward fold is the minimiser of an energy

    E(x, rho) = (k_bar/2) sum (|edge| - L0)^2  +  sum (k/2)(theta_hinge(x) - rho)^2

i.e. stiff bars keep panels rigid while each hinge is pulled to its target angle
``rho``. At the equilibrium ``x*`` we have ``grad_x E = 0`` for *any* crease
graph - not just chains or Miuras.

To get ``d(output)/d(rho)`` we do NOT backpropagate through the solver iterations
(huge memory). Instead we use the implicit function theorem at the fixed point:
differentiating ``grad_x E(x*, rho) = 0`` gives

    H dx*/drho = -d(grad_x E)/drho ,   H = d^2E/dx^2  (the Hessian at x*),

so we solve one linear system. ``E`` is invariant to rigid motion, so ``H`` has a
6-D null space; we use the least-squares (pseudo-inverse) solution, which is
exact for *gauge-invariant* outputs (distances, angles) - the ones that matter.

The Hessian is assembled analytically and *locally*: every bar and every hinge
touches at most four vertices, so ``H`` is sparse and the whole assembly is
``O(V)`` rather than the ``O(V^2)`` of finite-differencing the global gradient.
Concretely each block is:

    * bars    - the exact 3x3 axial-spring stiffness (closed form below);
    * hinges  - the exact rank term ``k (grad theta)(grad theta)^T`` plus the
                curvature term ``k (theta - rho) d^2 theta``. Only the 12x12
                dihedral curvature is filled by a compact *local* central
                difference of the analytic hinge gradient (24 O(1) evals per
                hinge) - still ``O(V)``, and it matches the old dense numerical
                Hessian to finite-difference precision (see the tests).

If SciPy is present the Hessian is returned as ``scipy.sparse``; otherwise a dense
array is assembled with the identical blocks. ``_num_hessian`` is kept for the
cross-check.
"""

from __future__ import annotations

import numpy as np

from foldforge.sim.mesh import dihedral_angle, dihedral_grad

try:                                        # optional accelerator
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
except Exception:                           # pragma: no cover
    _sp = None
    _spla = None


def _stencils(mesh):
    H = [(int(h.edge[0]), int(h.edge[1]), int(h.wings[0]), int(h.wings[1]))
         for h in mesh.hinges]
    kc = np.array([1.0 if h.is_crease else 1.5 for h in mesh.hinges])
    return H, kc


def energy_grad(mesh, x, targets, k_bar=8.0):
    """Analytic gradient of the fold energy w.r.t. all vertex coordinates (flat)."""
    H, kc = _stencils(mesh)
    x = x.reshape(-1, 3)
    g = np.zeros_like(x)
    a, b = mesh.bars[:, 0], mesh.bars[:, 1]
    d = x[b] - x[a]
    ln = np.linalg.norm(d, axis=1)
    fb = (k_bar * (ln - mesh.rest_lengths) / ln)[:, None] * d
    np.add.at(g, a, fb)
    np.add.at(g, b, -fb)
    for hi, (i, j, k, l) in enumerate(H):
        th = dihedral_angle(x[i], x[j], x[k], x[l])
        gg = dihedral_grad(x[i], x[j], x[k], x[l])
        c = kc[hi] * (th - targets[hi])
        g[i] += c * gg[0]; g[j] += c * gg[1]; g[k] += c * gg[2]; g[l] += c * gg[3]
    return g.reshape(-1)


def _num_hessian(mesh, x, targets, k_bar, eps=1e-5):
    """Dense Hessian by finite-differencing the analytic gradient (O(n^2))."""
    n = len(x)
    M = np.zeros((n, n))
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        M[:, i] = (energy_grad(mesh, xp, targets, k_bar)
                   - energy_grad(mesh, xm, targets, k_bar)) / (2 * eps)
    return 0.5 * (M + M.T)


def _bar_block(d, ln, rest, k_bar):
    """Exact 3x3 axial-spring stiffness ``d(force_on_b)/d(d)`` for one bar.

    With ``force_on_b = k (ln - rest)/ln * d`` and ``d = x[b] - x[a]`` this is
    ``k[ (ln-rest)/ln I + rest/ln^3 d d^T ]`` - the standard spring Hessian.
    """
    I3 = np.eye(3)
    return k_bar * ((ln - rest) / ln * I3 + rest / ln ** 3 * np.outer(d, d))


def _dihedral_hessian(p1, p2, p3, p4, eps=1e-6):
    """12x12 second derivative of the dihedral angle at one hinge.

    Filled by a compact local central difference of the *analytic* gradient
    ``dihedral_grad`` over just the 12 local DOFs - O(1) per hinge, and exact to
    finite-difference precision. Kept symmetric.
    """
    pts = [np.asarray(p, float) for p in (p1, p2, p3, p4)]
    Hh = np.zeros((12, 12))
    for c in range(12):
        node, comp = divmod(c, 3)
        pp = [p.copy() for p in pts]; pp[node][comp] += eps
        pm = [p.copy() for p in pts]; pm[node][comp] -= eps
        gp = np.concatenate(dihedral_grad(*pp))
        gm = np.concatenate(dihedral_grad(*pm))
        Hh[:, c] = (gp - gm) / (2 * eps)
    return 0.5 * (Hh + Hh.T)


def energy_hessian(mesh, x, targets, k_bar=8.0, *, sparse=None):
    """Analytic Hessian ``d^2 E / dx^2`` at ``x`` (flat coords).

    Assembled block-locally in O(V). Returns a ``scipy.sparse`` matrix when SciPy
    is available (and ``sparse`` is not ``False``), otherwise a dense array. Set
    ``sparse=False`` to force a dense result, ``sparse=True`` to require SciPy.
    """
    if sparse is None:
        sparse = _sp is not None
    if sparse and _sp is None:
        raise RuntimeError("energy_hessian(sparse=True) needs SciPy")

    H, kc = _stencils(mesh)
    xv = x.reshape(-1, 3)
    n = xv.size
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def scatter(block, nodes):
        """Add a (3k x 3k) block over vertex indices ``nodes``."""
        for bi, ni in enumerate(nodes):
            for bj, nj in enumerate(nodes):
                sub = block[3 * bi:3 * bi + 3, 3 * bj:3 * bj + 3]
                for a in range(3):
                    for b in range(3):
                        v = sub[a, b]
                        if v != 0.0:
                            rows.append(3 * ni + a)
                            cols.append(3 * nj + b)
                            data.append(v)

    # --- bars: exact axial-spring blocks ---
    a_idx, b_idx = mesh.bars[:, 0], mesh.bars[:, 1]
    for e in range(len(a_idx)):
        ai, bi = int(a_idx[e]), int(b_idx[e])
        d = xv[bi] - xv[ai]
        ln = float(np.linalg.norm(d))
        K = _bar_block(d, ln, mesh.rest_lengths[e], k_bar)
        # d(g[a])/d(x[a]) = -K, d(g[a])/d(x[b]) = +K, and symmetric.
        block = np.block([[-K, K], [K, -K]])
        scatter(block, (ai, bi))

    # --- hinges: rank term + curvature term ---
    for hi, (i, j, k, l) in enumerate(H):
        gg = dihedral_grad(xv[i], xv[j], xv[k], xv[l])
        g12 = np.concatenate(gg)                       # (12,) grad theta
        th = dihedral_angle(xv[i], xv[j], xv[k], xv[l])
        c = kc[hi] * (th - targets[hi])
        block = kc[hi] * np.outer(g12, g12) + c * _dihedral_hessian(
            xv[i], xv[j], xv[k], xv[l])
        scatter(block, (i, j, k, l))

    if sparse:
        M = _sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        return 0.5 * (M + M.T)
    M = np.zeros((n, n))
    np.add.at(M, (np.array(rows, int), np.array(cols, int)), np.array(data))
    return 0.5 * (M + M.T)


def _lstsq_solve(H, b):
    """Least-squares (min-norm) solve of the gauge-singular ``H x = b``.

    ``b`` may be a vector or a matrix (columns solved independently). Dense uses
    the SVD pseudo-inverse; sparse uses LSQR per column.
    """
    if _sp is not None and _sp.issparse(H):
        B = np.asarray(b)
        if B.ndim == 1:
            return _spla.lsqr(H, B, atol=1e-12, btol=1e-12)[0]
        out = np.empty((H.shape[1], B.shape[1]))
        for c in range(B.shape[1]):
            out[:, c] = _spla.lsqr(H, B[:, c], atol=1e-12, btol=1e-12)[0]
        return out
    return np.linalg.lstsq(np.asarray(H), np.asarray(b), rcond=1e-8)[0]


def equilibrium(mesh, targets, *, k_bar=8.0, iters=60, seed=0, sparse=False):
    """Solve for the folded equilibrium ``x*`` (damped Newton on the energy)."""
    x = mesh.vertices.copy()
    x[:, 2] += np.random.default_rng(seed).uniform(-1e-2, 1e-2, len(x))
    x = x.reshape(-1)
    for _ in range(iters):
        g = energy_grad(mesh, x, targets, k_bar)
        if np.linalg.norm(g) < 1e-11:
            break
        Hs = energy_hessian(mesh, x, targets, k_bar, sparse=sparse)
        dx = _lstsq_solve(Hs, -g)
        x = x + 0.9 * dx
    return x.reshape(-1, 3)


def implicit_grad(mesh, targets, output_grad, *, x_star=None, k_bar=8.0,
                  eps=1e-5, sparse=False):
    """Gradient of a scalar output w.r.t. the hinge ``targets``, via the IFT.

    ``output_grad(x_flat) -> (3V,)`` is the gradient of the (gauge-invariant)
    scalar output w.r.t. the vertex coordinates. Returns ``d(output)/d(targets)``.
    """
    if x_star is None:
        x_star = equilibrium(mesh, targets, k_bar=k_bar, sparse=sparse)
    xf = x_star.reshape(-1)
    H = energy_hessian(mesh, xf, targets, k_bar, sparse=sparse)
    m = len(targets)
    Mx = np.zeros((len(xf), m))
    for hi in range(m):
        tp = np.array(targets, float); tp[hi] += eps
        tm = np.array(targets, float); tm[hi] -= eps
        Mx[:, hi] = (energy_grad(mesh, xf, tp, k_bar)
                     - energy_grad(mesh, xf, tm, k_bar)) / (2 * eps)
    dxdrho = _lstsq_solve(H, -Mx)
    return output_grad(xf) @ dxdrho


def distance_output(a: int, b: int):
    """Gauge-invariant output: squared distance between vertices ``a`` and ``b``.

    Returns ``(value_fn, grad_fn)`` for use with :func:`implicit_grad`.
    """
    def value(x):
        x = x.reshape(-1, 3)
        return float(np.sum((x[a] - x[b]) ** 2))

    def grad(x):
        x = x.reshape(-1, 3)
        g = np.zeros_like(x)
        diff = 2 * (x[a] - x[b])
        g[a] = diff; g[b] = -diff
        return g.reshape(-1)

    return value, grad
