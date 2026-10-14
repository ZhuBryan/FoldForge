"""Amortized (learned) inverse design (v3) - pure numpy.

v1/v2 (:mod:`foldforge.diff.inverse`) solve inverse design by running an
optimiser *per target*: every new shape pays a fresh gradient-descent loop.
v3 trains a small neural net ONCE that maps a target shape descriptor straight
to fold parameters, so a new target is answered by a single forward pass - no
per-shape optimisation. This is the "amortized" / learned inverse design idea:
pay a training cost up front, then design in microseconds.

The parameterised family is the differentiable Miura (``foldforge.diff.miura``):
four design parameters ``(a, b, gamma, h)`` fold to a shape whose descriptor we
take to be the folded footprint ``(width, depth, height)`` (a smooth closed form
of the parameters, see :func:`descriptor_from_params`). The net learns the
inverse map ``descriptor -> params``.

Training loss (option **b**, self-supervised THROUGH the simulator - the real
"differentiable inverse design" point): we do NOT regress against ground-truth
parameters. Instead, for each target descriptor the net predicts params, we
*fold* those predicted params with the differentiable forward, and the loss is
the squared error between the folded-back descriptor and the target::

    descriptor --MLP--> latent y --transform--> params --fold--> descriptor'
                         loss = || descriptor' - descriptor ||^2

The gradient of the (transform + fold) stage w.r.t. the latent ``y`` is taken by
finite differences over the closed-form forward (4 latent dims, batched - the
same FD style the repo already uses in ``miura.fit_miura``); the gradient of the
MLP is hand-derived backprop. Because the loss only needs the forward fold, no
ground-truth params are required - the simulator *is* the supervisor. (We still
record the sampling params so we can report an honest train/test gap and a
reconstruction scatter, but they never enter the loss.)

The latent -> params transform keeps every predicted design foldable by
construction (bounded ``a, b, gamma`` via sigmoids and ``h`` as a fraction of
the flat-fold limit ``a*sin(gamma)``), so the net can never emit an invalid fold.

Honesty / scope: this is amortized inverse design over the *Miura family's
footprint distribution it is trained on*. It generalises to new targets drawn
from that family (the held-out split below) and returns instantly, but it is NOT
a general shape solver: off-distribution descriptors (shapes no Miura footprint
can produce) degrade gracefully to the nearest reachable Miura, exactly as the
4-DOF model allows - the same fundamental limit v1/v2 have, now paid once.
The footprint is a 3-number descriptor of a 4-parameter family, so the inverse
is a *manifold* (many params give the same footprint); the net learns one
consistent right-inverse, which is why we report reconstruction error (the
honest, well-defined metric) rather than per-parameter recovery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- foldable parameter ranges (the training distribution) -------------------
A_MIN, A_MAX = 0.6, 1.4
B_MIN, B_MAX = 0.6, 1.4
G_MIN, G_MAX = np.radians(30.0), np.radians(75.0)
H_FRAC_MIN, H_FRAC_SPAN = 0.05, 0.90   # h = frac * a*sin(gamma), frac in [.05,.95]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


# --------------------------------------------------------------------------
# The differentiable forward: params -> folded descriptor (closed form)
# --------------------------------------------------------------------------

def descriptor_from_params(P, rows, cols):
    """Folded footprint descriptor ``(width, depth, height)`` of Miura params.

    ``P`` has shape ``(..., 4)`` = ``(a, b, gamma, h)``. This is the exact,
    batched closed form of :func:`foldforge.diff.miura.footprint` (verified
    equal in the tests): for an ``rows x cols`` tessellation,

        w  = sqrt(a^2 - h^2);  sx = a*b*cos(gamma)/w;  d = sqrt(b^2 - sx^2)
        width = rows*w + sx;   depth = cols*d;         height = h
    """
    P = np.asarray(P, dtype=float)
    a, b, g, h = P[..., 0], P[..., 1], P[..., 2], P[..., 3]
    w = np.sqrt(np.maximum(a ** 2 - h ** 2, 1e-12))
    sx = a * b * np.cos(g) / w
    d = np.sqrt(np.maximum(b ** 2 - sx ** 2, 0.0))
    width = rows * w + sx
    depth = cols * d
    height = h
    return np.stack([width, depth, height], axis=-1)


def params_from_latent(Y):
    """Map an unconstrained latent ``Y`` (``(..., 4)``) to foldable Miura params.

    Smooth, bounded transforms guarantee every predicted design is a valid fold:
    ``a, b, gamma`` are sigmoid-squashed into their ranges, and ``h`` is a
    fraction of the flat-fold limit ``a*sin(gamma)`` so ``0 < h < a*sin(gamma)``
    always holds. Fully differentiable.
    """
    Y = np.asarray(Y, dtype=float)
    a = A_MIN + (A_MAX - A_MIN) * _sigmoid(Y[..., 0])
    g = G_MIN + (G_MAX - G_MIN) * _sigmoid(Y[..., 1])
    b = B_MIN + (B_MAX - B_MIN) * _sigmoid(Y[..., 2])
    frac = H_FRAC_MIN + H_FRAC_SPAN * _sigmoid(Y[..., 3])
    h = frac * a * np.sin(g)
    return np.stack([a, b, g, h], axis=-1)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

@dataclass
class Dataset:
    X_train: np.ndarray      # (n_train, 3) target descriptors (net input)
    P_train: np.ndarray      # (n_train, 4) sampling params (reference only)
    X_test: np.ndarray
    P_test: np.ndarray
    x_mean: np.ndarray       # input normalisation (fit on train)
    x_std: np.ndarray
    d_mean: np.ndarray       # descriptor-space scale for the loss
    d_std: np.ndarray
    rows: int
    cols: int


def make_dataset(n=600, rows=4, cols=4, test_frac=0.2, seed=0):
    """Sample ``n`` random foldable Miura params and their folded descriptors.

    Returns a :class:`Dataset` with a held-out test split. The descriptors are
    the only thing the net sees; the params are kept purely so we can report an
    honest generalisation gap and a reconstruction scatter - they never enter
    the (self-supervised) loss.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(A_MIN, A_MAX, n)
    b = rng.uniform(B_MIN, B_MAX, n)
    g = rng.uniform(G_MIN, G_MAX, n)
    frac = rng.uniform(0.1, 0.9, n)
    h = frac * a * np.sin(g)
    P = np.stack([a, b, g, h], axis=1)
    X = descriptor_from_params(P, rows, cols)

    n_test = int(round(n * test_frac))
    idx = rng.permutation(n)
    te, tr = idx[:n_test], idx[n_test:]
    x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-9
    return Dataset(
        X_train=X[tr], P_train=P[tr], X_test=X[te], P_test=P[te],
        x_mean=x_mean, x_std=x_std, d_mean=X[tr].mean(0), d_std=X[tr].std(0) + 1e-9,
        rows=rows, cols=cols)


# --------------------------------------------------------------------------
# The MLP (pure numpy, hand-derived backprop)
# --------------------------------------------------------------------------

@dataclass
class MLP:
    """Tiny 2-hidden-layer tanh MLP: descriptor (3) -> latent (4)."""

    W1: np.ndarray; b1: np.ndarray
    W2: np.ndarray; b2: np.ndarray
    W3: np.ndarray; b3: np.ndarray

    @staticmethod
    def init(hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        s = lambda m, k: rng.normal(0, np.sqrt(2.0 / m), (m, k))
        return MLP(
            W1=s(3, hidden), b1=np.zeros(hidden),
            W2=s(hidden, hidden), b2=np.zeros(hidden),
            W3=s(hidden, 4), b3=np.zeros(4))

    def forward(self, x):
        """Return ``(y, cache)``; ``y`` is the latent, ``cache`` feeds backprop."""
        z1 = x @ self.W1 + self.b1; h1 = np.tanh(z1)
        z2 = h1 @ self.W2 + self.b2; h2 = np.tanh(z2)
        y = h2 @ self.W3 + self.b3
        return y, (x, h1, h2)

    def backward(self, cache, dy):
        """Backprop output-gradient ``dy`` (dL/dy) into parameter gradients."""
        x, h1, h2 = cache
        gW3 = h2.T @ dy;            gb3 = dy.sum(0)
        dh2 = dy @ self.W3.T * (1 - h2 ** 2)
        gW2 = h1.T @ dh2;           gb2 = dh2.sum(0)
        dh1 = dh2 @ self.W2.T * (1 - h1 ** 2)
        gW1 = x.T @ dh1;            gb1 = dh1.sum(0)
        return MLP(gW1, gb1, gW2, gb2, gW3, gb3)

    # vector helpers for the Adam loop
    def _items(self):
        return (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)

    def axpy(self, other, step):
        a = self._items(); o = other._items()
        return MLP(*[p + step * q for p, q in zip(a, o)])


def predict_params(mlp, X, ds):
    """Instant inverse design: descriptors ``X`` -> predicted Miura params.

    One MLP forward pass + the bounded transform. ``ds`` supplies the input
    normalisation. Vectorised over any number of targets.
    """
    Xn = (np.asarray(X, float) - ds.x_mean) / ds.x_std
    y, _ = mlp.forward(Xn)
    return params_from_latent(y)


# --------------------------------------------------------------------------
# Training (self-supervised through the simulator)
# --------------------------------------------------------------------------

def _sim_jacobian(y, target_n, ds):
    """FD Jacobian of the normalised folded-descriptor w.r.t. the latent ``y``.

    Folds through ``params_from_latent`` then ``descriptor_from_params`` (both
    closed form), normalises by the descriptor scale, and returns both the
    per-sample residual ``(B,3)`` and Jacobian ``J`` ``(B,3,4)`` by central
    differences on the 4 latent dims (batched; 8 forward folds total). This is
    the gradient of the differentiable forward used to push the loss back into
    the net.
    """
    def fold_n(yy):
        P = params_from_latent(yy)
        D = descriptor_from_params(P, ds.rows, ds.cols)
        return (D - ds.d_mean) / ds.d_std      # normalised descriptor

    D0 = fold_n(y)                             # (B,3)
    res = D0 - target_n                        # (B,3)
    B = y.shape[0]
    J = np.zeros((B, 3, 4))
    eps = 1e-5
    for k in range(4):
        yp = y.copy(); yp[:, k] += eps
        ym = y.copy(); ym[:, k] -= eps
        J[:, :, k] = (fold_n(yp) - fold_n(ym)) / (2 * eps)
    return res, J


def train(ds, *, hidden=32, epochs=150, batch=64, lr=5e-3, seed=0):
    """Train the amortized inverse-design MLP. Returns ``(mlp, history)``.

    ``history`` is the mean per-target reconstruction loss (normalised
    descriptor MSE) per epoch. Self-supervised: the loss is the folded-back
    descriptor error (option b), so ground-truth params are never used.
    """
    mlp = MLP.init(hidden=hidden, seed=seed)
    rng = np.random.default_rng(seed + 1)
    Xn = (ds.X_train - ds.x_mean) / ds.x_std
    Tn = (ds.X_train - ds.d_mean) / ds.d_std           # targets in loss space
    n = len(Xn)

    # Adam state over the flat parameter list
    mom = MLP(*[np.zeros_like(p) for p in mlp._items()])
    vel = MLP(*[np.zeros_like(p) for p in mlp._items()])
    b1, b2, aeps = 0.9, 0.999, 1e-8
    history = []
    step = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        ep_loss = 0.0
        for s in range(0, n, batch):
            bi = order[s:s + batch]
            xb, tb = Xn[bi], Tn[bi]
            y, cache = mlp.forward(xb)
            res, J = _sim_jacobian(y, tb, ds)          # (B,3), (B,3,4)
            loss = float(np.mean(np.sum(res ** 2, axis=1)))
            ep_loss += loss * len(bi)
            # dL/dy = (2/B) res @ J   (per sample res_c * J_{c,k}), mean over batch
            dL_dres = (2.0 / len(bi)) * res            # (B,3)
            dy = np.einsum("bc,bck->bk", dL_dres, J)   # (B,4)
            grad = mlp.backward(cache, dy)
            step += 1
            new = []
            items = mlp._items(); gs = grad._items()
            ms = mom._items(); vs = vel._items()
            upd_m, upd_v, upd_p = [], [], []
            for p, gp, m, v in zip(items, gs, ms, vs):
                m = b1 * m + (1 - b1) * gp
                v = b2 * v + (1 - b2) * gp ** 2
                mhat = m / (1 - b1 ** step)
                vhat = v / (1 - b2 ** step)
                p = p - lr * mhat / (np.sqrt(vhat) + aeps)
                upd_m.append(m); upd_v.append(v); upd_p.append(p)
            mlp = MLP(*upd_p); mom = MLP(*upd_m); vel = MLP(*upd_v)
        history.append(ep_loss / n)
    return mlp, np.array(history)


# --------------------------------------------------------------------------
# Evaluation: fold the predicted params and measure reconstruction error
# --------------------------------------------------------------------------

def reconstruction_error(mlp, X, ds):
    """RMS reconstruction error of the learned inverse on descriptors ``X``.

    Predicts params, folds them back to descriptors, and returns both the
    per-target RMS error and the mean (over targets) of the per-target
    RMS - i.e. how close the folded prediction lands to the requested shape.
    """
    X = np.asarray(X, float)
    P = predict_params(mlp, X, ds)
    D = descriptor_from_params(P, ds.rows, ds.cols)
    per = np.sqrt(np.mean((D - X) ** 2, axis=1))
    return per, float(per.mean())
