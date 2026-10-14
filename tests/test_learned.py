"""Tests for v3 amortized (learned) inverse design (``foldforge.diff.learned``).

Seeded, small net + dataset so the whole module trains in ~1s. Checks the three
claims: the self-supervised loss actually goes down, the trained net generalises
to held-out targets (low reconstruction error, small train/test gap), and a
single inference is orders of magnitude faster than the per-shape optimiser.
"""

import time

import numpy as np

from foldforge.diff import learned as L
from foldforge.diff.miura import fit_miura, footprint


def test_descriptor_matches_footprint():
    """The batched closed-form descriptor equals miura.footprint exactly."""
    ds = L.make_dataset(n=20, rows=4, cols=4, seed=1)
    for p in ds.P_train[:8]:
        d1 = L.descriptor_from_params(p, 4, 4)
        d2 = np.array(footprint(4, 4, *p))
        assert np.allclose(d1, d2, atol=1e-9)


def test_predicted_params_are_foldable():
    """The latent->params transform always yields a valid fold (h < a*sin g)."""
    ds = L.make_dataset(n=60, rows=4, cols=4, seed=2)
    mlp = L.MLP.init(hidden=16, seed=0)
    P = L.predict_params(mlp, ds.X_test, ds)
    a, b, g, h = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    assert np.all(a > 0) and np.all(b > 0)
    assert np.all(h > 0) and np.all(h < a * np.sin(g))


def test_training_reduces_loss():
    """Self-supervised (through-the-sim) loss drops substantially."""
    ds = L.make_dataset(n=300, rows=4, cols=4, seed=0)
    _, hist = L.train(ds, hidden=24, epochs=80, batch=64, lr=5e-3)
    assert hist[-1] < 0.2 * hist[0]        # loss cut by >5x
    assert hist[-1] < 0.1                  # and ends small


def test_heldout_generalisation():
    """Held-out reconstruction error is low and the train/test gap is small."""
    ds = L.make_dataset(n=300, rows=4, cols=4, seed=0)
    mlp, _ = L.train(ds, hidden=24, epochs=80, batch=64, lr=5e-3)
    _, m_tr = L.reconstruction_error(mlp, ds.X_train, ds)
    _, m_te = L.reconstruction_error(mlp, ds.X_test, ds)
    # descriptor scale is O(width~4): require <0.12 RMS (~3%) on held-out.
    assert m_te < 0.12
    # honest generalisation: test error is not far above train error.
    assert m_te < m_tr + 0.05


def test_inference_much_faster_than_optimizer():
    """One learned inference is >=100x faster than one per-shape optimise."""
    ds = L.make_dataset(n=120, rows=4, cols=4, seed=0)
    mlp, _ = L.train(ds, hidden=24, epochs=40, batch=64, lr=5e-3)
    tgt = ds.X_test[0]

    t = time.perf_counter()
    for _ in range(200):
        L.predict_params(mlp, tgt[None, :], ds)
    net_t = (time.perf_counter() - t) / 200

    t = time.perf_counter()
    fit_miura(tgt, rows=4, cols=4, iters=400)
    opt_t = time.perf_counter() - t

    assert opt_t / net_t > 100
