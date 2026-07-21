"""Tests for the M4 generative model and its batched simulator-in-the-loop math."""

import numpy as np

from foldforge.generative import (
    FoldGenerator, make_dataset, batch_fold, batch_recon_grad,
)
from foldforge.diff.kinematics import fold_chain, spine_jacobian


def test_batch_fold_matches_single():
    a = np.random.default_rng(0).uniform(-0.5, 0.5, (4, 8))
    sp, _ = batch_fold(a)
    for i in range(4):
        assert np.allclose(sp[i], fold_chain(a[i]).spine)


def test_batch_recon_grad_matches_analytic():
    rng = np.random.default_rng(2)
    n = 10
    a = rng.uniform(-0.4, 0.4, (3, n))
    tgt = rng.standard_normal((3, n + 1, 2))
    _, gb = batch_recon_grad(a, tgt)
    for i in range(3):
        sp = fold_chain(a[i]).spine
        resid = sp - tgt[i]
        J = spine_jacobian(a[i])
        gi = (2.0 / (n + 1)) * np.einsum("kd,kdj->j", resid, J)
        assert np.allclose(gi, gb[i], atol=1e-9)


def test_training_beats_random():
    n = 12
    seg = 1.0
    rng = np.random.default_rng(0)
    Xtr = make_dataset(400, n, seg, rng)
    Xte = make_dataset(150, n, seg, rng)
    gen = FoldGenerator(n=n, seg=seg, hidden=48)
    hist = gen.train(Xtr, epochs=25)
    assert hist[-1] < hist[0]                      # learning happened
    y = gen._forward(np.concatenate([Xte, np.zeros((len(Xte), gen.noise_dim))], 1))[0]
    Lnet, _ = batch_recon_grad(y, Xte.reshape(len(Xte), -1, 2), seg)
    yr = rng.uniform(-0.35, 0.35, (len(Xte), n))
    Lrand, _ = batch_recon_grad(yr, Xte.reshape(len(Xte), -1, 2), seg)
    assert Lnet < Lrand * 0.5                      # clearly better than random


def test_batch_recon_grad_matches_finite_differences():
    """Independent FD check of the batched gradient (not vs the Jacobian)."""
    rng = np.random.default_rng(5)
    n = 9
    a = rng.uniform(-0.4, 0.4, (2, n))
    tgt = rng.standard_normal((2, n + 1, 2))
    _, g = batch_recon_grad(a, tgt)
    eps = 1e-6

    def per_sample_loss(ai, ti):
        sp, _ = batch_fold(ai[None])
        resid = sp[0] - ti
        return np.mean(np.sum(resid ** 2, axis=1))

    for i in range(2):
        for j in range(n):
            ap = a[i].copy(); ap[j] += eps
            am = a[i].copy(); am[j] -= eps
            fd = (per_sample_loss(ap, tgt[i]) - per_sample_loss(am, tgt[i])) / (2 * eps)
            assert abs(fd - g[i][j]) < 1e-6
