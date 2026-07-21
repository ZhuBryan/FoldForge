"""Tests for M3 inverse design: the optimiser should recover folds that match."""

import numpy as np

from foldforge.design import (
    fit_chain, chamfer_distance, target_arch, target_wave, resample_arclength,
)


def test_chamfer_zero_for_identical():
    a = np.random.default_rng(0).standard_normal((10, 2))
    assert chamfer_distance(a, a) < 1e-12


def test_resample_preserves_endpoints():
    c = np.stack([np.linspace(0, 5, 50), np.zeros(50)], axis=1)
    r = resample_arclength(c, 10)
    assert len(r) == 11
    assert np.allclose(r[0], c[0]) and np.allclose(r[-1], c[-1])


def test_fit_arch_converges():
    tgt = target_arch(24)
    r = fit_chain(tgt, iters=600, lr=0.05)
    assert r.losses[-1] < r.losses[0] * 0.05      # big loss reduction
    assert r.chamfer < 0.25                        # close match


def test_fit_wave_converges():
    r = fit_chain(target_wave(24), iters=600, lr=0.05)
    assert r.chamfer < 0.2
