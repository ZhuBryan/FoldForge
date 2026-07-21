"""Tests for the exact differentiable 2D Miura surface."""

import numpy as np

from foldforge.diff.miura import (
    flat_miura, folded_miura, fold_limit, footprint, fit_miura,
)


def _edge_lengths(V):
    R, C, _ = V.shape
    L = []
    for i in range(R):
        for j in range(C):
            if i + 1 < R:
                L.append(np.linalg.norm(V[i + 1, j] - V[i, j]))
            if j + 1 < C:
                L.append(np.linalg.norm(V[i, j + 1] - V[i, j]))
    return np.sort(L)


def test_folded_miura_is_exactly_isometric():
    g = np.radians(63)
    flat = _edge_lengths(flat_miura(4, 5, 1.2, 0.9, g))
    for h in (0.1, 0.3, 0.5, 0.7):
        fold = _edge_lengths(folded_miura(4, 5, 1.2, 0.9, g, h))
        assert np.abs(flat - fold).max() < 1e-12          # exact rigid fold


def test_facets_stay_planar():
    V = folded_miura(4, 5, 1.1, 0.95, np.radians(60), 0.5)
    worst = 0.0
    for i in range(4):
        for j in range(5):
            p = [V[i, j], V[i + 1, j], V[i + 1, j + 1], V[i, j + 1]]
            n = np.cross(p[1] - p[0], p[2] - p[0]); n = n / np.linalg.norm(n)
            worst = max(worst, abs(np.dot(p[3] - p[0], n)))
    assert worst < 1e-12


def test_flat_state_has_zero_height():
    V = folded_miura(4, 4, 1.0, 1.0, np.radians(60), 1e-9)
    assert np.ptp(V[:, :, 2]) < 1e-6


def test_inverse_design_recovers_footprint():
    tgt = footprint(6, 6, 1.1, 0.95, np.radians(55), 0.45)
    _, hist, ach = fit_miura(tgt, rows=6, cols=6, iters=500)
    assert hist[-1] < hist[0] * 1e-3
    assert np.allclose(ach, tgt, atol=0.05)
