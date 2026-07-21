"""Tests for folding arbitrary input: image / function / 3D points -> origami."""

import numpy as np

from foldforge.origamize import (
    origamize_image, origamize_function, origamize_points,
    heightmap_from_image, heightmap_from_function, heightmap_from_points,
)


def test_heightmap_from_image_normalised():
    arr = np.random.default_rng(0).random((40, 50))
    Z = heightmap_from_image(arr, grid=(12, 16))
    assert Z.shape == (12, 16)
    assert abs(Z.min()) < 1e-6 and abs(Z.max() - 1) < 1e-6


def test_origamize_image_runs():
    arr = np.zeros((30, 40)); arr[10:20, 12:28] = 1.0      # a bright bar
    r = origamize_image(arr, grid=(12, 16), iters=200)
    assert r.error < 1.0 and r.folded.shape[1] == 3


def test_origamize_function_matches_a_bump():
    r = origamize_function(lambda x, y: np.exp(-3 * (x ** 2 + y ** 2)),
                           grid=(14, 18), iters=300)
    assert r.error < 0.3


def test_heightmap_from_points_fills_grid():
    rng = np.random.default_rng(1)
    pts = rng.random((500, 3))
    Z = heightmap_from_points(pts, grid=(10, 12))
    assert Z.shape == (10, 12) and not np.isnan(Z).any()
