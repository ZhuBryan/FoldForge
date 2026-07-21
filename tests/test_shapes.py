"""Tests for the wild-shape library: text, terrain, OBJ meshes."""

import numpy as np

from foldforge.origamize import (
    text_heightfield, terrain_heightfield, load_obj,
    origamize_heightfield, origamize_points,
)


def test_text_heightfield_has_raised_marks():
    Z = text_heightfield("FOLD", grid=(16, 32))
    assert Z.shape == (16, 32)
    assert Z.max() > 0.5 and Z.min() >= 0.0      # letters are raised over background


def test_terrain_is_normalised_and_varied():
    Z = terrain_heightfield(grid=(20, 24), seed=1)
    assert Z.shape == (20, 24)
    assert abs(Z.min()) < 1e-6 and abs(Z.max() - 1) < 1e-6
    assert Z.std() > 0.05                          # actually has relief


def test_load_obj_and_origamize(tmp_path):
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, (1500, 2))
    P = np.stack([(1 + 0.4 * np.cos(th[:, 1])) * np.cos(th[:, 0]),
                  (1 + 0.4 * np.cos(th[:, 1])) * np.sin(th[:, 0]),
                  0.4 * np.sin(th[:, 1])], axis=1)
    obj = tmp_path / "t.obj"
    obj.write_text("\n".join(f"v {x} {y} {z}" for x, y, z in P))
    V = load_obj(str(obj))
    assert V.shape == (1500, 3)
    r = origamize_points(V, grid=(14, 18))
    assert r.error < 1.0 and r.folded.shape[1] == 3


def test_origamize_terrain_is_accurate():
    r = origamize_heightfield(terrain_heightfield(grid=(16, 20)) * 6.0)
    assert r.error < 0.3                           # corrugation tracks the terrain
