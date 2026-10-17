"""Curved-crease origami: the folded surface is a genuine bending (isometry).

A curved crease must fold flat paper into a *curved developable* surface with no
stretch. These tests verify exactly that: arc lengths / ruling lengths and the
first fundamental form are preserved (<1e-6), the result is genuinely 3-D (not a
trivial flat answer), and the concentric pleat preserves its circle radii.
"""

import numpy as np

from foldforge.design import curved_fold, concentric_pleat


def _circular_arc(n=150, radius=1.6, span=1.6):
    a = np.linspace(-span / 2, span / 2, n)
    return np.stack([radius * np.sin(a), radius * (1 - np.cos(a))], axis=1)


# --------------------------------------------------------------------------- #
# 1. single symmetric curved fold
# --------------------------------------------------------------------------- #
def test_single_fold_is_isometric():
    f = curved_fold(_circular_arc(), fold_angle=0.5)
    # the whole point: a bending, not a stretch
    assert f.developability_error < 1e-6


def test_single_fold_is_genuinely_3d():
    flat = curved_fold(_circular_arc(), fold_angle=0.01)
    folded = curved_fold(_circular_arc(), fold_angle=0.6)
    # near-zero fold stays essentially flat; a real fold lifts out of the plane
    assert flat.meta["z_range"] < 0.05
    assert folded.meta["z_range"] > 0.1
    # the folded crease has the same arc length but curls up (chord shrinks)
    flat_len = np.linalg.norm(np.diff(folded.crease_flat, axis=0), axis=1).sum()
    fold_len = np.linalg.norm(np.diff(folded.crease_folded, axis=0), axis=1).sum()
    assert abs(flat_len - fold_len) / flat_len < 1e-3           # arc length preserved
    flat_chord = np.linalg.norm(folded.crease_flat[-1] - folded.crease_flat[0])
    fold_chord = np.linalg.norm(folded.crease_folded[-1] - folded.crease_folded[0])
    assert fold_chord < flat_chord - 0.05                       # genuinely curled


def test_ruling_lengths_preserved_exactly():
    f = curved_fold(_circular_arc(), fold_angle=0.6, n_s=120, n_u=30)
    ns, nu = f.meta["n_s"], f.meta["n_u"]
    flat = f.flat.reshape(ns, nu, 3)
    folded = f.folded.reshape(ns, nu, 3)
    # the across-width ruling is straight paper; its total arc length (summed
    # along the row, through the fold kink) is intrinsic => identical flat/folded.
    flat_w = np.linalg.norm(np.diff(flat, axis=1), axis=2).sum(axis=1)
    fold_w = np.linalg.norm(np.diff(folded, axis=1), axis=2).sum(axis=1)
    assert np.max(np.abs(flat_w - fold_w)) < 1e-9


def test_flat_fold_angle_reproduces_flat_sheet():
    f = curved_fold(_circular_arc(), fold_angle=1e-6)
    # theta -> 0 reproduces the unfolded flat sheet: planar (z=0) and isometric
    # (a planar isometric image of the flat sheet IS the flat sheet, up to a
    # rigid motion -- the folded crease re-bases its start tangent to +x).
    assert np.max(np.abs(f.folded[:, 2])) < 1e-4
    assert f.developability_error < 1e-6


def test_width_capped_below_curvature_radius():
    # a tightly curved crease: requested width would cross the centre of curvature
    f = curved_fold(_circular_arc(radius=0.5), fold_angle=0.4, width=2.0)
    assert f.meta["width"] < f.meta["width_requested"]
    assert f.developability_error < 1e-5


# --------------------------------------------------------------------------- #
# 2. concentric circular pleat (conical-strip model)
# --------------------------------------------------------------------------- #
def test_pleat_preserves_radii():
    radii = np.linspace(0.4, 1.0, 7)
    angles = np.deg2rad([25, -25, 25, -25, 25, -25])
    p = concentric_pleat(radii, angles)
    flat_r = np.linalg.norm(p.flat[:, :2], axis=1)
    fold_r = np.linalg.norm(p.folded[:, :2], axis=1)
    assert np.max(np.abs(flat_r - fold_r)) < 1e-9          # radii exact


def test_pleat_ripples_in_3d():
    radii = np.linspace(0.4, 1.0, 7)
    angles = np.deg2rad([25, -25, 25, -25, 25, -25])
    p = concentric_pleat(radii, angles)
    z = np.array(p.meta["z"])
    # alternating mountain/valley => heights go up and down, not monotone
    dz = np.diff(z)
    assert np.any(dz > 0) and np.any(dz < 0)
    assert p.meta["z_range"] > 0.02


def test_pleat_radial_strain_is_the_reported_error():
    radii = np.linspace(0.5, 1.0, 6)
    angles = np.full(5, np.deg2rad(20))
    p = concentric_pleat(radii, angles)
    # conical-strip model: error == sec(angle) - 1 (honest approximation)
    assert np.isclose(p.developability_error, 1 / np.cos(np.deg2rad(20)) - 1)


def test_pleat_angle_count_validated():
    import pytest
    with pytest.raises(ValueError):
        concentric_pleat(np.linspace(0.4, 1.0, 5), np.zeros(10))
