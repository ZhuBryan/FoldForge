"""Tests for the origamizer: 3D target -> foldable crease pattern."""

import numpy as np

from foldforge.origamize import (
    origamize_profile, origamize_heightfield,
    profile_dome, profile_ridge, heightfield_dome,
)
from foldforge import foldability_report
from foldforge.geometry.fold_io import write_fold, read_fold


def test_profile_origamize_is_accurate_and_foldable():
    r = origamize_profile(profile_dome(), n_pleats=26)
    assert r.error < 0.12                                  # close to the target
    assert foldability_report(r.pattern).flat_foldable      # a valid pleated sheet
    assert len(r.angles) == 26


def test_fold_angles_signs_match_assignment():
    r = origamize_profile(profile_ridge(), n_pleats=24)
    for (a, b), kind, ang in zip(r.pattern.edges, r.pattern.assignment,
                                 r.pattern.fold_angle):
        if kind == "M":
            assert ang <= 0
        if kind == "V":
            assert ang >= 0


def test_origamized_pattern_roundtrips_fold(tmp_path):
    r = origamize_profile(profile_dome(), n_pleats=20)
    p = tmp_path / "o.fold"
    write_fold(r.pattern, p)
    back = read_fold(p)
    assert np.allclose(back.vertices, r.pattern.vertices)
    assert back.assignment == r.pattern.assignment


def test_heightfield_origamize_runs():
    r = origamize_heightfield(heightfield_dome(nx=16, ny=12), iters=300)
    assert r.error < 1.0 and r.folded.shape[1] == 3
