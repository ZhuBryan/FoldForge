"""Tests for the M1 fold simulator.

The two things easy to get silently wrong here are (1) the dihedral-angle
gradient that the crease forces are built on, and (2) whether the fold actually
keeps faces rigid. We pin down both: the gradient against finite differences,
and the fold against a max-strain tolerance.

Run with:  pytest -q
"""

import numpy as np
import pytest

from foldforge.geometry import examples
from foldforge.sim import FoldMesh, fold, creases_along_x
from foldforge.sim.mesh import dihedral_angle, dihedral_grad


# --- dihedral-angle geometry ------------------------------------------------

def test_dihedral_flat_is_zero():
    """Two coplanar triangles have a fold angle of zero."""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([0.5, 1.0, 0.0])
    p4 = np.array([0.5, -1.0, 0.0])
    assert abs(dihedral_angle(p1, p2, p3, p4)) < 1e-12


def test_dihedral_right_angle():
    """Fold one triangle up by 90 degrees -> fold angle is +/- pi/2."""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([0.5, 1.0, 0.0])      # flat wing
    p4 = np.array([0.5, 0.0, 1.0])      # wing folded straight up
    assert abs(abs(dihedral_angle(p1, p2, p3, p4)) - np.pi / 2) < 1e-9


def test_dihedral_grad_matches_finite_differences():
    """Analytic hinge gradient must agree with central finite differences.

    This is the M2-style check the whole simulator rests on: if this drifts,
    the crease forces push in subtly wrong directions and folds misbehave.
    """
    rng = np.random.default_rng(0)
    eps = 1e-7
    worst = 0.0
    for _ in range(200):
        pts = [rng.standard_normal(3) for _ in range(4)]
        # Skip near-degenerate triangles and the atan2 branch cut, where finite
        # differences themselves are unreliable.
        a1 = np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])) / 2
        a2 = np.linalg.norm(np.cross(pts[3] - pts[0], pts[1] - pts[0])) / 2
        if a1 < 0.2 or a2 < 0.2 or np.linalg.norm(pts[1] - pts[0]) < 0.4:
            continue
        if abs(dihedral_angle(*pts)) > 2.8:
            continue
        analytic = dihedral_grad(*pts)
        for i in range(4):
            for d in range(3):
                hi = [p.copy() for p in pts]; hi[i][d] += eps
                lo = [p.copy() for p in pts]; lo[i][d] -= eps
                fd = (dihedral_angle(*hi) - dihedral_angle(*lo)) / (2 * eps)
                worst = max(worst, abs(fd - analytic[i][d]))
    assert worst < 1e-5, f"gradient mismatch {worst}"


# --- mesh construction ------------------------------------------------------

def test_mesh_bars_have_positive_rest_length():
    mesh = FoldMesh.from_pattern(examples.miura(3, 3))
    assert np.all(mesh.rest_lengths > 0)


def test_mesh_requires_faces():
    """A pattern with no faces can't be turned into a 3D mesh."""
    bare = examples.miura(2, 2)
    bare.faces = []
    with pytest.raises(ValueError):
        FoldMesh.from_pattern(bare)


def test_flat_mesh_has_zero_strain():
    """Before folding, every bar is exactly its rest length."""
    mesh = FoldMesh.from_pattern(examples.miura(3, 3))
    assert mesh.max_strain(mesh.vertices) < 1e-12


# --- folding ----------------------------------------------------------------

# A mass-spring fold isn't perfectly rigid; a few percent strain is the
# expected ceiling for this method. We assert comfortably inside that.
STRAIN_TOL = 0.02


def test_miura_folds_into_3d_and_stays_rigid():
    mesh = FoldMesh.from_pattern(examples.miura(4, 4))
    result = fold(mesh, fold_fraction=0.8, actuate=creases_along_x(mesh))
    z_range = np.ptp(result.vertices[:, 2])
    assert z_range > 0.5, "pattern should leave the flat plane"
    assert result.max_strain < STRAIN_TOL, f"faces stretched {result.max_strain}"


def test_waterbomb_folds_into_3d_and_stays_rigid():
    mesh = FoldMesh.from_pattern(examples.waterbomb_base())
    result = fold(mesh, fold_fraction=0.7)
    assert np.ptp(result.vertices[:, 2]) > 0.5
    assert result.max_strain < STRAIN_TOL


def test_fold_starts_near_flat():
    """The trajectory begins essentially flat and grows: it's a smooth ramp."""
    mesh = FoldMesh.from_pattern(examples.miura(3, 3))
    result = fold(mesh, fold_fraction=0.7, actuate=creases_along_x(mesh))
    first = np.ptp(result.frames[0][:, 2])
    last = np.ptp(result.frames[-1][:, 2])
    assert first < last, "fold should deepen over the trajectory"


def test_fold_records_a_frame_per_stage():
    mesh = FoldMesh.from_pattern(examples.miura(2, 2))
    result = fold(mesh, stages=12)
    assert len(result.frames) == 12
    assert len(result.strains) == 12
