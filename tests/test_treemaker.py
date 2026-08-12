"""Tests for the TreeMaker-lite figurative design path.

Covers the packing (non-overlap, determinism, on-paper), the FOLD export round
trip, and flat-foldability of the trees that fully fold flat - the 3-flap
triangle, the 4-flap tangential quad *and* the 5-flap pentagon, all filled by the
one universal (gusset) molecule so every interior vertex passes Kawasaki +
Maekawa with exact flap lengths. Only the river tree (overlapping discs) is left
as an honest *partial* base (see the module docstring).
"""

import numpy as np
import pytest

from foldforge.design import (
    get_tree, pack_tree, crease_pattern, design_base, flap_length_errors,
    BUILTIN_TREES,
)
from foldforge.geometry.fold_io import write_fold, read_fold
from foldforge.geometry.foldability import foldability_report


def _fully_folds(name, flap_tol=0.02):
    """True iff every interior vertex passes both theorems and flaps are exact."""
    packing, pattern = design_base(get_tree(name))
    report = foldability_report(pattern)
    err = flap_length_errors(packing, pattern)
    passes = all(v.kawasaki and v.maekawa for v in report.vertices)
    return passes and err.max() < flap_tol, report, err


def test_packing_non_overlap_and_on_paper():
    packing = pack_tree(get_tree("three-flap"))
    c, r = packing.centers, packing.radii
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            assert np.linalg.norm(c[i] - c[j]) >= r[i] + r[j] - 1e-5
    assert np.all(c - r[:, None] >= -1e-5)
    assert np.all(c + r[:, None] <= 1 + 1e-5)


def test_packing_is_deterministic():
    a = pack_tree(get_tree("four-flap"))
    b = pack_tree(get_tree("four-flap"))
    assert np.allclose(a.centers, b.centers)
    assert a.scale == pytest.approx(b.scale)


def test_fold_export_round_trips(tmp_path):
    _, pattern = design_base(get_tree("three-flap"))
    path = tmp_path / "base.fold"
    write_fold(pattern, path)
    back = read_fold(path)
    assert back.n_vertices == pattern.n_vertices
    assert back.n_edges == pattern.n_edges
    assert list(back.assignment) == list(pattern.assignment)


def test_three_flap_is_flat_foldable():
    """The single-triangle base folds flat: every interior vertex passes."""
    packing, pattern = design_base(get_tree("three-flap"))
    report = foldability_report(pattern)
    assert len(report.vertices) == 1              # one incentre, all else on border
    assert report.flat_foldable
    for v in report.vertices:
        assert v.kawasaki is True
        assert v.maekawa is True


def test_four_flap_is_flat_foldable():
    """The 4-flap tangential quad folds flat end to end: one incircle molecule.

    This is the multi-flap case the old per-triangle assembly could not fold.
    """
    packing, pattern = design_base(get_tree("four-flap"))
    report = foldability_report(pattern)
    assert len(report.vertices) == 1              # single incentre, rest on border
    assert report.flat_foldable
    for v in report.vertices:
        assert v.kawasaki is True
        assert v.maekawa is True


@pytest.mark.parametrize("name", ["three-flap", "four-flap", "five-flap"])
def test_working_trees_fold_flat_with_exact_flaps(name):
    """The trees we claim work pass both theorems everywhere; flaps are exact."""
    ok, report, err = _fully_folds(name)
    assert ok, f"{name}: not fully flat-foldable"
    assert err.max() < 1e-3, f"{name}: flap error {err.max():.4f}"


def test_five_flap_universal_molecule_folds_flat():
    """The 5-flap pentagon folds flat via the universal (gusset) molecule.

    This is the multi-flap case (n >= 5) that the per-triangle / single-incircle
    molecules could not resolve: a non-tangential active polygon split by a gusset
    into tangential parts. Every interior vertex must pass Kawasaki AND Maekawa,
    and every flap length must be within 2% of its tree target.
    """
    packing, pattern = design_base(get_tree("five-flap"))
    report = foldability_report(pattern)
    assert len(report.vertices) >= 5          # many interior vertices, not one incentre
    assert report.flat_foldable
    for v in report.vertices:
        assert v.kawasaki is True, f"vertex {v.vertex} fails Kawasaki"
        assert v.maekawa is True, f"vertex {v.vertex} fails Maekawa"
    err = flap_length_errors(packing, pattern)
    assert err.max() < 0.02, f"five-flap flap error {err.max():.4f}"
    assert len(pattern.faces) > 0             # faces populated for the FOLD viewer


def test_river_packing_leaves_open_band_no_overlap():
    """The river packing is exact: same-hub discs tangent, cross-river ones gapped.

    Radius == the leaf's terminal edge, so two leaves sharing a hub are tangent
    (no overlap by the spine) and cross-river leaves sit an extra river band
    apart. This pins the root-cause fix (radius was the full path-to-root before).
    """
    from foldforge.design.treemaker import pack_tree
    tree = get_tree("river-four")
    packing = pack_tree(tree)
    c, r = packing.centers, packing.radii
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            assert np.linalg.norm(c[i] - c[j]) >= r[i] + r[j] - 1e-5   # no overlap
    # exactly two cross-river pairs are held an extra band apart (> tangent).
    gapped = 0
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            if np.linalg.norm(c[i] - c[j]) > r[i] + r[j] + 1e-4:
                gapped += 1
    assert gapped == 4                              # the 4 cross-river leaf pairs


def test_single_spine_river_folds_flat():
    """A body with limbs (single-spine river) folds flat end to end.

    Every interior vertex passes Kawasaki AND Maekawa and every flap length is
    exact (root-side flaps r_i, far-side flaps r_i + spine). This is the piece
    that takes the design path past single-hub star trees.
    """
    packing, pattern = design_base(get_tree("river-four"))
    report = foldability_report(pattern)
    assert len(report.vertices) == 2                # the two sub-hub nodes
    assert report.flat_foldable
    for v in report.vertices:
        assert v.kawasaki is True, f"vertex {v.vertex} fails Kawasaki"
        assert v.maekawa is True, f"vertex {v.vertex} fails Maekawa"
    err = flap_length_errors(packing, pattern)
    assert err.max() < 0.02, f"river flap error {err.max():.4f}"
    assert len(pattern.faces) > 0                   # faces populated for the viewer


def test_asymmetric_river_is_honestly_partial():
    """Honest ceiling: asymmetric / many-leaf rivers do NOT fully fold yet.

    The two-node river molecule only closes for the symmetric single-spine case;
    asymmetric sub-hubs need extra gusset nodes it does not emit, so the self-
    check fails and it falls back to the partial base. This pins that boundary so
    a regression that silently 'fixed' or faked it would fail loudly.
    """
    from foldforge.design.treemaker import river_tree, design_base
    tree = river_tree(0.5, [1.2, 0.8], [1.0, 0.9], "river-asym")
    packing, pattern = design_base(tree)
    report = foldability_report(pattern)
    passes = all(v.kawasaki and v.maekawa for v in report.vertices)
    assert not passes, "asymmetric river unexpectedly folds flat - update claims"
    assert len(report.vertices) > 0                 # it still builds a real pattern
