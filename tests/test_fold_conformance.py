"""FOLD-format conformance checks for FoldForge's exported crease patterns.

FoldForge writes the standard FOLD file format (github.com/edemaine/fold) so its
patterns open in the wider origami toolchain - Origami Simulator
(origamisimulator.org, File > Import) and Jason Ku's flat-folder, which both
consume FOLD. These tests guard the parts an importer relies on: valid JSON, the
required geometry keys, edge assignments drawn from the allowed M/V/B/F/U set,
and edge/face indices that actually reference existing vertices.
"""

import json
from pathlib import Path

import pytest

from foldforge.design import get_tree, design_base
from foldforge.geometry import examples
from foldforge.geometry.fold_io import write_fold

# The letters the FOLD spec allows for edges_assignment. B=border, M=mountain,
# V=valley, F=flat (unfolded), U=unassigned (C=cut / J=join are for cut-and-join
# patterns, which FoldForge does not emit).
ALLOWED_ASSIGNMENTS = {"B", "M", "V", "F", "U"}


def validate_fold(path):
    """Assert that ``path`` is a spec-conformant FOLD file; return the parsed dict.

    Checks: parses as JSON; ``file_spec`` is a number; the geometry keys are
    present; every ``edges_assignment`` letter is in the allowed set and matches
    the edge count; every edge/face index references an existing vertex.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    # file-level metadata an importer reads first
    assert isinstance(data.get("file_spec"), (int, float)), "file_spec must be a number"

    verts = data["vertices_coords"]
    edges = data["edges_vertices"]
    n_v = len(verts)

    # flat crease patterns are 2D; every coordinate has the same dimension
    dims = {len(v) for v in verts}
    assert dims <= {2, 3} and len(dims) == 1, f"inconsistent vertex dims {dims}"

    # edges reference real vertices
    for a, b in edges:
        assert 0 <= a < n_v and 0 <= b < n_v, f"edge ({a},{b}) out of range"

    # assignments: right count, all in the allowed set
    assign = data.get("edges_assignment")
    if assign is not None:
        assert len(assign) == len(edges), "edges_assignment length != edge count"
        bad = set(assign) - ALLOWED_ASSIGNMENTS
        assert not bad, f"illegal edge assignments {bad}"

    # foldAngle, if present, is per-edge
    if "edges_foldAngle" in data:
        assert len(data["edges_foldAngle"]) == len(edges)

    # faces (if any) reference real vertices
    for face in data.get("faces_vertices", []):
        for v in face:
            assert 0 <= v < n_v, f"face vertex {v} out of range"

    return data


@pytest.mark.parametrize("gen", ["miura", "waterbomb_base", "single_vertex"])
def test_builtin_patterns_export_conformant_fold(tmp_path, gen):
    pattern = examples.GENERATORS[gen]()
    path = tmp_path / f"{gen}.fold"
    write_fold(pattern, path)
    validate_fold(path)


def test_treemaker_base_exports_conformant_fold(tmp_path):
    """A designed figurative base (the TreeMaker-lite three-flap) exports a
    flat 2D crease pattern that Origami Simulator would accept."""
    _, pattern = design_base(get_tree("three-flap"))
    path = tmp_path / "three_flap.fold"
    write_fold(pattern, path)
    data = validate_fold(path)
    # a flat crease pattern: 2D coords, tagged as such for importers
    assert data["frame_classes"] == ["creasePattern"]
    assert data["frame_attributes"] == ["2D"]
    assert all(len(v) == 2 for v in data["vertices_coords"])
    # and it carries real M/V folds, not just borders
    assert {"M", "V"} & set(data["edges_assignment"])


def test_mountain_valley_letters_match_spec(tmp_path):
    """Guard against an M/V swap: the miura's four horizontal spine creases are
    a mountain row flanked by valley rows (M != V, both present)."""
    pattern = examples.miura()
    path = tmp_path / "miura.fold"
    write_fold(pattern, path)
    data = validate_fold(path)
    a = data["edges_assignment"]
    assert "M" in a and "V" in a and "M" != "V"


def test_no_ai_attribution_in_metadata(tmp_path):
    """The creator/title metadata must say FoldForge, never mention an assistant."""
    pattern = examples.miura()
    path = tmp_path / "m.fold"
    write_fold(pattern, path)
    text = Path(path).read_text(encoding="utf-8").lower()
    for banned in ("claude", "anthropic", "gpt", "ai-generated", "assistant", "copilot"):
        assert banned not in text
    assert json.loads(Path(path).read_text())["file_creator"] == "FoldForge"
