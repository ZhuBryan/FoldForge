"""Tests for folded 3D export (OBJ / STL)."""

import numpy as np

from foldforge.geometry import examples
from foldforge.sim import FoldMesh, fold, creases_along_x
from foldforge.origamize import origamize_heightfield, heightfield_dome
from foldforge.fabricate import to_obj, to_stl


def test_obj_from_fold(tmp_path):
    m = FoldMesh.from_pattern(examples.miura(4, 4))
    r = fold(m, fold_fraction=0.7, actuate=creases_along_x(m), stages=10, relax_iters=15)
    p = tmp_path / "f.obj"
    to_obj(r.vertices, m.triangles, p)
    text = p.read_text()
    nverts = text.count("\nv ") + (1 if text.startswith("v ") else 0)
    faces = [list(map(int, l.split()[1:])) for l in text.splitlines() if l.startswith("f ")]
    assert nverts == len(r.vertices)
    assert max(max(f) for f in faces) <= nverts        # 1-indexed, in range
    assert min(min(f) for f in faces) >= 1


def test_stl_is_valid(tmp_path):
    m = FoldMesh.from_pattern(examples.miura(3, 3))
    r = fold(m, fold_fraction=0.6, actuate=creases_along_x(m), stages=8, relax_iters=12)
    p = tmp_path / "f.stl"
    to_stl(r.vertices, m.triangles, p)
    text = p.read_text()
    assert text.startswith("solid") and text.strip().endswith("endsolid foldforge")
    assert text.count("facet normal") == len(m.triangles)


def test_origami_result_carries_triangles(tmp_path):
    r = origamize_heightfield(heightfield_dome(nx=14, ny=12))
    assert r.triangles is not None and r.triangles.shape[1] == 3
    p = tmp_path / "d.stl"
    to_stl(r.folded, r.triangles, p)
    assert p.read_text().count("facet normal") == len(r.triangles)
