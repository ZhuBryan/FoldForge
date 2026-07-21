"""Tests for glTF (.glb) export and per-vertex mountain/valley colours."""

import json
import struct

import numpy as np

from foldforge.geometry import examples
from foldforge.sim import FoldMesh, fold, creases_along_x
from foldforge.fabricate import to_gltf, to_obj, mountain_valley_colors


def _folded():
    m = FoldMesh.from_pattern(examples.miura(3, 3))
    r = fold(m, fold_fraction=0.6, actuate=creases_along_x(m),
             stages=8, relax_iters=12)
    return m, r


def test_glb_is_structurally_valid(tmp_path):
    m, r = _folded()
    p = tmp_path / "f.glb"
    to_gltf(r.vertices, m.triangles, p)
    data = p.read_bytes()

    magic, version, total = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67          # 'glTF'
    assert version == 2
    assert total == len(data)           # header length matches the file

    # JSON chunk
    jlen, jtype = struct.unpack("<II", data[12:20])
    assert jtype == 0x4E4F534A          # 'JSON'
    assert jlen % 4 == 0                 # 4-byte aligned
    gltf = json.loads(data[20:20 + jlen])

    # BIN chunk directly after the JSON chunk, filling the rest of the file
    off = 20 + jlen
    blen, btype = struct.unpack("<II", data[off:off + 8])
    assert btype == 0x004E4942          # 'BIN\0'
    assert blen % 4 == 0
    assert off + 8 + blen == len(data)
    assert gltf["buffers"][0]["byteLength"] <= blen

    # accessors: POSITION has one entry per vertex, indices == 3 * triangles
    prim = gltf["meshes"][0]["primitives"][0]
    pos = gltf["accessors"][prim["attributes"]["POSITION"]]
    idx = gltf["accessors"][prim["indices"]]
    assert pos["type"] == "VEC3" and pos["count"] == len(r.vertices)
    assert "min" in pos and "max" in pos              # spec-required bounds
    assert idx["type"] == "SCALAR" and idx["count"] == m.triangles.size


def test_glb_carries_vertex_colors(tmp_path):
    m, r = _folded()
    colors = mountain_valley_colors(m)
    assert colors.shape == (len(m.vertices), 3)
    p = tmp_path / "c.glb"
    to_gltf(r.vertices, m.triangles, p, vertex_colors=colors)
    data = p.read_bytes()
    jlen = struct.unpack("<II", data[12:20])[0]
    gltf = json.loads(data[20:20 + jlen])
    attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
    assert "COLOR_0" in attrs
    col = gltf["accessors"][attrs["COLOR_0"]]
    assert col["type"] == "VEC4" and col["count"] == len(m.vertices)


def test_obj_vertex_colors_written(tmp_path):
    m, r = _folded()
    colors = mountain_valley_colors(m)
    p = tmp_path / "c.obj"
    to_obj(r.vertices, m.triangles, p, vertex_colors=colors)
    vlines = [l for l in p.read_text().splitlines() if l.startswith("v ")]
    assert len(vlines) == len(r.vertices)
    assert all(len(l.split()) == 7 for l in vlines)   # v x y z r g b
    # at least mountain, valley and neutral tints appear
    assert np.unique(colors, axis=0).shape[0] >= 2
