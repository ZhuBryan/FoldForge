"""Fabrication export: turn a crease pattern into laser-cutter-ready vectors.

Hand-written SVG and DXF (no third-party libraries - they are just text). Edges
are split onto standard fabrication layers by assignment:

    mountain -> red    valley -> blue    border/cut -> black    flat/score -> grey

So the output drops straight onto a laser or vinyl cutter: cut the black border,
score the coloured folds.

The folded 3D exporters (OBJ / STL / glTF) live at the bottom. glTF is written as
a self-contained binary ``.glb`` by hand (``struct`` + ``json``, no dependency),
and OBJ / glTF can carry per-vertex colours - use :func:`mountain_valley_colors`
to tint each vertex by the crease type it touches.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from foldforge.geometry.crease_graph import CreasePattern

# assignment -> (layer name, hex colour). Border is the sheet outline = the cut.
_LAYER = {
    "M": ("mountain", "#ff0000"),
    "V": ("valley", "#0000ff"),
    "B": ("cut", "#000000"),
    "F": ("score", "#888888"),
    "U": ("unassigned", "#888888"),
}

# Triangulating facet diagonals get their own layer (they make each quad's two
# triangles planar). Distinct green so they read apart from the M/V/cut/score
# structural creases. Populated from ``pattern.metadata["facet_edges"]``.
_FACET_LAYER = ("facet", "#1a9850")

# RGB (0..1) for per-vertex mountain/valley tinting.
_MV_RGB = {
    "M": (0.85, 0.15, 0.15),   # mountain -> red
    "V": (0.15, 0.30, 0.85),   # valley   -> blue
    None: (0.75, 0.75, 0.75),  # neither  -> grey
}


def _scaled_xy(pattern: CreasePattern, size_mm: float, margin_mm: float):
    """Vertices scaled so the longest side is ``size_mm`` (plus margin), and the
    canvas (width, height) in mm. SVG y points down, so y is flipped.
    """
    xy = pattern.vertices[:, :2].astype(float)
    lo = xy.min(0)
    span = xy.max(0) - lo
    scale = size_mm / max(span.max(), 1e-9)
    p = (xy - lo) * scale + margin_mm
    w = span[0] * scale + 2 * margin_mm
    h = span[1] * scale + 2 * margin_mm
    p[:, 1] = h - p[:, 1]
    return p, w, h


def to_svg(pattern: CreasePattern, path: str | Path, *, size_mm: float = 200.0,
           margin_mm: float = 10.0, stroke_mm: float = 0.3,
           outline_only: bool = False) -> None:
    """Write ``pattern`` as a layered SVG sized in millimetres (one Inkscape
    layer per fold type, so a cutter can map layers to cut/score operations).

    With ``outline_only=True`` the border/cut layer is drawn as a single sheet
    rectangle (the bounding box) instead of every internal panel edge, so a
    coarse corrugation prints as a clean hand-fold sheet: the paper outline plus
    the coloured mountain/valley creases, with no misleading grid of "cut"
    lines. Fold (M/V/facet/score) creases are unchanged.
    """
    p, w, h = _scaled_xy(pattern, size_mm, margin_mm)
    facet = set(int(i) for i in pattern.metadata.get("facet_edges", []))
    layers: dict = {}
    for k, ((a, b), kind) in enumerate(zip(pattern.edges, pattern.assignment)):
        if outline_only and kind == "B" and k not in facet:
            continue                                    # replaced by one outline rect
        name, colour = _FACET_LAYER if k in facet else _LAYER.get(kind, _LAYER["U"])
        layers.setdefault((name, colour), []).append(
            f'<line x1="{p[a,0]:.3f}" y1="{p[a,1]:.3f}" '
            f'x2="{p[b,0]:.3f}" y2="{p[b,1]:.3f}" stroke="{colour}" '
            f'stroke-width="{stroke_mm}"/>'
        )
    if outline_only:                                    # one clean sheet outline
        name, colour = _LAYER["B"]
        x0, y0 = margin_mm, margin_mm
        layers.setdefault((name, colour), []).append(
            f'<rect x="{x0:.3f}" y="{y0:.3f}" width="{w - 2 * margin_mm:.3f}" '
            f'height="{h - 2 * margin_mm:.3f}" fill="none" stroke="{colour}" '
            f'stroke-width="{stroke_mm}"/>'
        )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'width="{w:.2f}mm" height="{h:.2f}mm" viewBox="0 0 {w:.3f} {h:.3f}">'
    ]
    for (name, colour), lines in layers.items():
        out.append(f'<g inkscape:groupmode="layer" inkscape:label="{name}" id="{name}">')
        out.extend(lines)
        out.append("</g>")
    out.append("</svg>")
    Path(path).write_text("\n".join(out), encoding="utf-8")


def to_dxf(pattern: CreasePattern, path: str | Path, *, size_mm: float = 200.0,
           margin_mm: float = 10.0) -> None:
    """Write ``pattern`` as a minimal ASCII DXF (R12) with one layer per fold type."""
    p, _, _ = _scaled_xy(pattern, size_mm, margin_mm)
    facet = set(int(i) for i in pattern.metadata.get("facet_edges", []))
    e = ["0", "SECTION", "2", "ENTITIES"]
    for k, ((a, b), kind) in enumerate(zip(pattern.edges, pattern.assignment)):
        name = _FACET_LAYER[0] if k in facet else _LAYER.get(kind, _LAYER["U"])[0]
        e += ["0", "LINE", "8", name,
              "10", f"{p[a,0]:.4f}", "20", f"{p[a,1]:.4f}", "30", "0.0",
              "11", f"{p[b,0]:.4f}", "21", f"{p[b,1]:.4f}", "31", "0.0"]
    e += ["0", "ENDSEC", "0", "EOF"]
    Path(path).write_text("\n".join(e) + "\n", encoding="utf-8")


# --- per-vertex mountain/valley colours -------------------------------------

def mountain_valley_colors(mesh) -> np.ndarray:
    """Per-vertex RGB (V, 3) tinting each vertex by the crease type it touches.

    A vertex on any mountain crease goes red, on a valley crease blue (mountain
    wins ties), everything else grey. Handy as ``vertex_colors`` for
    :func:`to_obj` / :func:`to_gltf` so a viewer shows the fold pattern in 3D.
    """
    n = len(mesh.vertices)
    kind = [None] * n                       # None < "V" < "M" priority
    for h in mesh.hinges:
        if not h.is_crease:
            continue
        k = "M" if h.target < 0 else "V"    # mountains fold with a negative target
        for v in (int(h.edge[0]), int(h.edge[1])):
            if kind[v] != "M":              # don't downgrade a mountain vertex
                kind[v] = "M" if k == "M" else (kind[v] or "V")
    return np.array([_MV_RGB[k] for k in kind], dtype=float)


# --- folded 3D geometry export (for 3D printing / external viewers) ----------

def to_obj(vertices, faces, path: str | Path, vertex_colors=None) -> None:
    """Write a folded 3D mesh to a Wavefront OBJ (faces may be tris or quads).

    If ``vertex_colors`` (V, 3 RGB in 0..1) is given, each vertex line carries
    ``v x y z r g b`` - the widely-read vertex-colour OBJ extension (MeshLab,
    Blender). Pass :func:`mountain_valley_colors` to tint by fold type.
    """
    v = np.asarray(vertices, dtype=float)
    if vertex_colors is None:
        lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    else:
        c = np.asarray(vertex_colors, dtype=float)
        lines = [f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}"
                 for (x, y, z), (r, g, b) in zip(v, c)]
    lines += ["f " + " ".join(str(i + 1) for i in face) for face in faces]  # OBJ is 1-indexed
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_stl(vertices, triangles, path: str | Path, name: str = "foldforge") -> None:
    """Write a folded 3D mesh to an ASCII STL (triangles only) - ready to 3D print."""
    v = np.asarray(vertices, dtype=float)
    out = [f"solid {name}"]
    for tri in triangles:
        a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        n = n / norm if norm > 1e-12 else n
        out.append(f"facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
        out.append("outer loop")
        for p in (a, b, c):
            out.append(f"vertex {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
        out.append("endloop")
        out.append("endfacet")
    out.append(f"endsolid {name}")
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


# --- glTF (binary .glb) export ----------------------------------------------

# glTF component/type constants (from the spec) we need.
_GL_FLOAT = 5126
_GL_UINT = 5125
_GL_ARRAY_BUFFER = 34962
_GL_ELEMENT_ARRAY_BUFFER = 34963


def _pad4(buf: bytearray, fill: bytes = b"\x00") -> None:
    """Pad a chunk to a 4-byte boundary in place (GLB requires it)."""
    while len(buf) % 4:
        buf += fill


def to_gltf(vertices, triangles, path: str | Path, vertex_colors=None) -> None:
    """Write a folded 3D mesh to a self-contained binary glTF (``.glb``).

    A single buffer holds float32 ``POSITION`` (with the spec-required min/max),
    uint32 triangle indices, and - if ``vertex_colors`` (V, 3 in 0..1) is given -
    a float32 ``COLOR_0`` VEC4 attribute (alpha 1). Written by hand with
    ``struct`` + ``json``; no third-party dependency. Pass
    :func:`mountain_valley_colors` to bake the fold pattern into the mesh.
    """
    v = np.asarray(vertices, dtype=np.float32)
    if len(v) == 0:
        raise ValueError("cannot export an empty mesh to glTF (no vertices)")
    tris = np.asarray(triangles, dtype=np.uint32).reshape(-1)

    # --- binary buffer: positions, then indices, then (optional) colours ---
    blob = bytearray()
    accessors = []
    buffer_views = []

    def add_view(raw: bytes, target: int) -> int:
        _pad4(blob)
        offset = len(blob)
        blob.extend(raw)
        buffer_views.append({"buffer": 0, "byteOffset": offset,
                             "byteLength": len(raw), "target": target})
        return len(buffer_views) - 1

    pos_view = add_view(v.tobytes(), _GL_ARRAY_BUFFER)
    accessors.append({
        "bufferView": pos_view, "componentType": _GL_FLOAT, "count": len(v),
        "type": "VEC3",
        "min": v.min(0).astype(float).tolist(),
        "max": v.max(0).astype(float).tolist(),
    })
    pos_acc = len(accessors) - 1

    idx_view = add_view(tris.tobytes(), _GL_ELEMENT_ARRAY_BUFFER)
    accessors.append({"bufferView": idx_view, "componentType": _GL_UINT,
                      "count": len(tris), "type": "SCALAR"})
    idx_acc = len(accessors) - 1

    attributes = {"POSITION": pos_acc}
    if vertex_colors is not None:
        c = np.asarray(vertex_colors, dtype=np.float32).reshape(len(v), -1)
        if c.shape[1] == 3:                 # promote RGB -> RGBA (alpha 1)
            c = np.concatenate([c, np.ones((len(c), 1), np.float32)], axis=1)
        col_view = add_view(c.tobytes(), _GL_ARRAY_BUFFER)
        accessors.append({"bufferView": col_view, "componentType": _GL_FLOAT,
                          "count": len(v), "type": "VEC4"})
        attributes["COLOR_0"] = len(accessors) - 1

    gltf = {
        "asset": {"version": "2.0", "generator": "foldforge"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [
            {"attributes": attributes, "indices": idx_acc, "mode": 4}]}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    # --- assemble the GLB container: header + JSON chunk + BIN chunk ---
    json_bytes = bytearray(json.dumps(gltf, separators=(",", ":")).encode("utf-8"))
    _pad4(json_bytes, b" ")                 # JSON chunk pads with spaces
    bin_bytes = bytearray(blob)
    _pad4(bin_bytes)                        # BIN chunk pads with zeros

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, total)            # 'glTF', v2, size
    out += struct.pack("<II", len(json_bytes), 0x4E4F534A)      # JSON chunk header
    out += json_bytes
    out += struct.pack("<II", len(bin_bytes), 0x004E4942)       # BIN chunk header
    out += bin_bytes
    Path(path).write_bytes(bytes(out))
