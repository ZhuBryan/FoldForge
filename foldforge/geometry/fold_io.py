"""Read and write the FOLD file format (the standard for crease patterns).

FOLD is a small JSON schema by Demaine et al. (github.com/edemaine/fold). Using
it means FoldForge files open in other origami tools, and vice versa. We only
touch the handful of fields M0 needs; unknown fields are preserved on read so we
don't silently throw away data.

The fields we use:
    vertices_coords    list of [x, y] (or [x, y, z]) points
    edges_vertices     list of [v1, v2] index pairs
    edges_assignment   list of "M"/"V"/"B"/"F"/"U"
    edges_foldAngle    optional list of fold angles in degrees
    faces_vertices     optional list of vertex-index loops
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from foldforge.geometry.crease_graph import CreasePattern

# Keys we map onto CreasePattern fields; everything else is kept in metadata.
_HANDLED_KEYS = {
    "vertices_coords",
    "edges_vertices",
    "edges_assignment",
    "edges_foldAngle",
    "faces_vertices",
}


def read_fold(path: str | Path) -> CreasePattern:
    """Load a ``.fold`` file into a :class:`CreasePattern`.

    Missing edge assignments default to "U" (unassigned). Any FOLD field we
    don't model (file_title, frame metadata, ...) is stashed in
    ``pattern.metadata`` so a later ``write_fold`` round-trips it unchanged.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    vertices = np.asarray(data["vertices_coords"], dtype=float)
    edges = np.asarray(data["edges_vertices"], dtype=int)

    assignment = data.get("edges_assignment")
    if assignment is None:
        assignment = ["U"] * len(edges)
    assignment = [str(a).upper() for a in assignment]

    fold_angle = data.get("edges_foldAngle")
    fold_angle = np.asarray(fold_angle, dtype=float) if fold_angle is not None else None

    faces = [list(map(int, f)) for f in data.get("faces_vertices", [])]

    metadata = {k: v for k, v in data.items() if k not in _HANDLED_KEYS}
    metadata.setdefault("name", data.get("file_title", Path(path).stem))

    return CreasePattern(
        vertices=vertices,
        edges=edges,
        assignment=assignment,
        faces=faces,
        fold_angle=fold_angle,
        metadata=metadata,
    )


def write_fold(pattern: CreasePattern, path: str | Path) -> None:
    """Write a :class:`CreasePattern` back out as a ``.fold`` file."""
    # Start from preserved metadata so unknown fields survive a read/write trip.
    data = dict(pattern.metadata)
    data.setdefault("file_spec", 1.1)
    data.setdefault("file_creator", "FoldForge")
    data["vertices_coords"] = pattern.vertices.tolist()
    data["edges_vertices"] = pattern.edges.tolist()
    data["edges_assignment"] = list(pattern.assignment)
    if pattern.fold_angle is not None:
        data["edges_foldAngle"] = pattern.fold_angle.tolist()
    if pattern.faces:
        data["faces_vertices"] = pattern.faces

    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
