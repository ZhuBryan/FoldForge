"""Origamizer tuck-folding demo: a flat sheet -> a 3D square pyramid.

Produces examples/output/origamizer.png with three panels:
  1. the flat crease pattern (faces + tuck pleats, mountain/valley coloured),
  2. the folded 3D result,
  3. the target polyhedron,
annotated with the face-position / dihedral / isometry errors.

    PYTHONPATH=. python examples/origamizer_demo.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from foldforge.design import origamize_polyhedron, square_pyramid
from foldforge.geometry.render import render_pattern

OUT = os.path.join(os.path.dirname(__file__), "output")


def _draw_solid(ax, verts, faces, title, folded=None):
    panels = [verts[f] for f in faces]
    ax.add_collection3d(
        Poly3DCollection(panels, facecolor="#f0c674", edgecolor="#8a6d1a",
                         linewidths=1.0, alpha=0.9)
    )
    span = verts.max(axis=0) - verts.min(axis=0)
    import numpy as np
    ax.set_box_aspect(np.where(span > 1e-9, span, 1.0))
    ax.view_init(elev=22, azim=-55)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)


def main():
    os.makedirs(OUT, exist_ok=True)
    target = square_pyramid()
    res = origamize_polyhedron(target)
    rep = res.report
    print(rep)

    fig = plt.figure(figsize=(16, 5.5))

    # 1. flat crease pattern
    ax1 = fig.add_subplot(1, 3, 1)
    render_pattern(res.pattern, ax=ax1,
                   title="flat crease pattern (4 tuck pleats)")

    # 2. folded result (fold the actual crease-pattern faces)
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw_solid(ax2, res.folded, res.pattern.faces, "folded result")

    # 3. target polyhedron
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    _draw_solid(ax3, res.target_vertices, res.target_faces, "target square pyramid")

    fig.suptitle(
        "Origamizer tuck-folding: flat sheet -> square pyramid     "
        f"face-position err {rep.face_position_error:.1e}   "
        f"dihedral err {rep.dihedral_error_deg:.1e} deg   "
        f"isometry err {rep.isometry_error:.1e}   "
        f"tucked {rep.area_tucked:.2f} of {rep.area_flat:.2f} paper",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT, "origamizer.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    main()
