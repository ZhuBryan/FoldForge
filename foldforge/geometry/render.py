"""Draw a flat crease pattern with the usual origami colour code.

Convention used by most origami tools (and the FOLD viewer):
    mountain -> red, valley -> blue, border -> black, flat/unassigned -> grey.

This is matplotlib only: good enough to eyeball a pattern in a notebook or save
a PNG for the README. The interactive 3D viewer is a later milestone (M6).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from foldforge.geometry.crease_graph import CreasePattern

# assignment -> (colour, linestyle). Mountains dash-dot, valleys dashed: the
# pattern stays readable in greyscale / print, not just by colour.
_STYLE = {
    "M": ("#d62728", "dashdot"),
    "V": ("#1f77b4", "dashed"),
    "B": ("#000000", "solid"),
    "F": ("#bbbbbb", "solid"),
    "U": ("#bbbbbb", "dotted"),
}


def render_pattern(
    pattern: CreasePattern,
    ax: "plt.Axes | None" = None,
    show_vertices: bool = False,
    title: str | None = None,
):
    """Render ``pattern`` flat and return the matplotlib ``Axes``.

    Args:
        pattern:        the crease pattern to draw.
        ax:             draw into this Axes if given, else make a new figure.
        show_vertices:  scatter the vertices (handy when debugging geometry).
        title:          plot title; defaults to the pattern's name.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    verts = pattern.vertices

    # One LineCollection per style keeps the draw fast and the legend clean.
    for kind, (colour, style) in _STYLE.items():
        segments = [
            [verts[a, :2], verts[b, :2]]
            for (a, b), k in zip(pattern.edges, pattern.assignment)
            if k == kind
        ]
        if segments:
            ax.add_collection(
                LineCollection(
                    segments, colors=colour, linestyles=style, linewidths=1.8,
                    label=_LEGEND[kind],
                )
            )

    if show_vertices:
        ax.scatter(verts[:, 0], verts[:, 1], s=12, color="#333333", zorder=3)

    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.05)
    ax.set_title(title or pattern.metadata.get("name", "crease pattern"))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return ax


_LEGEND = {
    "M": "mountain",
    "V": "valley",
    "B": "border",
    "F": "flat",
    "U": "unassigned",
}


def render_folded(mesh, positions, ax=None, elev: float = 28.0,
                  azim: float = -60.0, title: str | None = None):
    """Draw a folded 3D mesh: paper panels plus coloured crease lines.

    Args:
        mesh:       a FoldMesh (gives us the triangles and creases).
        positions:  (V, 3) folded vertex positions (from a FoldResult).
        ax:         a 3D Axes to draw into, or None to make one.
        elev, azim: camera angles.
        title:      plot title.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

    panels = [positions[tri] for tri in mesh.triangles]
    ax.add_collection3d(Poly3DCollection(
        panels, facecolor="#ece3d0", edgecolor="#b0a890",
        linewidths=0.3, alpha=0.95,
    ))

    # Crease lines: mountains fold to a negative target angle, valleys positive.
    for h in mesh.hinges:
        if not h.is_crease:
            continue
        seg = positions[list(h.edge)]
        colour = "#d62728" if h.target < 0 else "#1f77b4"
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=colour, linewidth=2.0)

    # Equal aspect: 3D matplotlib needs the box aspect set from the data ranges.
    span = positions.max(axis=0) - positions.min(axis=0)
    ax.set_box_aspect(np.where(span > 1e-9, span, 1.0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax
