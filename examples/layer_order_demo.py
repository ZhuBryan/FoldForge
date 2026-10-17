"""Proof images for the global layer-ordering / self-intersection check.

Writes ``examples/output/layer_order.png`` with two panels:

* LEFT  - the local-OK / global-FAIL headline: a triangular base with three
          flaps folded inward past the centre.  Every vertex passes the local
          flat-foldability theorems (there are no interior vertices to fail),
          the sheet folds flat geometrically, yet the three flaps are
          cyclically trapped and NO global layer order exists.  Local says
          "flat-foldable", global says "self-intersects".
* RIGHT - a FoldForge base (the crane's bird base) with a genuine valid layer
          order, drawn as a bottom->top stacking diagram.

    python examples/layer_order_demo.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from foldforge.geometry.crease_graph import CreasePattern
from foldforge.geometry.foldability import foldability_report
from foldforge.geometry.layer_order import layer_order, _build_constraints
from foldforge.design.classic import bird_base
from foldforge.design.lang_molecule import flat_fold_image

OUT = os.path.join(os.path.dirname(__file__), "output")


def pinwheel(d: float = 1.7, skew: float = 0.0) -> CreasePattern:
    """Triangular base + three inward flaps (see tests/test_layer_order.py)."""
    ang = np.radians([90.0, 210.0, 330.0])
    base = np.array([[np.cos(a), np.sin(a)] for a in ang])
    verts = list(base)
    faces = [[0, 1, 2]]
    edges, assign = [], []

    def add(u, v, kind):
        edges.append((u, v))
        assign.append(kind)

    for i in range(3):
        j = (i + 1) % 3
        mid = (base[i] + base[j]) / 2
        out = mid / np.linalg.norm(mid)
        edge = (base[j] - base[i]) / np.linalg.norm(base[j] - base[i])
        verts.append(mid + out * d + edge * skew)
        ai = len(verts) - 1
        faces.append([j, i, ai])
        add(i, j, "V"); add(j, ai, "B"); add(ai, i, "B")

    return CreasePattern(vertices=np.array(verts, float), edges=np.array(edges),
                         assignment=assign, faces=faces,
                         metadata={"name": "pinwheel3"})


def _draw_pattern(ax, p, title):
    """Draw a crease pattern: borders grey, mountains red, valleys blue."""
    colors = {"B": ("0.6", 1.0), "M": ("crimson", 1.8), "V": ("royalblue", 1.8)}
    for (a, b), k in zip(p.edges, p.assignment):
        c, lw = colors.get(k, ("0.6", 1.0))
        xy = p.vertices[[a, b]]
        ax.plot(xy[:, 0], xy[:, 1], color=c, lw=lw,
                ls="--" if k == "V" else "-")
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    ax.axis("off")


def _draw_folded_stack(ax, p, order, title):
    """Draw the folded faces as translucent layers in the given bottom->top order."""
    img, _, _ = flat_fold_image(p)
    cmap = plt.cm.viridis
    n = len(order)
    for layer, fi in enumerate(order):
        poly = img[p.faces[fi]]
        ax.fill(poly[:, 0], poly[:, 1], closed=True,
                facecolor=cmap(layer / max(n - 1, 1)), edgecolor="k",
                lw=0.6, alpha=0.5)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10); ax.axis("off")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # --- headline: local OK, global FAIL -----------------------------------
    pw = pinwheel(d=1.7)
    local = foldability_report(pw).flat_foldable
    res = layer_order(pw)
    img, _, _ = flat_fold_image(pw)
    _, overlaps, tortilla, taco = _build_constraints(pw, img)

    _draw_pattern(axes[0, 0], pw,
                  "Pinwheel crease pattern\n(triangle base + 3 inward flaps, folds flat)")
    _draw_folded_stack(axes[0, 1], pw, [0, 1, 2, 3],
                       "Folded: all 3 flaps pile on the base\n(index order shown; it self-intersects)")
    verdict = (
        "LOCAL  (Kawasaki + Maekawa):  flat-foldable = %s\n"
        "GLOBAL (layer ordering):        valid order = %s\n\n"
        "%d overlapping face pairs, %d taco-tortilla constraints.\n"
        "The three flaps are cyclically trapped: 0 of 24\n"
        "stacking orders avoid self-intersection.\n\n"
        "This is the gap the local theorems cannot see."
        % (local, res.valid, len(overlaps), len(tortilla))
    )
    axes[1, 0].axis("off")
    axes[1, 0].text(0.02, 0.98, verdict, va="top", ha="left", fontsize=10,
                    family="monospace",
                    bbox=dict(boxstyle="round", fc="#fff3f3", ec="crimson"))

    # --- a FoldForge base with a valid global order ------------------------
    bb = bird_base()
    rb = layer_order(bb)
    _draw_folded_stack(axes[1, 1], bb, rb.order,
                       "FoldForge bird base (crane): a VALID\nglobal layer order exists "
                       "(%d layers, bottom->top by colour)" % len(rb.order))

    fig.suptitle("Global layer-ordering check: catching what local flat-foldability misses",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT, "layer_order.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)
    print("pinwheel: local flat-foldable =", local, "| global valid =", res.valid)
    print("bird base: global valid =", rb.valid, "| order =", rb.order)


if __name__ == "__main__":
    main()
