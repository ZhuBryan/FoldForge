"""Proof image for the general river work (Stage 3, gate relaxed in Stage 4).

Runs every built-in river base end to end and draws, for each, its circle
packing (river bands shaded), its crease pattern (mountain/valley/border, spine
creases thickened, sub-hub nodes ringed) and an honest Kawasaki / Maekawa / flap
verdict. All four fold flat end to end (symmetric, mirror-symmetric and fully
asymmetric single spines, plus the linear 3-hub caterpillar - thanks to the
angle-aware packing; the caterpillar carries ~1% flap error on its far flaps).

    python examples/treemaker_river_general.py

writes examples/output/treemaker_river_general.png
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from foldforge.design import get_tree, design_base, flap_length_errors
from foldforge.geometry.foldability import foldability_report

OUT = os.path.join(os.path.dirname(__file__), "output")
_MV = {"M": "#d62728", "V": "#1f77b4", "B": "#333333", "F": "#999", "U": "#999"}

RIVERS = [
    ("river-four", "symmetric single spine\n[1,1 | 1,1]"),
    ("river-mirror", "mirror-symmetric single spine\n[1.1,0.9 | 0.9,1.1]"),
    ("river-asym", "fully asymmetric single spine\n[1.2,0.8 | 1.0,0.9]"),
    ("river-caterpillar", "multi-junction (3 hubs, 2 spines)\ncaterpillar"),
]


def _stats(name):
    packing, pattern = design_base(get_tree(name))
    err = flap_length_errors(packing, pattern)
    report = foldability_report(pattern)
    n = len(report.vertices)
    nk = sum(1 for x in report.vertices if x.kawasaki)
    nm = sum(1 for x in report.vertices if x.maekawa)
    works = (nk == n and nm == n and err.max() < 0.02)
    return packing, pattern, err, (n, nk, nm), works


def _packing_panel(ax, packing):
    tree = packing.tree
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="black", lw=1.2))
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]
    hubcol, cen = {}, packing.centers
    for (x, y), r, leaf in zip(cen, packing.radii, packing.leaves):
        hub = tree.parent_hub(leaf)
        col = hubcol.setdefault(hub, palette[len(hubcol) % len(palette)])
        ax.add_patch(plt.Circle((x, y), r, fill=False, ec=col, lw=1.6))
        ax.plot([x], [y], "o", color="#d62728", ms=4)
        ax.text(x, y + 0.035, f"{leaf}", ha="center", fontsize=7, color=col)
    # shade every open river band = a cross-hub pair held more than tangent apart.
    scale = packing.scale
    for i in range(len(packing.leaves)):
        for j in range(i + 1, len(packing.leaves)):
            li, lj = packing.leaves[i], packing.leaves[j]
            band = scale * tree.path_length(li, lj) - (packing.radii[i] + packing.radii[j])
            if band > 1e-3:
                mid = (cen[i] + cen[j]) / 2
                ax.plot([cen[i, 0], cen[j, 0]], [cen[i, 1], cen[j, 1]],
                        "-", color="#d67d1d", lw=0.8, alpha=0.5)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"packing (scale={scale:.3f})", fontsize=9)


def _crease_panel(ax, pattern):
    v = pattern.vertices
    iv = set(pattern.interior_vertices())
    for (a, b), kind in zip(pattern.edges, pattern.assignment):
        both = a in iv and b in iv
        lw = 3.0 if (both and kind in "MV") else (1.8 if kind in "MV" else 1.2)
        ax.plot([v[a, 0], v[b, 0]], [v[a, 1], v[b, 1]],
                color=_MV.get(kind, "#999"), lw=lw)
    ax.plot(v[:, 0], v[:, 1], "o", color="black", ms=2)
    for i in iv:
        ax.plot(v[i, 0], v[i, 1], "o", mfc="none", mec="#d67d1d", ms=11, mew=1.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("crease pattern\n(red=M, blue=V, black=border; thick=interior)",
                 fontsize=8)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(len(RIVERS), 2, figsize=(9.5, 4.3 * len(RIVERS)))
    print("General river bases (Stage 4):")
    for row, (name, desc) in enumerate(RIVERS):
        packing, pattern, err, (n, nk, nm), works = _stats(name)
        _packing_panel(axes[row, 0], packing)
        _crease_panel(axes[row, 1], pattern)
        verdict = "FOLDS FLAT END TO END" if works else "PARTIAL (honest boundary)"
        box = (f"{name}\n{desc}\n"
               f"interior vertices: {n}\n"
               f"Kawasaki: {nk}/{n}   Maekawa: {nm}/{n}\n"
               f"max flap error: {err.max() * 100:.2f}%\n{verdict}")
        axes[row, 1].text(
            1.04, 0.5, box, transform=axes[row, 1].transAxes, va="center",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="#eaffea" if works else "#fff2e6",
                      ec="#2ca02c" if works else "#d67d1d"))
        print(f"  {name:20s} K={nk}/{n} M={nm}/{n} flap={err.max()*100:.2f}% "
              f"-> {'FLAT' if works else 'PARTIAL'}")
    fig.suptitle("TreeMaker-lite general rivers: what the unified molecule folds",
                 fontsize=13, y=0.997)
    path = os.path.join(OUT, "treemaker_river_general.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
