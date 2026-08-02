"""Proof images for the TreeMaker-lite figurative design path.

Runs the flagship 3-flap tree end to end and writes, into examples/output/:
    treemaker_packing.png   the optimised circle packing in the unit square
    treemaker_crease.svg     M/V-coloured crease pattern (via fabricate.to_svg)
    treemaker_crease.png     a rendered copy of the same pattern
    treemaker_base.png       a 2-D schematic of the folded uniaxial base

    python examples/treemaker_demo.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from foldforge.design import (
    get_tree, design_base, flap_length_errors, folded_schematic,
)
from foldforge.fabricate import to_svg
from foldforge.geometry.foldability import foldability_report

OUT = os.path.join(os.path.dirname(__file__), "output")
_MV = {"M": "#d62728", "V": "#1f77b4", "B": "#333333", "F": "#999999", "U": "#999999"}


def packing_png(packing, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="black", lw=1.5))
    for (x, y), r, leaf in zip(packing.centers, packing.radii, packing.leaves):
        ax.add_patch(plt.Circle((x, y), r, fill=False, ec="#1f77b4", lw=1.8))
        ax.plot([x], [y], "o", color="#d62728", ms=5)
        ax.text(x, y + 0.03, f"flap {leaf}", ha="center", fontsize=9)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect("equal")
    ax.set_title(f"circle packing  (scale = {packing.scale:.3f})")
    ax.axis("off")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def crease_png(pattern, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    v = pattern.vertices
    for (a, b), kind in zip(pattern.edges, pattern.assignment):
        ax.plot([v[a, 0], v[b, 0]], [v[a, 1], v[b, 1]],
                color=_MV.get(kind, "#999"), lw=2.0 if kind in "MV" else 1.4)
    ax.plot(v[:, 0], v[:, 1], "o", color="black", ms=3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("rabbit-ear crease pattern\n(red=mountain, blue=valley, black=border)")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def base_png(packing, path):
    spine, tips = folded_schematic(packing)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(spine[:, 0], spine[:, 1], "-", color="black", lw=2.5, label="axis (spine)")
    for (sx, sy), (tx, ty), leaf in zip(spine, tips, packing.leaves):
        ax.plot([sx, tx], [sy, ty], "-", color="#1f77b4", lw=3)
        ax.plot([tx], [ty], "o", color="#d62728", ms=7)
        ax.text(tx + 0.03, ty, f"flap {leaf}\nlen {ty:.3f}", va="center", fontsize=9)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("folded uniaxial base (schematic): flaps hang off one axis")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    tree = get_tree("three-flap")
    packing, pattern = design_base(tree)
    err = flap_length_errors(packing, pattern)
    report = foldability_report(pattern)
    n = len(report.vertices)
    nk = sum(1 for x in report.vertices if x.kawasaki)
    nm = sum(1 for x in report.vertices if x.maekawa)

    packing_png(packing, os.path.join(OUT, "treemaker_packing.png"))
    to_svg(pattern, os.path.join(OUT, "treemaker_crease.svg"))
    crease_png(pattern, os.path.join(OUT, "treemaker_crease.png"))
    base_png(packing, os.path.join(OUT, "treemaker_base.png"))

    print("three-flap base, end to end:")
    print(f"  packing scale   : {packing.scale:.4f}")
    print(f"  flap len error  : max {err.max() * 100:.2f}%  mean {err.mean() * 100:.2f}%")
    print(f"  Kawasaki        : {nk}/{n} interior vertices pass")
    print(f"  Maekawa         : {nm}/{n} interior vertices pass")
    print(f"  wrote 4 files into {OUT}")


if __name__ == "__main__":
    main()
