"""Proof of the photo -> stick-figure tree -> uniaxial base bridge.

For each sample it renders ``photo_to_tree_<name>.png``: the photo, its GrabCut
silhouette, the medial-axis skeleton over the silhouette, and the pruned
stick-figure metric tree (limbs) drawn on the photo. For the butterfly it also
renders ``photo_to_base_<name>.png``: the packed leaf circles and the resulting
crease pattern the base folds from.

    python examples/photo_to_tree.py

Honest by construction: it prints, per sample, the leaf count found versus the
subject's real limb count and whether the designed base folds flat.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foldforge.design import design_from_image
from foldforge.origamize import silhouette_mask

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "output")
SAMPLES = os.path.join(HERE, "samples")

# (name, expected real limbs, optional GrabCut rect override in orig px coords)
CASES = [
    ("starfish", 5, (288, 144, 384, 432)),   # arms; auto-seed fails on wood grain
    ("butterfly", 4, None),                   # four wings
    ("swallowtail", 4, None),                 # four wings (+ tails read as flaps)
]


def _draw_tree(ax, ext):
    """Overlay the pruned stick-figure tree: edges as lines, leaves as dots."""
    coords = ext.node_coords            # orig node id -> (row, col)
    adj = ext.tree_adj                  # pruned adjacency, orig ids
    drawn = set()
    for u, nbrs in adj.items():
        if u not in coords:
            continue
        for v, _ in nbrs:
            if v not in coords or (min(u, v), max(u, v)) in drawn:
                continue
            drawn.add((min(u, v), max(u, v)))
            (r0, c0), (r1, c1) = coords[u], coords[v]
            ax.plot([c0, c1], [r0, r1], "-", color="#ff3b3b", lw=2.5, zorder=3)
    for u, nbrs in adj.items():
        if u not in coords:
            continue
        r, c = coords[u]
        leaf = len(nbrs) == 1
        ax.plot(c, r, "o", ms=9 if leaf else 6,
                color="#ffd400" if leaf else "#ff3b3b",
                mec="black", mew=0.8, zorder=4)


def make_tree_figure(name, expected, rect):
    img = plt.imread(os.path.join(SAMPLES, f"{name}.jpg"))
    d = design_from_image(os.path.join(SAMPLES, f"{name}.jpg"),
                          expected_leaves=expected, rect=rect)
    ext = d.extraction
    mask = silhouette_mask(os.path.join(SAMPLES, f"{name}.jpg"), rect=rect)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img); axes[0].set_title(f"1. photo: {name}")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("2. silhouette (GrabCut)")
    axes[2].imshow(mask, cmap="gray")
    ys, xs = np.where(ext.skeleton)
    axes[2].scatter(xs, ys, s=0.5, c="#00e0ff")
    axes[2].set_title("3. medial-axis skeleton")
    axes[3].imshow(img)
    _draw_tree(axes[3], ext)
    axes[3].set_title(f"4. pruned tree: {ext.n_leaves} limbs")
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fold = "folds flat" if d.fold_flat else "partial base"
    fig.suptitle(
        f"{name}: {ext.n_leaves} leaves found (real subject ~{expected}); "
        f"{ext.n_leaves}-flap base {fold}", fontsize=14)
    fig.tight_layout()
    path = os.path.join(OUT, f"photo_to_tree_{name}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return d, path


def make_base_figure(name, d):
    """Packed leaf circles + crease pattern for a designed base."""
    from foldforge.geometry.render import render_pattern
    pk, pat = d.packing, d.pattern
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    ax = axes[0]
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="black"))
    for (cx, cy), r in zip(pk.centers, pk.radii):
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec="#1f77b4", lw=1.8))
        ax.plot(cx, cy, "o", color="#1f77b4", ms=5)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal"); ax.set_title(f"circle packing ({len(pk.centers)} flaps)")
    ax.set_xticks([]); ax.set_yticks([])
    render_pattern(pat, ax=axes[1])
    axes[1].set_aspect("equal")
    axes[1].set_title("crease pattern (M red / V blue)")
    fig.suptitle(
        f"{name}: {len(pk.centers)}-flap uniaxial base  "
        f"(folds flat: {d.fold_flat}, max flap error {d.max_flap_error * 100:.1f}%)",
        fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT, f"photo_to_base_{name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    print("photo -> stick tree -> base, per sample:\n")
    designs = {}
    for name, expected, rect in CASES:
        d, path = make_tree_figure(name, expected, rect)
        designs[name] = d
        e = d.extraction
        print(f"{name:11s} leaves={e.n_leaves} (real ~{expected})  "
              f"edges={len(e.edge_lengths)}  fold_flat={d.fold_flat}  "
              f"maxflaperr={d.max_flap_error * 100:4.1f}%")
        print(f"            {e.note}")
        print(f"            base: {d.note}")
        print(f"            -> {os.path.basename(path)}")
    # butterfly base figure (the one case that folds flat end to end)
    bpath = make_base_figure("butterfly", designs["butterfly"])
    print(f"\nbase figure -> {os.path.basename(bpath)}")


if __name__ == "__main__":
    main()
