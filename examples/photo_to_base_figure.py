"""Render the end-to-end photo -> flat-foldable base figure.

Runs the full pipeline on one photo (segmentation -> skeleton -> stick-figure
tree -> circle packing -> universal-molecule crease pattern) and lays the five
stages out left to right with the measured fold-validity.

Usage: python examples/photo_to_base_figure.py [sample_name]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from foldforge.design import design_from_image

_MV = {"M": "#d62728", "V": "#1f77b4", "B": "#333333", "F": "#33aa55"}


def main(name="starfish"):
    path = f"examples/samples/{name}.jpg"
    d = design_from_image(path, expected_leaves=5 if name == "starfish" else None)
    ex, pk, cp = d.extraction, d.packing, d.pattern

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))

    # 1 - photo
    axes[0].imshow(Image.open(path))
    axes[0].set_title("1 - photo")

    # 2 - silhouette (mask recovered from the distance transform)
    mask = ex.dist > 0
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("2 - silhouette")

    # 3 - stick-figure tree over the silhouette
    axes[2].imshow(mask, cmap="gray")
    nc = ex.node_coords
    for a, nbrs in ex.tree_adj.items():
        for nb in nbrs:                       # neighbours are (node_id, length)
            b = nb[0]
            axes[2].plot([nc[a][1], nc[b][1]], [nc[a][0], nc[b][0]],
                         "-", color="#ff3b30", lw=2.0)
    axes[2].plot([c[1] for c in nc.values()], [c[0] for c in nc.values()],
                 "o", color="#ffcc00", ms=6, mec="black")
    axes[2].set_title(f"3 - stick figure ({ex.n_leaves} limbs)")

    # 4 - circle packing
    axes[3].add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="black"))
    for c, r in zip(pk.centers, pk.radii):
        axes[3].add_patch(plt.Circle(c, r, fill=False, ec="#1f77b4", lw=1.5))
        axes[3].plot([c[0]], [c[1]], "o", color="#d62728", ms=5)
    axes[3].set_xlim(-0.05, 1.05); axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_aspect("equal"); axes[3].set_title(f"4 - circle packing (scale {pk.scale:.3f})")

    # 5 - universal-molecule crease pattern
    v = cp.vertices
    for (a, b), kind in zip(cp.edges, cp.assignment):
        axes[4].plot([v[a][0], v[b][0]], [v[a][1], v[b][1]],
                     color=_MV.get(kind, "#999"), lw=2.0 if kind in "MV" else 1.4)
    axes[4].plot(v[:, 0], v[:, 1], "o", color="black", ms=2)
    axes[4].set_aspect("equal")
    axes[4].set_title("5 - crease pattern\n(red=mountain, blue=valley)")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    verdict = ("FOLDS FLAT" if d.fold_flat else "PARTIAL")
    fig.suptitle(
        f"Photo -> flat-foldable origami base   |   {name}: "
        f"{ex.n_leaves}-flap base - Kawasaki {d.kawasaki_pass}/{d.n_interior} - "
        f"Maekawa {d.maekawa_pass}/{d.n_interior} - flap error "
        f"{d.max_flap_error*100:.1f}% - {verdict}",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = f"examples/output/photo_to_foldable_base_{name}.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)
    print(f"{name}: leaves={ex.n_leaves} fold_flat={d.fold_flat} "
          f"Kawasaki={d.kawasaki_pass}/{d.n_interior} "
          f"Maekawa={d.maekawa_pass}/{d.n_interior} "
          f"flap_err={d.max_flap_error*100:.3f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "starfish")
