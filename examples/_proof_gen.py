"""Generate photo->tree and photo->base proof panels for the skeleton bridge."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foldforge.design.skeleton import tree_from_image, design_from_image

OUT = "examples/output"


def draw_tree(ax, coords, adj, color="#ff3b30"):
    """Draw a stick-figure tree (edges then nodes) using (row, col) coords."""
    seen = set()
    for u in adj:
        if u not in coords:
            continue
        for v, _ in adj[u]:
            if v not in coords:
                continue
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            (r0, c0), (r1, c1) = coords[u], coords[v]
            ax.plot([c0, c1], [r0, r1], "-", color=color, lw=2.6, zorder=3)
    for u in adj:
        if u not in coords:
            continue
        r, c = coords[u]
        leaf = len(adj[u]) == 1
        ax.plot(c, r, "o", color="#ffcc00" if leaf else color,
                ms=10 if leaf else 6, mec="k", mew=0.8, zorder=4)


def panel(name):
    ext = tree_from_image(f"examples/samples/{name}.jpg")
    photo = plt.imread(f"examples/samples/{name}.jpg")
    mask = np.asarray(ext.mask) > 0
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
    ax[0].imshow(photo); ax[0].set_title(f"1. photo: {name}")
    ax[1].imshow(mask, cmap="gray"); ax[1].set_title("2. silhouette mask")
    ax[2].imshow(mask, cmap="gray")
    ys, xs = np.where(ext.skeleton)
    ax[2].scatter(xs, ys, s=1.0, c="#00d0ff")
    ax[2].set_title("3. medial-axis skeleton")
    ax[3].imshow(mask, cmap="gray")
    draw_tree(ax[3], ext.node_coords, ext.tree_adj)
    ax[3].set_title(f"4. pruned stick-figure tree: {ext.n_leaves} limbs")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"photo -> stick-figure tree  |  {name}: {ext.note}", fontsize=13)
    fig.tight_layout()
    p = f"{OUT}/photo_to_tree_{name}.png"
    fig.savefig(p, dpi=85, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p, "leaves=", ext.n_leaves)


def base_panel(name):
    d = design_from_image(f"examples/samples/{name}.jpg")
    ext, pk, pt = d.extraction, d.packing, d.pattern
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))

    mask = np.asarray(ext.mask) > 0
    ax[0].imshow(mask, cmap="gray")
    draw_tree(ax[0], ext.node_coords, ext.tree_adj)
    ax[0].set_title(f"1. extracted tree: {ext.n_leaves} limbs")
    ax[0].axis("off")

    ax[1].set_aspect("equal")
    ax[1].add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="#333", lw=1))
    for (cx, cy), rr in zip(pk.centers, pk.radii):
        ax[1].add_patch(plt.Circle((cx, cy), rr, fill=False, ec="#ff3b30", lw=1.6))
        ax[1].plot(cx, cy, "o", color="#ff3b30", ms=4)
    ax[1].set_xlim(-0.02, 1.02); ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_title("2. packed flap circles")

    ax[2].set_aspect("equal")
    V = np.asarray(pt.vertices)
    col = {"M": "#d00", "V": "#06c", "B": "#333", "F": "#bbb", "U": "#999"}
    for (i, j), a in zip(pt.edges, pt.assignment):
        ax[2].plot([V[i][0], V[j][0]], [V[i][1], V[j][1]], "-",
                   color=col.get(str(a), "#999"), lw=1.3)
    ax[2].set_title("3. crease pattern (M=red, V=blue, B=black)")
    ax[2].axis("equal"); ax[2].axis("off")

    fold = "folds FLAT" if d.fold_flat else "PARTIAL (not flat-foldable)"
    fig.suptitle(
        f"photo -> tree -> base  |  {name}: {ext.n_leaves}-flap base {fold}\n"
        f"{d.note}", fontsize=10)
    fig.tight_layout()
    p = f"{OUT}/photo_to_base_{name}.png"
    fig.savefig(p, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)
    print("FOLD_VALIDITY:", d.note)


if __name__ == "__main__":
    which = sys.argv[1]
    if which.startswith("base:"):
        base_panel(which.split(":", 1)[1])
    else:
        panel(which)
