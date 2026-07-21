"""Regenerate the sample-animal gallery: solids + showcase figure.

Folds each photo in examples/samples/ into a closed origami solid and renders
examples/output/animals_showcase.png (photo -> height-field heatmap -> solid).
Sources are public-domain/CC images from Wikimedia Commons (see README).

Run from the repo root:  python examples/make_samples.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from foldforge.origamize import origamize_silhouette
from foldforge.fabricate import to_stl

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples")
OUT = os.path.join(HERE, "output")

# rect (original-pixel coords) only where the busy background defeats auto-rect
ANIMALS = [
    ("cat", None),
    ("dog", None),
    ("elephant", (230, 800, 1470, 1500)),
    ("butterfly", None),
]

HEAT = LinearSegmentedColormap.from_list(
    "studio", ["#0a0d12", "#1830c8", "#00b7c2", "#3fd66b", "#ffd23c", "#ff5a3c", "#c81818"])


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, rect in ANIMALS:
        path = os.path.join(SAMPLES, name + ".jpg")
        res, relief = origamize_silhouette(path, folds=30, closed=True, rect=rect)
        v, t = res.solid
        to_stl(v, t, os.path.join(OUT, name + "_solid.stl"))
        rows.append((name, path, relief, res.error, v, t))
        print(f"{name}: fold error {res.error:.3f}, {len(t)} triangles")

    fig = plt.figure(figsize=(13, 4.1 * len(rows)))
    for r, (name, path, relief, err, v, t) in enumerate(rows):
        ax = fig.add_subplot(len(rows), 3, r * 3 + 1)
        ax.imshow(Image.open(path)); ax.set_axis_off()
        ax.set_title(f"{name} — photo" if r == 0 else name, fontsize=12, loc="left")

        ax = fig.add_subplot(len(rows), 3, r * 3 + 2)
        R = relief.copy(); R[R < 0.02] = np.nan
        ax.imshow(relief * 0, cmap="gray", vmin=0, vmax=1)
        ax.imshow(R, cmap=HEAT, vmin=0, vmax=1); ax.set_axis_off()
        if r == 0: ax.set_title("height-field heatmap", fontsize=12, loc="left")

        ax = fig.add_subplot(len(rows), 3, r * 3 + 3, projection="3d")
        T = v[t]
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        lam = np.clip(n @ np.array([0.4, -0.5, 0.75]), 0.12, 1)
        base = np.array([0.85, 0.77, 0.60])
        ax.add_collection3d(Poly3DCollection(
            T, facecolors=np.clip(lam[:, None] * base, 0, 1), edgecolor="none"))
        lo, hi = v.min(0), v.max(0); c = (lo + hi) / 2; rad = (hi - lo).max() / 2
        ax.set_xlim(c[0] - rad, c[0] + rad); ax.set_ylim(c[1] - rad, c[1] + rad)
        ax.set_zlim(c[2] - rad, c[2] + rad)
        ax.view_init(28, -55); ax.set_axis_off()
        if r == 0: ax.set_title("folded closed solid", fontsize=12, loc="left")
        ax.text2D(0.02, 0.02, f"fold error {err:.3f}",
                  transform=ax.transAxes, fontsize=9, color="#555")

    plt.suptitle("FoldForge — photo → segmented height field → folded origami solid",
                 fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUT, "animals_showcase.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
