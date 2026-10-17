"""Curved-crease origami demo -> examples/output/curved_crease.png.

Two curved folds, each shown as flat crease pattern + folded 3-D surface:
 (a) a single symmetric curved crease (circular-arc crease), and
 (b) the iconic concentric-circle pleat.
Both are annotated with the developability (isometry) error - the proof that a
curved crease bends flat paper into a *curved* surface with (essentially) zero
stretch.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foldforge.design import curved_fold, concentric_pleat

OUT = "examples/output"
os.makedirs(OUT, exist_ok=True)


def main():
    # --- (a) single symmetric curved fold of a circular-arc crease ----------
    a = np.linspace(-0.85, 0.85, 160)
    arc = np.stack([1.6 * np.sin(a), 1.6 * (1 - np.cos(a))], axis=1)
    f = curved_fold(arc, fold_angle=0.6, width=0.3, n_s=900)
    ns, nu = f.meta["n_s"], f.meta["n_u"]
    Ff = f.flat.reshape(ns, nu, 3)
    Fo = f.folded.reshape(ns, nu, 3)

    # --- (b) concentric circular pleat --------------------------------------
    radii = np.linspace(0.25, 1.0, 9)
    angles = np.deg2rad([1, -1, 1, -1, 1, -1, 1, -1]) * 32
    p = concentric_pleat(radii, angles, n_theta=160)
    nth = p.meta["n_theta"]
    nr = len(radii)
    Po = p.folded.reshape(nr, nth, 3)

    fig = plt.figure(figsize=(15, 8.2))
    fig.suptitle("Curved-crease origami: folding flat paper along smooth curves "
                 "into curved developable surfaces", fontsize=14, y=0.98)

    # (a1) flat crease pattern
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(Ff[:, nu // 2, 0], Ff[:, nu // 2, 1], "r-", lw=2.5, label="curved crease")
    for j in range(0, ns, ns // 22):                        # a sample of rulings
        ax.plot(Ff[j, :, 0], Ff[j, :, 1], color="#9aa7b2", lw=0.6)
    ax.set_aspect("equal"); ax.set_title("(a) flat pattern: 1 curved crease + rulings")
    ax.legend(loc="upper center", fontsize=8); ax.axis("off")

    # (a2) folded 3-D surface
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    ax.plot_surface(Fo[:, :, 0], Fo[:, :, 1], Fo[:, :, 2],
                    color="#4c9be8", alpha=0.9, linewidth=0, antialiased=True,
                    rcount=60, ccount=30)
    ax.plot(Fo[:, nu // 2, 0], Fo[:, nu // 2, 1], Fo[:, nu // 2, 2], "r-", lw=2.5)
    ax.set_title("folded: a CURVED developable surface")
    _equal_3d(ax, Fo.reshape(-1, 3)); ax.view_init(elev=28, azim=-60)

    # (a3) the headline number
    ax = fig.add_subplot(2, 3, 3); ax.axis("off")
    ax.text(0.02, 0.9,
            "Single symmetric curved fold\n"
            "(rulings perpendicular to the crease)\n\n"
            f"fold half-angle  theta = {f.meta['theta']:.2f} rad\n"
            f"flat crease curvature -> x{1/np.cos(f.meta['theta']):.2f} in 3-D\n\n"
            f"developability (isometry) error:\n"
            f"    {f.developability_error:.2e}\n"
            "  = max |E,F,G|_flat - |E,F,G|_folded\n\n"
            "~0  ->  a genuine BENDING of the sheet\n"
            "(zero stretch), not a deformation.",
            fontsize=10.5, va="top", family="monospace")

    # (b1) flat pattern: concentric circles
    ax = fig.add_subplot(2, 3, 4)
    th = np.linspace(0, 2 * np.pi, 200)
    for i, r in enumerate(radii):
        style = "-" if i == 0 or i == nr - 1 else ("--" if i % 2 else "-")
        col = "#333" if i in (0, nr - 1) else ("#d1495b" if i % 2 else "#2a6f97")
        ax.plot(r * np.cos(th), r * np.sin(th), style, color=col, lw=1.6)
    ax.set_aspect("equal")
    ax.set_title("(b) flat pattern: concentric M/V circles")
    ax.axis("off")

    # (b2) folded pleat
    ax = fig.add_subplot(2, 3, 5, projection="3d")
    ax.plot_surface(Po[:, :, 0], Po[:, :, 1], Po[:, :, 2],
                    cmap="viridis", linewidth=0, antialiased=True,
                    rcount=nr, ccount=nth)
    ax.set_title("folded: rippled circular pleat")
    _equal_3d(ax, Po.reshape(-1, 3)); ax.view_init(elev=18, azim=-60)

    # (b3) the pleat numbers
    ax = fig.add_subplot(2, 3, 6); ax.axis("off")
    ax.text(0.02, 0.9,
            "Concentric circular pleat\n"
            "(conical-strip model)\n\n"
            f"{p.meta['n_strips']} annular strips, alternating M/V\n"
            f"each strip -> a cone frustum\n\n"
            f"crease-circle radius error:\n"
            f"    {p.meta['radius_error']:.1e}  (exact)\n\n"
            f"radial strain (model error):\n"
            f"    {p.developability_error*100:.1f}%\n\n"
            "Honest: the exact pleat has NO axisymmetric\n"
            "isometric embedding - real paper buckles into\n"
            "a non-axisymmetric wavy shape (not modelled).",
            fontsize=10.5, va="top", family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = f"{OUT}/curved_crease.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)
    print(f"single fold developability_error = {f.developability_error:.3e}")
    print(f"pleat radius_error = {p.meta['radius_error']:.3e}, "
          f"radial_strain = {p.developability_error:.3%}")


def _equal_3d(ax, pts):
    """Set equal aspect on a 3-D axis (matplotlib doesn't do it for us)."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = (lo + hi) / 2
    r = (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


if __name__ == "__main__":
    main()
