"""Proof figure for the deployable-structure demo.

A Miura-ori flat sheet that stows compactly and deploys to a target shape - the
aerospace application of origami (solar arrays, antennas). Produces
``examples/output/deployable.png`` with four panels:

  (a) the array at stowed / mid / deployed (3D), one single-DOF mechanism;
  (b) in-plane footprint area vs the deployment parameter t (the compaction);
  (c) max crease-edge deviation across the sweep (rigid-foldability: ~1e-15);
  (d) an inverse-designed array whose deployed footprint hits a target
      (target vs achieved bounding box), annotated with its compaction ratio.

This is the geometric / kinematic design layer: compaction, single-DOF
rigid-foldability and inverse design to a target. It models no material
thickness, hinge stiffness or actuation force - the front end of
deployable-structure design, not a structural/FEA tool.

Run:  PYTHONPATH=. python examples/deployable_demo.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from foldforge.design.deployable import MiuraDeployable, inverse_design_deployed

OUT = "examples/output"


def _panels(V):
    """Quad facets of a folded vertex grid (R+1, C+1, 3) as polygon list."""
    R, C = V.shape[0] - 1, V.shape[1] - 1
    return [[V[i, j], V[i + 1, j], V[i + 1, j + 1], V[i, j + 1]]
            for i in range(R) for j in range(C)]


def _draw_array(ax, V, title):
    polys = Poly3DCollection(_panels(V), facecolor="#4C78A8", edgecolor="#1b2a3a",
                             linewidths=0.4, alpha=0.9)
    ax.add_collection3d(polys)
    pts = V.reshape(-1, 3)
    for k, lim in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        c = pts[:, k]
        m = 0.5 * (c.min() + c.max()); r = max(c.max() - c.min(), 1.0) * 0.6
        lim(m - r, m + r)
    ax.set_box_aspect((1, 1, 0.5))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def main():
    os.makedirs(OUT, exist_ok=True)
    dep = MiuraDeployable(rows=6, cols=6)
    s = dep.summary()

    fig = plt.figure(figsize=(13, 9))

    # (a) stowed / mid / deployed 3D
    for col, t, name in ((0, 0.0, "stowed  (t=0)"),
                         (1, 0.5, "mid-deploy  (t=0.5)"),
                         (2, 1.0, "deployed  (t=1)")):
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        _draw_array(ax, dep.vertices(t), name)
    fig.text(0.5, 0.955,
             "Single-DOF Miura deployable: one parameter unfolds the whole array",
             ha="center", fontsize=12, weight="bold")

    # (b) footprint area vs t
    ts = np.linspace(0, 1, 60)
    areas = [dep.footprint_area(t) for t in ts]
    axb = fig.add_subplot(2, 3, 4)
    axb.plot(ts, areas, color="#E45756", lw=2)
    axb.fill_between(ts, areas, alpha=0.15, color="#E45756")
    axb.set_xlabel("deployment parameter t"); axb.set_ylabel("footprint area (x-y)")
    axb.set_title("Compaction: footprint grows as it deploys", fontsize=9)
    axb.annotate(f"compaction ratio\ndeployed / stowed = {s['compaction_ratio']:.1f}x",
                 xy=(1.0, areas[-1]), xytext=(0.35, areas[-1] * 0.72),
                 fontsize=10, weight="bold",
                 arrowprops=dict(arrowstyle="->", color="#333"))
    axb.scatter([0, 1], [areas[0], areas[-1]], color="#333", zorder=5)

    # (c) rigid-foldability: edge deviation across sweep
    L0 = dep.flat_edge_lengths()
    devs = [float(np.max(np.abs(dep.edge_lengths(t) - L0))) for t in ts]
    axc = fig.add_subplot(2, 3, 5)
    axc.plot(ts, np.maximum(devs, 1e-18), color="#54A24B", lw=2)
    axc.set_yscale("log"); axc.set_ylim(1e-18, 1e-6)
    axc.set_xlabel("deployment parameter t")
    axc.set_ylabel("max |edge - flat edge|")
    axc.set_title(f"Rigid-foldable: max edge deviation = {s['max_edge_deviation']:.1e}\n"
                  "(panels stay rigid - isometric motion)", fontsize=9)
    axc.axhline(1e-9, ls="--", color="#888", lw=1)

    # (d) inverse design to a target deployed footprint
    target = (12.0, 9.0)
    dd, res, achieved = inverse_design_deployed(target, iters=400, lr=0.05)
    axd = fig.add_subplot(2, 3, 6)
    tw, td = target; aw, ad = achieved
    axd.add_patch(plt.Rectangle((0, 0), tw, td, fill=False, edgecolor="#333",
                                lw=2, ls="--", label="target"))
    axd.add_patch(plt.Rectangle((0, 0), aw, ad, fill=True, facecolor="#4C78A8",
                                alpha=0.35, edgecolor="#1b2a3a", label="achieved"))
    axd.set_xlim(-0.5, tw + 1.5); axd.set_ylim(-0.5, td + 1.5)
    axd.set_aspect("equal"); axd.legend(loc="upper right", fontsize=8)
    axd.set_title("Inverse design: deployed footprint hits target\n"
                  f"target {tw:.0f}x{td:.0f}  ->  achieved {aw:.2f}x{ad:.2f}  "
                  f"(comp {dd.compaction_ratio():.1f}x)", fontsize=9)
    axd.set_xlabel("width"); axd.set_ylabel("depth")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(OUT, "deployable.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)
    print(f"compaction ratio      : {s['compaction_ratio']:.2f}x")
    print(f"max edge deviation    : {s['max_edge_deviation']:.2e}")
    print(f"deployed surface area : {s['deployed_surface_area']:.2f}")
    print(f"inverse design loss   : {res.history[0]:.3e} -> {res.final_loss:.3e}")
    print(f"target {target} -> achieved ({achieved[0]:.3f}, {achieved[1]:.3f})")


if __name__ == "__main__":
    main()
