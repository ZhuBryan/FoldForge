"""Proof image for the sheet-metal bending-sequence planner.

Writes examples/output/bending_plan.png for the ``coil`` part - a strip with
three parallel bends where the bend ORDER decides whether a press brake can
reach each bend. Four panels:

    1. the flat blank with its three bend lines (L1, L2, L3);
    2. the fully formed part (a rigid fold);
    3. the NAIVE order (L1, L2, L3): after the inner bend L1 is made, the outer
       bend L2's line has been lifted off the die - the tool can no longer seat
       it. The lifted line and blocked tool are drawn in red, with the total
       collision count;
    4. the PLANNED order (L3, L2, L1): outermost-first, every bend reachable,
       zero collisions, with the sequence listed.

    python examples/bending_planner_demo.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from foldforge.fabricate import bending as B

OUT = os.path.join(os.path.dirname(__file__), "output")


def _draw_panels(ax, part, pos, color="#8fb3d9", alpha=0.9):
    polys = [pos[tri] for tri in part.triangles]
    pc = Poly3DCollection(polys, facecolor=color, edgecolor="#2c3e50",
                          linewidths=0.6, alpha=alpha)
    ax.add_collection3d(pc)


def _draw_bend_lines(ax, part, pos, highlight=None):
    for i, b in enumerate(part.bends):
        q0, q1 = pos[b.axis[0]], pos[b.axis[1]]
        c = "#d94f4f" if (highlight is not None and i == highlight) else "#f39c12"
        ax.plot(*zip(q0, q1), color=c, lw=4, zorder=10)
        mid = (q0 + q1) / 2
        ax.text(mid[0], mid[1], mid[2] + 0.15, b.name, color=c,
                fontsize=11, fontweight="bold", ha="center")


def _draw_tool(ax, part, bend_idx, pos, color):
    b = part.bends[bend_idx]
    q0, q1 = pos[b.axis[0]], pos[b.axis[1]]
    size = float(np.linalg.norm(part.vertices.max(0) - part.vertices.min(0)))
    slab = B._tool_slab(q0, q1, 0.9 * size, 0.04 * size, 0.01 * size, inset=0.03)
    pc = Poly3DCollection(slab, facecolor=color, edgecolor=color,
                          linewidths=0.4, alpha=0.25)
    ax.add_collection3d(pc)


def _style(ax, title):
    ax.set_title(title, fontsize=11)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 2.5); ax.set_zlim(0, 3)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-62)


def main():
    os.makedirs(OUT, exist_ok=True)
    part = B.coil()
    naive = B.naive_plan(part)
    plan = B.plan_bend_order(part)
    allb = set(range(len(part.bends)))

    fig = plt.figure(figsize=(13, 10))
    fig.suptitle("Sheet-metal bending-sequence planner - part: coil "
                 "(three parallel bends)", fontsize=14, fontweight="bold")

    # 1. flat blank with bend lines
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    _draw_panels(ax, part, part.vertices, color="#dfe6ee")
    _draw_bend_lines(ax, part, part.vertices)
    _style(ax, "1. Flat blank + bend lines")

    # 2. formed part
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    _draw_panels(ax, part, part.fold_state(allb))
    _style(ax, "2. Formed part (rigid fold, collision-free)")

    # 3. naive order: after L1, L2's line is lifted -> tool blocked
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    after_L1 = part.fold_state({0})
    _draw_panels(ax, part, after_L1, color="#c6d7e8")
    _draw_bend_lines(ax, part, after_L1, highlight=1)      # L2 lifted, red
    _draw_tool(ax, part, 1, after_L1, color="#d94f4f")     # blocked tool
    _style(ax, f"3. NAIVE {naive.named_order()}: step 2 blocked\n"
               f"L2 lifted off the die - total interference "
               f"= {naive.total_interference}")

    # 4. planned order: outermost first, clean
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    _draw_panels(ax, part, part.fold_state(allb))
    _draw_tool(ax, part, 2, part.vertices, color="#27ae60")  # L3 reached flat
    _style(ax, f"4. PLANNED {plan.named_order()}: 0 collisions\n"
               f"every bend reachable (outermost-first)")
    seq = "  ->  ".join(plan.named_order())
    ax.text2D(0.5, -0.08, f"planned sequence:  {seq}", transform=ax.transAxes,
              ha="center", fontsize=11, color="#27ae60", fontweight="bold")

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    path = os.path.join(OUT, "bending_plan.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)
    print(f"naive order {naive.named_order()}: interference "
          f"{naive.interference}, feasible={naive.feasible}")
    print(f"planned order {plan.named_order()}: interference "
          f"{plan.interference}, feasible={plan.feasible}")


if __name__ == "__main__":
    main()
