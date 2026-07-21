#!/usr/bin/env python3
"""Audit the six built-in studio shapes: print a per-shape final-vs-target table
and render a contact sheet of the final folded frames to
``examples/output/studio_shapes_audit.png`` so a human can eyeball them.

Run:  python examples/audit_shapes.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_studio as ms


def _set_equal(ax, V):
    span = (V.max(0) - V.min(0)).max() / 2.0
    mid = (V.max(0) + V.min(0)) / 2.0
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)


def main():
    data = ms.build_data()
    order = data["order"]

    fig = plt.figure(figsize=(13, 8))
    for k, name in enumerate(order):
        S = data["shapes"][name]
        final = np.array(S["frames"][-1])
        tris = np.array(S["triangles"])
        span = final.max(0) - final.min(0)
        zt = ratio = float("nan")
        rms = "   -  "
        if S["target"]:
            T = np.array(S["target"]).reshape(-1, 3)
            zt = float(T.max(0)[2] - T.min(0)[2])
            ratio = span[2] / max(zt, 1e-9)
            # correspondence-free: mean distance from each folded vertex to the
            # nearest target point (the target is stored as line segments, so its
            # points repeat - nearest-neighbour is the honest metric here)
            P = np.unique(T, axis=0)
            d = np.linalg.norm(final[:, None, :] - P[None, :, :], axis=2).min(1)
            rms = f"{d.mean():.4f}"
        print(f"{name:8s} x={span[0]:6.2f} y={span[1]:6.2f} z={span[2]:6.2f} "
              f"| tgt z={zt:6.2f} ratio={ratio:5.2f} rms={rms}")

        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        pc = Poly3DCollection(final[tris], facecolor="#d9b98a", edgecolor="#5b4a33",
                              linewidths=0.15, alpha=1.0)
        ax.add_collection3d(pc)
        _set_equal(ax, final)
        ax.set_title(f"{name}  (z-span {span[2]:.1f})", fontsize=10)
        ax.set_axis_off()
        ax.view_init(elev=28, azim=-58)

    fig.suptitle("FoldForge studio - final folded frame of each built-in shape", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output",
                       "studio_shapes_audit.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
