#!/usr/bin/env python3
"""Inverse design v2: fit an arbitrary target with foldability in the loss.

Fits a crease mesh (flat pattern Q + folded surface P, optimised jointly) to a
bumpy, asymmetric, NON-Miura height field, twice: once with the Kawasaki
flat-foldability penalty off (w_kawasaki = 0) and once with it on. The figure
shows the target, both fitted surfaces, the loss curves, and a table of the
fidelity-vs-foldability tradeoff - the penalty lowers the Kawasaki residual
(more flat-foldable) at a small, quantified fidelity cost.

Run:  PYTHONPATH=. python examples/inverse_design_v2.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foldforge.diff.inverse import inverse_design_shape

OUT = os.path.join(os.path.dirname(__file__), "output")


def bumpy_target(R, C):
    """Seeded asymmetric non-developable height field (two offset bumps + tilt)."""
    xs = np.linspace(-1, 1, C); ys = np.linspace(-1, 1, R)
    X, Y = np.meshgrid(xs, ys)
    return (np.exp(-((X - 0.3) ** 2 + (Y + 0.2) ** 2) * 3)
            + 0.6 * np.exp(-((X + 0.5) ** 2 + (Y - 0.4) ** 2) * 6) + 0.2 * X)


def _surf(ax, P, R, C, title, cmap):
    G = P.reshape(R, C, 3)
    ax.plot_surface(G[..., 0], G[..., 1], G[..., 2], cmap=cmap,
                    edgecolor="k", linewidth=0.2, alpha=0.95)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def main():
    os.makedirs(OUT, exist_ok=True)
    R, C = 11, 11
    Z = bumpy_target(R, C)

    off = inverse_design_shape(Z, (R, C), w_fit=1.0, w_iso=4.0,
                               w_kawasaki=0.0, iters=700, lr=0.02)
    on = inverse_design_shape(Z, (R, C), w_fit=1.0, w_iso=4.0,
                              w_kawasaki=0.6, iters=700, lr=0.02)

    print("inverse design v2 - fidelity vs foldability tradeoff")
    print(f"{'':16}{'fidelity(RMSE)':>16}{'kawasaki(rad)':>16}{'max strain':>12}")
    for name, r in (("w_kawasaki=0", off), ("w_kawasaki=0.6", on)):
        print(f"{name:16}{r.fidelity:16.4f}{r.kawasaki_residual:16.4f}"
              f"{r.max_strain:12.4f}")
    d_fid = (on.fidelity - off.fidelity) / max(off.fidelity, 1e-9) * 100
    d_kaw = (1 - on.kawasaki_residual / max(off.kawasaki_residual, 1e-9)) * 100
    print(f"penalty effect: Kawasaki residual -{d_kaw:.0f}%  "
          f"for +{d_fid:.0f}% fidelity RMSE")

    fig = plt.figure(figsize=(13, 8))
    fig.suptitle("Inverse design v2: arbitrary target + flat-foldability penalty",
                 fontsize=13, fontweight="bold")

    axt = fig.add_subplot(2, 3, 1, projection="3d")
    xs = np.linspace(0, C - 1, C); ys = np.linspace(0, R - 1, R)
    X, Y = np.meshgrid(xs, ys)
    axt.plot_surface(X, Y, Z / np.ptp(Z) * 2.5, cmap="viridis",
                     edgecolor="k", linewidth=0.2)
    axt.set_title("target (non-Miura height field)", fontsize=10)
    axt.set_xticks([]); axt.set_yticks([]); axt.set_zticks([])

    _surf(fig.add_subplot(2, 3, 2, projection="3d"), off.P, R, C,
          f"fit, penalty OFF\nRMSE={off.fidelity:.3f}  Kawasaki={off.kawasaki_residual:.3f} rad",
          "plasma")
    _surf(fig.add_subplot(2, 3, 3, projection="3d"), on.P, R, C,
          f"fit, penalty ON\nRMSE={on.fidelity:.3f}  Kawasaki={on.kawasaki_residual:.3f} rad",
          "plasma")

    # loss curves
    axl = fig.add_subplot(2, 3, 4)
    axl.semilogy(off.history, label="total loss (penalty off)", color="#d1495b")
    axl.semilogy(on.history, label="total loss (penalty on)", color="#30638e")
    axl.semilogy(off.fid_history, "--", color="#d1495b", alpha=0.5,
                 label="fidelity term (off)")
    axl.set_xlabel("iteration"); axl.set_ylabel("loss (log)")
    axl.set_title("convergence", fontsize=10); axl.legend(fontsize=7)

    # flat patterns Q (foldability is a property of these)
    for col, (name, r) in enumerate((("Q, penalty off", off),
                                      ("Q, penalty on", on))):
        axq = fig.add_subplot(2, 6, 11 + col)
        Q = r.Q.reshape(R, C, 3)
        for j in range(R):
            axq.plot(Q[j, :, 0], Q[j, :, 1], "-", color="#888", lw=0.5)
        for i in range(C):
            axq.plot(Q[:, i, 0], Q[:, i, 1], "-", color="#888", lw=0.5)
        axq.set_title(f"{name}\nKawasaki {r.kawasaki_residual:.3f} rad", fontsize=8)
        axq.set_aspect("equal"); axq.set_xticks([]); axq.set_yticks([])

    # tradeoff table
    axtab = fig.add_subplot(2, 3, 5); axtab.axis("off")
    rows = [["", "fidelity\n(RMSE)", "Kawasaki\n(rad)", "max\nstrain"],
            ["penalty off", f"{off.fidelity:.4f}", f"{off.kawasaki_residual:.4f}",
             f"{off.max_strain:.3f}"],
            ["penalty on", f"{on.fidelity:.4f}", f"{on.kawasaki_residual:.4f}",
             f"{on.max_strain:.3f}"]]
    tab = axtab.table(cellText=rows, loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.8)
    axtab.set_title(f"tradeoff: Kawasaki residual -{d_kaw:.0f}%  "
                    f"(fidelity +{d_fid:.0f}%)", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUT, "inverse_design_v2.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    main()
