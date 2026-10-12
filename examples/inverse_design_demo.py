"""Proof figure for M3 differentiable inverse design.

Produces ``examples/output/inverse_design.png`` with three panels:

  (a) a target folded chain (made by unknown angles) with the recovered chain
      overlaid - visually identical because recovery is exact;
  (b) the loss-vs-iteration curve for the recovery, log scale;
  (c) a Miura inverse-fit: recovering the four design parameters from a folded
      footprint, with the loss curve and the recovered-vs-true error annotated.

Run:  PYTHONPATH=. python examples/inverse_design_demo.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foldforge.diff.kinematics import fold_chain
from foldforge.diff.miura import footprint
from foldforge.diff.inverse import recover_fold_angles, inverse_fit

OUT = "examples/output"


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- 1. Fold-angle recovery -----------------------------------------
    rng = np.random.default_rng(42)
    true_angles = rng.uniform(-0.6, 0.6, 5)
    target_spine = fold_chain(true_angles, seg=1.0).spine
    rec = recover_fold_angles(target_spine, seg=1.0, iters=3000, lr=0.05, seed=0)
    rec_spine = fold_chain(rec.params, seg=1.0).spine
    angle_err = np.max(np.abs(rec.params - true_angles))

    # ---- 2. Miura parameter recovery from a folded footprint ------------
    true_miura = (1.2, 0.9, np.radians(55), 0.5)      # a, b, gamma, h
    target_fp = footprint(6, 6, *true_miura)
    mres, achieved = inverse_fit(target_fp, iters=400, lr=0.05)
    true_p = np.array(true_miura)
    # gamma compared in degrees for a readable error
    perr = np.abs(mres.params - true_p)
    perr_readable = perr.copy(); perr_readable[2] = np.degrees(perr[2])

    # ---- figure ---------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) target vs recovered folded chain
    ax[0].plot(target_spine[:, 0], target_spine[:, 1], "-o", color="#8e8e93",
               lw=7, ms=11, alpha=0.5, label="target (unknown angles)")
    ax[0].plot(rec_spine[:, 0], rec_spine[:, 1], "--x", color="#ff3b30",
               lw=1.8, ms=9, mew=2, label="recovered")
    ax[0].set_title("(a) Fold-chain recovery\nmax angle error = %.1e rad"
                    % angle_err)
    ax[0].set_aspect("equal"); ax[0].legend(loc="best", fontsize=9)
    ax[0].set_xlabel("x"); ax[0].set_ylabel("z")
    ax[0].grid(alpha=0.25)

    # (b) recovery loss curve
    ax[1].semilogy(rec.history, color="#007aff", lw=2)
    ax[1].set_title("(b) Recovery loss vs iteration\n"
                    "sum of squared point distances")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("loss (log)")
    ax[1].grid(alpha=0.25, which="both")

    # (c) Miura fit loss + parameter table
    ax[2].semilogy(mres.history, color="#34c759", lw=2)
    ax[2].set_title("(c) Miura param recovery from footprint\n"
                    "loss %.2e -> %.2e" % (mres.history[0], mres.final_loss))
    ax[2].set_xlabel("iteration"); ax[2].set_ylabel("footprint loss (log)")
    ax[2].grid(alpha=0.25, which="both")
    names = ["a", "b", "gamma(deg)", "h"]
    tp = [true_miura[0], true_miura[1], np.degrees(true_miura[2]), true_miura[3]]
    rp = [mres.params[0], mres.params[1], np.degrees(mres.params[2]),
          mres.params[3]]
    lines = ["param    true    recovered   |err|"]
    for nm, t, r, e in zip(names, tp, rp, perr_readable):
        lines.append("%-9s %6.3f  %8.3f  %8.1e" % (nm, t, r, e))
    ax[2].text(0.5, 0.60, "\n".join(lines), transform=ax[2].transAxes,
               family="monospace", fontsize=9, va="top", ha="center",
               bbox=dict(boxstyle="round", fc="#f2f2f7", ec="#c7c7cc"))

    fig.suptitle("FoldForge - differentiable inverse design (M3): "
                 "recover crease/fold parameters from a folded shape",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, "inverse_design.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("wrote", path)
    print("fold-angle max error: %.2e rad" % angle_err)
    print("miura footprint loss: %.3e -> %.3e" % (mres.history[0],
                                                   mres.final_loss))
    print("miura param |err| (a,b,gamma_deg,h):",
          np.array2string(perr_readable, precision=2))


if __name__ == "__main__":
    main()
