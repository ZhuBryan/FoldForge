"""v3 amortized (learned) inverse design - the instant-generalisation demo.

Trains one small numpy MLP that maps a target Miura footprint descriptor
``(w, d, h)`` straight to fold parameters ``(a, b, gamma, h)``, self-supervised
through the differentiable fold forward (no ground-truth params in the loss).
Then, on HELD-OUT targets, it predicts params in microseconds and we fold them
back to measure reconstruction error - versus the per-shape optimiser
(``miura.fit_miura``), which must run a fresh gradient-descent loop per target.

Produces ``examples/output/inverse_design_v3.png``:
  (1) training loss curve,
  (2) held-out reconstruction error per target (learned, folded-back),
  (3) a headline table: learned held-out error + inference time vs the
      per-shape optimiser's error + time, and the speedup.
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from foldforge.diff import learned as L
from foldforge.diff.miura import fit_miura

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def main():
    ds = L.make_dataset(n=600, rows=4, cols=4, test_frac=0.2, seed=0)

    t0 = time.perf_counter()
    mlp, history = L.train(ds, hidden=32, epochs=150, batch=64, lr=5e-3)
    train_time = time.perf_counter() - t0

    per_tr, m_tr = L.reconstruction_error(mlp, ds.X_train, ds)
    per_te, m_te = L.reconstruction_error(mlp, ds.X_test, ds)

    # learned inference time per held-out target (amortized: one forward pass)
    reps = 200
    t0 = time.perf_counter()
    for _ in range(reps):
        L.predict_params(mlp, ds.X_test, ds)
    net_time = (time.perf_counter() - t0) / reps / len(ds.X_test)

    # per-shape optimiser on a held-out subset: its error + wall time per target
    sub = ds.X_test[:20]
    opt_errs = []
    t0 = time.perf_counter()
    for tgt in sub:
        _, _, ach = fit_miura(tgt, rows=4, cols=4, iters=400)
        opt_errs.append(np.sqrt(np.mean((np.array(ach) - tgt) ** 2)))
    opt_time = (time.perf_counter() - t0) / len(sub)
    opt_err = float(np.mean(opt_errs))
    speedup = opt_time / net_time

    print("train time       %.2fs" % train_time)
    print("loss  %.3f -> %.4f" % (history[0], history[-1]))
    print("recon RMS  train %.4f  test %.4f  (gap %.4f)"
          % (m_tr, m_te, m_te - m_tr))
    print("net inference    %.3e s/target" % net_time)
    print("optimiser        %.3e s/target  recon RMS %.4f" % (opt_time, opt_err))
    print("speedup          %.0fx" % speedup)

    # ---- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(14, 4.6))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(history, lw=1.8, color="#1f77b4")
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("self-supervised loss (norm. MSE)")
    ax1.set_title("Training (through the fold sim)")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(per_tr, bins=25, alpha=0.6, label="train (n=%d)" % len(per_tr),
             color="#2ca02c", density=True)
    ax2.hist(per_te, bins=25, alpha=0.6, label="held-out (n=%d)" % len(per_te),
             color="#d62728", density=True)
    ax2.axvline(m_te, color="#d62728", ls="--", lw=1)
    ax2.set_xlabel("folded-back reconstruction RMS")
    ax2.set_ylabel("density")
    ax2.set_title("Instant generalisation (held-out)")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(1, 3, 3); ax3.axis("off")
    rows = [
        ["", "recon RMS", "time / target"],
        ["learned (v3)", "%.4f" % m_te, "%.1e s" % net_time],
        ["optimiser (v1/v2)", "%.4f" % opt_err, "%.1e s" % opt_time],
        ["", "", "%.0fx faster" % speedup],
    ]
    tbl = ax3.table(cellText=rows, cellLoc="center", loc="center",
                    colWidths=[0.42, 0.29, 0.29])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
    for j in range(3):
        tbl[(0, j)].set_facecolor("#eeeeee")
        tbl[(0, j)].set_text_props(weight="bold")
    ax3.set_title("Held-out: learned vs per-shape optimiser", pad=18)
    ax3.text(0.5, 0.08,
             "trained once (%.1fs); then ~%.0fx faster per query.\n"
             "optimiser is near-exact on reachable targets but pays per shape."
             % (train_time, speedup),
             ha="center", va="center", transform=ax3.transAxes, fontsize=8.5,
             color="#555555")

    fig.tight_layout()
    path = os.path.join(OUT, "inverse_design_v3.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    main()
