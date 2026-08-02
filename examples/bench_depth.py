"""Benchmark monocular-depth backends: MiDaS vs Depth Anything V2.

Runs each depth model over the sample photos in ``examples/samples/`` and folds
the estimated relief, then reports a per-sample table of the same signals the
README already uses to compare depth backends:

* ``fold_err`` - the corrugation engine's fold error (``result.error``): how
  faithfully one crease pattern reproduces the relief. Lower is smoother; a
  *higher* value on a truthful relief just means it carries more real high-
  frequency detail than a single corrugation can absorb (see README).
* ``sharpness`` - mean gradient magnitude of the masked depth relief (Sobel).
  This is the "resolves depth edges N% sharper" proxy: a crisper depth map with
  better-preserved subject edges scores higher.
* ``sec`` - wall-clock seconds for ``estimate_depth`` (model load excluded via a
  warm-up call, so this is inference only).

Both backends normalise to the same 0..1 (1 = nearest) convention, so the
numbers are directly comparable. MiDaS_small needs only PyTorch; the Depth
Anything V2 models need ``transformers`` (``pip install transformers``) and a
one-time ~99 MB / ~371 MB download. Any model that can't load is skipped with a
note rather than faking a number.

Run from the repo root:  python examples/bench_depth.py
"""

import os
import time

import numpy as np

from foldforge.origamize import estimate_depth, origamize_depth
from foldforge.origamize.vision import silhouette_mask

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples")
OUT = os.path.join(HERE, "output")

# name -> GrabCut rect (only where a busy background defeats the auto-rect),
# mirroring examples/make_samples.py so the mask matches the gallery.
SAMPLE_RECTS = {
    "swallowtail": None,
    "moth": None,
    "ginkgo": None,
    "starfish": (120, 25, 835, 700),
    "fish": None,
    "butterfly": None,
}

MODELS = ["MiDaS_small", "depth_anything_v2_small", "depth_anything_v2_base"]


def _sharpness(depth: np.ndarray, mask: np.ndarray) -> float:
    """Mean Sobel gradient magnitude of the depth *inside* the subject mask."""
    import cv2
    m = mask > 0
    if not m.any():
        return float("nan")
    gx = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return float(np.hypot(gx, gy)[m].mean())


def _bench_model(model_type: str):
    """Return {name: (fold_err, sharpness, sec)} or None if the model can't load."""
    rows = {}
    warmed = False
    for name, rect in SAMPLE_RECTS.items():
        path = os.path.join(SAMPLES, name + ".jpg")
        if not os.path.exists(path):
            continue
        try:
            if not warmed:                 # first call also loads + caches weights
                estimate_depth(path, model_type=model_type)
                warmed = True
            t0 = time.perf_counter()
            depth = estimate_depth(path, model_type=model_type)
            sec = time.perf_counter() - t0
        except ImportError as exc:
            print(f"  skip {model_type}: {exc}")
            return None
        mask = silhouette_mask(path, rect=rect)
        res, _ = origamize_depth(path, folds=30, rect=rect, model_type=model_type)
        rows[name] = (res.error, _sharpness(depth, mask), sec)
        print(f"  {model_type:24s} {name:11s} "
              f"err={res.error:.3f} sharp={rows[name][1]:.4f} {sec:.2f}s")
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for model_type in MODELS:
        print(f"Benchmarking {model_type} ...")
        rows = _bench_model(model_type)
        if rows:
            results[model_type] = rows

    if not results:
        print("No depth model could run - install torch (and transformers for "
              "Depth Anything). No table written.")
        return

    names = list(SAMPLE_RECTS)
    lines = ["# Depth backend benchmark: MiDaS vs Depth Anything V2", "",
             "Per-sample fold error (engine `result.error`), depth-edge sharpness "
             "(mean masked Sobel gradient), and inference seconds. Same 0..1 "
             "(1=near) convention for every backend, so columns are comparable.", ""]
    for metric_i, label in ((0, "Fold error (lower = smoother relief)"),
                            (1, "Depth-edge sharpness (higher = crisper detail)"),
                            (2, "Inference seconds")):
        header = "| sample | " + " | ".join(results) + " |"
        sep = "|" + "---|" * (len(results) + 1)
        lines += [f"## {label}", "", header, sep]
        for name in names:
            cells = []
            for m in results:
                v = results[m].get(name)
                cells.append(f"{v[metric_i]:.4f}" if v else "-")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        # mean row
        means = []
        for m in results:
            vals = [results[m][n][metric_i] for n in names if n in results[m]]
            means.append(f"{np.mean(vals):.4f}" if vals else "-")
        lines += [f"| **mean** | " + " | ".join(means) + " |", ""]

    out = os.path.join(OUT, "depth_bench.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
