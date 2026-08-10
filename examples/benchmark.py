"""Standard fit-error benchmark: corrugation (1D) vs miura2d (2D) engines.

Runs a *fixed* set of targets through both fold engines and records the same
normalised mid-surface fit error the rest of the repo uses (see
:func:`foldforge.origamize.surface_fit_error`, via
:func:`foldforge.origamize.compare_engines`). It exists so every future change
stays measurable: re-run it and diff the table.

    python examples/benchmark.py            # synthetic + photo targets
    python examples/benchmark.py --no-photos  # synthetic only (no OpenCV needed)

Targets
-------
* **Synthetic** height fields the repo already ships (``heightfield_dome`` /
  ``heightfield_saddle`` / ``heightfield_ripple``). These are the shapes in the
  README's engine-comparison table and are reproduced here exactly (no OpenCV or
  torch needed).
* **Photo** silhouettes (butterfly, starfish, fish). To keep this runnable
  anywhere with **no torch/MiDaS**, the photo height field is the *silhouette
  inflation* (:func:`foldforge.origamize.vision.inflate`, a cv2 distance
  transform), not a depth estimate. So these rows are honest but *distinct* from
  the README's depth-based photo rows (zebra/cat/dog/elephant) - they measure the
  engines on an inflated-silhouette relief, downsampled to a small grid.

Outputs ``examples/output/benchmark.md`` (the table) and
``examples/output/benchmark.png`` (grouped bar chart). All numbers are computed,
never baked.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foldforge.origamize import compare_engines
from foldforge.origamize.surface import (
    heightfield_dome, heightfield_saddle, heightfield_ripple,
)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "output")
SAMPLES = os.path.join(HERE, "samples")

# README's known miura2d values on the shared synthetic shapes, for a sanity
# check printed alongside the fresh numbers (the table this benchmark subsumes).
README_MIURA = {"hemisphere": 0.017, "saddle": 0.001, "radial ripple": 0.023}


def synthetic_targets():
    """The three synthetic height fields (name -> Z), matching the README."""
    return [
        ("hemisphere", heightfield_dome()),
        ("saddle", heightfield_saddle()),
        ("radial ripple", heightfield_ripple()),
    ]


def photo_targets(max_cells: int = 32):
    """Silhouette-inflation height fields for the sample photos (needs OpenCV).

    Yields ``(name, Z)`` with ``Z`` a small (<= ``max_cells`` on the long side)
    height field so ``compare_engines`` stays CPU-fast. Skips silently (with a
    printed note by the caller) if OpenCV / the samples are unavailable.
    """
    import cv2
    from foldforge.origamize import silhouette_mask
    from foldforge.origamize.vision import inflate
    # (name, GrabCut rect override) - starfish's auto-seed latches onto wood grain
    cases = [("butterfly", None), ("starfish", (288, 144, 384, 432)),
             ("fish", None)]
    for name, rect in cases:
        path = os.path.join(SAMPLES, f"{name}.jpg")
        if not os.path.exists(path):
            continue
        mask = silhouette_mask(path, rect=rect)
        Z = inflate(mask)                         # 0..1 rounded relief, full res
        ys, xs = np.where(mask)
        Z = Z[ys.min():ys.max() + 1, xs.min():xs.max() + 1]   # crop to subject
        h, w = Z.shape
        s = max_cells / max(h, w)
        cols = max(3, int(round(w * s)))
        rows = max(3, int(round(h * s)))
        Zs = cv2.resize(Z, (cols, rows), interpolation=cv2.INTER_AREA)
        yield name, Zs


def run(with_photos: bool = True):
    rows = []
    targets = list(synthetic_targets())
    if with_photos:
        try:
            targets += list(photo_targets())
        except Exception as exc:                  # cv2 missing or samples gone
            print(f"(skipping photo targets: {exc})")
    for name, Z in targets:
        c = compare_engines(Z)
        corr = c["corrugation_error"]
        miura = c["miura2d_error"]
        ratio = corr / (miura + 1e-9)
        rows.append({
            "target": name, "corr": corr, "miura": miura, "ratio": ratio,
            "mean_strain": c["miura2d_mean_strain"],
            "max_strain": c["miura2d_max_strain"],
        })
        print(f"{name:14s} corr={corr:.3f}  miura2d={miura:.3f}  "
              f"ratio={ratio:5.1f}x  strain={c['miura2d_mean_strain']*100:.2f}%/"
              f"{c['miura2d_max_strain']*100:.1f}%")
    return rows


def write_markdown(rows, path):
    lines = [
        "# FoldForge engine benchmark",
        "",
        "Normalised mid-surface fit error (lower is better) of the two "
        "origamize engines on a fixed target set, via `compare_engines` "
        "(`surface_fit_error`, the same metric both engines are scored with). "
        "`ratio` = corrugation / miura2d. Regenerate with "
        "`python examples/benchmark.py`.",
        "",
        "| target | corrugation (1D) | miura2d (2D) | ratio | miura2d strain (mean / max) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target']} | {r['corr']:.3f} | **{r['miura']:.3f}** | "
            f"{r['ratio']:.1f}x | {r['mean_strain']*100:.2f}% / "
            f"{r['max_strain']*100:.1f}% |")
    lines += [
        "",
        "Synthetic rows (hemisphere / saddle / radial ripple) reproduce the "
        "README's engine-comparison table exactly. Photo rows use the no-torch "
        "silhouette-inflation relief (cv2 distance transform), downsampled to a "
        "small grid - honest, but distinct from the README's depth-based photo "
        "rows.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_chart(rows, path):
    names = [r["target"] for r in rows]
    corr = [r["corr"] for r in rows]
    miura = [r["miura"] for r in rows]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 4.5))
    ax.bar(x - w / 2, corr, w, label="corrugation (1D)", color="#c44e52")
    ax.bar(x + w / 2, miura, w, label="miura2d (2D)", color="#4c72b0")
    for xi, (cv, mv) in enumerate(zip(corr, miura)):
        ax.text(xi - w / 2, cv, f"{cv:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, mv, f"{mv:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("normalised mid-surface fit error (lower is better)")
    ax.set_title("Fold engine fit error: corrugation vs miura2d")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-photos", action="store_true",
                    help="synthetic targets only (no OpenCV needed)")
    args = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    print("FoldForge engine benchmark (corrugation vs miura2d):\n")
    rows = run(with_photos=not args.no_photos)

    # sanity check against the README's known miura2d values
    print("\nREADME cross-check (miura2d):")
    for r in rows:
        known = README_MIURA.get(r["target"])
        if known is not None:
            ok = "ok" if abs(r["miura"] - known) < 0.01 else "DIFFERS"
            print(f"  {r['target']:14s} got {r['miura']:.3f}  "
                  f"README {known:.3f}  [{ok}]")

    md = os.path.join(OUT, "benchmark.md")
    png = os.path.join(OUT, "benchmark.png")
    write_markdown(rows, md)
    write_chart(rows, png)
    print(f"\nwrote {md}\nwrote {png}")


if __name__ == "__main__":
    main()
