"""Printable step-by-step folding instructions from a crease pattern.

Turn a crease pattern (a FOLD file, or any :class:`CreasePattern` the engines
produce) into a one-page instruction sheet: the flat sheet drawn once per step,
with the creases folded so far highlighted in their mountain/valley colours and
a one-line caption per step ("Valley-fold along crease 3"). Reuses the same
red = mountain / blue = valley convention as :mod:`foldforge.fabricate.export`.

Ordering: the distinct mountain/valley fold lines are merged (collinear M/V
edges on one infinite line are a single fold, exactly as
:func:`foldforge.origamize.surface.crease_stats` counts them) and then folded in
a deterministic **left-to-right (or bottom-to-top) sweep** across the sheet -
the natural order a corrugation's parallel pleats fold in. This keeps a coarse,
hand-foldable pattern to a dozen clear steps without needing the collision-aware
:func:`foldforge.sim.sequencing.fold_sequence` solver (overkill for pleats).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from foldforge.geometry.crease_graph import CreasePattern

# Mountain/valley stroke colours (match the fabrication SVG layers).
_M_COLOR = "#d62728"        # mountain -> red
_V_COLOR = "#1f5fd6"        # valley   -> blue
_GHOST = "#c9c9c9"          # not-yet-folded creases (light grey)
_SHEET = "#333333"          # sheet outline


def _fold_lines(pattern: CreasePattern):
    """Distinct mountain/valley fold lines, in deterministic sweep order.

    Collinear M/V edges lying on one infinite line are merged into a single fold
    (an extruded pleat is one crease across the sheet). Each returned line is a
    dict with its ``kind`` ("M"/"V"), the flat ``segments`` [(p, q), ...] to draw,
    and a centroid. Lines are sorted along whichever axis they spread across most
    (parallel pleats then read left-to-right / bottom-to-top).
    """
    V = np.asarray(pattern.vertices, dtype=float)[:, :2]
    groups: dict = {}
    for (a, b), k in zip(np.asarray(pattern.edges), pattern.assignment):
        if k not in ("M", "V"):
            continue
        p, q = V[int(a)], V[int(b)]
        d = q - p
        n = float(np.hypot(d[0], d[1]))
        if n < 1e-9:
            continue
        theta = np.arctan2(d[1], d[0]) % np.pi          # line direction mod pi
        nx, ny = -np.sin(theta), np.cos(theta)          # unit normal for offset
        offset = nx * p[0] + ny * p[1]
        key = (k, round(theta, 2), round(offset, 1))
        groups.setdefault(key, {"kind": k, "segments": []})
        groups[key]["segments"].append((p, q))

    lines = []
    for g in groups.values():
        pts = np.array([pt for seg in g["segments"] for pt in seg])
        c = pts.mean(0)
        lines.append({"kind": g["kind"], "segments": g["segments"],
                      "cx": float(c[0]), "cy": float(c[1])})
    if lines:
        cx = np.array([l["cx"] for l in lines])
        cy = np.array([l["cy"] for l in lines])
        if cx.var() >= cy.var():                        # spread mostly in x -> sweep x
            lines.sort(key=lambda l: (l["cx"], l["cy"]))
        else:
            lines.sort(key=lambda l: (l["cy"], l["cx"]))
    return lines


def _panel_svg(pattern, lines, upto, ox, oy, size, scale, lo, height):
    """One step diagram: sheet outline + creases, folds 1..upto highlighted."""
    V = np.asarray(pattern.vertices, dtype=float)[:, :2]

    def tx(p):                                          # flat coords -> panel px
        x = ox + (p[0] - lo[0]) * scale
        y = oy + (height - (p[1] - lo[1]) * scale)      # SVG y points down
        return x, y

    out = []
    # sheet outline (bounding box of the pattern)
    span = (V.max(0) - lo) * scale
    out.append(f'<rect x="{ox:.1f}" y="{oy + height - span[1]:.1f}" '
               f'width="{span[0]:.1f}" height="{span[1]:.1f}" fill="#fafafa" '
               f'stroke="{_SHEET}" stroke-width="1.2"/>')
    # every crease as a faint ghost, so the full pattern is always visible
    for l in lines:
        for p, q in l["segments"]:
            x1, y1 = tx(p); x2, y2 = tx(q)
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                       f'y2="{y2:.1f}" stroke="{_GHOST}" stroke-width="1"/>')
    # creases folded so far, in their M/V colour (current step emphasised)
    for i, l in enumerate(lines[:upto]):
        colour = _M_COLOR if l["kind"] == "M" else _V_COLOR
        w = 3.0 if i == upto - 1 else 1.8
        dash = '' if l["kind"] == "M" else ' stroke-dasharray="4,3"'
        for p, q in l["segments"]:
            x1, y1 = tx(p); x2, y2 = tx(q)
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                       f'y2="{y2:.1f}" stroke="{colour}" stroke-width="{w}"{dash}/>')
    return "\n".join(out)


def fold_instructions_svg(pattern: CreasePattern, path: str | Path, *,
                          cols: int = 4, panel: float = 150.0,
                          title: str = "Folding instructions") -> int:
    """Write a one-page step-by-step folding instruction sheet as SVG.

    ``pattern`` is a crease pattern (read a FOLD file with
    :func:`foldforge.geometry.fold_io.read_fold`). The sheet has a title, a
    mountain/valley legend, an overview of the full crease pattern, then one
    numbered panel per fold: panel ``k`` shows the flat sheet with creases
    ``1..k`` highlighted (crease ``k`` emphasised) and a caption
    ("Valley-fold along crease k"). Returns the number of fold steps.

    ``cols`` panels per row, ``panel`` the panel size in px. Colours match the
    fabrication SVG (red = mountain, blue = valley; valley folds are dashed).
    """
    lines = _fold_lines(pattern)
    n = len(lines)
    if n == 0:
        raise ValueError("pattern has no mountain/valley creases to fold")

    V = np.asarray(pattern.vertices, dtype=float)[:, :2]
    lo = V.min(0)
    span = (V.max(0) - lo)
    margin = 0.12 * panel
    draw = panel - 2 * margin
    scale = draw / max(span.max(), 1e-9)
    dh = span[1] * scale                                # drawn sheet height (px)

    cap_h = 34.0                                        # caption strip per panel
    pad = 16.0
    cell_w = panel + pad
    cell_h = panel + cap_h + pad
    head_h = 78.0

    # overview panel (all creases) is step 0; then one panel per fold.
    n_cells = n + 1
    rows = (n_cells + cols - 1) // cols
    W = cols * cell_w + pad
    H = head_h + rows * cell_h + pad

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" '
           f'height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}" '
           f'font-family="Helvetica,Arial,sans-serif">']
    svg.append(f'<rect width="{W:.0f}" height="{H:.0f}" fill="white"/>')
    n_m = sum(1 for l in lines if l["kind"] == "M")
    n_v = n - n_m
    svg.append(f'<text x="{pad}" y="30" font-size="20" font-weight="bold">'
               f'{title}</text>')
    svg.append(f'<text x="{pad}" y="52" font-size="12" fill="#555">'
               f'{n} folds ({n_m} mountain, {n_v} valley) - '
               f'fold in numbered order</text>')
    # legend
    lx = pad
    svg.append(f'<line x1="{lx}" y1="66" x2="{lx + 26}" y2="66" '
               f'stroke="{_M_COLOR}" stroke-width="3"/>')
    svg.append(f'<text x="{lx + 32}" y="70" font-size="12">Mountain (fold away)'
               f'</text>')
    svg.append(f'<line x1="{lx + 170}" y1="66" x2="{lx + 196}" y2="66" '
               f'stroke="{_V_COLOR}" stroke-width="3" stroke-dasharray="4,3"/>')
    svg.append(f'<text x="{lx + 202}" y="70" font-size="12">Valley (fold toward '
               f'you)</text>')

    def cell_origin(cell):
        r, c = divmod(cell, cols)
        return pad + c * cell_w, head_h + r * cell_h

    def caption(cell, text, sub=False):
        ox, oy = cell_origin(cell)
        ty = oy + panel + 20
        colour = "#111" if not sub else "#111"
        svg.append(f'<text x="{ox + panel / 2:.1f}" y="{ty:.1f}" '
                   f'font-size="12" text-anchor="middle" fill="{colour}">'
                   f'{text}</text>')

    # step 0: the full crease pattern
    ox, oy = cell_origin(0)
    svg.append(_panel_svg(pattern, lines, n, ox, oy, panel, scale, lo, dh))
    caption(0, "Crease pattern (all folds)")

    for k in range(1, n + 1):
        ox, oy = cell_origin(k)
        svg.append(_panel_svg(pattern, lines, k, ox, oy, panel, scale, lo, dh))
        kind = lines[k - 1]["kind"]
        verb = "Mountain-fold" if kind == "M" else "Valley-fold"
        caption(k, f"{k}. {verb} crease {k}")

    svg.append("</svg>")
    Path(path).write_text("\n".join(svg), encoding="utf-8")
    return n
