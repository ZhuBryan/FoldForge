"""Tests for the step-by-step folding-instructions generator and rect-sheet export."""

import xml.dom.minidom as minidom

import numpy as np

from foldforge.geometry import examples
from foldforge.fabricate import fold_instructions_svg
from foldforge.fabricate.instructions import _fold_lines


def test_instructions_valid_svg_with_steps(tmp_path):
    pat = examples.miura()
    p = tmp_path / "steps.svg"
    n = fold_instructions_svg(pat, p)
    assert n >= 2                                         # at least two fold steps
    doc = minidom.parse(str(p))                           # raises if malformed XML
    assert doc.getElementsByTagName("svg")                # a real SVG root
    text = p.read_text()
    assert "Folding instructions" in text                 # titled sheet
    assert "Mountain" in text and "Valley" in text        # M/V legend present


def test_step_count_matches_distinct_fold_lines(tmp_path):
    pat = examples.miura()
    n = fold_instructions_svg(pat, tmp_path / "s.svg")
    assert n == len(_fold_lines(pat))                     # one step per distinct fold


def test_empty_pattern_raises(tmp_path):
    # single_vertex has creases; strip them to an all-border pattern instead.
    pat = examples.single_vertex()
    pat.assignment = ["B"] * len(pat.assignment)
    try:
        fold_instructions_svg(pat, tmp_path / "e.svg")
    except ValueError:
        return
    raise AssertionError("expected ValueError for a pattern with no M/V creases")


def _fill_ratio(v, f, nx=120, ny=120):
    """Fraction of the XY bounding box covered by the projected mesh footprint."""
    v = np.asarray(v)[:, :2]
    f = np.asarray(f)
    lo, hi = v.min(0), v.max(0)
    gx = np.linspace(lo[0], hi[0], nx)
    gy = np.linspace(lo[1], hi[1], ny)
    GX, GY = np.meshgrid(gx, gy)
    P = np.stack([GX.ravel(), GY.ravel()], 1)
    inside = np.zeros(len(P), bool)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    for i in range(len(f)):
        A, B, C = a[i], b[i], c[i]
        d = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        if abs(d) < 1e-12:
            continue
        s = ((B[1] - C[1]) * (P[:, 0] - C[0]) + (C[0] - B[0]) * (P[:, 1] - C[1])) / d
        t = ((C[1] - A[1]) * (P[:, 0] - C[0]) + (A[0] - C[0]) * (P[:, 1] - C[1])) / d
        inside |= (s >= -1e-9) & (t >= -1e-9) & (1 - s - t >= -1e-9)
    return inside.mean()


def test_rect_sheet_fills_the_rectangle():
    """rect_sheet keeps the full rectangular sheet; trimmed hugs the silhouette."""
    import cv2
    from foldforge.origamize import origamize_silhouette
    # A synthetic image: a bright *disc* on a dark background. A round silhouette
    # leaves the sheet corners empty, so trimming visibly shrinks the footprint
    # while rect_sheet keeps the whole rectangle.
    img = np.zeros((180, 220, 3), np.uint8)
    cv2.circle(img, (110, 90), 70, (230, 230, 230), -1)
    trimmed, _ = origamize_silhouette(img, folds=None, closed=True,
                                      foldable="easy", rect_sheet=False)
    rect, _ = origamize_silhouette(img, folds=None, closed=True,
                                   foldable="easy", rect_sheet=True)
    fr_rect = _fill_ratio(*rect.solid)
    fr_trim = _fill_ratio(*trimmed.solid)
    assert fr_rect > 0.95                                 # essentially a full rectangle
    assert fr_rect > fr_trim + 0.05                       # and fuller than the trimmed one
