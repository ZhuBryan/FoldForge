"""Auto-rect: locate an off-centre subject without a hand-tuned GrabCut box.

The old default seeded GrabCut with a fixed centre box, so an off-centre subject
was missed. :func:`foldforge.origamize.vision._auto_rect` seeds the box from an
edge-density saliency map instead. Skipped if OpenCV / SciPy are unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")


def _offcentre_textured_subject():
    """A high-texture (striped) subject in the lower-right on a smooth background."""
    import cv2
    H, W = 200, 280
    yy = np.linspace(60, 150, H)[:, None] * np.ones((1, W))
    bg = np.clip(yy, 0, 255).astype("uint8")                 # smooth ramp: few edges
    img = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = int(W * 0.58), int(H * 0.55), int(W * 0.92), int(H * 0.92)
    for x in range(x0, x1):                                   # vertical stripes = strong edges
        img[y0:y1, x] = (30, 30, 30) if ((x // 6) % 2 == 0) else (235, 235, 235)
    return img, (x0, y0, x1, y1)


def test_auto_rect_boxes_the_offcentre_subject():
    from foldforge.origamize.vision import _auto_rect
    img, (x0, y0, x1, y1) = _offcentre_textured_subject()
    rx, ry, rw, rh = _auto_rect(img)
    cx, cy = rx + rw / 2, ry + rh / 2
    sx, sy = (x0 + x1) / 2, (y0 + y1) / 2
    H, W = img.shape[:2]
    # the seeded box is centred on the subject, not on the frame centre
    assert abs(cx - sx) < 0.18 * W and abs(cy - sy) < 0.18 * H
    assert cx > 0.5 * W and cy > 0.5 * H                      # lower-right, where the subject is


def _full_width_grain_band():
    """A full-width textured band across the top (think wood grain) over a smooth
    field: the saliency map lights up edge to edge, so the largest edge-density
    blob is the *background* band, not the subject."""
    import cv2
    H, W = 220, 320
    img = np.full((H, W, 3), 150, np.uint8)                  # smooth field
    for y in range(0, int(H * 0.45)):                        # horizontal grain, full width
        img[y, :] = (60, 60, 60) if ((y // 3) % 2 == 0) else (210, 210, 210)
    return img


def test_auto_rect_rejects_full_frame_band():
    """A saliency blob spanning the full frame width has bled into a textured
    background; the seed must fall back to the centred inset box, not return the
    edge-to-edge band. Regression for the starfish-on-wood-grain under-count."""
    from foldforge.origamize.vision import _auto_rect
    img = _full_width_grain_band()
    H, W = img.shape[:2]
    rx, ry, rw, rh = _auto_rect(img)
    assert rw < 0.98 * W and rh < 0.98 * H                   # not a full-frame band
    cx, cy = rx + rw / 2, ry + rh / 2                        # centred fallback box
    assert abs(cx - 0.5 * W) < 0.1 * W and abs(cy - 0.5 * H) < 0.1 * H


def test_auto_mask_captures_offcentre_subject():
    from foldforge.origamize.vision import silhouette_mask
    img, (x0, y0, x1, y1) = _offcentre_textured_subject()
    m = silhouette_mask(img)                                  # rect=None -> auto
    assert 0.02 < m.mean() < 0.6
    inside = m[y0:y1, x0:x1].mean()
    outside = (m.sum() - m[y0:y1, x0:x1].sum()) / max(1, m.size - (x1 - x0) * (y1 - y0))
    assert inside > 0.5 and inside > 3 * outside              # foreground sits on the subject
