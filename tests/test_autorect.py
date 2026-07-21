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


def test_auto_mask_captures_offcentre_subject():
    from foldforge.origamize.vision import silhouette_mask
    img, (x0, y0, x1, y1) = _offcentre_textured_subject()
    m = silhouette_mask(img)                                  # rect=None -> auto
    assert 0.02 < m.mean() < 0.6
    inside = m[y0:y1, x0:x1].mean()
    outside = (m.sum() - m[y0:y1, x0:x1].sum()) / max(1, m.size - (x1 - x0) * (y1 - y0))
    assert inside > 0.5 and inside > 3 * outside              # foreground sits on the subject
