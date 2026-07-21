"""Bilateral symmetrization for photo reliefs.

A real photo of a symmetric subject (a butterfly, a face-on animal) is almost
never shot perfectly square-on: the camera is a few degrees off-axis and the
subject sits a little left or right of centre, so the extracted silhouette and
height field come out lopsided - one wing bigger than the other. This module
finds the subject's best mirror axis and *symmetrizes* the relief about it, so
the folded model comes out with matching sides.

The approach is deliberately simple and deterministic:

  1. **Find the axis.** Search a small grid of rotations and mirror-line offsets
     for the vertical (``"y"`` - left/right) or horizontal (``"x"`` - top/bottom)
     mirror that maximises the silhouette's self-overlap (mask mirror-IoU).
  2. **Align it to vertical/horizontal.** Rotate the relief and mask by the small
     angle found, so the mirror line is an image axis.
  3. **Average with the mirror.** ``sym = (A + mirror(A)) / 2`` - the robust,
     parameter-free choice. Both sides become identical by construction.

``mode``:

* ``"off"`` - do nothing (the default; keeps every existing result byte-identical).
* ``"auto"`` - detect *left/right* (bilateral) symmetry and apply it only if the
  subject is plausibly symmetric (mask mirror-IoU >= ``threshold``); a subject
  that is not clearly left/right symmetric (an animal turned or in side profile)
  is left untouched. Auto deliberately considers only the vertical mirror axis,
  since bilateral symmetry of an upright photo subject means left/right - a
  top/bottom fold is almost never what is wanted, so it is opt-in via ``"x"``.
* ``"x"`` / ``"y"`` - force top/bottom or left/right symmetry regardless.
"""

from __future__ import annotations

import numpy as np

# Default search grid: +/-12 deg of camera tilt in 1.5-deg steps.
_ANGLES = np.arange(-12.0, 12.01, 1.5)
_SEARCH = 200          # longest side (px) the axis search downsamples to
_THRESHOLD = 0.80      # min mask mirror-IoU for "auto" to fire


def _mirror_about(A: np.ndarray, c: int, axis: int):
    """Reflect ``A`` about integer index ``c`` along ``axis`` (out-of-range -> 0).

    Returns ``(mirrored, valid)`` where ``valid`` marks the indices whose mirror
    source lies inside the array.
    """
    n = A.shape[axis]
    idx = (2 * c - np.arange(n)).astype(int)
    valid = (idx >= 0) & (idx < n)
    src = np.clip(idx, 0, n - 1)
    take = np.take(A, src, axis=axis).astype(float)
    shape = [1, 1]
    shape[axis] = n
    take = np.where(valid.reshape(shape), take, 0.0)
    return take, valid


def _iou(mask: np.ndarray, c: int, axis: int) -> float:
    mir, _ = _mirror_about(mask, c, axis)
    a = mask > 0
    b = mir > 0
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def _find_axis(mask: np.ndarray, which: str, angles, search: int = _SEARCH):
    """Best mirror for one axis. Returns ``(iou, angle_deg, center_frac)``.

    ``which`` is ``"y"`` (vertical mirror line -> left/right symmetry, reflect
    columns) or ``"x"`` (horizontal mirror line -> top/bottom, reflect rows).
    ``center_frac`` is the mirror line as a fraction of the reflected axis'
    length, so it maps back to any resolution.
    """
    import cv2
    from scipy.ndimage import rotate as ndrotate

    axis = 1 if which == "y" else 0
    H, W = mask.shape
    s = search / float(max(H, W))
    m0 = cv2.resize(mask.astype(np.uint8),
                    (max(1, int(round(W * s))), max(1, int(round(H * s)))),
                    interpolation=cv2.INTER_NEAREST)
    n = m0.shape[axis]
    off = max(4, int(0.10 * n))
    best = (-1.0, 0.0, 0.5)
    for th in angles:
        m = ndrotate(m0.astype(float), th, reshape=False, order=0, prefilter=False)
        m = (m > 0.5).astype(np.uint8)
        coords = np.where(m > 0)[axis]
        c0 = int(round(coords.mean())) if coords.size else n // 2
        for c in range(max(0, c0 - off), min(n, c0 + off + 1)):
            i = _iou(m, c, axis)
            if i > best[0]:
                best = (i, float(th), c / float(n))
    return best


def measure_symmetry(mask: np.ndarray, which: str = "y", angles=None) -> dict:
    """Best mirror-IoU / angle / centre for ``which`` axis, without modifying.

    Handy for reporting how (a)symmetric a raw silhouette is.
    """
    angles = _ANGLES if angles is None else angles
    iou, angle, frac = _find_axis(mask, which, angles)
    return {"axis": which, "iou": iou, "angle": angle, "center_frac": frac}


def _apply(relief: np.ndarray, mask: np.ndarray, which: str, angle: float,
           frac: float):
    """Rotate to align the axis, then average relief and mask with their mirror."""
    from scipy.ndimage import rotate as ndrotate

    axis = 1 if which == "y" else 0
    if abs(angle) > 1e-9:
        relief = ndrotate(relief.astype(float), angle, reshape=False, order=1,
                          prefilter=False)
        mask = ndrotate(mask.astype(float), angle, reshape=False, order=0,
                        prefilter=False)
        mask = (mask > 0.5).astype(np.uint8)
    else:
        relief = relief.astype(float)
    n = relief.shape[axis]
    c = int(round(frac * n))

    def sym(A):
        mir, valid = _mirror_about(A, c, axis)
        shape = [1, 1]
        shape[axis] = n
        v = valid.reshape(shape)
        return np.where(v, 0.5 * (A.astype(float) + mir), A.astype(float))

    r = sym(relief)
    m = (sym(mask.astype(float)) >= 0.5).astype(np.uint8)
    r = r * m                      # keep the relief inside the symmetric silhouette
    return r, m


def symmetrize(relief: np.ndarray, mask: np.ndarray, mode: str = "off",
               threshold: float = _THRESHOLD, angles=None, return_info: bool = False):
    """Symmetrize ``relief`` and ``mask`` about their best mirror axis.

    ``mode`` is ``"off"`` / ``"auto"`` / ``"x"`` / ``"y"`` (see the module
    docstring). Returns ``(relief, mask)`` or, with ``return_info``,
    ``(relief, mask, info)`` where ``info`` reports the chosen ``axis``, the
    measured ``iou``, the ``angle`` applied, and whether it was ``applied``.
    """
    angles = _ANGLES if angles is None else angles
    info = {"axis": None, "iou": None, "angle": 0.0, "center_frac": None,
            "applied": False}
    if mode in (None, "off"):
        return (relief, mask, info) if return_info else (relief, mask)
    if mask is None or not np.any(mask > 0):
        return (relief, mask, info) if return_info else (relief, mask)

    if mode in ("x", "y"):
        iou, angle, frac = _find_axis(mask, mode, angles)
        which = mode
        apply = True
    elif mode == "auto":
        # Bilateral symmetry of an upright subject is left/right (vertical axis).
        iou, angle, frac = _find_axis(mask, "y", angles)
        which = "y"
        apply = iou >= threshold
    else:
        raise ValueError(
            f"unknown symmetry mode {mode!r}; choose 'off', 'auto', 'x' or 'y'")

    info.update(axis=which, iou=float(iou), angle=float(angle),
                center_frac=float(frac))
    if apply:
        relief, mask = _apply(relief, mask, which, angle, frac)
        info["applied"] = True
    return (relief, mask, info) if return_info else (relief, mask)
