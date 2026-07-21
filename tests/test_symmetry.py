"""Bilateral symmetrization of photo reliefs (butterfly wings that match).

Skipped automatically if OpenCV / SciPy aren't installed (needed by the axis
search and the image pipeline).
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")


def _lr_mirror_rmse(Z):
    """Left/right mirror RMSE about the field's centre, range-normalised."""
    rng = float(Z.max() - Z.min()) + 1e-9
    return float(np.sqrt(np.mean((Z - np.fliplr(Z)) ** 2)) / rng)


def _tilted_symmetric_mask():
    """A left/right-symmetric blob, then rotated a few degrees off-axis - a
    stand-in for a butterfly shot slightly tilted."""
    from scipy.ndimage import rotate as ndrotate
    m = np.zeros((120, 140), np.uint8)
    m[40:90, 45:95] = 1                     # body, symmetric about col 70
    m[55:75, 20:45] = 1                     # left wing
    m[55:75, 95:120] = 1                    # right wing (mirror)
    m = ndrotate(m.astype(float), 6.0, reshape=False, order=0, prefilter=False)
    return (m > 0.5).astype(np.uint8)


def _asymmetric_mask():
    """A right triangle: strongly left/right asymmetric, no good vertical mirror."""
    m = np.zeros((90, 90), np.uint8)
    for j in range(90):
        m[j, : int(8 + 0.85 * j)] = 1
    return m


# --- the helper ------------------------------------------------------------

def test_off_is_identity():
    from foldforge.origamize.symmetry import symmetrize
    m = _tilted_symmetric_mask()
    r = m.astype(float) * 0.7
    r2, m2 = symmetrize(r, m, mode="off")
    assert r2 is r and m2 is m               # untouched, same objects


def _self_symmetry_residual(field, info):
    """Max |field - mirror(field)| about the *detected* axis (which need not be
    the array's geometric centre)."""
    from foldforge.origamize.symmetry import _mirror_about
    axis = 1 if info["axis"] == "y" else 0
    n = field.shape[axis]
    c = int(round(info["center_frac"] * n))
    mir, valid = _mirror_about(field, c, axis)
    take = np.take(field, np.arange(n)[valid], axis=axis)
    take_m = np.take(mir, np.arange(n)[valid], axis=axis)
    return float(np.abs(take - take_m).max())


def test_force_y_makes_field_left_right_symmetric():
    from foldforge.origamize.symmetry import symmetrize
    m = _tilted_symmetric_mask()
    r = m.astype(float) * 0.7
    r2, m2, info = symmetrize(r, m, mode="y", return_info=True)
    assert info["applied"] and info["axis"] == "y"
    # exact left/right symmetry about the detected mirror axis, by construction
    assert _self_symmetry_residual(r2, info) < 1e-9


def test_force_x_symmetrizes_top_bottom():
    from foldforge.origamize.symmetry import symmetrize
    m = np.zeros((90, 90), np.uint8)
    m[15:45, 30:60] = 1                       # blob only in the top half
    r = m.astype(float)
    r2, m2, info = symmetrize(r, m, mode="x", return_info=True)
    assert info["applied"] and info["axis"] == "x"
    assert _self_symmetry_residual(r2, info) < 1e-9


def test_auto_fires_on_symmetric_subject():
    from foldforge.origamize.symmetry import symmetrize
    m = _tilted_symmetric_mask()
    r = m.astype(float)
    r2, m2, info = symmetrize(r, m, mode="auto", return_info=True)
    assert info["applied"] and info["axis"] == "y"
    assert info["iou"] >= 0.80


def test_auto_declines_asymmetric_subject():
    from foldforge.origamize.symmetry import symmetrize
    m = _asymmetric_mask()
    r = m.astype(float)
    r2, m2, info = symmetrize(r, m, mode="auto", return_info=True)
    assert not info["applied"]               # asymmetric -> left untouched
    assert r2 is r and m2 is m


def test_auto_handles_empty_mask():
    from foldforge.origamize.symmetry import symmetrize
    m = np.zeros((40, 40), np.uint8)
    r = m.astype(float)
    r2, m2, info = symmetrize(r, m, mode="auto", return_info=True)
    assert not info["applied"]


def test_bad_mode_raises():
    from foldforge.origamize.symmetry import symmetrize
    m = _tilted_symmetric_mask()
    with pytest.raises(ValueError):
        symmetrize(m.astype(float), m, mode="diagonal")


# --- integration through the photo pipeline --------------------------------

def _tilted_symmetric_photo():
    """A symmetric two-winged subject on a noisy background, shot a bit tilted."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import rotate as ndrotate
    rng = np.random.default_rng(3)
    W, H = 220, 190
    bg = np.full((H, W), 190.0) + rng.normal(0, 14, (H, W))
    im = Image.fromarray(np.clip(bg, 0, 255).astype("uint8")).convert("RGB")
    d = ImageDraw.Draw(im)
    cx = 110
    d.ellipse([cx - 12, 60, cx + 12, 140], fill=(40, 35, 60))         # body
    d.polygon([(cx, 75), (cx - 70, 55), (cx - 55, 120)], fill=(40, 35, 60))   # L wing
    d.polygon([(cx, 75), (cx + 70, 55), (cx + 55, 120)], fill=(40, 35, 60))   # R wing
    arr = np.asarray(im).astype(float)
    arr = ndrotate(arr, 7.0, reshape=False, order=1, mode="nearest")
    return np.clip(arr, 0, 255).astype("uint8")


def test_relief_symmetry_off_is_unchanged(monkeypatch):
    """Default off must leave the estimated relief byte-identical.

    GrabCut is not bit-reproducible run to run, so the segmentation is pinned to
    a fixed mask; this isolates the *only* thing the symmetry feature could
    change on the off path (it changes nothing)."""
    import foldforge.origamize.vision as V
    from foldforge.origamize.vision import relief_from_image
    photo = _tilted_symmetric_photo()
    fixed = np.zeros(photo.shape[:2], np.uint8)
    fixed[50:150, 30:190] = 1
    monkeypatch.setattr(V, "_mask_from_bgr", lambda *a, **k: fixed.copy())
    a = relief_from_image(photo, grid=(30, 30))                 # default
    b = relief_from_image(photo, grid=(30, 30), symmetry="off")
    assert np.array_equal(a, b)
    c = relief_from_image(photo, grid=(30, 30), symmetry="y")
    assert not np.array_equal(a, c)                             # y actually changes it


def test_relief_symmetry_y_matches_the_wings():
    from foldforge.origamize.vision import relief_from_image
    photo = _tilted_symmetric_photo()
    raw = relief_from_image(photo, grid=(30, 30))
    sym = relief_from_image(photo, grid=(30, 30), symmetry="y")
    assert _lr_mirror_rmse(sym) < _lr_mirror_rmse(raw) * 0.5
    assert _lr_mirror_rmse(sym) < 0.02


def test_origamize_silhouette_symmetry_passthrough():
    from foldforge.origamize import origamize_silhouette
    photo = _tilted_symmetric_photo()
    result, relief = origamize_silhouette(photo, grid=(26, 30), symmetry="y")
    assert relief.shape == (26, 30)
    assert _lr_mirror_rmse(relief) < 0.02
    assert result.folded.shape[1] == 3
