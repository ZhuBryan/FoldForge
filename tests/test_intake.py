"""Robust image-intake tests: big photos fold fast, transparency and 16-bit
images are read correctly, and missing files fail loudly.

These guard the failures users hit feeding real files (phone photos, PNGs with
alpha, 16-bit scans) to the origami loaders. Skipped if OpenCV / SciPy / PIL
aren't installed.
"""

import time

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")
pytest.importorskip("PIL")


def _subject_photo(w, h):
    """A bright elliptical subject on a darker textured background (RGB uint8)."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(0)
    bg = np.clip(70 + rng.normal(0, 12, (h, w, 3)), 0, 255).astype("uint8")
    im = Image.fromarray(bg)
    d = ImageDraw.Draw(im)
    d.ellipse([int(w * 0.28), int(h * 0.24), int(w * 0.72), int(h * 0.78)],
              fill=(210, 205, 190))
    return im


def test_huge_image_folds_fast(tmp_path):
    """A multi-megapixel photo must origamize in well under 40s (it is
    downscaled before GrabCut) and yield a sane, non-empty subject mask."""
    from foldforge.origamize import origamize_silhouette, silhouette_mask
    p = tmp_path / "huge.jpg"
    _subject_photo(2600, 2200).save(p, quality=90)
    t = time.time()
    result, relief = origamize_silhouette(str(p), grid=(24, 28))
    dt = time.time() - t
    assert dt < 30.0, f"origamize_silhouette on a big photo took {dt:.1f}s"
    assert relief.shape == (24, 28)
    m = silhouette_mask(str(p))
    assert 0.03 < m.mean() < 0.9        # found a subject, not all/none


def test_rgba_transparency_not_read_as_black(tmp_path):
    """Transparent PNG pixels must composite onto white, not read as black."""
    from PIL import Image
    from foldforge.origamize.vision import _read_source_rgb
    w = h = 64
    rgba = np.zeros((h, w, 4), dtype="uint8")
    rgba[..., 3] = 0                                  # fully transparent everywhere
    rgba[16:48, 16:48, :3] = (200, 40, 40)            # an opaque red square
    rgba[16:48, 16:48, 3] = 255
    p = tmp_path / "trans.png"
    Image.fromarray(rgba, "RGBA").save(p)

    got = _read_source_rgb(str(p)).astype(float)
    # transparent border must be white, not black
    assert got[0, 0].mean() > 240, "transparent pixels read as black"
    # a black-composited control differs materially in the transparent region
    a = rgba[..., 3:4].astype(float) / 255.0
    black = (rgba[..., :3].astype(float) * a)
    assert abs(got.mean() - black.mean()) > 20.0


def test_16bit_png_gives_non_constant_heightfield(tmp_path):
    """A 16-bit grayscale PNG must normalise to a varied (non-degenerate)
    height field, not collapse to a constant that yields error=0."""
    from PIL import Image
    from foldforge.origamize.io import heightmap_from_image
    w = h = 96
    grad = np.linspace(0, 65535, w)[None, :] * np.ones((h, 1))
    Image.fromarray(grad.astype("uint16")).save(tmp_path / "g16.png")
    Z = heightmap_from_image(str(tmp_path / "g16.png"), grid=(16, 16))
    assert Z.std() > 0.05, "16-bit image collapsed to a constant height field"


def test_missing_file_raises_filenotfound(tmp_path):
    """A missing path raises FileNotFoundError naming the file, not a cryptic
    decode error or silent success."""
    from foldforge.origamize.io import heightmap_from_image
    from foldforge.origamize.vision import _read_source_rgb
    missing = str(tmp_path / "does_not_exist.png")
    with pytest.raises(FileNotFoundError):
        heightmap_from_image(missing)
    with pytest.raises(FileNotFoundError):
        _read_source_rgb(missing)
