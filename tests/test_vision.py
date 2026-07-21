"""Image -> shape -> origami: segment a subject and fold its estimated shape.

Skipped automatically if OpenCV / SciPy aren't installed.
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")


def _cat_on_noisy_bg():
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(1)
    W, H = 200, 170
    bg = (np.linspace(120, 200, W)[None, :] * np.ones((H, 1))) + rng.normal(0, 18, (H, W))
    im = Image.fromarray(np.clip(bg, 0, 255).astype("uint8")).convert("RGB")
    d = ImageDraw.Draw(im)
    d.ellipse([60, 80, 150, 150], fill=(35, 30, 30))
    d.ellipse([120, 55, 165, 100], fill=(35, 30, 30))
    d.polygon([(128, 60), (136, 38), (146, 62)], fill=(35, 30, 30))
    return np.asarray(im)


def test_silhouette_mask_finds_a_centred_subject():
    from foldforge.origamize import silhouette_mask
    m = silhouette_mask(_cat_on_noisy_bg())
    assert 0.05 < m.mean() < 0.7        # found a subject, not all/none of the frame


def test_origamize_silhouette_folds_the_shape():
    from foldforge.origamize import origamize_silhouette
    result, relief = origamize_silhouette(_cat_on_noisy_bg(), grid=(26, 30), height=6)
    assert relief.shape == (26, 30)
    assert relief.max() > 0.9 and relief.min() < 0.1        # inflated, normalised
    assert result.error < 1.0 and result.folded.shape[1] == 3


def test_inflate_peaks_in_the_interior():
    from foldforge.origamize import inflate
    mask = np.zeros((40, 40), np.uint8)
    mask[8:32, 8:32] = 1
    relief = inflate(mask)
    assert relief[20, 20] > relief[9, 9]    # centre puffs higher than the edge


def test_round_inflation_matches_hemisphere_better_than_power():
    """The default rounded profile reproduces a disc's analytic hemisphere far
    more faithfully than the legacy ``u**0.5`` power law."""
    from foldforge.origamize.vision import inflate
    H = W = 200
    cy = cx = 100
    R = 80
    yy, xx = np.mgrid[0:H, 0:W]
    rho = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inside = rho <= R
    mask = inside.astype(np.uint8)
    true = np.zeros((H, W))
    true[inside] = np.sqrt(np.clip(R ** 2 - rho[inside] ** 2, 0, None))
    true /= true.max()

    def rmse(field):
        f = field[inside] / (field[inside].max() + 1e-9)
        return float(np.sqrt(np.mean((f - true[inside]) ** 2)))

    err_round = rmse(inflate(mask, profile="round"))     # default
    err_power = rmse(inflate(mask, profile="power", power=0.5))
    assert err_round < 0.03                              # hugs the true hemisphere
    assert err_round < err_power / 3                     # a big, clear improvement


def test_relief_default_smooth_uses_round_profile(monkeypatch):
    """`relief_from_image`'s default (smooth) style must inflate with the round
    (spherical-cap) profile - a silent revert to the legacy power law would be a
    fidelity regression, so pin it. Only the 'origami' style keeps 'power'."""
    import foldforge.origamize.vision as vision
    seen = []
    real = vision.inflate

    def spy(mask, power=0.5, profile="round"):
        seen.append(profile)
        return real(mask, power, profile=profile)

    monkeypatch.setattr(vision, "inflate", spy)
    img = _cat_on_noisy_bg()
    vision.relief_from_image(img, grid=(16, 16))          # default style="smooth"
    assert seen and seen[0] == "round"
    seen.clear()
    vision.relief_from_image(img, grid=(16, 16), style="origami")
    assert seen and seen[0] == "power"                    # low-poly keeps the flat law


def test_robust01_ignores_a_single_outlier():
    """Robust percentile normalisation keeps a relief's interior contrast even
    when one outlier cell would blow out plain min/max scaling."""
    from foldforge.origamize.vision import _robust01
    z = np.linspace(0.0, 1.0, 400).reshape(20, 20).copy()
    clean_mid = float(z[z > 0.01].mean())
    z[0, 0] = 12.0                                       # inject a lone spike
    mm = (z - z.min()) / (z.max() - z.min())            # naive min/max
    rb = _robust01(z)
    assert mm[z < 10].mean() < 0.15                     # spike crushes naive contrast
    assert abs(rb[z < 10].mean() - clean_mid) < 0.05    # robust keeps it intact
