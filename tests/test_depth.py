"""Monocular-depth origami: MiDaS / DPT depth -> masked -> folded.

Skipped automatically if PyTorch (or the vision stack) isn't installed; PyTorch
is an optional dependency. With torch present the MiDaS repo/weights download
once via torch.hub and are cached, so these stay fast on repeat runs. The
DPT_Hybrid case additionally needs ``timm`` and its ~470 MB checkpoint, and is
skipped cleanly when either is missing.
"""

import os

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("scipy")


def _subject_on_noisy_bg(W=160, H=130):
    """A dark blob 'subject' on a noisy gradient background (GrabCut-friendly)."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(0)
    bg = (np.linspace(120, 200, W)[None, :] * np.ones((H, 1))) + rng.normal(0, 16, (H, W))
    im = Image.fromarray(np.clip(bg, 0, 255).astype("uint8")).convert("RGB")
    d = ImageDraw.Draw(im)
    d.ellipse([45, 55, 120, 110], fill=(30, 28, 28))
    d.ellipse([95, 40, 135, 78], fill=(30, 28, 28))
    return np.asarray(im)


def _dpt_available():
    """DPT_Hybrid needs timm and its cached checkpoint; skip if either absent."""
    import importlib.util
    if importlib.util.find_spec("timm") is None:
        return False
    import torch
    ckpt = os.path.join(torch.hub.get_dir(), "checkpoints", "dpt_hybrid_384.pt")
    return os.path.exists(ckpt)


def test_estimate_depth_shape_and_range():
    from foldforge.origamize import estimate_depth
    img = _subject_on_noisy_bg()
    d = estimate_depth(img)
    assert d.shape == img.shape[:2]                 # per-pixel, same H x W
    assert d.dtype == np.float64
    assert 0.0 <= d.min() and d.max() <= 1.0        # normalised
    assert d.max() - d.min() > 0.1                  # not a constant map


def test_origamize_depth_folds_the_subject():
    from foldforge.origamize import origamize_depth
    result, relief = origamize_depth(_subject_on_noisy_bg(), grid=(20, 24), height=6)
    assert relief.shape == (20, 24)
    assert relief.max() > 0.9 and relief.min() < 0.1        # normalised height field
    assert result.folded.shape[1] == 3 and np.isfinite(result.error)
    assert result.triangles.shape[1] == 3                  # ready for to_stl


def test_estimate_depth_rejects_unknown_model():
    from foldforge.origamize import estimate_depth
    with pytest.raises(ValueError):
        estimate_depth(_subject_on_noisy_bg(), model_type="DPT_Nonexistent")


@pytest.mark.skipif(not _dpt_available(),
                    reason="DPT_Hybrid needs timm + the dpt_hybrid_384.pt checkpoint")
def test_dpt_hybrid_depth_shape_and_range():
    from foldforge.origamize import estimate_depth
    img = _subject_on_noisy_bg()
    d = estimate_depth(img, model_type="DPT_Hybrid")
    assert d.shape == img.shape[:2]                 # per-pixel, same H x W
    assert d.dtype == np.float64
    assert 0.0 <= d.min() and d.max() <= 1.0        # normalised 0..1
    assert d.max() - d.min() > 0.1                  # not a constant map
