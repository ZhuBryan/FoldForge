"""Fold-count knob, aspect-preservation, and origami style.

Exercises the new controls on tiny synthetic photos (no network). Skipped if
OpenCV / SciPy / PIL are unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")
pytest.importorskip("PIL")


def _portrait_subject(W=150, H=230):
    """A tall (portrait) blob on a noisy background -> non-square crop."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(3)
    bg = (np.linspace(120, 200, W)[None, :] * np.ones((H, 1))) + rng.normal(0, 16, (H, W))
    im = Image.fromarray(np.clip(bg, 0, 255).astype("uint8")).convert("RGB")
    ImageDraw.Draw(im).ellipse([int(W * 0.32), int(H * 0.12),
                                int(W * 0.68), int(H * 0.88)], fill=(32, 28, 28))
    return np.asarray(im)


def _coplanar_fraction(folded, triangles, deg=2.0):
    """Fraction of interior shared edges whose two faces are near-coplanar."""
    V = np.asarray(folded, float)
    T = np.asarray(triangles, int)
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln < 1e-12, 1.0, ln)
    edge_faces = {}
    for fi, (a, b, c) in enumerate(T):
        for e in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault(frozenset((int(e[0]), int(e[1]))), []).append(fi)
    shared = [f for f in edge_faces.values() if len(f) == 2]
    if not shared:
        return 0.0
    cos_t = np.cos(np.deg2rad(deg))
    near = sum(1 for f0, f1 in shared if abs(float(n[f0] @ n[f1])) >= cos_t)
    return near / len(shared)


def test_folds_knob_monotonic_triangle_count():
    from foldforge.origamize import origamize_silhouette
    img = _portrait_subject()
    counts = []
    for folds in (8, 16, 28):
        r, _ = origamize_silhouette(img, folds=folds)
        counts.append(len(r.triangles))
    assert counts[0] < counts[1] < counts[2]     # more folds -> more triangles


def test_output_aspect_matches_crop_aspect():
    from foldforge.origamize import origamize_silhouette, silhouette_mask
    img = _portrait_subject()
    m = silhouette_mask(img)
    ys, xs = np.where(m > 0)
    crop_aspect = (np.ptp(xs) + 1) / (np.ptp(ys) + 1)      # width / height
    r, _ = origamize_silhouette(img, folds=24)
    t = r.target.reshape(-1, 3)
    out_aspect = np.ptp(t[:, 0]) / np.ptp(t[:, 1])          # length / width
    assert abs(out_aspect - crop_aspect) / crop_aspect < 0.10
    assert out_aspect < 0.9        # a portrait subject folds taller-than-wide


def test_origami_style_is_more_planar_than_smooth():
    from foldforge.origamize import origamize_silhouette
    img = _portrait_subject()
    rs, _ = origamize_silhouette(img, folds=12, style="smooth")
    ro, _ = origamize_silhouette(img, folds=12, style="origami")
    # Tight tolerance: origami's posterised plateaus are *exactly* flat, while a
    # smooth dome is only gently curved (its neighbours drift past 2 degrees).
    fs = _coplanar_fraction(rs.folded, rs.triangles, deg=2.0)
    fo = _coplanar_fraction(ro.folded, ro.triangles, deg=2.0)
    assert fo > fs                       # posterised relief -> more flat facets
    assert fo > 0.35                     # a meaningful share of near-coplanar pairs


def test_origami_and_folds_close_into_solid():
    from foldforge.origamize import origamize_silhouette
    img = _portrait_subject()
    r, _ = origamize_silhouette(img, folds=10, style="origami", closed=True)
    assert r.solid is not None
    V, T = r.solid
    assert len(T) > len(r.triangles)     # closed solid is thicker than open sheet
