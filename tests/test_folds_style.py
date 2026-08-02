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


# --- hand-fold budget: crease count + difficulty ----------------------------

def test_crease_stats_merges_collinear_pleats():
    """A corrugation's extruded pleats each span many panel edges but count as
    one fold line; crease_stats reports the fold-line count, not the edge count."""
    from foldforge.origamize import origamize_heightfield, heightfield_dome, crease_stats
    r = origamize_heightfield(heightfield_dome(12, 8))
    count, diff = crease_stats(r.pattern)
    mv_edges = sum(1 for a in r.pattern.assignment if a in ("M", "V"))
    assert count < mv_edges              # collinear edges merged into fold lines
    assert count == r.crease_count       # result exposes the same number
    assert diff == r.difficulty in ("Easy", "Medium", "Hard (machine)")


def test_budget_folds_presets_and_labels():
    from foldforge.origamize import budget_folds, difficulty_label
    assert budget_folds("easy") < budget_folds("medium")
    assert budget_folds("hard", default=40) == 40        # hard keeps current detail
    assert budget_folds(None, default=33) == 33          # back-compat: no budget
    assert difficulty_label(12) == "Easy"
    assert difficulty_label(300) == "Hard (machine)"


def test_easy_budget_far_fewer_creases_than_detailed():
    """The hand-fold budget genuinely coarsens: Easy yields far fewer creases
    than the detailed default, and every result reports crease_count/difficulty."""
    from foldforge.origamize import origamize_silhouette
    img = _portrait_subject()
    easy, _ = origamize_silhouette(img, foldable="easy")
    hard, _ = origamize_silhouette(img, foldable="hard")
    assert easy.crease_count < hard.crease_count
    assert len(easy.triangles) < len(hard.triangles)
    assert easy.difficulty == "Easy"
    for r in (easy, hard):
        assert isinstance(r.crease_count, int) and r.crease_count >= 0
        assert r.difficulty in ("Easy", "Medium", "Hard (machine)")


def test_svg_outline_only_is_clean_hand_fold_sheet():
    """outline_only replaces the internal panel-edge clutter with one sheet
    rectangle, keeping the coloured M/V creases: a printable hand-fold sheet."""
    import tempfile, os
    from foldforge.origamize import origamize_heightfield, heightfield_dome
    from foldforge.fabricate import to_svg
    r = origamize_heightfield(heightfield_dome(14, 8))
    with tempfile.TemporaryDirectory() as d:
        full = os.path.join(d, "full.svg"); clean = os.path.join(d, "clean.svg")
        to_svg(r.pattern, full)
        to_svg(r.pattern, clean, outline_only=True)
        tf, tc = open(full).read(), open(clean).read()
    assert tc.count("<line") < tf.count("<line")     # panel clutter removed
    assert tc.count("<rect") == 1                     # one sheet outline
    assert "#ff0000" in tc or "#0000ff" in tc         # M/V creases still present
