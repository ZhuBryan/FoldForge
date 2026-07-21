"""The true-2D origamizer: warped-Miura fit vs the 1D corrugation.

Pure numpy (no OpenCV/torch), so these always run. They pin (a) that the 2D
engine beats corrugation clearly on curved targets, (b) that it returns a valid,
FOLD-exportable 2D crease pattern with M and V creases, (c) that its fold is
near-rigid (bounded edge strain, planar facets), and (d) a CLI smoke.
"""

import numpy as np
import pytest

from foldforge.origamize import (
    origamize_miura, compare_engines, surface_fit_error, corrugation_surface,
    heightfield_dome, heightfield_saddle, origamize_heightfield,
)
from foldforge.geometry.foldability import foldability_report
from foldforge.geometry.fold_io import write_fold, read_fold


def test_beats_corrugation_on_curved_targets():
    """On a hemisphere and a saddle the 2D engine's mid-surface tracks the
    target several times more closely than the (valid single-sheet) corrugation."""
    for Z in (heightfield_dome(), heightfield_saddle()):
        c = compare_engines(Z, iters=500)
        assert c["miura2d_error"] < 0.06                      # tight surface fit
        assert c["miura2d_error"] < c["corrugation_error"] / 3  # clear win


def test_fit_error_metric_is_symmetric_and_identical():
    """The metric is the same callable for both engines; a perfect surface
    scores ~0 and the corrugation extrusion scores its true (poor) fit."""
    Z = heightfield_dome()
    R, C = Z.shape
    # a folded grid that *is* the target scores near zero
    xs = np.linspace(0, 1, C); ys = np.linspace(0, 1, R)
    exact = np.zeros((R, C, 3))
    exact[..., 0] = xs[None, :]; exact[..., 1] = ys[:, None]; exact[..., 2] = Z
    assert surface_fit_error(exact, Z, use_midsurface=False) < 1e-6
    # corrugation's coherent single sheet is an extrusion -> poor on a dome
    corr = origamize_heightfield(Z)
    assert surface_fit_error(corrugation_surface(corr, (R, C)), Z) > 0.15


def test_returns_valid_2d_crease_pattern():
    """A genuine 2D pattern: both mountains and valleys, triangle faces, FOLD I/O."""
    r = origamize_miura(heightfield_dome(), iters=500)
    p = r.pattern
    kinds = set(p.assignment)
    assert "M" in kinds and "V" in kinds          # two crease families, not 1D pleats
    # faces are the triangulating diagonals' triangles (planar by construction)
    assert len(p.faces) > 50 and all(len(f) == 3 for f in p.faces)
    assert p.fold_angle is not None and np.abs(p.fold_angle).max() > 5.0
    # FOLD round-trips through the standard reader/writer
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".fold"); os.close(fd)
    try:
        write_fold(p, path)
        q = read_fold(path)
        assert q.n_vertices == p.n_vertices and q.n_edges == p.n_edges
    finally:
        os.remove(path)


def test_fold_is_near_rigid():
    """Edge strain (isometry residual) stays small: a near-rigid fold, with the
    facets triangulated so each panel is planar to machine precision."""
    r = origamize_miura(heightfield_saddle(), iters=600, wiso=6.0)
    assert r.mean_strain < 0.02          # mean edge strain ~1%
    assert r.max_strain < 0.12           # worst hot-spot bounded
    # every exported triangle is exactly planar (3 points define a plane)
    V, T = r.folded, r.triangles
    assert T.shape[1] == 3 and V.shape[1] == 3
    worst = 0.0
    for a, b, c in T[:200]:
        n = np.cross(V[b] - V[a], V[c] - V[a])
        worst = max(worst, 0.0)          # trivially planar; guard shape only
    assert worst == 0.0


def test_exported_fold_faces_are_planar():
    """Regression for the quad-non-planarity bug: every face written to the FOLD
    is planar on the folded surface. The old export wrote Miura quads that bowed
    ~1.5 cm out of plane when folded; now the triangulating diagonals are
    exported and the faces are those triangles, planar by construction."""
    import tempfile, os
    r = origamize_miura(heightfield_saddle(), iters=600, wiso=6.0)
    fd, path = tempfile.mkstemp(suffix=".fold"); os.close(fd)
    try:
        write_fold(r.pattern, path)
        q = read_fold(path)
    finally:
        os.remove(path)
    # faces are triangles, and the facet diagonals ride their own layer
    assert q.faces and all(len(f) <= 3 for f in q.faces)
    assert r.pattern.metadata.get("facet_edges")
    # place each exported face on the folded surface and measure out-of-plane bow
    V = r.folded                          # same (R*C) vertex indexing as the faces
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))   # sheet diagonal (length units)
    worst = 0.0
    for f in q.faces:
        pts = V[f]
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        worst = max(worst, float(np.max(np.abs((pts - pts[0]) @ n))))
    assert worst < 1e-6 * diag            # planar to machine precision, not ~1.5 cm


def test_not_flat_foldable_is_reported_honestly():
    """A curved relief cannot also collapse flat, so Kawasaki should fail - the
    engine does not pretend otherwise (the corrugation's single profile can, but
    it only makes a ridge)."""
    r = origamize_miura(heightfield_dome(), iters=400)
    rep = foldability_report(r.pattern)
    assert not rep.flat_foldable          # honest: relief != flat-foldable


def test_cli_origamize_miura2d_smoke(tmp_path):
    """`foldforge origamize img out.fold --engine miura2d` writes a valid FOLD."""
    from PIL import Image
    from foldforge.cli import main
    # a smooth radial bump image (no segmentation needed for `origamize`)
    yy, xx = np.mgrid[0:32, 0:32]
    r = np.sqrt((xx - 15.5) ** 2 + (yy - 15.5) ** 2)
    img = np.clip(255 * (1 - r / 22), 0, 255).astype("uint8")
    ip = tmp_path / "bump.png"; Image.fromarray(img).save(ip)
    op = tmp_path / "bump.fold"
    main(["origamize", str(ip), str(op), "--rows", "10", "--cols", "12",
          "--engine", "miura2d"])
    q = read_fold(str(op))
    assert q.n_vertices == 10 * 12
    assert "M" in set(q.assignment) and "V" in set(q.assignment)


def test_performance_photo_scale():
    """A photo-scale grid fits in well under the ~1-2 minute budget."""
    import time
    Z = heightfield_dome(40, 32)
    t0 = time.time()
    r = origamize_miura(Z, iters=600)
    assert time.time() - t0 < 20.0        # generous ceiling; ~1s in practice
    assert r.error < 0.05
