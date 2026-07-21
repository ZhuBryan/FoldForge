"""`foldforge fold <photo> <out>`: one-command photo -> folded 3D mesh.

Exercises the CLI end to end on a tiny synthetic photo (no network) and checks
its human-readable errors. Skipped if OpenCV / SciPy / PIL are unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")
pytest.importorskip("PIL")


def _synth_photo(path):
    from PIL import Image, ImageDraw
    W, H = 170, 150
    rng = np.random.default_rng(0)
    bg = np.clip(70 + rng.normal(0, 10, (H, W, 3)), 0, 255).astype("uint8")
    im = Image.fromarray(bg)
    ImageDraw.Draw(im).ellipse([int(W * 0.42), int(H * 0.28),
                                int(W * 0.86), int(H * 0.82)], fill=(215, 205, 185))
    im.save(path)


def test_fold_writes_watertight_stl(tmp_path, capsys):
    from foldforge.cli import main
    img = tmp_path / "subj.png"; _synth_photo(img)
    out = tmp_path / "subj.stl"
    main(["fold", str(img), str(out), "--grid", "24"])
    assert out.exists() and out.stat().st_size > 10_000          # a real, non-trivial mesh
    assert out.read_text().startswith("solid")
    summary = capsys.readouterr().out
    assert "closed solid" in summary and "coverage" in summary


def test_fold_open_sheet_and_obj(tmp_path):
    from foldforge.cli import main
    img = tmp_path / "s.png"; _synth_photo(img)
    out = tmp_path / "s.obj"
    main(["fold", str(img), str(out), "--grid", "20", "--open-sheet"])
    assert out.exists() and out.read_text().startswith(("#", "v ", "o "))


def test_fold_missing_file_is_friendly(tmp_path):
    from foldforge.cli import main
    with pytest.raises(SystemExit) as e:
        main(["fold", str(tmp_path / "nope.png"), str(tmp_path / "o.stl")])
    assert "not found" in str(e.value).lower()


def test_fold_bad_extension_is_friendly(tmp_path):
    from foldforge.cli import main
    img = tmp_path / "s.png"; _synth_photo(img)
    with pytest.raises(SystemExit) as e:
        main(["fold", str(img), str(tmp_path / "o.ply")])
    assert "stl" in str(e.value).lower()


def test_fold_folds_and_style_parse(tmp_path, capsys):
    from foldforge.cli import main
    img = tmp_path / "s.png"; _synth_photo(img)
    out = tmp_path / "s.stl"
    main(["fold", str(img), str(out), "--folds", "12", "--style", "origami"])
    assert out.exists() and out.read_text().startswith("solid")
    summary = capsys.readouterr().out
    assert "origami style" in summary and "folds=12" in summary


def test_fold_detail_preset_parse(tmp_path, capsys):
    from foldforge.cli import main
    img = tmp_path / "s.png"; _synth_photo(img)
    out = tmp_path / "s.stl"
    main(["fold", str(img), str(out), "--detail", "rough"])
    assert out.exists()
    assert "folds=12" in capsys.readouterr().out          # rough == 12 folds
