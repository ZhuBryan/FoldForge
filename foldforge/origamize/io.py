"""Turn *any* input into something foldable: image, function, or 3D points.

Each adapter produces a height field ``Z`` (a 2D array), which the origamizer
(:func:`foldforge.origamize.origamize_heightfield`) then decomposes into a
pleated, foldable sheet. So "fold this image / this surface / this mesh" all
reduce to one path.

    origamize_image("photo.png")          # brightness -> relief -> origami
    origamize_function(lambda x, y: ...)   # any analytic surface
    origamize_points(mesh_vertices)        # a top-down height field of a mesh
"""

from __future__ import annotations

import numpy as np

from foldforge.origamize.surface import origamize_heightfield, OrigamiResult


def fold_heightfield(Z, length=24.0, width=20.0, engine="corrugation",
                     iters=500, **miura_kw) -> OrigamiResult:
    """Decompose a (already height-scaled) field ``Z`` with the chosen engine.

    ``engine="corrugation"`` (default) keeps the exact 1D pleated-strip path;
    ``engine="miura2d"`` fits a genuine 2D warped-Miura tessellation
    (:func:`foldforge.origamize.miura_fit.origamize_miura`) whose folded
    mid-surface tracks the field far more faithfully on curved shapes. Shared by
    every intake (image / function / points / silhouette / depth) so the engine
    switch is one keyword everywhere.
    """
    import numpy as np
    Z = np.asarray(Z, dtype=float)
    if engine in ("miura2d", "miura", "2d"):
        from foldforge.origamize.miura_fit import origamize_miura
        height = float(Z.max() - Z.min()) or 1.0
        return origamize_miura(Z, length=length, width=width, height=height,
                               **miura_kw)
    return origamize_heightfield(Z, length=length, width=width, iters=iters)


def _resize(img: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Resize a 2D array to (gh, gw). Uses PIL if available, else bilinear numpy."""
    try:
        from PIL import Image
        im = Image.fromarray((img * 255).clip(0, 255).astype("uint8"))
        return np.asarray(im.resize((gw, gh), Image.BILINEAR), dtype=float) / 255.0
    except Exception:
        H, W = img.shape
        yi = np.linspace(0, H - 1, gh); xi = np.linspace(0, W - 1, gw)
        y0 = np.floor(yi).astype(int); y1 = np.clip(y0 + 1, 0, H - 1); wy = (yi - y0)[:, None]
        x0 = np.floor(xi).astype(int); x1 = np.clip(x0 + 1, 0, W - 1); wx = (xi - x0)[None, :]
        top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
        bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
        return top * (1 - wy) + bot * wy


def _read_gray01(path: str) -> np.ndarray:
    """Load an image *path* as a grayscale 0..1 array via the robust central
    reader (16-bit, alpha, palette/CMYK, EXIF rotation), falling back to
    matplotlib if the vision stack is unavailable. Missing files raise
    ``FileNotFoundError``."""
    try:
        from foldforge.origamize.vision import _read_source_rgb
    except Exception:
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"image file not found: {path!r}")
        import matplotlib.image as mpimg
        img = mpimg.imread(path).astype(float)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        if img.max() > 1.0:
            img = img / 255.0
        return img
    rgb = _read_source_rgb(path)                 # HxWx3 uint8, all formats normalised
    return rgb.astype(float).mean(axis=2) / 255.0


def heightmap_from_image(source, grid=(18, 24), invert=False) -> np.ndarray:
    """Load an image (path or array) as a normalised 0..1 height field.

    Colour images are converted to grayscale; brightness becomes height (set
    ``invert=True`` to make dark = high, e.g. for line art). Image *paths* go
    through the robust loader (16-bit, alpha, palette/CMYK, EXIF orientation);
    a missing path raises ``FileNotFoundError``.
    """
    if isinstance(source, str):
        img = _read_gray01(source)
    else:
        img = np.asarray(source, dtype=float)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)         # grayscale
        if img.max() > 1.0:
            img = img / 255.0
    Z = _resize(img, grid[0], grid[1])
    if invert:
        Z = 1.0 - Z
    Z = Z - Z.min()
    return Z / (Z.max() + 1e-9)


def heightmap_from_function(f, grid=(18, 24), extent=(-1, 1, -1, 1)) -> np.ndarray:
    """Sample ``z = f(x, y)`` on a grid into a normalised height field."""
    x = np.linspace(extent[0], extent[1], grid[1])
    y = np.linspace(extent[2], extent[3], grid[0])
    Y, X = np.meshgrid(y, x, indexing="ij")
    Z = np.asarray(f(X, Y), dtype=float)
    Z = Z - Z.min()
    return Z / (Z.max() + 1e-9)


def heightmap_from_points(points, grid=(18, 24)) -> np.ndarray:
    """Top-down height field from a 3D point cloud / mesh vertices ``(N, 3)``.

    Bins the xy-plane into cells and takes the max z per cell (a height-map
    silhouette of the model); empty cells are filled from their neighbours.
    """
    P = np.asarray(points, dtype=float)
    xy = P[:, :2]; z = P[:, 2]
    lo, hi = xy.min(0), xy.max(0)
    u = (xy - lo) / (hi - lo + 1e-9)
    gi = np.clip((u[:, 1] * (grid[0] - 1)).astype(int), 0, grid[0] - 1)
    gj = np.clip((u[:, 0] * (grid[1] - 1)).astype(int), 0, grid[1] - 1)
    Z = np.full(grid, np.nan)
    for a, b, zz in zip(gi, gj, z):
        if np.isnan(Z[a, b]) or zz > Z[a, b]:
            Z[a, b] = zz
    # fill empty cells by iterative neighbour averaging
    for _ in range(grid[0] + grid[1]):
        nan = np.isnan(Z)
        if not nan.any():
            break
        filled = np.where(nan, 0.0, Z)
        cnt = (~nan).astype(float)
        acc = np.zeros_like(Z); wacc = np.zeros_like(Z)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            acc += np.roll(filled, (dy, dx), (0, 1)); wacc += np.roll(cnt, (dy, dx), (0, 1))
        Z = np.where(nan & (wacc > 0), acc / (wacc + 1e-9), Z)
    Z = np.nan_to_num(Z, nan=np.nanmin(Z))
    Z = Z - Z.min()
    return Z / (Z.max() + 1e-9)


def origamize_image(source, grid=(18, 24), height=6.0, length=24.0, width=20.0,
                    invert=False, iters=500, engine="corrugation") -> OrigamiResult:
    """Estimate a relief from an image and decompose it into a foldable sheet.

    ``engine="miura2d"`` fits a true 2D Miura tessellation instead of the 1D
    corrugation (default); see :func:`fold_heightfield`.
    """
    Z = heightmap_from_image(source, grid, invert) * height
    return fold_heightfield(Z, length, width, engine=engine, iters=iters)


def origamize_function(f, grid=(18, 24), extent=(-1, 1, -1, 1), height=6.0,
                       length=24.0, width=20.0, iters=500,
                       engine="corrugation") -> OrigamiResult:
    """Decompose an analytic surface ``z = f(x, y)`` into a foldable sheet."""
    Z = heightmap_from_function(f, grid, extent) * height
    return fold_heightfield(Z, length, width, engine=engine, iters=iters)


def origamize_points(points, grid=(18, 24), height=6.0, length=24.0, width=20.0,
                     iters=500, engine="corrugation") -> OrigamiResult:
    """Decompose a 3D point cloud / mesh (as a height map) into a foldable sheet."""
    Z = heightmap_from_points(points, grid) * height
    return fold_heightfield(Z, length, width, engine=engine, iters=iters)
