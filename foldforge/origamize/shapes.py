"""A library of wild targets to fold: text, procedural terrain, and OBJ meshes.

Each returns a height field (or, for meshes, 3D points) that the origamizer
turns into a foldable sheet - so "fold this word / this terrain / this 3D model"
all just work.
"""

from __future__ import annotations

import numpy as np


def text_heightfield(text: str, grid=(20, 40), pad: int = 6) -> np.ndarray:
    """Render ``text`` to a normalised height field (letters raised)."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = grid[1] * 16, grid[0] * 16
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", H - 2 * pad * 4)
    except Exception:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), text, font=font)
    d.text(((W - (bbox[2] - bbox[0])) / 2 - bbox[0],
            (H - (bbox[3] - bbox[1])) / 2 - bbox[1]), text, fill=255, font=font)
    return np.asarray(im.resize((grid[1], grid[0]), Image.BILINEAR), dtype=float) / 255.0


def terrain_heightfield(grid=(24, 30), octaves: int = 4, seed: int = 0) -> np.ndarray:
    """Procedural fractal terrain: summed, smoothed random octaves, normalised."""
    rng = np.random.default_rng(seed)
    H, W = grid
    Z = np.zeros((H, W))
    amp = 1.0
    for o in range(octaves):
        n = 2 ** (o + 1)
        coarse = rng.standard_normal((min(n, H), min(n, W)))
        yi = np.linspace(0, coarse.shape[0] - 1, H)
        xi = np.linspace(0, coarse.shape[1] - 1, W)
        y0 = np.floor(yi).astype(int); y1 = np.clip(y0 + 1, 0, coarse.shape[0] - 1)
        x0 = np.floor(xi).astype(int); x1 = np.clip(x0 + 1, 0, coarse.shape[1] - 1)
        wy = (yi - y0)[:, None]; wx = (xi - x0)[None, :]
        up = (coarse[y0][:, x0] * (1 - wx) + coarse[y0][:, x1] * wx) * (1 - wy) + \
             (coarse[y1][:, x0] * (1 - wx) + coarse[y1][:, x1] * wx) * wy
        Z += amp * up
        amp *= 0.5
    Z = Z - Z.min()
    return Z / (Z.max() + 1e-9)


def load_obj(path: str) -> np.ndarray:
    """Read vertex positions ``(N, 3)`` from a Wavefront OBJ file."""
    verts = []
    for line in open(path, "r", encoding="utf-8"):
        if line.startswith("v "):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=float)
