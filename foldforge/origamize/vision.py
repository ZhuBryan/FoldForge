"""Image -> shape -> origami: estimate a subject's 3D form from a photo and fold it.

Folding a photo by raw brightness folds the background too. To fold *the animal*,
we estimate its shape:

  1. segment the subject from the background with GrabCut (OpenCV);
  2. "inflate" the silhouette into a rounded bas-relief - the distance to the
     boundary, raised to a power, so the middle of the body puffs up like a
     balloon (the classic silhouette-inflation trick);
  3. crop to the subject and hand the height field to the origamizer.

The result is a 3D-ish relief of the animal that folds from a flat sheet.
When ``rect`` is not given, an edge-density saliency heuristic seeds GrabCut's
rectangle on the subject (so an off-centre subject works out of the box); pass
``rect=(x, y, w, h)`` to override it.

Needs OpenCV and SciPy (``pip install opencv-python scipy``).
"""

from __future__ import annotations

import numpy as np

from foldforge.origamize.surface import (
    origamize_heightfield, OrigamiResult, close_relief, trim_background_triangles,
)

# Largest side (in pixels) we hand to GrabCut / depth nets. A 12-megapixel phone
# photo makes GrabCut take the better part of a minute; downscaling first keeps
# it well under a second with no visible loss for a coarse origami relief.
_MAX_SIDE = 1000


def _pil_to_rgb_uint8(im):
    """Normalise a PIL image to an ``HxWx3`` uint8 RGB array.

    Handles high-bit-depth (16/32-bit ``I``/``F``) by rescaling to 0..255,
    composites any alpha onto white, and converts palette / CMYK / grayscale
    via PIL so their colours survive.
    """
    from PIL import Image
    mode = im.mode
    if mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
        a = np.asarray(im).astype(np.float64)
        a = a - a.min()
        g = (a / (a.max() + 1e-9) * 255.0).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)
    if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        return np.asarray(Image.alpha_composite(bg, im).convert("RGB"))
    return np.asarray(im.convert("RGB"))


def _array_to_rgb_uint8(arr):
    """Normalise an array (any dtype / 1,3,4 channels) to ``HxWx3`` uint8 RGB."""
    arr = np.asarray(arr)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        mx = float(arr.max()) if arr.size else 1.0
        arr = arr * 255.0 if mx <= 1.0 else arr
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:                     # 16-bit / int: rescale, don't wrap
        a = arr.astype(np.float64)
        if a.size and a.max() > 255:
            a = (a - a.min()) / (a.max() - a.min() + 1e-9) * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:                          # RGBA -> composite on white
        rgb = arr[..., :3].astype(np.float64)
        a = arr[..., 3:4].astype(np.float64) / 255.0
        arr = np.clip(rgb * a + 255.0 * (1.0 - a), 0, 255).astype(np.uint8)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] != 3:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def _read_source_rgb(source):
    """Load ``source`` (path or array) as an ``HxWx3`` uint8 RGB array.

    For a path we prefer PIL (honouring EXIF orientation and decoding palette /
    CMYK / 16-bit / alpha correctly), falling back to OpenCV if PIL cannot open
    it. A missing path raises ``FileNotFoundError`` and an undecodable one
    ``ValueError`` - both naming the file.
    """
    if isinstance(source, str):
        import os
        if not os.path.exists(source):
            raise FileNotFoundError(f"image file not found: {source!r}")
        try:
            from PIL import Image, ImageOps
            im = Image.open(source)
            im.load()
            im = ImageOps.exif_transpose(im)        # honour camera orientation
            return _pil_to_rgb_uint8(im)
        except Exception:
            pass
        import cv2
        bgr = cv2.imread(source, cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise ValueError(
                f"could not decode image (unsupported or corrupt format): {source!r}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else bgr
        return _array_to_rgb_uint8(rgb)
    return _array_to_rgb_uint8(source)


def _load_bgr(source, max_side: int = _MAX_SIDE):
    """Load ``source`` as an OpenCV BGR uint8 array, downscaled for speed.

    Returns ``(bgr, scale)`` where ``scale`` is ``loaded_size / original_size``
    (``<= 1``): multiply an original-pixel rectangle by it to match the returned,
    possibly downscaled, image. Downscaling so the longest side is ``<=
    max_side`` keeps GrabCut / depth fast on multi-megapixel phone photos.
    """
    import cv2
    rgb = _read_source_rgb(source)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    scale = 1.0
    longest = max(h, w)
    if max_side and longest > max_side:
        scale = max_side / float(longest)
        bgr = cv2.resize(bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                         interpolation=cv2.INTER_AREA)
    return bgr, scale


def _auto_rect(bgr, inset: float = 0.06):
    """Seed a GrabCut rectangle around the most salient subject.

    GrabCut needs a box that holds the subject and mostly excludes the
    background. The old default - a fixed centre box covering ~88% of the frame -
    fails whenever the subject is off-centre or small. Instead we find where the
    image is *busy*: blur the gradient magnitude into an edge-density map, keep
    its Otsu-bright region, and take the bounding box of the largest blob. A
    subject on a plainer background (an animal on grass, an object on a table)
    lights up, so the box tracks it. Falls back to the centred inset box when the
    heuristic finds nothing usable. Returns a rect in ``bgr``'s own pixel coords.
    """
    import cv2
    h, w = bgr.shape[:2]
    fallback = (int(w * inset), int(h * inset),
                int(w * (1 - 2 * inset)), int(h * (1 - 2 * inset)))
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        dens = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), max(h, w) * 0.02)
        dn = cv2.normalize(dens, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, th = cv2.threshold(dn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (int(w * 0.03) | 1, int(h * 0.03) | 1))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return fallback
        x, y, rw, rh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        m = int(0.02 * max(w, h))                       # small margin around the blob
        x = max(0, x - m); y = max(0, y - m)
        rw = min(w - x, rw + 2 * m); rh = min(h - y, rh + 2 * m)
        area = rw * rh / float(w * h)                   # reject whole-frame / sliver boxes
        if rw < 8 or rh < 8 or area < 0.02 or area > 0.98:
            return fallback
        return (int(x), int(y), int(rw), int(rh))
    except Exception:
        return fallback


def _mask_from_bgr(bgr, scale, iters=5, inset=0.06, rect=None):
    """GrabCut mask on an already-loaded BGR image. A given ``rect`` is in
    original pixel coords and rescaled by ``scale`` to match ``bgr``; when
    ``rect`` is None the box is seeded automatically by :func:`_auto_rect`."""
    import cv2
    h, w = bgr.shape[:2]
    if rect is None:
        gc_rect = _auto_rect(bgr, inset=inset)          # saliency-seeded, in bgr coords
    else:                                               # user rect is in ORIGINAL coords
        x, y, rw, rh = rect
        gc_rect = (int(round(x * scale)), int(round(y * scale)),
                   max(1, int(round(rw * scale))), max(1, int(round(rh * scale))))
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, gc_rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)


def silhouette_mask(source, iters: int = 5, inset: float = 0.06, rect=None) -> np.ndarray:
    """Foreground (subject) mask via GrabCut. Returns a 0/1 uint8 mask.

    ``rect=(x, y, w, h)`` overrides the auto-seeded rectangle and is given in the
    image's **original pixel coordinates** (it is rescaled internally if the
    image is downscaled for speed). With no ``rect`` an edge-density saliency
    heuristic locates the subject. The returned mask matches the (possibly
    downscaled) working resolution.
    """
    bgr, scale = _load_bgr(source)
    return _mask_from_bgr(bgr, scale, iters=iters, inset=inset, rect=rect)


def inflate(mask: np.ndarray, power: float = 0.5,
            profile: str = "round") -> np.ndarray:
    """Inflate a binary silhouette into a rounded height field (0..1).

    Let ``u`` be the (normalised) distance to the boundary, 0 at the silhouette
    edge and 1 at the deepest interior point.

    * ``profile="round"`` (default) raises the silhouette to a true **spherical
      cap**, ``sqrt(u * (2 - u))``. For a disc this reproduces an exact
      hemisphere: on a unit-disc test its height matches the analytic hemisphere
      to RMSE ~0.010, versus ~0.134 for the old ``u**0.5`` law - a markedly more
      faithful balloon (the classic silhouette-inflation target is a rounded, not
      merely puffed, body).
    * ``profile="power"`` keeps the legacy behaviour, ``u**power``; ``power=0.5``
      is the historical rounded-ish default.

    ``power`` is retained for backward compatibility and is used only when
    ``profile="power"``.
    """
    import cv2
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    d = d / (d.max() + 1e-9)
    if profile == "round":
        return np.sqrt(np.clip(d * (2.0 - d), 0.0, None))
    if profile == "power":
        return d ** power
    raise ValueError(f"unknown profile {profile!r}; choose 'round' or 'power'")


def _robust01(z: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Normalise ``z`` to 0..1 using robust percentile bounds instead of min/max.

    Plain ``(z - z.min()) / (z.max() - z.min())`` lets a single outlier cell - a
    specular highlight, a segmentation speck, a lone depth spike - stretch the
    range and crush the real relief's contrast into a sliver. Clipping to the
    ``[lo, hi]`` percentiles ignores such outliers, so the true relief keeps its
    dynamic range (on a synthetic relief with one injected outlier this preserves
    interior contrast at ~0.40 versus ~0.07 for min/max, where the clean value is
    ~0.34). With no outliers the two agree to within the clipped tails.
    """
    z = np.asarray(z, dtype=float)
    a, b = np.percentile(z, lo), np.percentile(z, hi)
    if b - a < 1e-9:                                    # near-constant field
        b = a + 1e-9
    return np.clip((z - a) / (b - a), 0.0, 1.0)


def _layout_from_folds(folds, crop_shape, size: float = 24.0):
    """Grid + physical sheet dims from a single ``folds`` knob and the crop.

    ``folds`` is the number of cells across the *longer* side of the subject
    crop; the shorter side gets proportionally fewer so the cells stay roughly
    square. The sheet's physical extent (``length`` along columns, ``width``
    along rows) keeps the crop's aspect with its longest side equal to ``size``.
    Returns ``((n_rows, n_cols), length, width)``.
    """
    ch, cw = max(int(crop_shape[0]), 1), max(int(crop_shape[1]), 1)
    folds = max(2, int(folds))
    if ch >= cw:                                    # portrait: rows are longer
        n_rows = folds
        n_cols = max(2, int(round(folds * cw / ch)))
        width, length = size, size * cw / ch
    else:                                           # landscape: cols are longer
        n_cols = folds
        n_rows = max(2, int(round(folds * ch / cw)))
        length, width = size, size * ch / cw
    return (n_rows, n_cols), float(length), float(width)


def _posterize(Z, levels: int = 5):
    """Quantise a 0..1 height field to ``levels`` uniform bands.

    Flattening the relief into a few discrete heights *before* corrugation turns
    smooth terrain into big planar plateaus joined by sharp steps - the folded
    result reads as a few large paper facets instead of dense micro-ripple.
    """
    levels = max(2, int(levels))
    return np.round(np.asarray(Z, float) * (levels - 1)) / (levels - 1)


def relief_from_image(source, grid=(30, 30), power: float = 0.5,
                      smooth: float = 1.0, detail: float = 0.35, rect=None,
                      return_mask: bool = False, folds=None, style: str = "smooth",
                      levels: int = 5, size: float = 24.0, return_dims: bool = False,
                      symmetry: str = "off"):
    """Estimate a normalised height field of the subject in ``source``.

    The height is the inflated silhouette (overall rounded shape) *modulated by
    interior shading* (``detail``): brighter, lit regions ride a little higher
    and shadows recess, so the folded model shows surface features (a face, a
    back, limbs) instead of a smooth blob. Shading is edge-preserving
    (bilateral) so features stay crisp. ``detail=0`` gives the pure balloon.

    ``folds`` (when given) is the single fold-count knob: it sets ``grid`` and
    the physical sheet dims from the subject crop's aspect (see
    :func:`_layout_from_folds`), overriding ``grid``/``size``. Smoothing is
    scaled down on coarse grids so a low ``folds`` stays crisp rather than
    blurring its few creases away. ``style="origami"`` drops the blur entirely
    and posterises the relief to ``levels`` height bands, yielding big flat
    facets with sharp creases instead of dense corrugation.

    ``symmetry`` mirror-symmetrizes the subject before folding (default
    ``"off"``): ``"auto"`` detects the best mirror axis and applies it only if the
    silhouette is plausibly symmetric, ``"x"``/``"y"`` force top-bottom / left-right
    symmetry (see :mod:`foldforge.origamize.symmetry`) - so a butterfly shot a few
    degrees off-axis comes out with matching wings.
    """
    from scipy.ndimage import gaussian_filter
    import cv2
    bgr, scale = _load_bgr(source)                 # decode once, reuse for mask
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = _mask_from_bgr(bgr, scale, rect=rect)
    # A rounded (spherical-cap) balloon is the more faithful silhouette inflation
    # for the smooth 3D-subject relief; the low-poly "origami" style keeps the
    # flatter power-law profile so its posterised facets stay large and planar.
    base = inflate(mask, power, profile=("power" if style == "origami" else "round"))
    shade = cv2.bilateralFilter(gray, 7, 45, 45).astype(float) / 255.0
    relief = base * ((1.0 - detail) + detail * shade) * mask   # modulate shape by shading
    if symmetry and symmetry != "off":                         # match a subject's sides
        from foldforge.origamize.symmetry import symmetrize
        relief, mask = symmetrize(relief, mask, mode=symmetry)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        crop, mcrop = relief, mask
    else:
        sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        crop, mcrop = relief[sl], mask[sl]

    length = width = float(size)
    if folds is not None:                          # single knob -> grid + aspect
        grid, length, width = _layout_from_folds(folds, crop.shape, size)

    # Origami style wants crisp steps (no blur); otherwise scale the blur with
    # grid fineness so a coarse/rough grid keeps its creases instead of smearing.
    if style == "origami":
        smooth_eff = 0.0
    else:
        smooth_eff = float(smooth) * (max(grid) / 40.0)
    if smooth_eff:
        crop = gaussian_filter(crop, sigma=smooth_eff)
    Z = cv2.resize(crop, (grid[1], grid[0]), interpolation=cv2.INTER_AREA)
    Z = _robust01(Z)                               # outlier-robust 0..1 (see _robust01)
    if style == "origami":                         # few flat facets, sharp creases
        Z = _posterize(Z, levels)
    out = [Z]
    if return_mask:                                    # subject coverage per grid cell
        gm = cv2.resize(mcrop.astype(np.float32), (grid[1], grid[0]),
                        interpolation=cv2.INTER_AREA)
        out.append((gm >= 0.5).astype(np.uint8))
    if return_dims:
        out += [length, width]
    return out[0] if len(out) == 1 else tuple(out)


def origamize_silhouette(source, grid=(30, 30), height: float = 6.0,
                         length: float = 24.0, width: float = 24.0,
                         power: float = 0.5, smooth: float = 1.0,
                         detail: float = 0.35, rect=None, closed: bool = False,
                         close_mode: str = "mirror", close_base: float = 0.0,
                         folds=None, style: str = "smooth", levels: int = 5,
                         engine: str = "corrugation", symmetry: str = "off",
                         foldable=None, rect_sheet: bool = False):
    """Estimate the subject's shape from an image and fold it.

    Returns ``(result, relief)``: an
    :class:`~foldforge.origamize.surface.OrigamiResult` and the estimated height
    field (handy for visualisation). ``detail`` blends in interior shading.

    ``foldable`` is the hand-fold budget preset (``"easy"``/``"medium"``/
    ``"hard"``); when given it sets ``folds`` to a small target so the crease
    pattern is genuinely foldable by hand (tens of creases, not hundreds).
    ``"hard"`` keeps the current detail. The result carries ``crease_count`` and
    ``difficulty``. This is still a coarse relief/corrugation, not figurative
    origami.

    ``folds`` is the fold-count knob (cells across the subject's longer side);
    it drives the grid resolution *and* makes the folded sheet keep the subject
    crop's aspect ratio (longest side ``= max(length, width)``) instead of a
    forced square. ``style="origami"`` posterises the relief into a few large
    planar facets with sharp creases (low-poly folded-paper look); ``"smooth"``
    (default) keeps the rounded relief. Lower ``folds`` => fewer, coarser folds.

    A photo shows only one side, so the raw relief is an *open* one-sided sheet
    with a hollow back. Pass ``closed=True`` to also build a watertight, printable
    solid by mirroring the relief into a back sheet and stitching the rim (see
    :func:`~foldforge.origamize.surface.close_relief`); it is attached as
    ``result.solid = (vertices, triangles)`` - a two-sheet folded model whose back
    is a mirrored *estimate*, not measured geometry. The flat background is trimmed
    first so the solid hugs the subject (a thin rim at the silhouette, not a wide
    flat plate). ``close_mode`` is ``"mirror"`` (puffed symmetric back) or
    ``"flat"`` (back flattened to ``z=close_base``).
    """
    from foldforge.origamize.surface import budget_folds
    folds = budget_folds(foldable, folds)
    size = max(float(length), float(width))
    Z, gmask, L, W = relief_from_image(
        source, grid=grid, power=power, smooth=smooth, detail=detail, rect=rect,
        return_mask=True, folds=folds, style=style, levels=levels, size=size,
        return_dims=True, symmetry=symmetry)
    if folds is None:                              # honour explicit dims (legacy)
        L, W = length, width
    from foldforge.origamize.io import fold_heightfield
    result = fold_heightfield(Z * height, length=L, width=W, engine=engine)
    if closed:
        # Trim the flat background before mirroring so the solid hugs the subject
        # (a thin rim at the silhouette) instead of a wide flat plate. Fall back
        # to the full sheet if the mask is empty or covers everything. With
        # ``rect_sheet=True`` the trim is skipped entirely, so the exported solid
        # keeps the FULL rectangular sheet - flat background paper at the baseline
        # with the subject relief rising within it, like a real folded sheet.
        keep = gmask.reshape(-1).astype(bool)
        tris = result.triangles
        if not rect_sheet and keep.any() and not keep.all():
            tris = trim_background_triangles(result.triangles, keep)
        pruned = OrigamiResult(result.pattern, result.folded, result.angles,
                               result.target, result.error, triangles=tris)
        result.solid = close_relief(pruned, mode=close_mode, base=close_base)
    return result, Z
