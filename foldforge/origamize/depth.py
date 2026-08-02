"""Monocular depth -> origami: fold a photo's estimated real 3D relief.

Silhouette inflation (:mod:`foldforge.origamize.vision`) puffs a subject into a
rounded balloon; a monocular depth network instead predicts genuine per-pixel
distance, so the folded model follows the subject's actual near/far structure.
This module runs MiDaS via ``torch.hub``, keeps only the subject with the same
GrabCut segmentation used by the silhouette folder, normalises the masked depth,
and hands the height field to the origamizer - so it returns the same
``(result, relief)`` pair and everything downstream (``to_stl``, rendering)
works unchanged.

Two models are wired:

* ``MiDaS_small`` (default, ~85 MB) - the MiDaS v2.1 small net. Needs neither
  ``timm`` nor ``torchvision``; fast on CPU. ImageNet-normalised input resized
  with the *upper_bound* rule so both sides are ``<= 256`` and multiples of 32.
* ``DPT_Hybrid`` (~470 MB) - the MiDaS v3 dense-prediction transformer. Sharper,
  higher-fidelity relief but heavier, and it needs ``timm`` for its ViT/ResNet
  backbone. Its own preprocessing mirrors MiDaS' ``dpt_transform``: a
  *minimal* keep-aspect resize to multiples of 32 near 384 and ``[-1, 1]``
  normalisation (mean/std 0.5), which is what the checkpoint was trained with.
* ``depth_anything_v2_small`` (~99 MB) / ``depth_anything_v2_base`` (~371 MB) -
  Depth Anything V2, loaded via the ``transformers`` depth-estimation pipeline.
  Often noticeably sharper than MiDaS small on fine detail. The pipeline owns its
  own preprocessing, so these bypass ``_CONFIG``; their raw output is disparity-
  like (larger = nearer, same convention as MiDaS' inverse depth) and is
  normalised to ``0..1`` (1 = nearest) so reliefs match the MiDaS path exactly.
  ``transformers`` is an optional, lazily-imported dependency.

PyTorch is optional: this module imports without it and raises a clear
``ImportError`` only when a depth model is actually requested (naming ``timm``
if that is what is missing for DPT). Checkpoints and hub repos download once and
are cached, so repeated calls are fast.
"""

from __future__ import annotations

import numpy as np

from foldforge.origamize.surface import origamize_heightfield, OrigamiResult, close_relief
from foldforge.origamize.vision import (
    silhouette_mask, _load_bgr, _layout_from_folds, _posterize,
)

# ImageNet normalisation MiDaS v2.1 was trained with.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Per-model preprocessing, mirroring MiDaS' transform table. ``size`` is the
# target the keep-aspect resize aims for; ``resize`` is the MiDaS rule.
_CONFIG: dict = {
    "MiDaS_small": {
        "size": 256, "resize": "upper_bound",
        "mean": _IMAGENET_MEAN, "std": _IMAGENET_STD,
    },
    "DPT_Hybrid": {
        "size": 384, "resize": "minimal",
        "mean": np.array([0.5, 0.5, 0.5]), "std": np.array([0.5, 0.5, 0.5]),
    },
}
_MODELS: dict = {}          # model_type -> loaded torch module (cached)


def _require_torch():
    try:
        import torch
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            "monocular-depth origami needs PyTorch. Install the CPU build, e.g. "
            "`pip install torch --index-url https://download.pytorch.org/whl/cpu`."
        ) from exc
    return torch


def _hub_repo(torch, repo: str, cache_key: str):
    """Cache a hub repo, put it *first* on ``sys.path``, return its path.

    Both MiDaS repos (v2.1 small and v3 master) ship a top-level ``midas``
    package, so whichever imports first wins in ``sys.modules``. We drop any
    cached ``midas`` modules and move the wanted repo to the front so the right
    package is imported; the already-built model objects keep working because
    they hold direct references to their classes, not the ``sys.modules`` cache.
    """
    import os
    import sys
    path = os.path.join(torch.hub.get_dir(), cache_key)
    # Drop any cached ``midas`` package and move this repo to the front *before*
    # touching torch.hub: master's hubconf eagerly imports ``midas.dpt_depth``,
    # which must resolve against this repo, not a previously loaded one.
    for name in [n for n in sys.modules if n == "midas" or n.startswith("midas.")]:
        del sys.modules[name]
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    if not os.path.isdir(path):                    # first run: fetch + cache repo
        torch.hub.list(repo, trust_repo=True)      # no weights pulled
    return path


def _download_if_missing(torch, url: str, name: str) -> str:
    import os
    ckpt = os.path.join(torch.hub.get_dir(), "checkpoints", name)
    if not os.path.exists(ckpt):
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.hub.download_url_to_file(url, ckpt)
    return ckpt


def _load_midas_small():
    """MiDaS v2.1 small: EfficientNet-Lite backbone, no timm/torchvision."""
    torch = _require_torch()
    # The small model's backbone lives in a second repo; cache both, no weights.
    _hub_repo(torch, "rwightman/gen-efficientnet-pytorch",
              "rwightman_gen-efficientnet-pytorch_master")
    _hub_repo(torch, "isl-org/MiDaS:v2_1", "isl-org_MiDaS_v2_1")
    from midas.midas_net_custom import MidasNet_small

    ckpt = _download_if_missing(
        torch,
        "https://github.com/isl-org/MiDaS/releases/download/v2_1/"
        "midas_v21_small_256.pt", "midas_v21_small_256.pt")
    # Passing ``path`` makes the constructor skip the backbone's ImageNet
    # download and load our full checkpoint instead.
    model = MidasNet_small(path=ckpt, features=64, backbone="efficientnet_lite3",
                           exportable=True, non_negative=True,
                           blocks={"expand": True})
    model.eval()
    return model


def _load_dpt_hybrid():
    """MiDaS v3 DPT-Hybrid: ViT-B + ResNet50 backbone (needs timm)."""
    torch = _require_torch()
    try:
        import timm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the DPT_Hybrid depth model needs the `timm` package for its "
            "ViT/ResNet backbone. Install it with `pip install timm` "
            "(or use model_type='MiDaS_small', which does not)."
        ) from exc
    _hub_repo(torch, "intel-isl/MiDaS", "intel-isl_MiDaS_master")
    from midas.dpt_depth import DPTDepthModel

    ckpt = _download_if_missing(
        torch,
        "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt",
        "dpt_hybrid_384.pt")
    # ``path`` loads our full checkpoint (backbone + head); the backbone is
    # built with pretrained=False so timm never fetches ImageNet weights.
    model = DPTDepthModel(path=ckpt, backbone="vitb_rn50_384", non_negative=True)
    model.eval()
    return model


# model_type -> zero-arg loader (extend this table to wire more nets).
_LOADERS = {
    "MiDaS_small": _load_midas_small,
    "DPT_Hybrid": _load_dpt_hybrid,
}


# Depth Anything V2 backends: model_type -> HuggingFace repo id. Loaded lazily
# via the transformers depth-estimation pipeline (optional dependency).
_DA_MODELS = {
    "depth_anything_v2_small": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth_anything_v2_base": "depth-anything/Depth-Anything-V2-Base-hf",
}
_DA_PIPELINES: dict = {}        # model_type -> loaded transformers pipeline (cached)


def _load_depth_anything(model_type: str):
    """Load and cache a Depth Anything V2 pipeline (needs ``transformers``)."""
    if model_type in _DA_PIPELINES:
        return _DA_PIPELINES[model_type]
    _require_torch()                       # clearest error if torch is missing
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "the Depth Anything v2 depth models need the `transformers` package. "
            "Install it with `pip install transformers` (or use the default "
            "model_type='MiDaS_small', which needs only PyTorch)."
        ) from exc
    pipe = pipeline("depth-estimation", model=_DA_MODELS[model_type], device=-1)
    _DA_PIPELINES[model_type] = pipe
    return pipe


def _estimate_depth_anything(image, model_type: str) -> np.ndarray:
    """Depth Anything V2 depth map, normalised to the MiDaS 0..1 (1=near) convention."""
    import cv2
    from PIL import Image
    pipe = _load_depth_anything(model_type)
    rgb = _load_rgb(image)                                  # HxWx3 RGB uint8
    h, w = rgb.shape[:2]
    out = pipe(Image.fromarray(rgb))
    # ``predicted_depth`` is the raw network output (disparity-like: larger =
    # nearer, matching MiDaS' inverse depth), before the pipeline's 0-255 vis.
    pred = out["predicted_depth"]
    d = np.asarray(pred).squeeze().astype(float)
    if d.shape != (h, w):                                  # net-res -> original
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    d = d - d.min()
    return d / (d.max() + 1e-9)                            # 1 = nearest, 0 = farthest


def _load_model(model_type: str = "MiDaS_small"):
    """Load and cache a depth model (repos + weights fetched once)."""
    if model_type in _MODELS:
        return _MODELS[model_type]
    if model_type not in _LOADERS:
        raise ValueError(
            f"unsupported model_type {model_type!r}; choose from "
            f"{', '.join(sorted(_LOADERS))}."
        )
    model = _LOADERS[model_type]()
    _MODELS[model_type] = model
    return model


def _net_dims(w: int, h: int, target: int = 384, method: str = "minimal",
              multiple: int = 32):
    """MiDaS keep-aspect resize dims, mirroring ``transforms.Resize.get_size``.

    ``upper_bound`` (small model) keeps both sides ``<= target``; ``minimal``
    (DPT) scales as little as possible toward ``target``. Both round each side
    to a multiple of ``multiple``.
    """
    import math
    sw, sh = target / w, target / h
    if method == "upper_bound":
        sw = sh = min(sw, sh)
    elif method == "lower_bound":
        sw = sh = max(sw, sh)
    elif method == "minimal":
        if abs(1 - sw) < abs(1 - sh):
            sh = sw
        else:
            sw = sh
    else:                                                   # pragma: no cover
        raise ValueError(f"unknown resize method {method!r}")

    def constrain(x: float, min_val: int, max_val) -> int:
        y = round(x / multiple) * multiple
        if max_val is not None and y > max_val:
            y = math.floor(x / multiple) * multiple
        if y < min_val:
            y = math.ceil(x / multiple) * multiple
        return int(y)

    if method == "upper_bound":
        return (constrain(sw * w, 0, target), constrain(sh * h, 0, target))
    if method == "lower_bound":
        return (constrain(sw * w, target, None), constrain(sh * h, target, None))
    # minimal: keep at least one block so a tiny image never collapses to 0.
    return (constrain(sw * w, multiple, None), constrain(sh * h, multiple, None))


def _load_rgb(source) -> np.ndarray:
    import cv2
    bgr, _ = _load_bgr(source)          # _load_bgr returns (bgr, scale)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def estimate_depth(image, model_type: str = "MiDaS_small") -> np.ndarray:
    """Estimate a normalised monocular depth map for ``image``.

    ``image`` is a path or an ``HxWx3`` RGB / ``HxW`` array. Returns an ``HxW``
    float array in ``0..1`` where **1 = nearest** the camera and 0 = farthest.
    ``model_type`` is ``'MiDaS_small'`` (default), ``'DPT_Hybrid'`` (sharper,
    heavier, needs ``timm``), or ``'depth_anything_v2_small'`` /
    ``'depth_anything_v2_base'`` (Depth Anything V2 via ``transformers``, often
    sharper on fine detail). Requires PyTorch; checkpoints download once.
    """
    if model_type in _DA_MODELS:
        return _estimate_depth_anything(image, model_type)
    torch = _require_torch()
    import cv2
    if model_type not in _CONFIG:
        raise ValueError(
            f"unsupported model_type {model_type!r}; choose from "
            f"{', '.join(sorted(_CONFIG))}."
        )
    cfg = _CONFIG[model_type]
    model = _load_model(model_type)
    rgb = _load_rgb(image)
    h, w = rgb.shape[:2]
    nw, nh = _net_dims(w, h, cfg["size"], cfg["resize"])
    img = cv2.resize(rgb.astype(np.float32) / 255.0, (nw, nh),
                     interpolation=cv2.INTER_CUBIC)
    img = (img - cfg["mean"]) / cfg["std"]
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32))
    with torch.no_grad():
        pred = model(x.unsqueeze(0))
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze()
    d = pred.cpu().numpy().astype(float)
    d = d - d.min()
    return d / (d.max() + 1e-9)


def depth_relief(source, grid=(30, 30), smooth: float = 1.0, rect=None,
                 model_type: str = "MiDaS_small", folds=None, style: str = "smooth",
                 levels: int = 5, size: float = 24.0, return_dims: bool = False,
                 symmetry: str = "off"):
    """Estimate a normalised height field of the subject from monocular depth.

    Runs a depth network, keeps only the subject (GrabCut silhouette),
    normalises the depth *within* the subject so its own near/far range fills
    ``0..1``, crops to the subject and resizes to ``grid``. Unlike the inflated
    silhouette this follows the estimated real relief. ``rect=(x, y, w, h)``
    points GrabCut at an off-centre subject.

    ``folds`` (the fold-count knob) sets the grid and the sheet's aspect from
    the subject crop; smoothing scales with grid fineness. ``style="origami"``
    drops the blur and posterises to ``levels`` height bands for a faceted,
    folded-paper look. Returns the height field, or ``(Z, length, width)`` when
    ``return_dims`` is set.
    """
    from scipy.ndimage import gaussian_filter
    import cv2
    depth = estimate_depth(source, model_type=model_type)      # HxW, 1 = near
    mask = silhouette_mask(source, rect=rect)
    m = mask > 0
    if m.any():                                                # subject's own range -> 0..1
        # Robust percentile bounds so one spurious near/far depth spike inside the
        # subject can't stretch the range and flatten its real relief (see
        # _robust01); a plain min/max here is outlier-sensitive.
        lo, hi = np.percentile(depth[m], [1.0, 99.0])
        depth = np.clip((depth - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    relief = depth * mask
    if symmetry and symmetry != "off":                         # match a subject's sides
        from foldforge.origamize.symmetry import symmetrize
        relief, mask = symmetrize(relief, mask, mode=symmetry)
        m = mask > 0
    ys, xs = np.where(m)
    crop = relief if len(xs) == 0 else relief[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    length = width = float(size)
    if folds is not None:
        grid, length, width = _layout_from_folds(folds, crop.shape, size)
    smooth_eff = 0.0 if style == "origami" else float(smooth) * (max(grid) / 40.0)
    if smooth_eff:
        crop = gaussian_filter(crop, sigma=smooth_eff)
    Z = cv2.resize(crop, (grid[1], grid[0]), interpolation=cv2.INTER_AREA)
    # NB: the depth was already robustly normalised to the subject's own
    # percentile range above (outlier-safe), so a second _robust01 here would be
    # redundant - and worse, it would re-stretch using the crop's percentiles,
    # which include background zeros in the bounding box. Left out on purpose.
    if style == "origami":
        Z = _posterize(Z, levels)
    return (Z, length, width) if return_dims else Z


def origamize_depth(source, grid=(30, 30), height: float = 6.0,
                    length: float = 24.0, width: float = 24.0,
                    smooth: float = 1.0, rect=None,
                    model_type: str = "MiDaS_small", closed: bool = False,
                    close_mode: str = "mirror", close_base: float = 0.0,
                    folds=None, style: str = "smooth", levels: int = 5,
                    engine: str = "corrugation", symmetry: str = "off",
                    foldable=None, rect_sheet: bool = False):
    """Estimate the subject's 3D relief from monocular depth and fold it.

    Mirrors :func:`foldforge.origamize.vision.origamize_silhouette` but uses a
    depth network instead of silhouette inflation, giving genuine estimated
    relief rather than a rounded balloon. Returns ``(result, relief)``: an
    :class:`~foldforge.origamize.surface.OrigamiResult` and the height field.

    ``folds`` is the fold-count knob (cells across the subject's longer side);
    it drives the grid and makes the sheet keep the subject crop's aspect
    instead of a forced square. ``style="origami"`` posterises the relief into
    a few large planar facets with sharp creases; ``"smooth"`` (default) keeps
    the rounded relief.

    Pass ``closed=True`` to also attach a watertight, printable solid at
    ``result.solid = (vertices, triangles)`` by mirroring the relief into a back
    sheet and stitching the rim (see
    :func:`~foldforge.origamize.surface.close_relief`). Because a photo shows one
    side, the back is a mirrored *estimate* - a two-sheet folded model, not
    single-sheet origami. ``close_mode`` is ``"mirror"`` or ``"flat"``.
    """
    from foldforge.origamize.surface import budget_folds
    folds = budget_folds(foldable, folds)
    size = max(float(length), float(width))
    Z, L, W = depth_relief(source, grid=grid, smooth=smooth, rect=rect,
                           model_type=model_type, folds=folds, style=style,
                           levels=levels, size=size, return_dims=True,
                           symmetry=symmetry)
    if folds is None:
        L, W = length, width
    from foldforge.origamize.io import fold_heightfield
    result = fold_heightfield(Z * height, length=L, width=W, engine=engine)
    if closed:
        # The depth path never trims background (the whole sheet is folded relief),
        # so ``rect_sheet`` is accepted for a uniform CLI but is a no-op here: the
        # exported solid already keeps the full rectangular sheet.
        result.solid = close_relief(result, mode=close_mode, base=close_base)
    return result, Z
