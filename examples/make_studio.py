#!/usr/bin/env python3
"""Generate studio/index.html - the self-contained Three.js FoldForge web studio.

The studio embeds a small trajectory database (six built-in shapes, each a
sequence of folding frames) plus a client-side origamizer so a visitor can drop
in their own photo and watch it fold in the browser. Everything is baked into a
single HTML file with Three.js inlined - no build step, no server, works offline.

Run:  python examples/make_studio.py   ->   writes studio/index.html

The six shapes are computed here from the FoldForge core (the same origamizer /
rigid-fold kinematics used everywhere else), so the studio always reflects the
current engine. Each shape is a *fold sweep*: we scale the closed-form fold
angles from ~0 (flat) to 1 (fully folded) and record the intermediate frames the
slider interpolates between.
"""

from __future__ import annotations

import base64
import io
import json
import os

import numpy as np

from foldforge.origamize.surface import (
    heightfield_dome, heightfield_saddle, profile_ridge,
)
from foldforge.design.inverse import resample_arclength, angles_from_curve
from foldforge.diff.kinematics import fold_chain
from foldforge.diff.miura import flat_miura, folded_miura, fold_limit

NF = 22          # frames per fold sweep
F0 = 0.04        # first frame's fold fraction (near-flat, matches the JS uploader)


# --------------------------------------------------------------------------- #
#  height-field targets (relief corrugations)                                 #
# --------------------------------------------------------------------------- #
def smiley_field(nx=20, ny=16, height=6.0):
    """A round face in relief: raised disc with two eye dimples and a smile."""
    x = np.linspace(-1, 1, nx); y = np.linspace(-1, 1, ny)
    Y, X = np.meshgrid(y, x, indexing="ij")
    face = np.clip(1 - (X ** 2 + Y ** 2), 0, None) ** 0.5
    eyeL = np.exp(-(((X + 0.35) ** 2 + (Y - 0.3) ** 2)) / 0.02)
    eyeR = np.exp(-(((X - 0.35) ** 2 + (Y - 0.3) ** 2)) / 0.02)
    mouth = np.exp(-((X ** 2 / 0.6 + (Y + 0.30 + 0.35 * X ** 2) ** 2) / 0.02))
    Z = np.clip(face - 0.9 * (eyeL + eyeR + mouth), 0, None)
    return height * Z / (Z.max() + 1e-9)


def two_peaks_field(nx=20, ny=16, height=6.0):
    """Two Gaussian peaks - a simple analytic function surface."""
    x = np.linspace(-1, 1, nx); y = np.linspace(-1, 1, ny)
    Y, X = np.meshgrid(y, x, indexing="ij")
    Z = (np.exp(-(((X + 0.4) ** 2 + (Y + 0.2) ** 2)) / 0.15)
         + 0.85 * np.exp(-(((X - 0.45) ** 2 + (Y - 0.25) ** 2)) / 0.12))
    return height * Z / (Z.max() + 1e-9)


def build_heightfield_shape(Z, label, length=24.0, width=20.0):
    """Origamize a height field row-by-row and record the fold sweep.

    Mirrors the in-browser image origamizer: each row becomes an exact fold
    chain whose angles trace that row's cross-section; scaling the angles by a
    fraction ``f`` gives a partially folded frame. The final fold's centroid is
    subtracted from every frame so the animation stays centred.
    """
    ny, nx = Z.shape
    xs = np.linspace(0, length, nx)
    ys = np.linspace(0, width, ny)

    rows, tgt = [], []
    for j in range(ny):
        prof = np.stack([xs, Z[j]], axis=1)
        rs = resample_arclength(prof, nx - 1)          # nx points
        ang, seg = angles_from_curve(rs)
        rows.append((ang, seg, rs[0]))     # keep the row's target start to re-anchor
        for i in range(nx):
            tgt.append([rs[i, 0], ys[j], rs[i, 1]])

    def idx(j, i):
        return j * nx + i

    final = []
    for j in range(ny):
        ang, seg, start = rows[j]
        sp = fold_chain(ang, seg=seg).spine
        sp = sp + (start - sp[0])                   # re-anchor to the row's true z (handles negative z)
        for i in range(nx):
            final.append([sp[i, 0], ys[j], sp[i, 1]])
    c = np.asarray(final).mean(axis=0)

    frames = []
    for fi in range(NF):
        f = F0 + (1 - F0) * fi / (NF - 1)
        V = []
        for j in range(ny):
            ang, seg, start = rows[j]
            sp = fold_chain(f * ang, seg=seg).spine
            sp = sp + (start - sp[0])               # keep every frame anchored to the row's z
            for i in range(nx):
                V.append([sp[i, 0] - c[0], ys[j] - c[1], sp[i, 1] - c[2]])
        frames.append(V)

    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            tris.append([idx(j, i), idx(j, i + 1), idx(j + 1, i + 1)])
            tris.append([idx(j, i), idx(j + 1, i + 1), idx(j + 1, i)])

    T = np.asarray(tgt) - c
    segl = []
    for j in range(ny):
        for i in range(nx - 1):
            segl += [*T[idx(j, i)], *T[idx(j, i + 1)]]
    for j in range(ny - 1):
        for i in range(nx):
            segl += [*T[idx(j, i)], *T[idx(j + 1, i)]]

    chain = {"kind": "rows",
             "rows": [{"a": a.tolist(), "s": float(sg), "sx": float(st[0]), "sz": float(st[1])}
                      for (a, sg, st) in rows],
             "ys": [float(v) for v in ys], "nx": int(nx), "ny": int(ny),
             "c": [float(v) for v in c]}
    return dict(frames=frames, triangles=tris, creases=[], target=segl, label=label,
                chain=chain)


# --------------------------------------------------------------------------- #
#  pleated profile (ridge) - carries mountain/valley creases                  #
# --------------------------------------------------------------------------- #
def build_ridge_shape(n_pleats=19, width=20.0, n_rows=8, height=5.0, length=24.0):
    prof = profile_ridge(n=200, length=length, height=height)
    target = resample_arclength(prof, n_pleats)
    angles, seg = angles_from_curve(target)
    n = len(target) - 1
    ys = np.linspace(0, width, n_rows)

    def idx(i, r):
        return i * n_rows + r

    sp1 = fold_chain(angles, seg=seg).spine
    sp1 = sp1 + (target[0] - sp1[0])
    final = np.array([[sp1[i, 0], ys[r], sp1[i, 1]]
                      for i in range(n + 1) for r in range(n_rows)])
    c = final.mean(axis=0)

    frames = []
    for fi in range(NF):
        f = F0 + (1 - F0) * fi / (NF - 1)
        sp = fold_chain(f * angles, seg=seg).spine
        sp = sp + (target[0] - sp[0])
        V = [[sp[i, 0] - c[0], ys[r] - c[1], sp[i, 1] - c[2]]
             for i in range(n + 1) for r in range(n_rows)]
        frames.append(V)

    tris = []
    for i in range(n):
        for r in range(n_rows - 1):
            a, b, cc, d = idx(i, r), idx(i + 1, r), idx(i + 1, r + 1), idx(i, r + 1)
            tris.append([a, b, cc]); tris.append([a, cc, d])

    creases = []
    for i in range(1, n):                              # interior pleat lines
        s = -1 if angles[i] < 0 else 1                 # sign -> mountain/valley
        for r in range(n_rows - 1):
            creases.append([idx(i, r), idx(i, r + 1), s])

    T = np.array([[target[i, 0], ys[r], target[i, 1]]
                  for i in range(n + 1) for r in range(n_rows)]) - c
    segl = []
    for i in range(n):
        for r in range(n_rows):
            segl += [*T[idx(i, r)], *T[idx(i + 1, r)]]

    chain = {"kind": "cols", "a": [float(v) for v in angles], "s": float(seg),
             "sx": float(target[0, 0]), "sz": float(target[0, 1]),
             "ys": [float(v) for v in ys], "np1": int(n + 1), "nrows": int(n_rows),
             "c": [float(v) for v in c]}
    return dict(frames=frames, triangles=tris, creases=creases, target=segl,
                label="Ridge pleats (exact, M/V)", chain=chain)


# --------------------------------------------------------------------------- #
#  Miura-ori tessellation - exact rigid fold, one DOF                         #
# --------------------------------------------------------------------------- #
def build_miura_shape(rows=6, cols=8, a=3.6, b=3.2, gamma_deg=72.0, frac=0.92):
    gamma = np.radians(gamma_deg)
    hmax = frac * fold_limit(a, gamma)         # fold deep enough to read as folded
    R, C = rows + 1, cols + 1

    def idx(i, j):
        return i * C + j

    c = folded_miura(rows, cols, a, b, gamma, h=hmax).reshape(-1, 3).mean(axis=0)

    frames = []
    for fi in range(NF):
        f = fi / (NF - 1)
        h = max(1e-3, f * hmax)
        V = folded_miura(rows, cols, a, b, gamma, h=h).reshape(-1, 3) - c
        frames.append(V.tolist())

    tris = []
    for i in range(rows):
        for j in range(cols):
            a0, b0, c0, d0 = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            tris.append([a0, b0, c0]); tris.append([a0, c0, d0])

    creases = []
    for i in range(R):                                 # b-direction zigzag folds
        for j in range(C - 1):
            creases.append([idx(i, j), idx(i, j + 1), 1 if i % 2 == 0 else -1])
    for i in range(R - 1):                             # a-direction folds
        for j in range(C):
            creases.append([idx(i, j), idx(i + 1, j), -1 if j % 2 == 0 else 1])

    return dict(frames=frames, triangles=tris, creases=creases, target=[],
                label="Miura-ori (tessellation)")


# --------------------------------------------------------------------------- #
def round_shape(sh, nd=4):
    def rnd(x):
        if isinstance(x, float):
            return round(x, nd)
        if isinstance(x, list):
            return [rnd(v) for v in x]
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        return x
    out = {}
    for k, v in sh.items():
        out[k] = rnd(v) if k in ("frames", "target", "chain") else v
    return out


# --------------------------------------------------------------------------- #
#  sample animals - the Python GrabCut relief pipeline baked as studio shapes  #
# --------------------------------------------------------------------------- #
# Each of the four photos in examples/samples/ is segmented with the same
# high-quality OpenCV GrabCut / silhouette-inflation pipeline that make_samples
# uses, then turned into a row-by-row fold chain (identical build path to the
# height-field shapes, so the fold slider and step-by-step mode work) plus a
# per-vertex UV map and the photo as an embedded JPEG texture, and a reduced-res
# segmentation blob so the heatmap panel lights up. Heavy deps (OpenCV/SciPy/PIL)
# are imported lazily inside these functions and the result is cached to
# _animals_baked.json, so build_data() itself stays pure-numpy for the tests.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SAMPLES = os.path.join(_HERE, "samples")
_ANIM_CACHE = os.path.join(_HERE, "_animals_baked.json")

# (key, studio label, GrabCut rect in original px | None for auto-seed,
#  symmetry mode "off"/"auto"/"y", source-image stem)
# The butterfly is baked twice: once raw and once bilaterally symmetrized (same
# photo) so the showcase gallery can present the symmetry before/after demo.
ANIMAL_SPECS = [
    ("cat", "Cat", None, "off", "cat"),
    ("dog", "Dog", None, "off", "dog"),
    ("elephant", "Elephant", (230, 800, 1470, 1500), "off", "elephant"),
    ("butterfly", "Butterfly", None, "off", "butterfly"),
    ("butterfly_sym", "Butterfly (symmetrized)", None, "y", "butterfly"),
]


def _clean_animal_mask(mask):
    """Largest component, morphological open/close, hole-fill.

    GrabCut occasionally leaves thin spikes (a cat's whiskers) or interior specks;
    an opening drops the spikes, a closing seals nicks, and a hole-fill keeps the
    silhouette solid so the relief has no craters.
    """
    import cv2
    from scipy.ndimage import binary_fill_holes
    m = mask.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    k = max(3, int(max(m.shape) * 0.012)) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    return binary_fill_holes(m).astype(np.uint8)


def _seg_blob(mask, hfield, H, W, seg_side=80):
    """Reduced-res mask + normalised height blob for the studio heatmap / pipeline
    strip. ``hfield`` is any per-pixel relief; it is masked, downsampled and
    normalised to [0,1]. Mirrors the packing the JS `ensureSeg` unpacks."""
    import cv2
    ss = seg_side / max(H, W)
    sw, sh = max(8, int(round(W * ss))), max(8, int(round(H * ss)))
    m2 = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
    h2 = cv2.resize((hfield * mask).astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    h2 = h2 / (h2.max() + 1e-9)
    ys2, xs2 = np.where(m2 > 0)
    if xs2.size == 0:
        xs2 = np.array([0]); ys2 = np.array([0])
    return {
        "w": int(sw), "h": int(sh),
        "mask": "".join("1" if v else "0" for v in m2.reshape(-1)),
        "height": [round(float(v), 2) for v in h2.reshape(-1)],
        "frac": round(float(m2.sum()) / (sw * sh), 3),
        "bbox": [int(xs2.min()), int(ys2.min()), int(xs2.max()), int(ys2.max())],
    }


def _bake_one_animal(name, label, rect, sym_mode="off", stem=None,
                     folds=28, seg_side=80, tex_side=384):
    import cv2
    from scipy.ndimage import gaussian_filter
    from PIL import Image
    from foldforge.origamize.vision import (
        _load_bgr, _mask_from_bgr, inflate, _layout_from_folds, _read_source_rgb,
    )
    from foldforge.origamize.symmetry import symmetrize
    path = os.path.join(_SAMPLES, (stem or name) + ".jpg")
    bgr, scale = _load_bgr(path)
    H, W = bgr.shape[:2]
    mask = _clean_animal_mask(_mask_from_bgr(bgr, scale, rect=rect))

    # relief = inflated silhouette modulated by interior shading (edge-preserving)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    base = inflate(mask)                # default round (spherical-cap) profile; power ignored
    shade = cv2.bilateralFilter(gray, 7, 45, 45).astype(float) / 255.0
    relief = base * (0.65 + 0.35 * shade) * mask

    # bilateral symmetrization (butterfly_sym): find the subject's mirror axis and
    # average the relief about it, so the folded model comes out with matching sides.
    raw_mask, raw_base = mask, base
    sym_info = None
    if sym_mode not in (None, "off"):
        relief, mask, sym_info = symmetrize(relief, mask, mode=sym_mode, return_info=True)
        base = relief                  # the symmetric relief IS the height field for the seg blob

    ys, xs = np.where(mask > 0)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    crop = relief[y0:y1 + 1, x0:x1 + 1]

    (nrows, ncols), L, Wd = _layout_from_folds(folds, crop.shape, size=24.0)
    sm = gaussian_filter(crop, sigma=max(ncols, nrows) / 40.0)
    Zg = cv2.resize(sm, (ncols, nrows), interpolation=cv2.INTER_AREA).astype(float)
    Zg -= Zg.min()
    Zg /= (Zg.max() + 1e-9)
    hscale = 0.38 * min(L, Wd)                      # relief scaled to the sheet (no tube-curl)
    shp = build_heightfield_shape(Zg * hscale, label, length=L, width=Wd)

    # the fold chain drives the animation, so keep only flat + fully folded frames
    shp["frames"] = [shp["frames"][0], shp["frames"][-1]]
    shp["target"] = []                              # skip the dense wireframe over a textured photo

    # per-vertex UVs (grid fraction) mapped onto the bbox-cropped photo texture
    uv = []
    for j in range(nrows):
        for i in range(ncols):
            uv += [round(i / (ncols - 1), 4), round(1 - j / (nrows - 1), 4)]
    shp["uv"] = uv
    ox0, oy0 = int(x0 / scale), int(y0 / scale)
    ox1, oy1 = int(round((x1 + 1) / scale)), int(round((y1 + 1) / scale))
    rgb = _read_source_rgb(path)
    tcrop = rgb[oy0:oy1, ox0:ox1]
    im = Image.fromarray(tcrop)
    s = tex_side / max(im.size)
    if s < 1:
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=72, optimize=True)
    shp["tex"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    # reduced-res full-frame segmentation for the heatmap panel / pipeline strip
    shp["seg"] = _seg_blob(mask, base, H, W, seg_side)
    # engine label (shown in the showcase gallery) + symmetry provenance
    shp["engine"] = "OpenCV GrabCut + silhouette inflation"
    if sym_info is not None:
        # a "before symmetry" blob so the pipeline strip can show the before/after pair
        shp["segRaw"] = _seg_blob(raw_mask, raw_base, H, W, seg_side)
        shp["sym"] = {"mode": sym_mode, "axis": sym_info["axis"],
                      "iou": round(float(sym_info["iou"]), 3),
                      "applied": bool(sym_info["applied"])}
        if sym_info["applied"]:
            shp["engine"] += " + bilateral symmetrization"
    shp["mode"] = "subject"
    shp["nx"] = int(ncols)
    shp["ny"] = int(nrows)
    return round_shape(shp)


def bake_animals(force=False):
    """Bake the four sample animals into _animals_baked.json (needs OpenCV/SciPy/PIL).

    Returns the {"shapes":..., "order":...} dict. Cached so a rebuild that only
    touches the HTML shell needn't re-run GrabCut; pass ``force=True`` to refresh.
    """
    shapes, order = {}, []
    for name, label, rect, sym_mode, stem in ANIMAL_SPECS:
        shapes[name] = _bake_one_animal(name, label, rect, sym_mode=sym_mode, stem=stem)
        order.append(name)
    data = {"shapes": shapes, "order": order}
    with open(_ANIM_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    return data


def build_data():
    shapes = {
        "dome":   build_heightfield_shape(heightfield_dome(20, 16), "Dome (relief corrugation)"),
        "image":  build_heightfield_shape(smiley_field(20, 16),     "Image → relief (smiley)"),
        "peak":   build_heightfield_shape(two_peaks_field(20, 16),  "Function surface (two peaks)"),
        "saddle": build_heightfield_shape(heightfield_saddle(20, 16), "Saddle (hypar)"),
        "ridge":  build_ridge_shape(),
        "miura":  build_miura_shape(),
    }
    order = ["dome", "image", "peak", "saddle", "ridge", "miura"]
    shapes = {k: round_shape(v) for k, v in shapes.items()}
    # append the baked sample animals (from cache) after the six geometric shapes
    if os.path.exists(_ANIM_CACHE):
        with open(_ANIM_CACHE, "r", encoding="utf-8") as f:
            anim = json.load(f)
        for k in anim["order"]:
            shapes[k] = anim["shapes"][k]
            order.append(k)
    return {"shapes": shapes, "order": order}


# --------------------------------------------------------------------------- #
#  HTML shell. __THREE__ -> inlined three.min.js, __DATA__ -> the JSON blob.    #
# --------------------------------------------------------------------------- #
HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FoldForge Studio</title>
<style>
 html,body{margin:0;height:100%;font-family:system-ui,Segoe UI,Arial;background:#0d1016;color:#e9e4d7;overflow:hidden}
 #hud{position:fixed;top:14px;left:18px;line-height:1.45}#hud small{color:#97a1b0}
 .legend{position:fixed;top:14px;right:18px;font-size:13px;text-align:right}
 .m{color:#ff6b6b}.v{color:#5b9bd5}.t{color:#39d3c7}
 #ui{position:fixed;left:0;right:0;bottom:0;padding:14px 20px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;
     background:linear-gradient(transparent,rgba(0,0,0,.65))}
 input[type=range]{flex:1;min-width:140px;accent-color:#d9a441}
 select,button{background:#1f2531;color:#e9e4d7;border:1px solid #39414f;border-radius:7px;padding:7px 12px;cursor:pointer;font-size:14px}
 button:hover,select:hover{background:#2a3140}
 #pct{min-width:42px;text-align:right;font-variant-numeric:tabular-nums}
 label.fold{display:flex;gap:9px;align-items:center;flex:1;min-width:200px}
 #cap{position:fixed;left:18px;bottom:66px;font-size:13px;color:#c8cedb}
 #status{position:fixed;left:18px;bottom:92px;max-width:72vw;font-size:13px;padding:6px 11px;border-radius:7px;background:rgba(20,26,36,.9);color:#e9e4d7;display:none}
 #status.err{background:rgba(122,32,32,.94)}#status.ok{background:rgba(30,82,52,.94)}
 body.drag::after{content:"Drop an image or .fold file";position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
   font-size:22px;color:#e9e4d7;background:rgba(13,16,22,.72);border:3px dashed #d9a441;pointer-events:none;z-index:60}
 /* pipeline strip: input photo -> height field -> symmetrized, stacked at right */
 #pipeline{position:fixed;right:18px;top:74px;width:168px;max-height:calc(100vh - 200px);overflow-y:auto;
   display:none;flex-direction:column;gap:8px;z-index:40}
 .pstage{background:rgba(20,26,36,.92);border:1px solid #39414f;border-radius:9px;padding:6px 7px 5px}
 .pstage.hidden{display:none}
 .plabel{font-size:11px;color:#97a1b0;margin:1px 2px 5px}
 .pstage canvas,.pstage img{width:154px;height:auto;max-height:150px;object-fit:contain;display:block;border-radius:5px;background:#0a0d12}
 #heatscale{height:8px;border-radius:4px;margin:5px 2px 1px;
   background:linear-gradient(90deg,#00007f,#0000ff,#00ffff,#00ff00,#ffff00,#ff0000,#7f0000)}
 #heatends{display:flex;justify-content:space-between;font-size:10px;color:#8b95a4;margin:2px 2px 0}
 /* showcase gallery: collapsible panel of baked sample results at left */
 #gallery{position:fixed;left:18px;top:74px;width:252px;max-height:calc(100vh - 200px);overflow-y:auto;
   background:rgba(16,20,28,.96);border:1px solid #39414f;border-radius:11px;padding:10px;display:none;z-index:45}
 #gallery h3{margin:2px 2px 9px;font-size:14px}
 .gcard{display:flex;gap:8px;align-items:center;padding:6px;border:1px solid #2a3140;border-radius:8px;margin-bottom:7px;background:rgba(31,37,49,.6)}
 .gcard img{width:56px;height:56px;object-fit:cover;border-radius:6px;background:#0a0d12;flex:none}
 .gmeta{font-size:11px;line-height:1.35;flex:1;min-width:0}
 .gmeta .geng{color:#8b95a4}.gmeta .gsym{color:#39d3c7}
 button.gload{padding:5px 10px;font-size:12px;flex:none}
 button.on{background:#3a4a2f;border-color:#5a7a3f}
</style></head><body>
<div id="hud"><b>FoldForge Studio</b><br><small>drag rotate &middot; scroll zoom &middot; pick a shape and fold it, or fold your own image</small></div>
<div class="legend"><span class="m">&#9632;</span>mountain <span class="v">&#9632;</span>valley <span class="t">&#9472;</span>target<br><small>exact rigid fold kinematics</small></div>
<div id="cap"></div>
<div id="pipeline">
 <div class="pstage hidden" id="pstage-input"><div class="plabel">1 &middot; input photo</div><img id="pimg" alt="input photo"></div>
 <div class="pstage hidden" id="pstage-height"><div class="plabel" id="heatlabel">2 &middot; height field</div><canvas id="heat" width="160" height="160"></canvas>
  <div id="heatscale"></div><div id="heatends"><span>low</span><span>high</span></div></div>
 <div class="pstage hidden" id="pstage-sym"><div class="plabel" id="symlabel">3 &middot; symmetrized</div><canvas id="heatsym" width="160" height="160"></canvas>
  <div id="heatscale"></div><div id="heatends"><span>low</span><span>high</span></div></div>
</div>
<div id="gallery"><h3>Showcase gallery</h3><div id="gallerylist"></div>
 <div style="font-size:11px;color:#8b95a4;margin:3px 2px">Baked offline with the OpenCV pipeline; the butterfly is shown raw and bilaterally symmetrized for a before/after.</div></div>
<div id="status"></div>
<div id="ui">
 <select id="shape"></select>
 <select id="detail" title="Fold detail for a dropped image: fewer/bigger folds vs. finer relief">
  <option value="10">Detail: Rough</option>
  <option value="16">Detail: Medium</option>
  <option value="24" selected>Detail: Fine</option>
  <option value="36">Detail: Extra</option>
 </select>
 <select id="foldmode" title="Fold every crease together, or one crease after another like instructions">
  <option value="all" selected>Fold: all at once</option>
  <option value="step">Fold: step by step</option>
 </select>
 <select id="shapemode" title="3D subject: segment the subject and fold its inflated silhouette. Brightness: fold the raw luminance of the whole frame.">
  <option value="subject" selected>Shape: 3D subject</option>
  <option value="bright">Shape: brightness relief</option>
 </select>
 <select id="symmode" title="Bilateral symmetry for a folded photo subject. Auto applies only when the mirror overlap (IoU) is at least 0.80; Force always applies; Off keeps the raw silhouette.">
  <option value="off">Symmetry: Off</option>
  <option value="auto" selected>Symmetry: Auto</option>
  <option value="force">Symmetry: Force</option>
 </select>
 <button id="heatBtn" class="on" title="Show/hide the photo pipeline strip (input &rarr; height field &rarr; symmetrized)">&#128200; Pipeline</button>
 <button id="galleryBtn" title="Open the showcase gallery of baked sample results">&#127912; Gallery</button>
 <button id="texBtn" title="Texture the folded sheet with the source photo. Off = blank paper (the reference photo stays in its own panel).">&#128444;&#65039; Photo texture</button>
 <button id="uploadBtn">&#128247; Fold an image</button>
 <input id="imgfile" type="file" accept="image/*" style="display:none">
 <button id="foldBtn">&#128193; Load .fold</button>
 <input id="foldfile" type="file" accept=".fold,application/json" style="display:none">
 <button id="dl">&#11015; OBJ</button>
 <button id="play">&#10073;&#10073; Pause</button>
 <label class="fold">fold <input id="slider" type="range" min="0" max="1" step="0.001" value="0"></label>
 <span id="pct">0%</span>
</div>
<script>__THREE__</script>
<script>
// surface any uncaught error into the status bar (first error only, kept small)
window.onerror=function(msg,src,ln,col,err){var s=document.getElementById('status'); if(s&&s.getAttribute('data-err')!=='1'){s.setAttribute('data-err','1'); s.textContent='Error: '+((err&&err.message)||msg); s.className='err'; s.style.display='block';} return false;};
// offline guard: if Three.js somehow did not load, show a plain-HTML banner (not console-only) and stop
if(typeof THREE==='undefined'){
 document.body.innerHTML='<div style="position:fixed;inset:0;display:flex;align-items:center;justify-content:center;padding:28px;font-family:system-ui,Segoe UI,Arial;font-size:18px;line-height:1.5;color:#e9e4d7;background:#0d1016;text-align:center">The 3D library (Three.js) failed to load, so the studio can&rsquo;t start.<br>This build is meant to be fully self-contained &mdash; re-generate it with examples/three.min.js present.</div>';
 throw new Error('THREE is undefined (three.min.js did not load)');
}
const DATA=__DATA__;
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0d1016);
const camera=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,0.05,2000); camera.up.set(0,0,1);
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(devicePixelRatio); renderer.setSize(innerWidth,innerHeight);
renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff,0.5));
const dl=new THREE.DirectionalLight(0xffffff,0.9); dl.position.set(4,-6,9);
dl.castShadow=true; dl.shadow.mapSize.set(2048,2048); dl.shadow.bias=-0.0008; scene.add(dl);
const dl2=new THREE.DirectionalLight(0x88aaff,0.22); dl2.position.set(-5,4,-3); scene.add(dl2);
// ground plane (receives the model's shadow); normal +z since camera.up is z
const ground=new THREE.Mesh(new THREE.PlaneGeometry(1,1),
  new THREE.MeshStandardMaterial({color:0x141a24,roughness:1,metalness:0}));
ground.receiveShadow=true; scene.add(ground);
const pivot=new THREE.Group(); scene.add(pivot);
let S,frames,V,pos,geo,creaseGeo,cPos,mesh;
// remove every child of the pivot AND free its GPU resources (geometry +
// material) so re-folding / switching shapes doesn't leak buffers each time.
function clearPivot(){
  while(pivot.children.length){
    const c=pivot.children[0]; pivot.remove(c);
    if(c.geometry) c.geometry.dispose();
    if(c.material){ (Array.isArray(c.material)?c.material:[c.material]).forEach(m=>m&&m.dispose()); }
  }
}
// a baked animal ships its texture as a JPEG data-URI and its seg blob as a
// packed mask string + height array; realise both into GPU/typed forms once.
function shapeTexture(S){
  if(!S.tex) return null;
  if(S._tex) return S._tex;
  if(typeof S.tex==='string'){ const im=new Image(); const tx=new THREE.Texture(im);
    tx.colorSpace=THREE.SRGBColorSpace; im.onload=()=>{tx.needsUpdate=true;}; im.src=S.tex; S._tex=tx; return tx; }
  S._tex=S.tex; return S.tex;                          // already a THREE.Texture (dropped-image path)
}
// Folded-sheet material. Default is blank paper; the photo texture is only used
// when the user turns on "Photo texture" (and the shape actually carries a photo).
function paperMaterial(){ return new THREE.MeshStandardMaterial({color:0xece3d0,side:THREE.DoubleSide,roughness:0.82,metalness:0,flatShading:true}); }
function materialFor(S){
  if(photoTex&&S&&S.uv&&S.tex)
    return new THREE.MeshStandardMaterial({map:shapeTexture(S),side:THREE.DoubleSide,roughness:0.85,metalness:0});
  return paperMaterial();
}
// The source photo for the pipeline strip's input stage: a baked animal ships a
// data-URI; a dropped image carries its <img> on the texture.
function photoSrc(S){
  if(S&&typeof S.tex==='string') return S.tex;
  if(S&&S.tex&&S.tex.image) return S.tex.image.src||S.tex.image.currentSrc||null;
  if(S&&S._tex&&S._tex.image) return S._tex.image.src||null;
  return null;
}
// realise a packed seg blob (mask string + height array) into typed arrays once
function ensureSeg(sg){
  if(!sg||sg._ready) return sg;
  if(typeof sg.mask==='string'){ const m=new Uint8Array(sg.w*sg.h);
    for(let i=0;i<m.length;i++) m[i]=(sg.mask.charCodeAt(i)===49)?1:0; sg.mask=m; }
  if(Array.isArray(sg.height)) sg.height=Float64Array.from(sg.height);
  sg._ready=true; return sg;
}
function buildShape(name){
  S=DATA.shapes[name]; frames=S.frames.map(f=>f.flat()); V=frames[0].length; clearPivot();
  geo=new THREE.BufferGeometry(); pos=new Float32Array(V);
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3)); geo.setIndex(S.triangles.flat());
  if(S.uv) geo.setAttribute('uv',new THREE.BufferAttribute(new Float32Array(S.uv),2)); // keep UVs so the photo-texture toggle can turn on any time
  mesh=new THREE.Mesh(geo,materialFor(S)); mesh.castShadow=true; mesh.receiveShadow=true; pivot.add(mesh);
  if(S.creases&&S.creases.length){
    creaseGeo=new THREE.BufferGeometry(); cPos=new Float32Array(S.creases.length*6); const cCol=new Float32Array(S.creases.length*6);
    S.creases.forEach((c,k)=>{const col=c[2]<0?[1,0.42,0.42]:[0.36,0.61,0.84]; for(let e=0;e<2;e++)cCol.set(col,(k*2+e)*3);});
    creaseGeo.setAttribute('position',new THREE.BufferAttribute(cPos,3)); creaseGeo.setAttribute('color',new THREE.BufferAttribute(cCol,3));
    pivot.add(new THREE.LineSegments(creaseGeo,new THREE.LineBasicMaterial({vertexColors:true})));
  } else creaseGeo=null;
  if(S.target&&S.target.length){
    const tg=new THREE.BufferGeometry(); tg.setAttribute('position',new THREE.BufferAttribute(new Float32Array(S.target),3));
    pivot.add(new THREE.LineSegments(tg,new THREE.LineBasicMaterial({color:0x39d3c7,transparent:true,opacity:0.32})));
  }
  let r=0; const last=frames[frames.length-1];
  for(let i=0;i<V;i+=3){const dd=Math.hypot(last[i],last[i+1],last[i+2]); if(dd>r)r=dd;}
  if(r<1e-3)r=1;
  camera.position.set(0,-r*2.4,r*1.7); camera.lookAt(0,0,0);
  // sit the ground just below the model and size the shadow camera to fit
  ground.position.set(0,0,-r*1.15); ground.scale.set(r*8,r*8,1);
  dl.position.set(r*0.7,-r*1.1,r*2.2);
  const sc=dl.shadow.camera; sc.left=-r*3; sc.right=r*3; sc.top=r*3; sc.bottom=-r*3;
  sc.near=0.05; sc.far=r*40; sc.updateProjectionMatrix();
  document.getElementById('cap').textContent=S.label||'';
  foldmode.disabled=!(S.chain||S.dynamic);   // step-by-step needs closed-form chain or the live PBD solver
  if(S.dynamic){ S.x=Float64Array.from(S.solver.x0); S.frac=0; S.seeded=false; }
  drawPipeline(S);                                    // input photo -> height field -> symmetrized (baked animals carry a seg blob)
  // photo-texture toggle only applies to shapes that actually carry a photo;
  // grey it out and clear its "on" state for textureless shapes (dome, .fold, ...)
  const canTex=!!(S&&S.tex&&S.uv);
  texBtn.disabled=!canTex;
  texBtn.classList.toggle('on', canTex&&photoTex);
}
// smoothstep activation for crease k of K under global fraction t (step-by-step folding)
function stepFrac(k,K,t){ if(K<=1)return t; const w=1/K,u=(t-k*w)/w,c=Math.min(1,Math.max(0,u)); return c*c*(3-2*c); }
// recompute a chain-backed shape (image / height-field / ridge) client-side, all-at-once or crease-by-crease
function computeChain(t,step){
  const ch=S.chain;
  if(ch.kind==='cols'){
    const ang=ch.a,seg=ch.s,ys=ch.ys,nrows=ch.nrows,np1=ch.np1,K=ang.length,cx=ch.c[0],cy=ch.c[1],cz=ch.c[2];
    const spx=new Float64Array(np1),spz=new Float64Array(np1); let h=0,x=ch.sx,z=ch.sz; spx[0]=x; spz[0]=z;
    for(let k=0;k<ang.length;k++){const f=step?stepFrac(k,K,t):t; h+=f*ang[k]; x+=seg*Math.cos(h); z+=seg*Math.sin(h); spx[k+1]=x; spz[k+1]=z;}
    let p=0; for(let i=0;i<np1;i++)for(let r=0;r<nrows;r++){pos[p++]=spx[i]-cx; pos[p++]=ys[r]-cy; pos[p++]=spz[i]-cz;}
  } else {
    const rows=ch.rows,ys=ch.ys,nx=ch.nx,ny=ch.ny,K=nx-1,cx=ch.c[0],cy=ch.c[1],cz=ch.c[2]; let p=0;
    for(let j=0;j<ny;j++){const r=rows[j],ang=r.a,seg=r.s; let h=0,x=r.sx,z=r.sz;
      pos[p++]=x-cx; pos[p++]=ys[j]-cy; pos[p++]=z-cz;
      for(let k=0;k<ang.length;k++){const f=step?stepFrac(k,K,t):t; h+=f*ang[k]; x+=seg*Math.cos(h); z+=seg*Math.sin(h);
        pos[p++]=x-cx; pos[p++]=ys[j]-cy; pos[p++]=z-cz;}}
  }
  geo.attributes.position.needsUpdate=true; geo.computeVertexNormals();
  if(creaseGeo){S.creases.forEach((c,kk)=>{for(let d=0;d<3;d++){cPos[kk*6+d]=pos[c[0]*3+d];cPos[kk*6+3+d]=pos[c[1]*3+d];}}); creaseGeo.attributes.position.needsUpdate=true;}
}
function setFrame(t){
  if(S.dynamic){ foldDynamic(t); return; }
  if(S.chain){ computeChain(t,stepMode); return; }
  const x=t*(frames.length-1),i=Math.floor(x),f=x-i,j=Math.min(i+1,frames.length-1);
  const A=frames[i],B=frames[j];
  for(let k=0;k<V;k++) pos[k]=A[k]*(1-f)+B[k]*f;
  geo.attributes.position.needsUpdate=true; geo.computeVertexNormals();
  if(creaseGeo){S.creases.forEach((c,kk)=>{for(let d=0;d<3;d++){cPos[kk*6+d]=pos[c[0]*3+d];cPos[kk*6+3+d]=pos[c[1]*3+d];}}); creaseGeo.attributes.position.needsUpdate=true;}
}
// ----- client-side origamizer: fold an uploaded image (closed form) -----
function imageHeightfield(img,nx,ny,height,levels){
  const cv=document.createElement('canvas'); cv.width=nx; cv.height=ny;
  const ctx=cv.getContext('2d');
  ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,nx,ny);                 // fill white: transparent PNG pixels read white, not black
  const iw=img.naturalWidth||img.width, ih=img.naturalHeight||img.height;
  const car=nx/ny, iar=iw/ih; let sx=0,sy=0,sw=iw,sh=ih;           // center-crop source to grid aspect (no squashing)
  if(iar>car){sw=ih*car; sx=(iw-sw)/2;} else {sh=iw/car; sy=(ih-sh)/2;}
  ctx.drawImage(img,sx,sy,sw,sh,0,0,nx,ny);
  const d=ctx.getImageData(0,0,nx,ny).data; const Z=[]; let mn=1e9,mx=-1e9;
  for(let j=0;j<ny;j++){const row=[]; for(let i=0;i<nx;i++){const k=(j*nx+i)*4;
    const lum=(d[k]*0.299+d[k+1]*0.587+d[k+2]*0.114)/255; row.push(lum); if(lum<mn)mn=lum; if(lum>mx)mx=lum;} Z.push(row);}
  for(let j=0;j<ny;j++)for(let i=0;i<nx;i++) Z[j][i]=(Z[j][i]-mn)/(mx-mn+1e-9)*height;
  if(levels&&levels>1){const L=levels-1; for(let j=0;j<ny;j++)for(let i=0;i<nx;i++) Z[j][i]=Math.round(Z[j][i]/height*L)/L*height;}  // posterize to big flat facets (rough mode)
  return Z;
}
function resampleProfile(xs,zs,npl){
  const pts=xs.map((x,i)=>[x,zs[i]]); const cum=[0];
  for(let i=1;i<pts.length;i++) cum.push(cum[i-1]+Math.hypot(pts[i][0]-pts[i-1][0],pts[i][1]-pts[i-1][1]));
  const total=cum[cum.length-1],out=[];
  for(let k=0;k<=npl;k++){const s=total*k/npl; let i=1; while(i<cum.length&&cum[i]<s)i++; if(i>=cum.length)i=cum.length-1;
    const t=(s-cum[i-1])/(cum[i]-cum[i-1]+1e-9);
    out.push([pts[i-1][0]+t*(pts[i][0]-pts[i-1][0]),pts[i-1][1]+t*(pts[i][1]-pts[i-1][1])]);}
  return out;
}
function anglesFromCurve(curve){
  const head=[]; let seg=0;
  for(let i=1;i<curve.length;i++){head.push(Math.atan2(curve[i][1]-curve[i-1][1],curve[i][0]-curve[i-1][0]));
    seg+=Math.hypot(curve[i][0]-curve[i-1][0],curve[i][1]-curve[i-1][1]);}
  const ang=[head[0]];
  for(let i=1;i<head.length;i++){let a=head[i]-head[i-1]; a=((a+Math.PI)%(2*Math.PI)+2*Math.PI)%(2*Math.PI)-Math.PI; ang.push(a);}
  return {angles:ang, seg:seg/(curve.length-1)};
}
function foldChain(angles,seg,f){let h=0,x=0,z=0; const sp=[[0,0]];
  for(let k=0;k<angles.length;k++){h+=f*angles[k]; x+=seg*Math.cos(h); z+=seg*Math.sin(h); sp.push([x,z]);} return sp;}
// ----- in-browser subject segmentation + silhouette inflation (3D subject mode) -----
function segmentSubject(img){
  const iw=img.naturalWidth||img.width, ih=img.naturalHeight||img.height;
  const sc=144/Math.max(iw,ih), w=Math.max(24,Math.round(iw*sc)), h=Math.max(24,Math.round(ih*sc)), N=w*h;
  const cv=document.createElement('canvas'); cv.width=w; cv.height=h;
  const ctx=cv.getContext('2d'); ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,w,h); ctx.drawImage(img,0,0,w,h);
  const d=ctx.getImageData(0,0,w,h).data;
  const lum=new Float64Array(N);
  for(let p=0;p<N;p++) lum[p]=(d[p*4]*0.299+d[p*4+1]*0.587+d[p*4+2]*0.114)/255;
  // background model: 2-cluster k-means over the border ring (handles sky+ground)
  const ring=[]; for(let x=0;x<w;x++){ring.push(x,(h-1)*w+x);} for(let y=1;y<h-1;y++){ring.push(y*w,y*w+w-1);}
  const px=p=>[d[p*4],d[p*4+1],d[p*4+2]];
  let c0=px(ring[0]), c1=px(ring[ring.length>>1]);
  const d2=(a,b)=>{const r=a[0]-b[0],g=a[1]-b[1],bl=a[2]-b[2]; return r*r+g*g+bl*bl;};
  for(let it=0;it<6;it++){
    const s0=[0,0,0],s1=[0,0,0]; let n0=0,n1=0;
    for(const p of ring){const c=px(p); if(d2(c,c0)<=d2(c,c1)){s0[0]+=c[0];s0[1]+=c[1];s0[2]+=c[2];n0++;} else {s1[0]+=c[0];s1[1]+=c[1];s1[2]+=c[2];n1++;}}
    if(n0)c0=[s0[0]/n0,s0[1]/n0,s0[2]/n0]; if(n1)c1=[s1[0]/n1,s1[1]/n1,s1[2]/n1];
  }
  // ring distance stats -> threshold
  let mu=0; const rd=ring.map(p=>Math.sqrt(Math.min(d2(px(p),c0),d2(px(p),c1))));
  rd.forEach(v=>mu+=v); mu/=rd.length;
  let sd=0; rd.forEach(v=>sd+=(v-mu)*(v-mu)); sd=Math.sqrt(sd/rd.length);
  const qx=new Int32Array(N);
  // build a cleaned largest-component mask at a given colour-distance threshold
  function maskAt(thr){
    let mask=new Uint8Array(N);
    for(let p=0;p<N;p++){ if(Math.sqrt(Math.min(d2(px(p),c0),d2(px(p),c1)))>thr) mask[p]=1; }
    // clean: two majority-vote passes (smoother, less jagged edges)
    for(let pass=0;pass<2;pass++){
      const m2=new Uint8Array(N);
      for(let y=0;y<h;y++)for(let x=0;x<w;x++){let s=0,n=0;
        for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const yy=y+dy,xx=x+dx;
          if(yy<0||xx<0||yy>=h||xx>=w)continue; n++; s+=mask[yy*w+xx];}
        m2[y*w+x]=(s*2>n)?1:0;}
      mask=m2;
    }
    // kill horizon/ground bands: a mask row spanning almost the full frame is background scenery
    for(let y=0;y<h;y++){let c=0; for(let x=0;x<w;x++)c+=mask[y*w+x];
      if(c>0.82*w) for(let x=0;x<w;x++)mask[y*w+x]=0;}
    // largest connected component (BFS)
    const label=new Int32Array(N).fill(-1); let best=-1,bestN=0,nlab=0;
    for(let seed=0;seed<N;seed++){
      if(!mask[seed]||label[seed]>=0)continue;
      let head=0,tail=0; qx[tail++]=seed; label[seed]=nlab; let cnt=0;
      while(head<tail){const p=qx[head++]; cnt++;
        const y=(p/w)|0,x=p%w;
        if(x>0&&mask[p-1]&&label[p-1]<0){label[p-1]=nlab;qx[tail++]=p-1;}
        if(x<w-1&&mask[p+1]&&label[p+1]<0){label[p+1]=nlab;qx[tail++]=p+1;}
        if(y>0&&mask[p-w]&&label[p-w]<0){label[p-w]=nlab;qx[tail++]=p-w;}
        if(y<h-1&&mask[p+w]&&label[p+w]<0){label[p+w]=nlab;qx[tail++]=p+w;}}
      if(cnt>bestN){bestN=cnt;best=nlab;} nlab++;
    }
    const out=new Uint8Array(N);
    for(let p=0;p<N;p++) out[p]=(mask[p]&&label[p]===best)?1:0;
    return {mask:out, frac:bestN/N};
  }
  // adaptive threshold: retry a couple of times to land the subject in 4%-65%
  // of the frame (too small -> lower the threshold to include more; too big -> raise it)
  const thr0=Math.max(26, mu+2*sd);
  let r=maskAt(thr0), thr=thr0;
  for(let it=0; it<3 && !(r.frac>=0.04 && r.frac<=0.65); it++){
    thr *= (r.frac<0.04)?0.7:1.4; r=maskAt(thr);
  }
  let mask=r.mask, frac=r.frac;
  if(frac<0.02){ // segmentation failed -> centred ellipse fallback (mirrors the Python fallback)
    mask=new Uint8Array(N);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){const u=(x-w/2)/(w*0.38),v=(y-h/2)/(h*0.38);
      if(u*u+v*v<=1)mask[y*w+x]=1;}
    frac=0.454;
  }
  // fill interior holes: flood-fill the background inward from the border through
  // non-mask pixels; any non-mask pixel never reached is enclosed -> promote it to
  // subject (holes would otherwise fold into deep craters).
  { const bg=new Uint8Array(N); let sp=0;
    for(let x=0;x<w;x++){ if(!mask[x]&&!bg[x]){bg[x]=1;qx[sp++]=x;} const b=(h-1)*w+x; if(!mask[b]&&!bg[b]){bg[b]=1;qx[sp++]=b;} }
    for(let y=0;y<h;y++){ const l=y*w; if(!mask[l]&&!bg[l]){bg[l]=1;qx[sp++]=l;} const rr=y*w+w-1; if(!mask[rr]&&!bg[rr]){bg[rr]=1;qx[sp++]=rr;} }
    while(sp){ const p=qx[--sp], y=(p/w)|0, x=p%w;
      if(x>0&&!mask[p-1]&&!bg[p-1]){bg[p-1]=1;qx[sp++]=p-1;}
      if(x<w-1&&!mask[p+1]&&!bg[p+1]){bg[p+1]=1;qx[sp++]=p+1;}
      if(y>0&&!mask[p-w]&&!bg[p-w]){bg[p-w]=1;qx[sp++]=p-w;}
      if(y<h-1&&!mask[p+w]&&!bg[p+w]){bg[p+w]=1;qx[sp++]=p+w;} }
    for(let p=0;p<N;p++) if(!mask[p]&&!bg[p]) mask[p]=1;
  }
  // chamfer distance transform (3-4 weights, two passes)
  const INF=1e9, dt=new Float64Array(N);
  for(let p=0;p<N;p++) dt[p]=mask[p]?INF:0;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){const p=y*w+x; if(!dt[p])continue;
    let m=dt[p];
    if(x>0)m=Math.min(m,dt[p-1]+3); if(y>0)m=Math.min(m,dt[p-w]+3);
    if(x>0&&y>0)m=Math.min(m,dt[p-w-1]+4); if(x<w-1&&y>0)m=Math.min(m,dt[p-w+1]+4);
    dt[p]=m;}
  for(let y=h-1;y>=0;y--)for(let x=w-1;x>=0;x--){const p=y*w+x; if(!mask[p])continue;
    let m=dt[p];
    if(x<w-1)m=Math.min(m,dt[p+1]+3); if(y<h-1)m=Math.min(m,dt[p+w]+3);
    if(x<w-1&&y<h-1)m=Math.min(m,dt[p+w+1]+4); if(x>0&&y<h-1)m=Math.min(m,dt[p+w-1]+4);
    dt[p]=m;}
  let dmax=0; for(let p=0;p<N;p++) if(mask[p]&&dt[p]>dmax)dmax=dt[p];
  const height=new Float64Array(N);
  for(let p=0;p<N;p++) if(mask[p]){const u=dt[p]/(dmax||1); height[p]=Math.sqrt(Math.max(0,u*(2-u)));} // spherical-cap inflation: a true rounded balloon (matches the Python inflate; ~14x closer to a hemisphere than the old smoothstep puff)
  // Luminance surface-detail is no longer baked into the field here (it used to
  // carve the zebra's stripes into ridges); it is re-applied, tamed and only in
  // smooth mode, at sample time in subjectHeightfield.
  // separable 1-2-1 blur (masked) -> a clean field to bilinear-sample from; the
  // crisp `height` is kept for posterized/rough detail.
  const heightS=new Float64Array(N), tmp=new Float64Array(N);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){const p=y*w+x; if(!mask[p]){tmp[p]=0;continue;}
    let s=2*height[p],wt=2; if(x>0&&mask[p-1]){s+=height[p-1];wt++;} if(x<w-1&&mask[p+1]){s+=height[p+1];wt++;} tmp[p]=s/wt;}
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){const p=y*w+x; if(!mask[p]){heightS[p]=0;continue;}
    let s=2*tmp[p],wt=2; if(y>0&&mask[p-w]){s+=tmp[p-w];wt++;} if(y<h-1&&mask[p+w]){s+=tmp[p+w];wt++;} heightS[p]=s/wt;}
  // bbox of the mask
  let x0=w,y0=h,x1=0,y1=0;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++) if(mask[y*w+x]){if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y;}
  return {w:w,h:h,mask:mask,height:height,heightS:heightS,lum:lum,frac:frac,bbox:[x0,y0,x1,y1]};
}
// bilinear sample of seg.height over the mask bbox -> ny x nx fold grid
function subjectHeightfield(seg,nx,ny,hscale,levels){
  const [bx0,by0,bx1,by1]=seg.bbox, bw=Math.max(1,bx1-bx0), bh=Math.max(1,by1-by0), Z=[];
  const rough=(levels&&levels>1);
  const HF=rough?seg.height:(seg.heightS||seg.height);   // crisp field for rough/posterize, blurred otherwise
  const modul=(!rough&&seg.lum)?0.06:0;                  // tamed +-6% luminance detail (off in rough mode)
  for(let j=0;j<ny;j++){const row=[];
    for(let i=0;i<nx;i++){
      const gx=bx0+(i/(nx-1))*bw, gy=by0+(j/(ny-1))*bh;
      const x0=Math.min(seg.w-1,Math.floor(gx)),y0=Math.min(seg.h-1,Math.floor(gy)),x1=Math.min(seg.w-1,x0+1),y1=Math.min(seg.h-1,y0+1);
      const fx=gx-x0,fy=gy-y0,S=(x,y)=>HF[y*seg.w+x];
      let v=((S(x0,y0)*(1-fx)+S(x1,y0)*fx)*(1-fy)+(S(x0,y1)*(1-fx)+S(x1,y1)*fx)*fy);
      if(modul){const L=(x,y)=>seg.lum[y*seg.w+x];
        const lu=((L(x0,y0)*(1-fx)+L(x1,y0)*fx)*(1-fy)+(L(x0,y1)*(1-fx)+L(x1,y1)*fx)*fy);
        v*=1+modul*(1-2*lu);}                            // darker regions ride slightly higher
      row.push(v*hscale);
    } Z.push(row);}
  if(nx>12){ // light 3x3 box smooth so the corrugation is clean (skip in rough/posterized mode)
    const Z2=Z.map(r=>r.slice());
    for(let j=1;j<ny-1;j++)for(let i=1;i<nx-1;i++){let s=0;
      for(let dj=-1;dj<=1;dj++)for(let di=-1;di<=1;di++)s+=Z[j+dj][i+di];
      Z2[j][i]=s/9;} 
    for(let j=0;j<ny;j++)for(let i=0;i<nx;i++)Z[j][i]=Z2[j][i];
  }
  if(levels&&levels>1){let mx=0; for(let j=0;j<ny;j++)for(let i=0;i<nx;i++)if(Z[j][i]>mx)mx=Z[j][i];
    const L=levels-1; if(mx>0) for(let j=0;j<ny;j++)for(let i=0;i<nx;i++) Z[j][i]=Math.round(Z[j][i]/mx*L)/L*mx;}
  return Z;
}
// ----- in-browser bilateral symmetrization (a modest port of origamize/symmetry.py) -----
// nearest-neighbour rotation of a w*h field by `deg` about its centre (for the small
// camera-tilt search); out-of-frame samples read 0.
function _rotNN(src,w,h,deg){ if(Math.abs(deg)<1e-6) return src;
  const r=deg*Math.PI/180,ca=Math.cos(r),sa=Math.sin(r),cx=(w-1)/2,cy=(h-1)/2,out=new Float64Array(w*h);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){const dx=x-cx,dy=y-cy;
    const sx=Math.round(cx+dx*ca-dy*sa),sy=Math.round(cy+dx*sa+dy*ca);
    if(sx>=0&&sx<w&&sy>=0&&sy<h) out[y*w+x]=src[sy*w+sx];}
  return out;
}
// mask mirror-IoU about vertical line x=c (reflect columns): the self-overlap score
function _mirrorIoU(mask,w,h,c){ let inter=0,uni=0;
  for(let y=0;y<h;y++){const row=y*w;
    for(let x=0;x<w;x++){const a=mask[row+x]?1:0,mx=2*c-x,b=(mx>=0&&mx<w&&mask[row+mx])?1:0;
      if(a|b)uni++; if(a&b)inter++;}}
  return uni?inter/uni:0;
}
// best vertical mirror over a small rotation + offset grid -> {iou, angle, frac}
function measureSym(seg){ const w=seg.w,h=seg.h; let best={iou:-1,angle:0,frac:0.5};
  const angles=[-10,-6,-3,0,3,6,10];
  for(const ang of angles){ const m=_rotNN(seg.mask,w,h,ang);
    let sx=0,n=0; for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(m[y*w+x]){sx+=x;n++;}
    if(!n)continue; const c0=Math.round(sx/n),off=Math.max(3,Math.round(0.08*w));
    for(let c=Math.max(0,c0-off);c<=Math.min(w-1,c0+off);c++){const iou=_mirrorIoU(m,w,h,c);
      if(iou>best.iou)best={iou:iou,angle:ang,frac:c/w};}
  }
  if(best.iou<0)best.iou=0; return best;
}
// rotate the fields to align the mirror to vertical, then average with the mirror
// (sym = (A+mirror(A))/2), keeping the relief inside the symmetric silhouette
function applySym(seg,info){ const w=seg.w,h=seg.h,N=w*h;
  const mask=_rotNN(seg.mask,w,h,info.angle),height=_rotNN(seg.height,w,h,info.angle),
        heightS=_rotNN(seg.heightS||seg.height,w,h,info.angle),lum=_rotNN(seg.lum||seg.height,w,h,info.angle);
  const c=Math.round(info.frac*w);
  const om=new Uint8Array(N),oh=new Float64Array(N),ohs=new Float64Array(N),ol=new Float64Array(N);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){const p=y*w+x,mx=2*c-x;
    if(mx>=0&&mx<w){const q=y*w+mx;
      om[p]=(mask[p]||mask[q])?1:0; oh[p]=0.5*(height[p]+height[q]);
      ohs[p]=0.5*(heightS[p]+heightS[q]); ol[p]=0.5*(lum[p]+lum[q]);}
    else{om[p]=mask[p]?1:0; oh[p]=height[p]; ohs[p]=heightS[p]; ol[p]=lum[p];}
  }
  for(let p=0;p<N;p++)if(!om[p]){oh[p]=0;ohs[p]=0;}
  let x0=w,y0=h,x1=0,y1=0,frac=0;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(om[y*w+x]){frac++;if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y;}
  if(frac===0){x0=0;y0=0;x1=w-1;y1=h-1;}
  return {w:w,h:h,mask:om,height:oh,heightS:ohs,lum:ol,frac:frac/N,bbox:[x0,y0,x1,y1],_ready:true};
}
function buildImageShape(img,nx,ny,mode,seg){
  nx=nx||24; ny=ny||18; mode=mode||'brightness';
  const height=6,length=24,width=length*ny/nx,NF=20;   // square cells so the fold keeps the subject's aspect
  const posterize=(nx<=12)?5:0;                         // rough detail -> few big folded facets
  const subj=(mode==='subject'&&seg);
  const hs=subj?0.38*Math.min(length,width):height;   // relief scaled to the sheet so full fold doesn't curl into a tube
  const Z=subj?subjectHeightfield(seg,nx,ny,hs,posterize):imageHeightfield(img,nx,ny,height,posterize);
  const xs=[]; for(let i=0;i<nx;i++) xs.push(length*i/(nx-1));
  const ys=[]; for(let j=0;j<ny;j++) ys.push(width*j/(ny-1));
  const rows=[]; const tgt=[];
  for(let j=0;j<ny;j++){const rs=resampleProfile(xs,Z[j],nx-1); const ac=anglesFromCurve(rs); rows.push(ac);
    for(let i=0;i<nx;i++) tgt.push([rs[i][0],ys[j],rs[i][1]]);}
  const F1=[]; for(let j=0;j<ny;j++){const sp=foldChain(rows[j].angles,rows[j].seg,1); for(let i=0;i<nx;i++) F1.push([sp[i][0],ys[j],sp[i][1]]);}
  let cx=0,cy=0,cz=0; F1.forEach(p=>{cx+=p[0];cy+=p[1];cz+=p[2];}); cx/=F1.length;cy/=F1.length;cz/=F1.length;
  const frames=[];
  for(let fi=0;fi<NF;fi++){const f=0.04+(1-0.04)*fi/(NF-1); const V=[];
    for(let j=0;j<ny;j++){const sp=foldChain(rows[j].angles,rows[j].seg,f); for(let i=0;i<nx;i++) V.push([sp[i][0]-cx,ys[j]-cy,sp[i][1]-cz]);} frames.push(V);}
  const idx=(j,i)=>j*nx+i; const tris=[];
  for(let j=0;j<ny-1;j++)for(let i=0;i<nx-1;i++) tris.push([idx(j,i),idx(j,i+1),idx(j+1,i+1)],[idx(j,i),idx(j+1,i+1),idx(j+1,i)]);
  const segl=[]; const T=tgt.map(p=>[p[0]-cx,p[1]-cy,p[2]-cz]);
  for(let j=0;j<ny;j++)for(let i=0;i<nx-1;i++) segl.push(...T[idx(j,i)],...T[idx(j,i+1)]);
  for(let j=0;j<ny-1;j++)for(let i=0;i<nx;i++) segl.push(...T[idx(j,i)],...T[idx(j+1,i)]);
  // per-vertex UVs (grid) + the source photo as a texture, so the fold is photoreal
  const uv=[];
  if(subj){const [bx0,by0,bx1,by1]=seg.bbox;
    for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){
      uv.push((bx0+(i/(nx-1))*(bx1-bx0))/(seg.w-1),1-(by0+(j/(ny-1))*(by1-by0))/(seg.h-1));}}
  else for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){uv.push(i/(nx-1),1-j/(ny-1));}
  const tex=new THREE.Texture(img); tex.needsUpdate=true; tex.colorSpace=THREE.SRGBColorSpace;
  const chain={kind:'rows',rows:rows.map(r=>({a:r.angles,s:r.seg,sx:0,sz:0})),ys:ys,nx:nx,ny:ny,c:[cx,cy,cz]};
  return {frames:frames,triangles:tris,creases:[],target:segl,uv:uv,tex:tex,chain:chain,nx:nx,ny:ny,seg:seg||null,mode:mode,label:subj?'Your image (3D subject)':'Your image (brightness relief)'};
}
// ===== embedded PBD rigid-origami solver (folds loaded .fold patterns live) =====
function foldModelFromFold(fold){
  const vc=fold.vertices_coords||[], ev=fold.edges_vertices||[],
        ea=fold.edges_assignment||[], efa=fold.edges_foldAngle||null, fv=fold.faces_vertices||[];
  let mnx=1e9,mny=1e9,mxx=-1e9,mxy=-1e9;
  vc.forEach(p=>{if(p[0]<mnx)mnx=p[0];if(p[1]<mny)mny=p[1];if(p[0]>mxx)mxx=p[0];if(p[1]>mxy)mxy=p[1];});
  const sc=20/Math.max(mxx-mnx,mxy-mny,1e-6), cx=(mnx+mxx)/2, cy=(mny+mxy)/2;
  const nv=vc.length, x0=new Float64Array(nv*3);
  for(let i=0;i<nv;i++){x0[i*3]=(vc[i][0]-cx)*sc;x0[i*3+1]=(vc[i][1]-cy)*sc;x0[i*3+2]=(vc[i].length>2?vc[i][2]:0)*sc;}
  const ekey=(a,b)=>a<b?a+","+b:b+","+a, edgeInfo={};
  ev.forEach((e,k)=>{edgeInfo[ekey(e[0],e[1])]={assign:(ea[k]||"").toUpperCase(),fold:(efa&&efa[k]!=null)?efa[k]:null};});
  const tris=[], edgeTris={};
  const addET=(a,b,ti,apex)=>{const k=ekey(a,b);(edgeTris[k]||(edgeTris[k]=[])).push([ti,apex]);};
  fv.forEach(f=>{const v0=f[0];
    for(let k=1;k<f.length-1;k++){const a=f[k],b=f[k+1],ti=tris.length;
      tris.push([v0,a,b]); addET(v0,a,ti,b); addET(a,b,ti,v0); addET(b,v0,ti,a);}});
  const bars=Object.keys(edgeTris).map(k=>k.split(",").map(Number));
  const rest=new Float64Array(bars.length);
  bars.forEach((e,i)=>{const a=e[0],b=e[1];rest[i]=Math.hypot(x0[a*3]-x0[b*3],x0[a*3+1]-x0[b*3+1],x0[a*3+2]-x0[b*3+2]);});
  const DEF=0.95*Math.PI, hinges=[], creases=[], bias=new Float64Array(nv);
  Object.keys(edgeTris).forEach(k=>{const pair=edgeTris[k]; if(pair.length!==2)return;
    const ij=k.split(",").map(Number), info=edgeInfo[k]||{assign:"",fold:null};
    const apexA=pair[0][1], apexB=pair[1][1];
    let s=0; if(info.assign==="M")s=-1; else if(info.assign==="V")s=1;
    if(info.assign==="B")return;
    if(s!==0){const tgt=(info.fold!=null)?info.fold*Math.PI/180:s*DEF;
      hinges.push({e:ij,w:[apexA,apexB],target:tgt,crease:true});
      creases.push([ij[0],ij[1],s]);
      bias[apexA]+=Math.sign(tgt); bias[apexB]+=Math.sign(tgt);
    } else hinges.push({e:ij,w:[apexA,apexB],target:0.0,crease:false});
  });
  return {nv,x0,bars,rest,hinges,tris,creases,bias};
}
const _G=new Float64Array(12);
function _dihedral(x,e,w,G){
  const i1=e[0]*3,i2=e[1]*3,i3=w[0]*3,i4=w[1]*3;
  const p1x=x[i1],p1y=x[i1+1],p1z=x[i1+2];
  const ex=x[i2]-p1x,ey=x[i2+1]-p1y,ez=x[i2+2]-p1z;
  const el=Math.hypot(ex,ey,ez)||1e-12, ehx=ex/el,ehy=ey/el,ehz=ez/el;
  const ax=x[i3]-p1x,ay=x[i3+1]-p1y,az=x[i3+2]-p1z;
  const bx=x[i4]-p1x,by=x[i4+1]-p1y,bz=x[i4+2]-p1z;
  let c1x=ey*az-ez*ay,c1y=ez*ax-ex*az,c1z=ex*ay-ey*ax;
  let c2x=by*ez-bz*ey,c2y=bz*ex-bx*ez,c2z=bx*ey-by*ex;
  const a1=Math.hypot(c1x,c1y,c1z)||1e-12, a2=Math.hypot(c2x,c2y,c2z)||1e-12;
  const n1x=c1x/a1,n1y=c1y/a1,n1z=c1z/a1, n2x=c2x/a2,n2y=c2y/a2,n2z=c2z/a2;
  const crx=n1y*n2z-n1z*n2y,cry=n1z*n2x-n1x*n2z,crz=n1x*n2y-n1y*n2x;
  const theta=Math.atan2(crx*ehx+cry*ehy+crz*ehz, n1x*n2x+n1y*n2y+n1z*n2z);
  if(G){const f3=(ax*ex+ay*ey+az*ez)/(el*el), f4=(bx*ex+by*ey+bz*ez)/(el*el);
    const h1=a1/el, h2=a2/el;
    const g3x=n1x/h1,g3y=n1y/h1,g3z=n1z/h1, g4x=n2x/h2,g4y=n2y/h2,g4z=n2z/h2;
    G[6]=g3x;G[7]=g3y;G[8]=g3z; G[9]=g4x;G[10]=g4y;G[11]=g4z;
    G[0]=-(1-f3)*g3x-(1-f4)*g4x;G[1]=-(1-f3)*g3y-(1-f4)*g4y;G[2]=-(1-f3)*g3z-(1-f4)*g4z;
    G[3]=-f3*g3x-f4*g4x;G[4]=-f3*g3y-f4*g4y;G[5]=-f3*g3z-f4*g4z;}
  return theta;
}
function _projHinge(x,e,w,target,k){
  const th=_dihedral(x,e,w,_G);
  let C=th-target; while(C>Math.PI)C-=2*Math.PI; while(C<-Math.PI)C+=2*Math.PI;
  const idx=[e[0],e[1],w[0],w[1]];
  let denom=0; for(let j=0;j<4;j++){const a=_G[j*3],b=_G[j*3+1],c=_G[j*3+2];denom+=a*a+b*b+c*c;}
  if(denom<1e-12)return; const lam=k*C/denom;
  for(let j=0;j<4;j++){const p=idx[j]*3;x[p]+=lam*_G[j*3];x[p+1]+=lam*_G[j*3+1];x[p+2]+=lam*_G[j*3+2];}
}
function relaxFold(m,x,fraction,iters,opts){
  opts=opts||{}; const projIters=opts.projIters==null?5:opts.projIters,
    kbend=opts.kbend==null?0.4:opts.kbend, kflat=opts.kflat==null?0.7:opts.kflat, hf=opts.hf||null;
  const bars=m.bars, rest=m.rest, hinges=m.hinges;
  for(let it=0;it<iters;it++){
    for(let hi=0;hi<hinges.length;hi++){const h=hinges[hi];
      if(h.crease)_projHinge(x,h.e,h.w,(hf?hf[hi]:fraction)*h.target,kbend); else _projHinge(x,h.e,h.w,0,kflat);}
    for(let p=0;p<projIters;p++){for(let bi=0;bi<bars.length;bi++){
      const a=bars[bi][0],b=bars[bi][1];
      const dx=x[b*3]-x[a*3],dy=x[b*3+1]-x[a*3+1],dz=x[b*3+2]-x[a*3+2];
      const L=Math.hypot(dx,dy,dz)||1e-12,c=0.5*(L-rest[bi])/L;
      const cx=c*dx,cy=c*dy,cz=c*dz;
      x[a*3]+=cx;x[a*3+1]+=cy;x[a*3+2]+=cz; x[b*3]-=cx;x[b*3+1]-=cy;x[b*3+2]-=cz;}}
  }
}
const FOPTS={projIters:5,kbend:0.4,kflat:0.7};
// ----- load a .fold crease pattern; fold it live with the PBD solver -----
function buildFoldShape(fold){
  const m=foldModelFromFold(fold);
  const flat=[]; for(let i=0;i<m.nv;i++) flat.push(m.x0[i*3],m.x0[i*3+1],m.x0[i*3+2]);
  const tris=m.tris.map(t=>[t[0],t[1],t[2]]);
  const creases=m.creases.map(c=>[c[0],c[1],c[2]]);
  const segl=[]; (fold.edges_vertices||[]).forEach(e=>{segl.push(m.x0[e[0]*3],m.x0[e[0]*3+1],m.x0[e[0]*3+2],m.x0[e[1]*3],m.x0[e[1]*3+1],m.x0[e[1]*3+2]);});
  return {frames:[flat],triangles:tris,creases:creases,target:segl,dynamic:true,solver:m,
          label:((fold.metadata&&fold.metadata.name)||fold.name||'Your .fold')+' — folding live'};
}
// warm-started live fold: drive applied fraction toward the slider value across frames
function foldDynamic(t){
  const m=S.solver, x=S.x;
  if(!S.seeded && (t>0.0005||S.frac>0)){ for(let i=0;i<m.nv;i++) x[i*3+2]+=0.05*m.bias[i]+0.01*Math.sin(i*12.9898); S.seeded=true; }
  let d=t-S.frac, md=0.03; if(d>md)d=md; else if(d<-md)d=-md; S.frac+=d;
  if(S.frac<0)S.frac=0; else if(S.frac>1)S.frac=1;
  let hf=null;
  if(stepMode){ let K=0; for(let hi=0;hi<m.hinges.length;hi++) if(m.hinges[hi].crease)K++;
    hf=new Array(m.hinges.length); let ci=0;
    for(let hi=0;hi<m.hinges.length;hi++){ if(m.hinges[hi].crease){hf[hi]=stepFrac(ci,K,S.frac); ci++;} else hf[hi]=0; } }
  const opt={projIters:5,kbend:0.4,kflat:0.7,hf:hf};
  relaxFold(m,x,S.frac,26,opt);
  if(Math.abs(t-S.frac)<1e-4) relaxFold(m,x,S.frac,18,opt); // extra settle iterations at target
  for(let k=0;k<V;k++) pos[k]=x[k];
  geo.attributes.position.needsUpdate=true; geo.computeVertexNormals();
  if(creaseGeo){S.creases.forEach((c,kk)=>{for(let dd=0;dd<3;dd++){cPos[kk*6+dd]=pos[c[0]*3+dd];cPos[kk*6+3+dd]=pos[c[1]*3+dd];}}); creaseGeo.attributes.position.needsUpdate=true;}
}
let rx=-1.0,rz=0.5,down=false,px=0,py=0; const el=renderer.domElement;
el.addEventListener('pointerdown',e=>{down=true;px=e.clientX;py=e.clientY});
addEventListener('pointerup',()=>down=false);
addEventListener('pointermove',e=>{if(!down)return;rz+=(e.clientX-px)*0.01;rx+=(e.clientY-py)*0.01;px=e.clientX;py=e.clientY;});
addEventListener('wheel',e=>{camera.position.multiplyScalar(1+Math.sign(e.deltaY)*0.08);},{passive:true});
let playing=true,t=0,dir=1,stepMode=false,lastImg=null,detailNx=24,shapeMode='subject',heatOn=true,photoTex=false,symMode='auto';
const slider=document.getElementById('slider'),pct=document.getElementById('pct'),play=document.getElementById('play'),sel=document.getElementById('shape');
const detail=document.getElementById('detail'),foldmode=document.getElementById('foldmode');
foldmode.addEventListener('change',()=>{stepMode=(foldmode.value==='step');});
detail.addEventListener('change',()=>{detailNx=parseInt(detail.value)||24; rebuildImage();});
const shapemode=document.getElementById('shapemode');
shapemode.addEventListener('change',()=>{shapeMode=shapemode.value; rebuildImage();});
const heatBtn=document.getElementById('heatBtn'),pipeline=document.getElementById('pipeline'),
      heatCanvas=document.getElementById('heat'),heatLabel=document.getElementById('heatlabel'),
      heatSymCanvas=document.getElementById('heatsym'),symLabel=document.getElementById('symlabel'),
      pimg=document.getElementById('pimg'),texBtn=document.getElementById('texBtn');
const stageInput=document.getElementById('pstage-input'),stageHeight=document.getElementById('pstage-height'),stageSym=document.getElementById('pstage-sym');
const symmode=document.getElementById('symmode');
symmode.addEventListener('change',()=>{symMode=symmode.value; rebuildImage();});
const galleryBtn=document.getElementById('galleryBtn'),galleryPanel=document.getElementById('gallery');
galleryBtn.addEventListener('click',()=>{const open=galleryPanel.style.display!=='block';
  galleryPanel.style.display=open?'block':'none'; galleryBtn.classList.toggle('on',open);});
heatBtn.addEventListener('click',()=>{heatOn=!heatOn; heatBtn.classList.toggle('on',heatOn); drawPipeline(S);});
// Photo-texture toggle: swap the current sheet between blank paper and the photo,
// no rebuild needed (UVs are already on the geometry).
texBtn.addEventListener('click',()=>{photoTex=!photoTex; texBtn.classList.toggle('on',photoTex);
  if(mesh){ const m=mesh.material; mesh.material=materialFor(S); if(m)m.dispose(); }});
function _heatColor(v){v=Math.min(1,Math.max(0,v));
  const S=[[0,0,127],[0,0,255],[0,255,255],[0,255,0],[255,255,0],[255,0,0],[127,0,0]];
  const u=v*(S.length-1),i=Math.floor(u),f=u-i,a=S[i],b=S[Math.min(S.length-1,i+1)];
  return [a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f];}
// paint one seg blob's height field (or brightness) as a jet heatmap on a canvas
function renderHeat(canvas,seg,subj){
  const CW=160,CH=160,ctx=canvas.getContext('2d'); canvas.width=CW;canvas.height=CH;
  ctx.fillStyle='#0a0d12'; ctx.fillRect(0,0,CW,CH);
  const w=seg.w,h=seg.h,sc=Math.min(CW/w,CH/h),dw=Math.max(1,Math.round(w*sc)),dh=Math.max(1,Math.round(h*sc)),
        ox=Math.round((CW-dw)/2),oy=Math.round((CH-dh)/2),im=ctx.createImageData(dw,dh);
  for(let y=0;y<dh;y++)for(let x=0;x<dw;x++){const sx=Math.min(w-1,Math.floor(x/sc)),sy=Math.min(h-1,Math.floor(y/sc)),sp=sy*w+sx,o=(y*dw+x)*4;
    if(subj&&!seg.mask[sp]){im.data[o]=10;im.data[o+1]=13;im.data[o+2]=18;im.data[o+3]=255;continue;}
    const val=subj?seg.height[sp]:(seg.lum?seg.lum[sp]:seg.height[sp]),c=_heatColor(val);
    im.data[o]=c[0];im.data[o+1]=c[1];im.data[o+2]=c[2];im.data[o+3]=255;}
  ctx.putImageData(im,ox,oy);
  if(subj){ctx.fillStyle='rgba(255,255,255,0.9)';
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){const p=y*w+x; if(!seg.mask[p])continue;
      const edge=(x===0||y===0||x===w-1||y===h-1||!seg.mask[p-1]||!seg.mask[p+1]||!seg.mask[p-w]||!seg.mask[p+w]);
      if(edge)ctx.fillRect(ox+x*sc,oy+y*sc,Math.max(1,sc),Math.max(1,sc));}}
}
// pipeline strip: input photo -> height field -> (if symmetry applied) symmetrized
function drawPipeline(shape){
  if(!heatOn||!shape){pipeline.style.display='none';return;}
  const seg=shape.seg; if(!seg){pipeline.style.display='none';return;}
  ensureSeg(seg); if(shape.segRaw) ensureSeg(shape.segRaw);
  const subj=(shape.mode==='subject'), applied=!!(shape.sym&&shape.sym.applied);
  // stage 1: input photo
  const src=photoSrc(shape);
  if(src){ pimg.src=src; stageInput.classList.remove('hidden'); }
  else { pimg.removeAttribute('src'); stageInput.classList.add('hidden'); }
  // stage 2: height field (the raw, pre-symmetry field when we have it)
  const base=(applied&&shape.segRaw)?shape.segRaw:seg;
  renderHeat(heatCanvas,base,subj);
  heatLabel.textContent=applied?'2 · height field (raw)'
    :(subj?('2 · height field · '+Math.round(seg.frac*100)+'% subject'):'2 · brightness relief');
  stageHeight.classList.remove('hidden');
  // stage 3: symmetrized height field
  if(applied){ renderHeat(heatSymCanvas,seg,true);
    symLabel.textContent='3 · symmetrized'+(shape.sym.iou!=null?(' · mirror-IoU '+shape.sym.iou.toFixed(2)):'');
    stageSym.classList.remove('hidden'); }
  else stageSym.classList.add('hidden');
  pipeline.style.display='flex';
}
function computeNy(img,nx){const iw=img.naturalWidth||img.width,ih=img.naturalHeight||img.height; return Math.max(6,Math.round(nx*ih/iw));}
function rebuildImage(){ if(!lastImg)return;
  if(!lastImg._seg) lastImg._seg=segmentSubject(lastImg);
  const rawSeg=lastImg._seg, nx=detailNx;
  // bilateral symmetry (subject mode): measure the mirror overlap, apply per the mode
  let seg=rawSeg, segRaw=null, sym=null;
  if(shapeMode==='subject' && symMode!=='off'){
    const meas=measureSym(rawSeg);
    const applied=(symMode==='force')||(meas.iou>=0.80);      // 0.80 = the Python auto threshold
    sym={iou:meas.iou,angle:meas.angle,applied:applied,mode:symMode};
    if(applied){ segRaw=rawSeg; seg=applySym(rawSeg,meas); }
  }
  let ny;
  if(shapeMode==='subject'){const bw=seg.bbox[2]-seg.bbox[0]+1,bh=seg.bbox[3]-seg.bbox[1]+1; ny=Math.max(6,Math.round(nx*bh/Math.max(1,bw)));}
  else ny=computeNy(lastImg,nx);
  const shape=buildImageShape(lastImg,nx,ny,shapeMode,seg);
  shape.segRaw=segRaw; shape.sym=sym; shape.engine='in-browser k-means + chamfer-DT inflation';
  const prevUp=DATA.shapes['__up'];        // dispose the previous fold's texture (overwrite leak)
  if(prevUp){ if(prevUp.tex&&prevUp.tex.dispose)prevUp.tex.dispose();
    if(prevUp._tex&&prevUp._tex!==prevUp.tex&&prevUp._tex.dispose)prevUp._tex.dispose(); }
  DATA.shapes['__up']=shape;
  if(!DATA.order.includes('__up')){DATA.order.push('__up'); const o=document.createElement('option');o.value='__up';o.textContent=shape.label;sel.appendChild(o);}
  else{const ex=[...sel.options].find(o=>o.value==='__up'); if(ex)ex.textContent=shape.label;}
  sel.value='__up'; buildShape('__up'); t=0;dir=1;playing=true;play.innerHTML='&#10073;&#10073; Pause';
  const nm=lastImg._name||'image', iw=lastImg.naturalWidth||lastImg.width, ih=lastImg.naturalHeight||lastImg.height;
  const info=(shapeMode==='subject')?('subject '+Math.round(rawSeg.frac*100)+'% of frame'):'brightness relief';
  let smsg='';
  if(sym) smsg=' · mirror-IoU '+sym.iou.toFixed(2)+(sym.applied?' (symmetrized)':' (< 0.80, left as-is)');
  setStatus('Folded '+nm+' — '+info+', '+iw+'×'+ih+', '+(nx-1)+' folds'+smsg,'ok');
}
DATA.order.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=DATA.shapes[n].label||n;sel.appendChild(o);});
// showcase gallery: one card per baked sample (marked by an `engine` label),
// each with a downscaled thumbnail + a Load button that builds the baked model
function loadBaked(n){ sel.value=n; buildShape(n); t=0;dir=1;playing=true;play.innerHTML='&#10073;&#10073; Pause'; }
function buildGallery(){
  const list=document.getElementById('gallerylist');
  DATA.order.forEach(n=>{const Sn=DATA.shapes[n]; if(!Sn||!Sn.engine)return;
    const card=document.createElement('div'); card.className='gcard';
    const im=document.createElement('img'); if(typeof Sn.tex==='string')im.src=Sn.tex; im.alt=Sn.label||n;
    const meta=document.createElement('div'); meta.className='gmeta';
    let html='<b>'+(Sn.label||n)+'</b><br><span class="geng">'+Sn.engine+'</span>';
    if(Sn.sym&&Sn.sym.iou!=null) html+='<br><span class="gsym">mirror-IoU '+Sn.sym.iou.toFixed(2)+(Sn.sym.applied?' · symmetrized':'')+'</span>';
    meta.innerHTML=html;
    const btn=document.createElement('button'); btn.className='gload'; btn.textContent='Load';
    btn.addEventListener('click',()=>loadBaked(n));
    card.appendChild(im); card.appendChild(meta); card.appendChild(btn); list.appendChild(card);
  });
}
buildGallery();
sel.addEventListener('change',()=>{buildShape(sel.value);t=0;dir=1;playing=true;play.innerHTML='&#10073;&#10073; Pause';});
slider.addEventListener('input',()=>{playing=false;play.innerHTML='&#9654; Play';t=parseFloat(slider.value);});
play.addEventListener('click',()=>{playing=!playing;play.innerHTML=playing?'&#10073;&#10073; Pause':'&#9654; Play';});
const statusEl=document.getElementById('status');
function setStatus(msg,kind){ if(!msg){statusEl.style.display='none';return;} statusEl.textContent=msg; statusEl.className=kind||''; statusEl.style.display='block'; }
// load an image File -> fold it; a visible status message on both success and failure (undecodable formats no longer fail silently)
function foldImageFile(file){
  if(!file)return; setStatus('Reading '+file.name+' \u2026');
  const url=URL.createObjectURL(file), img=new Image();
  img.onload=()=>{ URL.revokeObjectURL(url);
    try{ lastImg=img; img._name=file.name; rebuildImage(); }
    catch(err){ setStatus('Could not fold '+file.name+': '+err.message,'err'); }
  };
  img.onerror=()=>{ URL.revokeObjectURL(url);
    setStatus("Couldn't read "+file.name+" \u2014 browsers can't decode this format (iPhone HEIC/HEIF and TIFF aren't supported). Re-export it as JPG or PNG.",'err'); };
  img.src=url;
}
const fileInput=document.getElementById('imgfile');
document.getElementById('uploadBtn').addEventListener('click',()=>fileInput.click());
fileInput.addEventListener('change',e=>{foldImageFile(e.target.files[0]); fileInput.value='';});
// load .fold
function foldFoldFile(file){
  if(!file)return; setStatus('Reading '+file.name+' \u2026');
  const rd=new FileReader(); rd.onload=()=>{ try{const fold=JSON.parse(rd.result); const shape=buildFoldShape(fold);
    DATA.shapes['__fold']=shape;
    if(!DATA.order.includes('__fold')){DATA.order.push('__fold'); const o=document.createElement('option');o.value='__fold';o.textContent=shape.label;sel.appendChild(o);}
    else{const ex=[...sel.options].find(o=>o.value==='__fold'); if(ex)ex.textContent=shape.label;}
    sel.value='__fold'; buildShape('__fold'); t=0;dir=1;playing=false;play.innerHTML='&#9654; Play'; slider.value=0;
    setStatus('Loaded '+file.name,'ok');
  }catch(err){ setStatus('Could not parse '+file.name+': '+err.message,'err'); } };
  rd.onerror=()=>setStatus("Couldn't read "+file.name,'err');
  rd.readAsText(file);
}
const foldInput=document.getElementById('foldfile');
document.getElementById('foldBtn').addEventListener('click',()=>foldInput.click());
foldInput.addEventListener('change',e=>{foldFoldFile(e.target.files[0]); foldInput.value='';});
// drop-anywhere: preventDefault on EVERY dragover/drop so a file dropped anywhere on the
// page can never make the tab navigate to the image; route by extension to the existing handlers.
window.addEventListener('dragenter',e=>{e.preventDefault(); if(e.dataTransfer)e.dataTransfer.dropEffect='copy'; document.body.classList.add('drag');});
window.addEventListener('dragover',e=>{e.preventDefault(); if(e.dataTransfer)e.dataTransfer.dropEffect='copy'; document.body.classList.add('drag');});
window.addEventListener('dragleave',e=>{ if(e.relatedTarget===null) document.body.classList.remove('drag'); });
window.addEventListener('drop',e=>{e.preventDefault(); document.body.classList.remove('drag');
  const file=e.dataTransfer&&e.dataTransfer.files&&e.dataTransfer.files[0]; if(!file)return;
  if(/\.fold$/i.test(file.name)) foldFoldFile(file); else foldImageFile(file);
});
// download the current (interpolated) frame as an OBJ mesh
function currentOBJ(){
  let s='# FoldForge Studio export\no '+(sel.value||'shape')+'\n';
  for(let i=0;i<V;i+=3) s+='v '+pos[i].toFixed(5)+' '+pos[i+1].toFixed(5)+' '+pos[i+2].toFixed(5)+'\n';
  S.triangles.forEach(f=>{ s+='f '+(f[0]+1)+' '+(f[1]+1)+' '+(f[2]+1)+'\n'; });
  return s;
}
document.getElementById('dl').addEventListener('click',()=>{
  const blob=new Blob([currentOBJ()],{type:'text/plain'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download=(sel.value||'foldforge')+'.obj'; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(()=>URL.revokeObjectURL(a.href),1000);
});
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
// ready self-test: exercise the fold pipeline once (build the first baked shape and fold it),
// surfacing any failure visibly instead of leaving the page dead.
try{
  buildShape(DATA.order[0]);
  setFrame(0.5); setFrame(0);
  setStatus('Ready \u2014 drop a photo anywhere, or pick a shape','ok');
}catch(err){ setStatus('Startup self-test failed: '+((err&&err.message)||err),'err'); throw err; }
function loop(){requestAnimationFrame(loop);
  if(playing){t+=dir*0.006; if(t>=1){t=1;dir=-1} if(t<=0){t=0;dir=1} slider.value=t;}
  pct.textContent=Math.round(t*100)+'%'; setFrame(t); pivot.rotation.set(rx,0,rz); renderer.render(scene,camera);}
loop();
</script></body></html>"""


THREE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "three.min.js")


def load_three():
    """Return the minified Three.js source to inline. Fail loudly if it is missing
    (never silently fall back to a CDN - the whole point is an offline single file)."""
    if not os.path.exists(THREE_PATH):
        raise SystemExit(
            "ERROR: examples/three.min.js not found - cannot build a self-contained studio.\n"
            "The studio inlines Three.js so it works offline. Fetch the exact r128 build:\n"
            "  curl -L -o examples/three.min.js "
            "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js\n"
            "then re-run: python examples/make_studio.py"
        )
    with open(THREE_PATH, "r", encoding="utf-8") as f:
        src = f.read()
    # be robust even though r128's minified build contains no literal </script>
    return src.replace("</script>", "<\\/script>")


def main():
    try:
        bake_animals()                              # refresh the sample-animal cache (needs OpenCV)
    except Exception as e:                          # keep the six geometric shapes working regardless
        print("sample-animal bake skipped:", e)
    data = build_data()
    blob = json.dumps(data, separators=(",", ":"))
    three_src = load_three()
    html = HTML.replace("__THREE__", three_src).replace("__DATA__", blob)
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, os.pardir, "studio", "index.html")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", out, "(%.2f MB)" % (len(html) / 1e6))
    print("shapes:", ", ".join(data["order"]))


if __name__ == "__main__":
    main()
