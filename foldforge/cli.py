"""FoldForge command line: check and render crease patterns.

    foldforge fold   cat.jpg cat.stl            # photo -> folded 3D mesh (one command)
    foldforge check  path/to/pattern.fold      # report foldability
    foldforge render path/to/pattern.fold out.png
    foldforge gen    miura examples/miura.fold  # write a built-in example

Kept deliberately small; it is a thin wrapper over the library so the package
"feels like a real tool" without inventing features nothing uses yet.
"""

from __future__ import annotations

import argparse

from foldforge.geometry import examples
from foldforge.geometry.fold_io import read_fold, write_fold
from foldforge.geometry.foldability import foldability_report


def _check(args) -> None:
    pattern = read_fold(args.path)
    report = foldability_report(pattern)
    print(pattern)
    print(report.summary())


def _render(args) -> None:
    import matplotlib.pyplot as plt
    from foldforge.geometry.render import render_pattern

    pattern = read_fold(args.path)
    render_pattern(pattern, show_vertices=args.vertices)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


def _gen(args) -> None:
    if args.name not in examples.GENERATORS:
        raise SystemExit(
            f"unknown example {args.name!r}; choose from "
            f"{', '.join(examples.GENERATORS)}"
        )
    write_fold(examples.GENERATORS[args.name](), args.out)
    print(f"wrote {args.out}")


def _origamize(args) -> None:
    """Decompose an image into a foldable crease pattern (writes a FOLD file).

    With ``--silhouette`` it segments the subject and inflates its shape
    (great for animals/objects on a background); ``--depth`` instead estimates
    genuine 3D relief with a monocular depth model (needs torch); otherwise
    brightness = height.
    """
    if args.depth:
        from foldforge.origamize import origamize_depth
        result, _ = origamize_depth(args.image, grid=(args.rows, args.cols),
                                    model_type=args.depth_model, engine=args.engine,
                                    symmetry=args.symmetry, foldable=args.foldable)
    elif args.silhouette:
        from foldforge.origamize import origamize_silhouette
        result, _ = origamize_silhouette(args.image, grid=(args.rows, args.cols),
                                         engine=args.engine, symmetry=args.symmetry,
                                         foldable=args.foldable)
    else:
        from foldforge.origamize import origamize_image
        result = origamize_image(args.image, grid=(args.rows, args.cols),
                                 invert=args.invert, engine=args.engine,
                                 foldable=args.foldable)
    write_fold(result.pattern, args.out)
    print(f"wrote {args.out}  (match error {result.error:.3f})")
    print(f"  crease pattern : {result.crease_count} folds, {result.difficulty}")
    if args.engine == "miura2d" and hasattr(result, "max_strain"):
        print(f"  fold strain: mean {result.mean_strain * 100:.1f}%  "
              f"max {result.max_strain * 100:.1f}%")
        print("  note: relief pattern, not flat-foldable (cannot collapse flat)")


def _fold(args) -> None:
    """One command: turn a photo into a folded 3D mesh.

    Auto-segments the subject (edge-density-seeded GrabCut, no hand-tuned rect),
    estimates its relief (silhouette inflation, or ``--depth`` for a monocular
    depth model), folds it, and by default closes it into a watertight solid.
    The output format follows the file extension (.stl / .glb / .obj). Errors
    for missing or undecodable files are printed plainly, without a traceback.
    """
    import os

    ext = os.path.splitext(args.out)[1].lower()
    if ext not in (".stl", ".glb", ".gltf", ".obj"):
        raise SystemExit(
            f"unsupported output '{args.out}': use a .stl, .glb, or .obj extension")
    from foldforge.origamize import budget_folds
    closed = not args.open_sheet
    detail_map = {"rough": 12, "medium": 24, "fine": 40}
    folds = args.folds
    if args.detail:
        folds = detail_map[args.detail]
    if folds is None:                    # back-compat: --grid still sets resolution
        folds = args.grid
    if args.foldable:                    # hand-fold budget overrides all of the above
        folds = budget_folds(args.foldable, folds)
    style = args.style
    try:
        if args.depth:
            from foldforge.origamize import origamize_depth
            result, relief = origamize_depth(args.image, folds=folds, style=style,
                                             closed=closed, model_type=args.depth_model,
                                             engine=args.engine, symmetry=args.symmetry,
                                             rect_sheet=args.rect_sheet)
            how = f"depth ({args.depth_model})"
        else:
            from foldforge.origamize import origamize_silhouette
            result, relief = origamize_silhouette(args.image, folds=folds,
                                                  style=style, closed=closed,
                                                  engine=args.engine,
                                                  symmetry=args.symmetry,
                                                  rect_sheet=args.rect_sheet)
            how = "silhouette"
    except FileNotFoundError as exc:
        raise SystemExit(f"error: {exc}")
    except ValueError as exc:
        raise SystemExit(f"error: could not read '{args.image}': {exc}")
    except ImportError as exc:                           # e.g. --depth without torch
        raise SystemExit(f"error: {exc}")

    from foldforge.fabricate import to_stl, to_gltf, to_obj
    if closed and result.solid is not None:
        vertices, faces = result.solid
        kind = "closed solid (watertight)"
    else:
        vertices, faces = result.folded, result.triangles
        kind = "open sheet"
    if ext == ".stl":
        to_stl(vertices, faces, args.out)
    elif ext in (".glb", ".gltf"):
        to_gltf(vertices, faces, args.out)
    else:
        to_obj(vertices, faces, args.out)

    import numpy as np
    coverage = float((np.asarray(relief) > 1e-6).mean())
    size_kb = os.path.getsize(args.out) / 1024.0
    print(f"Folded {args.image} -> {args.out}  ({how}, {args.engine} engine, "
          f"{style} style, folds={folds})")
    print(f"  subject coverage : {coverage * 100:4.0f}% of the sheet  (auto-segmented)")
    print(f"  fold match error : {result.error:.3f}")
    if args.engine == "miura2d" and hasattr(result, "max_strain"):
        print(f"  fold strain      : mean {result.mean_strain * 100:.1f}%  "
              f"max {result.max_strain * 100:.1f}%  "
              f"(relief pattern, not flat-foldable)")
    print(f"  crease pattern   : {result.crease_count} folds, {result.difficulty} "
          f"(hand-fold difficulty)")
    print(f"  {kind}: {len(faces)} triangles, {size_kb:.0f} KB")


def _design(args) -> None:
    """Design a figurative uniaxial base from a built-in stick-figure tree.

    Circle-packs the tree's flaps (TreeMaker-lite), fills the Delaunay
    triangulation with rabbit-ear molecules, and writes the crease pattern as a
    layered SVG (M/V colours) plus a matching .fold next to it.
    """
    from foldforge.design import get_tree, design_base, flap_length_errors
    from foldforge.geometry.foldability import foldability_report
    from foldforge.fabricate import to_svg

    tree = get_tree(args.tree)
    packing, pattern = design_base(tree)
    if args.out.lower().endswith(".fold"):
        write_fold(pattern, args.out)
    else:
        to_svg(pattern, args.out)
    print(f"wrote {args.out}")
    err = flap_length_errors(packing, pattern)
    report = foldability_report(pattern)
    n_int = len(report.vertices)
    n_kaw = sum(1 for v in report.vertices if v.kawasaki)
    n_mae = sum(1 for v in report.vertices if v.maekawa)
    print(f"  packing scale     : {packing.scale:.4f}")
    print(f"  max flap error    : {err.max() * 100:.2f}%")
    print(f"  Kawasaki          : {n_kaw}/{n_int} interior vertices pass")
    print(f"  Maekawa           : {n_mae}/{n_int} interior vertices pass")


def _instructions(args) -> None:
    """Write a printable step-by-step folding instruction sheet (SVG).

    The input is either a ``.fold`` crease pattern or a photo: a photo is first
    turned into a hand-foldable crease pattern (silhouette relief, ``--foldable``
    budget, default ``easy``) so the instructions stay short. Each numbered panel
    highlights the creases folded so far in mountain/valley colours.
    """
    import os
    from foldforge.fabricate import fold_instructions_svg

    ext = os.path.splitext(args.pattern)[1].lower()
    if ext == ".fold":
        pattern = read_fold(args.pattern)
    else:                                                # treat as a photo
        from foldforge.origamize import origamize_silhouette
        result, _ = origamize_silhouette(args.pattern, foldable=args.foldable,
                                         engine=args.engine)
        pattern = result.pattern
    steps = fold_instructions_svg(pattern, args.out)
    print(f"wrote {args.out}  ({steps} fold steps)")


def _export(args) -> None:
    """Export a FOLD pattern to a layered SVG or DXF for a cutter."""
    from foldforge.fabricate import to_svg, to_dxf
    pattern = read_fold(args.path)
    if args.out.lower().endswith(".dxf"):
        to_dxf(pattern, args.out)
    else:
        to_svg(pattern, args.out, outline_only=args.outline)
    print(f"wrote {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="foldforge", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="report a pattern's foldability")
    p.add_argument("path")
    p.set_defaults(func=_check)

    p = sub.add_parser("render", help="render a pattern to an image")
    p.add_argument("path")
    p.add_argument("out")
    p.add_argument("--vertices", action="store_true", help="also draw vertices")
    p.set_defaults(func=_render)

    p = sub.add_parser("gen", help="write a built-in example .fold file")
    p.add_argument("name", help=", ".join(examples.GENERATORS))
    p.add_argument("out")
    p.set_defaults(func=_gen)

    p = sub.add_parser("origamize", help="fold an image into a crease pattern")
    p.add_argument("image", help="path to an image file")
    p.add_argument("out", help="output .fold path")
    p.add_argument("--rows", type=int, default=18)
    p.add_argument("--cols", type=int, default=24)
    p.add_argument("--invert", action="store_true", help="dark = high (line art)")
    p.add_argument("--silhouette", action="store_true",
                   help="estimate the subject's shape (segment + inflate)")
    p.add_argument("--depth", action="store_true",
                   help="estimate real 3D relief with a monocular depth model (needs torch)")
    p.add_argument("--depth-model",
                   choices=["MiDaS_small", "DPT_Hybrid",
                            "depth_anything_v2_small", "depth_anything_v2_base"],
                   default="MiDaS_small",
                   help="depth network for --depth. DPT_Hybrid is sharper but "
                        "heavier and needs timm; depth_anything_v2_small/_base use "
                        "Depth Anything V2 (needs transformers) and are often "
                        "sharper on fine detail; default MiDaS_small")
    p.add_argument("--engine", choices=["corrugation", "miura2d"],
                   default="corrugation",
                   help="corrugation (1D pleated strips, default) or miura2d "
                        "(true 2D warped-Miura tessellation, far better on curves)")
    p.add_argument("--foldable", choices=["easy", "medium", "hard"], default=None,
                   help="hand-fold budget: easy/medium cap the crease count so a "
                        "human can fold it (tens of creases); hard = full detail")
    p.add_argument("--symmetry", choices=["off", "auto", "x", "y"], default="off",
                   help="mirror-symmetrize the subject (silhouette/depth): auto "
                        "detects the axis, x/y force top-bottom / left-right "
                        "(a butterfly's wings match); default off")
    p.set_defaults(func=_origamize)

    p = sub.add_parser("fold", help="photo -> folded 3D mesh in one command")
    p.add_argument("image", help="path to a photo (jpg/png/...)")
    p.add_argument("out", help="output mesh: .stl, .glb, or .obj")
    p.add_argument("--grid", type=int, default=40, help="fold resolution (cells/side)")
    p.add_argument("--folds", type=int, default=None,
                   help="fold count across the subject's longer side "
                        "(fewer = rougher/simpler; default ~40). Overrides --grid.")
    p.add_argument("--detail", choices=["rough", "medium", "fine"], default=None,
                   help="fold-count preset: rough=12, medium=24, fine=40 "
                        "(alias for --folds)")
    p.add_argument("--foldable", choices=["easy", "medium", "hard"], default=None,
                   help="hand-fold budget: easy/medium cap the crease count so a "
                        "human can fold the pattern by hand (tens of creases, not "
                        "hundreds); hard keeps full detail. Overrides --folds/--detail. "
                        "Still a coarse relief/corrugation, not figurative origami.")
    p.add_argument("--style", choices=["smooth", "origami"], default="smooth",
                   help="smooth (rounded relief) or origami (few large flat "
                        "facets with sharp creases)")
    p.add_argument("--open-sheet", action="store_true",
                   help="keep the one-sided open relief (default: closed watertight solid)")
    p.add_argument("--rect-sheet", action="store_true",
                   help="keep the FULL rectangular sheet (flat background paper at "
                        "the baseline, subject relief within it) instead of trimming "
                        "the export to the subject silhouette")
    p.add_argument("--depth", action="store_true",
                   help="estimate real 3D relief with a monocular depth model (needs torch)")
    p.add_argument("--depth-model",
                   choices=["MiDaS_small", "DPT_Hybrid",
                            "depth_anything_v2_small", "depth_anything_v2_base"],
                   default="MiDaS_small", help="depth network for --depth "
                   "(depth_anything_v2_small/_base need transformers)")
    p.add_argument("--engine", choices=["corrugation", "miura2d"],
                   default="corrugation",
                   help="corrugation (1D pleated strips, default) or miura2d "
                        "(true 2D warped-Miura tessellation, far better on curves)")
    p.add_argument("--symmetry", choices=["off", "auto", "x", "y"], default="off",
                   help="mirror-symmetrize the subject so its sides match: auto "
                        "detects the axis (only fires if the subject looks "
                        "symmetric), x/y force top-bottom / left-right, off (default) "
                        "leaves it as shot. Great for a butterfly's wings.")
    p.set_defaults(func=_fold)

    p = sub.add_parser("design",
                       help="figurative origami: circle-pack a stick-figure tree")
    from foldforge.design import BUILTIN_TREES
    p.add_argument("tree", help=", ".join(BUILTIN_TREES))
    p.add_argument("out", help="output .svg (crease pattern) or .fold")
    p.set_defaults(func=_design)

    p = sub.add_parser("instructions",
                       help="printable step-by-step folding sheet (SVG)")
    p.add_argument("pattern", help="input .fold crease pattern, or a photo")
    p.add_argument("out", help="output .svg path")
    p.add_argument("--foldable", choices=["easy", "medium", "hard"], default="easy",
                   help="photo input only: hand-fold budget (default easy) so the "
                        "instructions stay a dozen steps, not hundreds")
    p.add_argument("--engine", choices=["corrugation", "miura2d"],
                   default="corrugation", help="photo input only: fold engine")
    p.set_defaults(func=_instructions)

    p = sub.add_parser("export", help="export a .fold to layered SVG/DXF (cutter)")
    p.add_argument("path", help="input .fold path")
    p.add_argument("out", help="output .svg or .dxf path")
    p.add_argument("--outline", action="store_true",
                   help="SVG only: draw one clean sheet outline instead of every "
                        "panel edge (a printable hand-fold sheet: outline + M/V creases)")
    p.set_defaults(func=_export)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
