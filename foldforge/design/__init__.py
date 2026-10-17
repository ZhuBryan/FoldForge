"""Inverse design for FoldForge (Milestone 3).

Given a target shape, recover the fold that produces it by gradient descent,
using the analytic gradients from :mod:`foldforge.diff`.

    from foldforge.design import fit_chain, target_arch, chamfer_distance
"""

from foldforge.design.inverse import (
    fit_chain,
    FitResult,
    chamfer_distance,
    angles_from_curve,
    resample_arclength,
    target_arch,
    target_wave,
    target_step,
)
from foldforge.design.treemaker import (
    MetricTree,
    star_tree,
    river_tree,
    get_tree,
    BUILTIN_TREES,
    Packing,
    pack_tree,
    crease_pattern,
    design_base,
    flap_length_errors,
    folded_schematic,
    reduced_dist,
)
from foldforge.design.classic import bird_base
from foldforge.design.curved_crease import (
    curved_fold,
    concentric_pleat,
    FoldedSurface,
)
from foldforge.design.skeleton import (
    tree_from_mask,
    tree_from_image,
    design_from_image,
    TreeExtraction,
    PhotoDesign,
)

__all__ = [
    "fit_chain",
    "FitResult",
    "chamfer_distance",
    "angles_from_curve",
    "resample_arclength",
    "target_arch",
    "target_wave",
    "target_step",
    "MetricTree",
    "star_tree",
    "river_tree",
    "get_tree",
    "BUILTIN_TREES",
    "Packing",
    "pack_tree",
    "crease_pattern",
    "design_base",
    "flap_length_errors",
    "folded_schematic",
    "reduced_dist",
    "bird_base",
    "curved_fold",
    "concentric_pleat",
    "FoldedSurface",
    "tree_from_mask",
    "tree_from_image",
    "design_from_image",
    "TreeExtraction",
    "PhotoDesign",
]
