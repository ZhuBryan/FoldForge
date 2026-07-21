"""Differentiable origami for FoldForge (Milestone 2).

Two complementary ways to get gradients through a fold, both pure numpy:

    kinematics  - a rigid *fold chain* whose folded shape is a smooth, closed-
                  form function of its fold angles, with analytic gradients
                  (verified against finite differences). This is the rigorous
                  "differentiable rigid origami" core.
    params      - finite-difference gradients over the few parameters of a
                  crease family (e.g. a Miura's sector angle + fold amount),
                  used to optimise the full M1 simulator in M3-M5.

    from foldforge.diff import fold_chain, apex_height, apex_height_grad
"""

from foldforge.diff.kinematics import (
    fold_chain,
    FoldChainResult,
    spine_jacobian,
    apex_height,
    apex_height_grad,
)
from foldforge.diff.miura import (
    flat_miura,
    folded_miura,
    fold_limit,
    footprint,
    footprint_grad_h,
    fit_miura,
)
from foldforge.diff.implicit import (
    equilibrium,
    energy_grad,
    implicit_grad,
    distance_output,
)

__all__ = [
    "fold_chain",
    "FoldChainResult",
    "spine_jacobian",
    "apex_height",
    "apex_height_grad",
    "flat_miura",
    "folded_miura",
    "fold_limit",
    "footprint",
    "footprint_grad_h",
    "fit_miura",
    "equilibrium",
    "energy_grad",
    "implicit_grad",
    "distance_output",
]
