"""Generative model for FoldForge (Milestone 4).

A small from-scratch network that proposes a fold for a target shape in one
shot (amortised inverse design), trained with the differentiable simulator in
the loop.

    from foldforge.generative import FoldGenerator, make_dataset
"""

from foldforge.generative.model import (
    FoldGenerator,
    make_dataset,
    batch_fold,
    batch_recon_grad,
    smooth_angles,
)

__all__ = [
    "FoldGenerator",
    "make_dataset",
    "batch_fold",
    "batch_recon_grad",
    "smooth_angles",
]
