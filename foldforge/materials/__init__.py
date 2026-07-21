"""Origami metamaterials for FoldForge (Milestone 5).

Kinematic mechanical properties of a Miura-ori as a function of fold state:
its (negative) Poisson's ratio, a stiffness proxy, deployment compactness, and
inverse design of the sector angle to a target response.

    from foldforge.materials import poisson_ratio, fit_sector_angle
"""

from foldforge.materials.mechanics import (
    cell_dims,
    poisson_ratio,
    poisson_curve,
    deployment_ratio,
    stiffness_proxy,
    fit_sector_angle,
)

__all__ = [
    "cell_dims",
    "poisson_ratio",
    "poisson_curve",
    "deployment_ratio",
    "stiffness_proxy",
    "fit_sector_angle",
]
