"""3D folding simulator for FoldForge (Milestone 1).

Turns a flat :class:`~foldforge.geometry.crease_graph.CreasePattern` into a 3D
folded shape using a dynamic-relaxation (mass-spring) solver in the style of
Ghassaei, Demaine & Gershenfeld's Origami Simulator (2018):

    * faces are kept rigid by stiff axial springs along every triangle edge,
    * each crease is a hinge driven toward a target fold angle,
    * the whole thing is relaxed to equilibrium as the fold ramps up.

    from foldforge.sim import FoldMesh, fold
    from foldforge.geometry import examples

    mesh = FoldMesh.from_pattern(examples.miura())
    result = fold(mesh, fold_fraction=0.9)
    result.vertices   # (V, 3) folded coordinates
"""

from foldforge.sim.mesh import FoldMesh, Hinge, dihedral_angle, dihedral_grad
from foldforge.sim.solver import fold, FoldResult, creases_along_x
from foldforge.sim.collision import (
    self_intersections, intersection_count, separation_penalty,
)
from foldforge.sim.sequencing import fold_sequence, FoldSequence

__all__ = [
    "FoldMesh",
    "Hinge",
    "dihedral_angle",
    "dihedral_grad",
    "fold",
    "FoldResult",
    "creases_along_x",
    "self_intersections",
    "intersection_count",
    "separation_penalty",
    "fold_sequence",
    "FoldSequence",
]
