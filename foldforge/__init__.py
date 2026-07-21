"""FoldForge: a differentiable computational origami engine.

Milestone 0 (this release) is the geometry + theory core:

    from foldforge import CreasePattern, read_fold, write_fold
    from foldforge import check_kawasaki, check_maekawa, foldability_report
    from foldforge import render_pattern
    from foldforge.geometry import examples

Everything here works on the *flat* crease pattern (the unfolded sheet). The
actual 3D folding simulator arrives in M1.
"""

from foldforge.geometry.crease_graph import CreasePattern
from foldforge.geometry.fold_io import read_fold, write_fold
from foldforge.geometry.foldability import (
    check_kawasaki,
    check_maekawa,
    foldability_report,
)
from foldforge.geometry.render import render_pattern

__all__ = [
    "CreasePattern",
    "read_fold",
    "write_fold",
    "check_kawasaki",
    "check_maekawa",
    "foldability_report",
    "render_pattern",
]

__version__ = "0.6.0"
