"""Origamize: decompose a 3D target into a foldable crease pattern.

The capstone feature - turn a shape into the pleated sheet that folds into it.

    from foldforge.origamize import origamize_profile, origamize_image, heightfield_dome
"""

from foldforge.origamize.surface import (
    origamize_profile, origamize_heightfield, OrigamiResult,
    profile_dome, profile_ridge, heightfield_dome, heightfield_saddle, heightfield_ripple,
    close_relief,
)
from foldforge.origamize.io import (
    origamize_image, origamize_function, origamize_points, fold_heightfield,
    heightmap_from_image, heightmap_from_function, heightmap_from_points,
)
from foldforge.origamize.miura_fit import (
    origamize_miura, Miura2DResult, surface_fit_error, compare_engines,
    corrugation_surface, midsurface,
)
from foldforge.origamize.shapes import text_heightfield, terrain_heightfield, load_obj
from foldforge.origamize.vision import (
    origamize_silhouette, silhouette_mask, relief_from_image, inflate,
)
from foldforge.origamize.depth import (
    estimate_depth, origamize_depth, depth_relief,
)
from foldforge.origamize.symmetry import symmetrize, measure_symmetry

__all__ = [
    "origamize_profile", "origamize_heightfield", "OrigamiResult",
    "profile_dome", "profile_ridge", "heightfield_dome", "heightfield_saddle",
    "heightfield_ripple", "close_relief", "origamize_image", "origamize_function", "origamize_points",
    "fold_heightfield", "origamize_miura", "Miura2DResult", "surface_fit_error",
    "compare_engines", "corrugation_surface", "midsurface",
    "heightmap_from_image", "heightmap_from_function", "heightmap_from_points",
    "text_heightfield", "terrain_heightfield", "load_obj",
    "origamize_silhouette", "silhouette_mask", "relief_from_image", "inflate",
    "estimate_depth", "origamize_depth", "depth_relief",
    "symmetrize", "measure_symmetry",
]
