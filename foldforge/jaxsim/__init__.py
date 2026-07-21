"""Optional JAX backend: automatic differentiation + GPU. Requires `pip install jax`."""

from foldforge.jaxsim.autodiff import (
    fold_chain, apex_height, apex_height_grad,
    folded_miura, miura_height, miura_height_grad,
)

__all__ = [
    "fold_chain", "apex_height", "apex_height_grad",
    "folded_miura", "miura_height", "miura_height_grad",
]
