"""Analysis routines, organized by domain.

Subpackages mirror the freud Python layout (so notebooks port with
near-mechanical renaming) and the underlying Rust crate layout
(``molrs_compute::{density, order, environment, …}``).
"""

from . import (
    cluster,
    density,
    diffraction,
    environment,
    ml,
    msd,
    order,
    pmft,
)

__all__ = [
    "cluster",
    "density",
    "diffraction",
    "environment",
    "ml",
    "msd",
    "order",
    "pmft",
]
