# Reconstruction module
# Converts detected 2D elements into 3D geometry
#
# Three reconstruction strategies:
# - Strategy A: External DNN APIs (Kaedim, Replicate, etc.)
# - Strategy B: Built-in basic reconstruction (single-view extrusion)
# - Strategy C: Multi-view DNN reconstruction (GaussianCAD, custom models)

from .builder import ModelBuilder
from .strategies import (
    ReconstructionStrategy,
    ExternalAPIStrategy,
    BasicExtrusionStrategy,
    MultiViewDNNStrategy,
    get_strategy,
)

__all__ = [
    "ModelBuilder",
    "ReconstructionStrategy",
    "ExternalAPIStrategy",
    "BasicExtrusionStrategy",
    "MultiViewDNNStrategy",
    "get_strategy",
]
