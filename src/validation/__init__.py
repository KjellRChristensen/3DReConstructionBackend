# Validation module
# Tools for comparing reconstructed 3D models against ground truth CAD data

from .cad_import import CADImporter, load_ground_truth
from .projection import ProjectionGenerator
from .metrics import MeshComparison, compute_metrics
from .pipeline import ValidationPipeline

__all__ = [
    "CADImporter",
    "load_ground_truth",
    "ProjectionGenerator",
    "MeshComparison",
    "compute_metrics",
    "ValidationPipeline",
]
