"""
Training Module for CAD2Program Fine-tuning

This module provides tools for:
1. Generating orthographic projections from 3D models
2. Creating training pairs (2D views + 3D ground truth)
3. Batch processing for dataset creation
"""

from .orthographic_renderer import (
    OrthographicRenderer,
    ViewType,
    RenderConfig,
    OrthographicView,
)

__all__ = [
    "OrthographicRenderer",
    "ViewType",
    "RenderConfig",
    "OrthographicView",
]
