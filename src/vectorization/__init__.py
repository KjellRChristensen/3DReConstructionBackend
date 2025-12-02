# Vectorization module
# Converts raster images to vector representations:
# - Edge detection
# - Line extraction
# - Contour detection
# - SVG/vector output

from .vectorizer import RasterToVector

__all__ = ["RasterToVector"]
