"""
Document Loader - Handles loading various input formats
"""
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InputFormat(Enum):
    PDF = "pdf"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    TIFF = "tiff"
    DWG = "dwg"
    DXF = "dxf"
    UNKNOWN = "unknown"


@dataclass
class LoadedDocument:
    """Container for loaded document data"""
    filepath: Path
    format: InputFormat
    images: list  # List of numpy arrays (pages/layers)
    metadata: dict
    vector_data: Optional[dict] = None  # For DWG/DXF


class DocumentLoader:
    """
    Unified loader for all supported input formats.

    Supported formats:
    - PDF: Uses pdf2image/PyMuPDF for raster, pdfplumber for vector
    - Images: PNG, JPG, TIFF via OpenCV/Pillow
    - CAD: DWG/DXF via ezdxf (DXF) or ODA converter (DWG)
    """

    SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.dwg', '.dxf'}

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def load(self, filepath: Union[str, Path]) -> LoadedDocument:
        """Load a document from file path"""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = filepath.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {ext}")

        format_type = self._detect_format(filepath)

        if format_type == InputFormat.PDF:
            return self._load_pdf(filepath)
        elif format_type in (InputFormat.PNG, InputFormat.JPG, InputFormat.JPEG, InputFormat.TIFF):
            return self._load_image(filepath, format_type)
        elif format_type in (InputFormat.DWG, InputFormat.DXF):
            return self._load_cad(filepath, format_type)
        else:
            raise ValueError(f"Cannot load format: {format_type}")

    def _detect_format(self, filepath: Path) -> InputFormat:
        """Detect file format from extension"""
        ext = filepath.suffix.lower().lstrip('.')
        try:
            return InputFormat(ext)
        except ValueError:
            if ext == 'tif':
                return InputFormat.TIFF
            return InputFormat.UNKNOWN

    def _load_pdf(self, filepath: Path) -> LoadedDocument:
        """Load PDF document"""
        # TODO: Implement PDF loading
        # - Use pdf2image for raster conversion
        # - Use pdfplumber/PyMuPDF for vector extraction
        logger.info(f"Loading PDF: {filepath}")
        raise NotImplementedError("PDF loading not yet implemented")

    def _load_image(self, filepath: Path, format_type: InputFormat) -> LoadedDocument:
        """Load raster image"""
        # TODO: Implement image loading via OpenCV
        logger.info(f"Loading image: {filepath}")
        raise NotImplementedError("Image loading not yet implemented")

    def _load_cad(self, filepath: Path, format_type: InputFormat) -> LoadedDocument:
        """Load CAD file (DWG/DXF)"""
        # TODO: Implement CAD loading
        # - DXF: Use ezdxf
        # - DWG: Use ODA File Converter or Teigha
        logger.info(f"Loading CAD file: {filepath}")
        raise NotImplementedError("CAD loading not yet implemented")
