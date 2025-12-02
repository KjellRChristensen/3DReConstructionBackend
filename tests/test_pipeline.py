"""
Tests for the reconstruction pipeline
"""
import pytest
from pathlib import Path


class TestPipelineConfig:
    """Test pipeline configuration"""

    def test_default_config(self):
        from src.pipeline import PipelineConfig

        config = PipelineConfig(input_path=Path("test.pdf"))

        assert config.dpi == 300
        assert config.wall_height == 2.8
        assert config.num_floors == 1
        assert "glb" in config.export_formats

    def test_custom_config(self):
        from src.pipeline import PipelineConfig

        config = PipelineConfig(
            input_path=Path("test.pdf"),
            wall_height=3.0,
            num_floors=2,
            export_formats=["obj", "ifc"],
        )

        assert config.wall_height == 3.0
        assert config.num_floors == 2
        assert config.export_formats == ["obj", "ifc"]


class TestDocumentLoader:
    """Test document loading"""

    def test_supported_formats(self):
        from src.ingestion.loader import DocumentLoader

        loader = DocumentLoader()
        assert ".pdf" in loader.SUPPORTED_EXTENSIONS
        assert ".png" in loader.SUPPORTED_EXTENSIONS
        assert ".dxf" in loader.SUPPORTED_EXTENSIONS

    def test_file_not_found(self):
        from src.ingestion.loader import DocumentLoader

        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.pdf")


class TestExporter:
    """Test model export"""

    def test_supported_formats(self):
        from src.export.exporter import ModelExporter, ExportFormat

        assert ExportFormat.OBJ.value == "obj"
        assert ExportFormat.GLB.value == "glb"
        assert ExportFormat.USDZ.value == "usdz"
