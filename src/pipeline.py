"""
Pipeline Orchestrator - Coordinates the full reconstruction workflow
"""
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    INGESTION = "ingestion"
    PREPROCESSING = "preprocessing"
    VECTORIZATION = "vectorization"
    RECOGNITION = "recognition"
    RECONSTRUCTION = "reconstruction"
    EXPORT = "export"


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution"""
    # Input
    input_path: Path
    input_format: Optional[str] = None  # Auto-detect if None

    # Processing options
    dpi: int = 300
    denoise: bool = True
    deskew: bool = True

    # Detection options
    detect_walls: bool = True
    detect_doors: bool = True
    detect_windows: bool = True
    detect_rooms: bool = True
    detect_dimensions: bool = True

    # 3D Model options
    wall_height: float = 2.8
    floor_thickness: float = 0.3
    num_floors: int = 1
    scale: Optional[float] = None  # Auto-detect from dimensions

    # Export
    output_dir: Path = Path("./output")
    export_formats: List[str] = None

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["glb", "obj"]


@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    success: bool
    output_files: List[Path]
    stages_completed: List[PipelineStage]
    error: Optional[str] = None
    timing: Optional[dict] = None
    metadata: Optional[dict] = None


class Pipeline:
    """
    Main pipeline orchestrator for 2D → 3D reconstruction.

    Pipeline stages:
    1. Ingestion: Load PDF/image/CAD file
    2. Preprocessing: Deskew, denoise, enhance
    3. Vectorization: Convert raster to vectors
    4. Recognition: Detect architectural elements
    5. Reconstruction: Build 3D geometry
    6. Export: Output to requested formats
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._progress_callback: Optional[Callable] = None
        self._current_stage: Optional[PipelineStage] = None
        self._timing: dict = {}

    def set_progress_callback(self, callback: Callable[[PipelineStage, float, str], None]):
        """
        Set callback for progress updates.

        Callback signature: (stage: PipelineStage, progress: float, message: str)
        """
        self._progress_callback = callback

    def _report_progress(self, progress: float, message: str):
        """Report progress to callback if set"""
        if self._progress_callback and self._current_stage:
            self._progress_callback(self._current_stage, progress, message)

    def run(self) -> PipelineResult:
        """
        Execute the full pipeline.

        Returns:
            PipelineResult with output files and status
        """
        logger.info(f"Starting pipeline for: {self.config.input_path}")
        stages_completed = []
        start_time = datetime.now()

        try:
            # Stage 1: Ingestion
            self._current_stage = PipelineStage.INGESTION
            self._report_progress(0.0, "Loading document...")
            document = self._run_ingestion()
            stages_completed.append(PipelineStage.INGESTION)

            # Stage 2: Preprocessing
            self._current_stage = PipelineStage.PREPROCESSING
            self._report_progress(0.0, "Preprocessing images...")
            processed_images = self._run_preprocessing(document)
            stages_completed.append(PipelineStage.PREPROCESSING)

            # Stage 3: Vectorization
            self._current_stage = PipelineStage.VECTORIZATION
            self._report_progress(0.0, "Converting to vectors...")
            vector_data = self._run_vectorization(processed_images)
            stages_completed.append(PipelineStage.VECTORIZATION)

            # Stage 4: Recognition
            self._current_stage = PipelineStage.RECOGNITION
            self._report_progress(0.0, "Detecting elements...")
            elements = self._run_recognition(processed_images, vector_data)
            stages_completed.append(PipelineStage.RECOGNITION)

            # Stage 5: Reconstruction
            self._current_stage = PipelineStage.RECONSTRUCTION
            self._report_progress(0.0, "Building 3D model...")
            model = self._run_reconstruction(elements)
            stages_completed.append(PipelineStage.RECONSTRUCTION)

            # Stage 6: Export
            self._current_stage = PipelineStage.EXPORT
            self._report_progress(0.0, "Exporting files...")
            output_files = self._run_export(model)
            stages_completed.append(PipelineStage.EXPORT)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Pipeline completed in {elapsed:.1f}s")

            return PipelineResult(
                success=True,
                output_files=output_files,
                stages_completed=stages_completed,
                timing=self._timing,
            )

        except Exception as e:
            logger.error(f"Pipeline failed at {self._current_stage}: {e}")
            return PipelineResult(
                success=False,
                output_files=[],
                stages_completed=stages_completed,
                error=str(e),
            )

    def _run_ingestion(self):
        """Stage 1: Load input document"""
        from .ingestion import DocumentLoader

        loader = DocumentLoader(dpi=self.config.dpi)
        self._report_progress(0.5, "Reading file...")

        # TODO: Implement actual loading
        # document = loader.load(self.config.input_path)

        self._report_progress(1.0, "Document loaded")
        raise NotImplementedError("Ingestion not yet implemented")

    def _run_preprocessing(self, document):
        """Stage 2: Preprocess images"""
        from .ingestion import ImagePreprocessor, PreprocessingConfig

        config = PreprocessingConfig(
            denoise=self.config.denoise,
            deskew=self.config.deskew,
        )
        preprocessor = ImagePreprocessor(config)

        # TODO: Process each page/image
        self._report_progress(1.0, "Preprocessing complete")
        raise NotImplementedError("Preprocessing not yet implemented")

    def _run_vectorization(self, images):
        """Stage 3: Convert to vectors"""
        from .vectorization import RasterToVector

        vectorizer = RasterToVector()

        # TODO: Vectorize each image
        self._report_progress(1.0, "Vectorization complete")
        raise NotImplementedError("Vectorization not yet implemented")

    def _run_recognition(self, images, vector_data):
        """Stage 4: Detect architectural elements"""
        from .recognition import ElementDetector

        detector = ElementDetector()

        # TODO: Run detection
        self._report_progress(1.0, "Detection complete")
        raise NotImplementedError("Recognition not yet implemented")

    def _run_reconstruction(self, elements):
        """Stage 5: Build 3D model"""
        from .reconstruction import ModelBuilder

        builder = ModelBuilder(
            default_wall_height=self.config.wall_height,
            default_floor_thickness=self.config.floor_thickness,
        )

        # TODO: Build model
        self._report_progress(1.0, "Model built")
        raise NotImplementedError("Reconstruction not yet implemented")

    def _run_export(self, model):
        """Stage 6: Export to requested formats"""
        from .export import ModelExporter

        exporter = ModelExporter(output_dir=self.config.output_dir)
        output_files = []

        for fmt in self.config.export_formats:
            self._report_progress(0.5, f"Exporting {fmt}...")
            # output_path = exporter.export(model, "model", fmt)
            # output_files.append(output_path)

        self._report_progress(1.0, "Export complete")
        raise NotImplementedError("Export not yet implemented")


def run_pipeline(
    input_path: str,
    output_dir: str = "./output",
    export_formats: List[str] = None,
    **kwargs
) -> PipelineResult:
    """
    Convenience function to run the pipeline.

    Args:
        input_path: Path to input file (PDF, image, or CAD)
        output_dir: Directory for output files
        export_formats: List of output formats
        **kwargs: Additional PipelineConfig options

    Returns:
        PipelineResult
    """
    config = PipelineConfig(
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        export_formats=export_formats or ["glb", "obj"],
        **kwargs
    )

    pipeline = Pipeline(config)
    return pipeline.run()
