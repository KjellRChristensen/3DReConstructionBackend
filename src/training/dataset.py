"""
Dataset Preparation for VLM Fine-tuning

Generates training datasets from 3D CAD models for fine-tuning
OpenECAD/TinyLLaVA models on custom CAD styles.

Dataset format follows TinyLLaVA/LLaVA conversation format:
[
    {
        "id": "unique_id",
        "image": "path/to/image.png",
        "conversations": [
            {"from": "human", "value": "<image>\nPrompt text"},
            {"from": "gpt", "value": "CAD code output"}
        ]
    }
]
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class CADCodeFormat(Enum):
    """Supported CAD code output formats for training"""
    OPENECAD = "openecad"      # OpenECAD command sequences
    CADQUERY = "cadquery"      # CadQuery Python code
    BUILD123D = "build123d"    # Build123d Python code


@dataclass
class TrainingSample:
    """Single training sample"""
    id: str
    image_path: Path
    cad_code: str
    code_format: CADCodeFormat
    source_model: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_conversation(self, prompt: str) -> Dict[str, Any]:
        """Convert to TinyLLaVA conversation format"""
        return {
            "id": self.id,
            "image": str(self.image_path.name),  # Relative path
            "conversations": [
                {"from": "human", "value": f"<image>\n{prompt}"},
                {"from": "gpt", "value": self.cad_code}
            ]
        }


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    output_dir: Path
    image_resolution: int = 512
    views: List[str] = field(default_factory=lambda: ["front", "isometric"])
    code_format: CADCodeFormat = CADCodeFormat.OPENECAD
    train_split: float = 0.9
    seed: int = 42
    max_workers: int = 4

    # Prompts for different views
    prompts: Dict[str, str] = field(default_factory=lambda: {
        "front": "Generate CAD construction code for this front view engineering drawing.",
        "top": "Generate CAD construction code for this top view engineering drawing.",
        "right": "Generate CAD construction code for this right side view engineering drawing.",
        "isometric": "Generate CAD construction code for this isometric view of a 3D model.",
        "multi": "Generate CAD construction code for this multi-view engineering drawing.",
    })


class CADCodeExtractor:
    """
    Extracts CAD code from various 3D model formats.

    For training, we need the CAD construction sequence that creates each model.
    This can come from:
    1. STEP/IGES files with construction history
    2. Manually annotated CAD code
    3. Parametric CAD files (FreeCAD, OpenSCAD)
    """

    @staticmethod
    def extract_from_step(step_path: Path) -> Optional[str]:
        """
        Extract construction history from STEP file.

        Note: Most STEP files don't contain full construction history.
        This is a best-effort extraction.
        """
        try:
            # Try using OCC to read STEP
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.IFSelect import IFSelect_RetDone

            reader = STEPControl_Reader()
            status = reader.ReadFile(str(step_path))

            if status != IFSelect_RetDone:
                return None

            reader.TransferRoots()
            shape = reader.OneShape()

            # Extract basic geometry info for code generation
            # This is simplified - real implementation would analyze topology
            return CADCodeExtractor._shape_to_openecad(shape)

        except ImportError:
            logger.warning("PythonOCC not available for STEP extraction")
            return None
        except Exception as e:
            logger.error(f"Error extracting from STEP: {e}")
            return None

    @staticmethod
    def _shape_to_openecad(shape) -> str:
        """Convert OCC shape to OpenECAD code (simplified)"""
        # This is a placeholder - real implementation would analyze
        # the shape topology and generate appropriate code
        lines = [
            "# Auto-extracted from STEP file",
            "plane_0 = add_sketchplane([0, 0, 0], [0, 0, 1], [1, 0, 0])",
        ]
        return "\n".join(lines)

    @staticmethod
    def from_openscad(scad_path: Path) -> Optional[str]:
        """Convert OpenSCAD file to CAD code"""
        try:
            with open(scad_path, 'r') as f:
                scad_code = f.read()

            # Convert OpenSCAD to CadQuery-style code
            # This is a simplified conversion
            return CADCodeExtractor._openscad_to_cadquery(scad_code)
        except Exception as e:
            logger.error(f"Error reading OpenSCAD: {e}")
            return None

    @staticmethod
    def _openscad_to_cadquery(scad_code: str) -> str:
        """Convert OpenSCAD to CadQuery (simplified)"""
        # Placeholder - would need proper parsing
        return f"# Converted from OpenSCAD\nimport cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)"

    @staticmethod
    def from_json_annotation(json_path: Path) -> Optional[str]:
        """Load CAD code from JSON annotation file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data.get("cad_code") or data.get("code")
        except Exception as e:
            logger.error(f"Error reading JSON annotation: {e}")
            return None


class DatasetGenerator:
    """
    Generates training datasets from 3D CAD models.

    Pipeline:
    1. Load 3D model
    2. Render orthographic views
    3. Extract/load CAD code
    4. Create training samples
    5. Save dataset in TinyLLaVA format
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_directories()

    def _setup_directories(self):
        """Create output directory structure"""
        self.images_dir = self.config.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.train_dir = self.config.output_dir / "train"
        self.train_dir.mkdir(exist_ok=True)

        self.val_dir = self.config.output_dir / "val"
        self.val_dir.mkdir(exist_ok=True)

    def generate_from_models(
        self,
        model_paths: List[Path],
        code_provider: Optional[Callable[[Path], Optional[str]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Generate dataset from a list of 3D model files.

        Args:
            model_paths: List of paths to 3D model files
            code_provider: Function that returns CAD code for a model path
                          If None, will attempt auto-extraction
            progress_callback: Called with (current, total) for progress updates

        Returns:
            Dataset statistics
        """
        from .orthographic_renderer import OrthographicRenderer, RenderConfig

        renderer = OrthographicRenderer(RenderConfig(
            resolution=self.config.image_resolution,
            show_hidden_lines=True,
        ))

        samples = []
        errors = []

        for i, model_path in enumerate(model_paths):
            if progress_callback:
                progress_callback(i, len(model_paths))

            try:
                sample = self._process_model(
                    model_path,
                    renderer,
                    code_provider
                )
                if sample:
                    samples.append(sample)
            except Exception as e:
                logger.error(f"Error processing {model_path}: {e}")
                errors.append({"path": str(model_path), "error": str(e)})

        # Split into train/val
        import random
        random.seed(self.config.seed)
        random.shuffle(samples)

        split_idx = int(len(samples) * self.config.train_split)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        # Save datasets
        self._save_dataset(train_samples, self.train_dir, "train")
        self._save_dataset(val_samples, self.val_dir, "val")

        stats = {
            "total_models": len(model_paths),
            "successful": len(samples),
            "errors": len(errors),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "output_dir": str(self.config.output_dir),
            "error_details": errors,
        }

        # Save stats
        with open(self.config.output_dir / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def _process_model(
        self,
        model_path: Path,
        renderer,
        code_provider: Optional[Callable],
    ) -> Optional[TrainingSample]:
        """Process a single model into training sample(s)"""

        # Generate unique ID
        model_hash = hashlib.md5(str(model_path).encode()).hexdigest()[:8]
        sample_id = f"{model_path.stem}_{model_hash}"

        # Get CAD code
        if code_provider:
            cad_code = code_provider(model_path)
        else:
            cad_code = self._auto_extract_code(model_path)

        if not cad_code:
            logger.warning(f"No CAD code available for {model_path}")
            return None

        # Render views
        try:
            views = renderer.render_model(model_path, self.config.views)
        except Exception as e:
            logger.error(f"Failed to render {model_path}: {e}")
            return None

        # Save images and create samples
        # For now, use the first view (typically front or isometric)
        primary_view = self.config.views[0]
        if primary_view in views:
            view_data = views[primary_view]
            image_filename = f"{sample_id}_{primary_view}.png"
            image_path = self.images_dir / image_filename

            # Save image
            import cv2
            cv2.imwrite(str(image_path), view_data.image)

            return TrainingSample(
                id=sample_id,
                image_path=image_path,
                cad_code=cad_code,
                code_format=self.config.code_format,
                source_model=model_path,
                metadata={
                    "view": primary_view,
                    "resolution": self.config.image_resolution,
                }
            )

        return None

    def _auto_extract_code(self, model_path: Path) -> Optional[str]:
        """Attempt to auto-extract CAD code from model"""
        suffix = model_path.suffix.lower()

        # Check for accompanying annotation file
        annotation_path = model_path.with_suffix('.json')
        if annotation_path.exists():
            code = CADCodeExtractor.from_json_annotation(annotation_path)
            if code:
                return code

        # Try format-specific extraction
        if suffix in ['.step', '.stp']:
            return CADCodeExtractor.extract_from_step(model_path)
        elif suffix == '.scad':
            return CADCodeExtractor.from_openscad(model_path)

        return None

    def _save_dataset(
        self,
        samples: List[TrainingSample],
        output_dir: Path,
        split_name: str
    ):
        """Save samples in TinyLLaVA format"""
        # Get appropriate prompt
        prompt = self.config.prompts.get(
            self.config.views[0],
            self.config.prompts["front"]
        )

        # Convert to conversation format
        conversations = [
            sample.to_conversation(prompt)
            for sample in samples
        ]

        # Save JSON
        output_path = output_dir / f"{split_name}.json"
        with open(output_path, 'w') as f:
            json.dump(conversations, f, indent=2)

        # Copy images to split directory
        split_images_dir = output_dir / "images"
        split_images_dir.mkdir(exist_ok=True)

        for sample in samples:
            if sample.image_path.exists():
                shutil.copy(
                    sample.image_path,
                    split_images_dir / sample.image_path.name
                )

        logger.info(f"Saved {len(samples)} samples to {output_path}")

    def generate_from_annotations(
        self,
        annotations_file: Path,
        images_dir: Path,
    ) -> Dict[str, Any]:
        """
        Generate dataset from pre-annotated data.

        Annotations file format:
        [
            {
                "image": "image1.png",
                "cad_code": "plane_0 = add_sketchplane(...)"
            },
            ...
        ]
        """
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        samples = []
        for i, ann in enumerate(annotations):
            image_path = images_dir / ann["image"]
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            # Copy image to dataset
            new_image_path = self.images_dir / image_path.name
            shutil.copy(image_path, new_image_path)

            sample = TrainingSample(
                id=f"ann_{i:05d}",
                image_path=new_image_path,
                cad_code=ann["cad_code"],
                code_format=self.config.code_format,
                metadata=ann.get("metadata", {}),
            )
            samples.append(sample)

        # Split and save
        import random
        random.seed(self.config.seed)
        random.shuffle(samples)

        split_idx = int(len(samples) * self.config.train_split)
        self._save_dataset(samples[:split_idx], self.train_dir, "train")
        self._save_dataset(samples[split_idx:], self.val_dir, "val")

        return {
            "total": len(annotations),
            "successful": len(samples),
            "train": split_idx,
            "val": len(samples) - split_idx,
        }


def create_dataset_from_directory(
    models_dir: Path,
    output_dir: Path,
    code_annotations: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to create dataset from a directory of models.

    Args:
        models_dir: Directory containing 3D model files
        output_dir: Where to save the dataset
        code_annotations: Optional JSON file with CAD code annotations
        **kwargs: Additional DatasetConfig parameters

    Returns:
        Dataset statistics
    """
    config = DatasetConfig(output_dir=Path(output_dir), **kwargs)
    generator = DatasetGenerator(config)

    # Find model files
    model_extensions = ['.step', '.stp', '.obj', '.stl', '.glb', '.gltf']
    model_paths = []
    for ext in model_extensions:
        model_paths.extend(Path(models_dir).glob(f"**/*{ext}"))

    if not model_paths:
        raise ValueError(f"No model files found in {models_dir}")

    # Load code annotations if provided
    code_map = {}
    if code_annotations and code_annotations.exists():
        with open(code_annotations, 'r') as f:
            annotations = json.load(f)
        code_map = {
            Path(a.get("model", a.get("path", ""))).stem: a.get("cad_code", a.get("code"))
            for a in annotations
        }

    def code_provider(model_path: Path) -> Optional[str]:
        return code_map.get(model_path.stem)

    return generator.generate_from_models(
        model_paths,
        code_provider=code_provider if code_map else None,
    )
