"""
Validation Pipeline - End-to-end validation of reconstruction against ground truth

Workflow:
1. Load ground truth 3D CAD model
2. Generate 2D projection (synthetic floor plan)
3. Run reconstruction on the 2D projection
4. Compare reconstructed 3D against ground truth
5. Report metrics
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Complete validation result"""
    ground_truth_file: str
    generated_2d_file: Optional[str]
    reconstructed_3d_file: Optional[str]

    # Metrics
    metrics_3d: Dict[str, float]
    metrics_2d: Optional[Dict[str, float]]

    # Timing
    projection_time: float
    reconstruction_time: float
    comparison_time: float
    total_time: float

    # Strategy used
    strategy_used: str

    # Success
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ground_truth_file": self.ground_truth_file,
            "generated_2d_file": self.generated_2d_file,
            "reconstructed_3d_file": self.reconstructed_3d_file,
            "metrics_3d": self.metrics_3d,
            "metrics_2d": self.metrics_2d,
            "timing": {
                "projection_time": self.projection_time,
                "reconstruction_time": self.reconstruction_time,
                "comparison_time": self.comparison_time,
                "total_time": self.total_time,
            },
            "strategy_used": self.strategy_used,
            "success": self.success,
            "error": self.error,
        }


class ValidationPipeline:
    """
    End-to-end validation pipeline for 3D reconstruction.

    Tests reconstruction quality by:
    1. Taking a ground truth 3D model
    2. Creating a 2D floor plan from it
    3. Running reconstruction
    4. Comparing output to ground truth
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "data/validation",
        strategy: str = "auto",
        wall_height: float = 2.8,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.strategy = strategy
        self.wall_height = wall_height

    async def validate(
        self,
        ground_truth_path: Union[str, Path],
        floor_height: float = 1.0,
        save_intermediates: bool = True,
    ) -> ValidationResult:
        """
        Run full validation pipeline on a ground truth 3D model.

        Args:
            ground_truth_path: Path to ground truth CAD file (IFC, OBJ, GLB, etc.)
            floor_height: Height at which to slice for floor plan
            save_intermediates: Whether to save intermediate files

        Returns:
            ValidationResult with all metrics
        """
        from .cad_import import CADImporter
        from .projection import ProjectionGenerator
        from .metrics import MeshComparison

        ground_truth_path = Path(ground_truth_path)
        start_time = time.time()

        try:
            # Step 1: Load ground truth
            logger.info(f"Loading ground truth: {ground_truth_path}")
            importer = CADImporter()
            gt_model = importer.load(ground_truth_path)
            gt_mesh = gt_model.mesh

            # Step 2: Generate 2D projection
            logger.info("Generating 2D floor plan from ground truth")
            proj_start = time.time()

            projector = ProjectionGenerator()
            projection = projector.generate_floor_plan(
                gt_mesh,
                floor_height=floor_height,
            )

            # Save 2D projection
            projection_name = ground_truth_path.stem + "_2d.png"
            projection_path = self.output_dir / projection_name
            projector.save_projection(projection, projection_path)

            projection_time = time.time() - proj_start

            # Step 3: Run reconstruction
            logger.info("Running reconstruction on generated floor plan")
            recon_start = time.time()

            reconstructed_mesh, strategy_used = await self._run_reconstruction(
                projection_path,
                projection.scale,
            )

            # Save reconstructed mesh
            recon_name = ground_truth_path.stem + "_reconstructed.glb"
            recon_path = self.output_dir / recon_name
            reconstructed_mesh.export(str(recon_path))

            reconstruction_time = time.time() - recon_start

            # Step 4: Compare meshes
            logger.info("Comparing reconstructed mesh to ground truth")
            compare_start = time.time()

            comparator = MeshComparison()
            comparison = comparator.compare(reconstructed_mesh, gt_mesh)

            comparison_time = time.time() - compare_start

            # Step 5: Also compare 2D projections
            gt_projection = projector.generate_floor_plan(
                gt_mesh, floor_height=floor_height
            )
            recon_projection = projector.generate_floor_plan(
                reconstructed_mesh, floor_height=floor_height
            )

            from .metrics import compute_2d_metrics
            metrics_2d = compute_2d_metrics(
                recon_projection.image,
                gt_projection.image,
            )

            total_time = time.time() - start_time

            return ValidationResult(
                ground_truth_file=str(ground_truth_path),
                generated_2d_file=str(projection_path),
                reconstructed_3d_file=str(recon_path),
                metrics_3d=comparison.to_dict(),
                metrics_2d=metrics_2d,
                projection_time=projection_time,
                reconstruction_time=reconstruction_time,
                comparison_time=comparison_time,
                total_time=total_time,
                strategy_used=strategy_used,
                success=True,
            )

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import traceback
            return ValidationResult(
                ground_truth_file=str(ground_truth_path),
                generated_2d_file=None,
                reconstructed_3d_file=None,
                metrics_3d={},
                metrics_2d=None,
                projection_time=0,
                reconstruction_time=0,
                comparison_time=0,
                total_time=time.time() - start_time,
                strategy_used="none",
                success=False,
                error=f"{str(e)}\n{traceback.format_exc()}",
            )

    async def _run_reconstruction(self, image_path: Path, scale: float):
        """Run reconstruction using configured strategy"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from reconstruction.strategies import (
            ReconstructionInput,
            ReconstructionManager,
            BasicExtrusionStrategy,
            MultiViewDNNStrategy,
            StrategyType,
        )

        # Set up reconstruction
        manager = ReconstructionManager()
        manager.register_strategy(BasicExtrusionStrategy())
        manager.register_strategy(MultiViewDNNStrategy(model_type="depth_estimation"))

        input_data = ReconstructionInput(
            primary_image=image_path,
            wall_height=self.wall_height,
            scale=scale,
        )

        # Determine strategy
        if self.strategy == "auto":
            preferred = None
        else:
            preferred = StrategyType(self.strategy)

        result = await manager.reconstruct(
            input_data,
            preferred_strategy=preferred,
            fallback=True,
        )

        if not result.success:
            raise RuntimeError(f"Reconstruction failed: {result.error}")

        # Convert result to mesh
        import trimesh
        import io
        mesh = trimesh.load(io.BytesIO(result.model_data), file_type=result.format)

        return mesh, result.strategy_used

    async def validate_batch(
        self,
        ground_truth_files: List[Union[str, Path]],
        floor_height: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Run validation on multiple ground truth files.

        Args:
            ground_truth_files: List of paths to CAD files
            floor_height: Height for floor plan slice

        Returns:
            Summary with per-file results and aggregate metrics
        """
        results = []
        aggregate_3d = {
            "chamfer_distance": [],
            "hausdorff_distance": [],
            "iou_3d": [],
            "f_score": [],
        }

        for gt_file in ground_truth_files:
            logger.info(f"Validating: {gt_file}")
            result = await self.validate(gt_file, floor_height=floor_height)
            results.append(result.to_dict())

            if result.success:
                for key in aggregate_3d:
                    if key in result.metrics_3d:
                        aggregate_3d[key].append(result.metrics_3d[key])

        # Compute aggregate statistics
        import numpy as np
        aggregate_stats = {}
        for key, values in aggregate_3d.items():
            if values:
                aggregate_stats[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        return {
            "results": results,
            "aggregate": aggregate_stats,
            "total_files": len(ground_truth_files),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
        }

    def save_results(self, results: Dict[str, Any], output_file: Union[str, Path]):
        """Save validation results to JSON file"""
        output_file = Path(output_file)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")


# Convenience function for API
async def run_validation(
    ground_truth_path: str,
    strategy: str = "auto",
    wall_height: float = 2.8,
    floor_height: float = 1.0,
) -> Dict[str, Any]:
    """
    Run validation on a single ground truth file.

    Returns dict suitable for JSON response.
    """
    pipeline = ValidationPipeline(strategy=strategy, wall_height=wall_height)
    result = await pipeline.validate(ground_truth_path, floor_height=floor_height)
    return result.to_dict()
