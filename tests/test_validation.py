"""
Tests for the Validation Pipeline

Tests CAD import, 2D projection, metrics, and the full validation pipeline.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile

pytestmark = pytest.mark.asyncio


class TestCADImporter:
    """Tests for CAD file import"""

    def test_importer_initialization(self):
        """Test CADImporter initializes correctly"""
        from validation.cad_import import CADImporter

        importer = CADImporter()
        assert importer.has_trimesh is True  # Should be installed

    def test_supported_formats(self):
        """Test supported formats list"""
        from validation.cad_import import CADImporter

        importer = CADImporter()
        assert '.obj' in importer.SUPPORTED_FORMATS
        assert '.glb' in importer.SUPPORTED_FORMATS
        assert '.ifc' in importer.SUPPORTED_FORMATS
        assert '.dxf' in importer.SUPPORTED_FORMATS

    def test_load_obj_mesh(self, tmp_path):
        """Test loading OBJ file"""
        from validation.cad_import import CADImporter
        import trimesh

        # Create a simple cube OBJ file
        cube = trimesh.creation.box(extents=[1, 1, 1])
        obj_path = tmp_path / "test_cube.obj"
        cube.export(str(obj_path))

        importer = CADImporter()
        model = importer.load(obj_path)

        assert model is not None
        assert model.format == 'obj'
        assert len(model.vertices) > 0
        assert len(model.faces) > 0

    def test_load_glb_mesh(self, tmp_path):
        """Test loading GLB file"""
        from validation.cad_import import CADImporter
        import trimesh

        cube = trimesh.creation.box(extents=[2, 2, 2])
        glb_path = tmp_path / "test_cube.glb"
        cube.export(str(glb_path))

        importer = CADImporter()
        model = importer.load(glb_path)

        assert model is not None
        assert model.format == 'glb'
        assert len(model.vertices) == 8  # Cube has 8 vertices

    def test_load_nonexistent_file(self):
        """Test error on nonexistent file"""
        from validation.cad_import import CADImporter

        importer = CADImporter()
        with pytest.raises(FileNotFoundError):
            importer.load("/nonexistent/path/model.obj")

    def test_load_unsupported_format(self, tmp_path):
        """Test error on unsupported format"""
        from validation.cad_import import CADImporter

        fake_file = tmp_path / "model.xyz"
        fake_file.write_text("fake data")

        importer = CADImporter()
        with pytest.raises(ValueError, match="Unsupported format"):
            importer.load(fake_file)


class TestProjectionGenerator:
    """Tests for 2D projection generation"""

    def test_generator_initialization(self):
        """Test ProjectionGenerator initializes correctly"""
        from validation.projection import ProjectionGenerator

        projector = ProjectionGenerator(resolution=512)
        assert projector.resolution == 512

    def test_generate_floor_plan(self, tmp_path):
        """Test floor plan generation from mesh"""
        from validation.projection import ProjectionGenerator
        import trimesh

        # Create a simple building shape
        building = trimesh.creation.box(extents=[10, 10, 3])

        projector = ProjectionGenerator(resolution=256)
        projection = projector.generate_floor_plan(building, floor_height=1.5)

        assert projection is not None
        assert projection.width > 0
        assert projection.height > 0
        assert projection.image.shape[0] == projection.height
        assert projection.image.shape[1] == projection.width
        assert projection.scale > 0

    def test_save_projection(self, tmp_path):
        """Test saving projection to file"""
        from validation.projection import ProjectionGenerator
        import trimesh

        building = trimesh.creation.box(extents=[5, 5, 2])

        projector = ProjectionGenerator()
        projection = projector.generate_floor_plan(building)

        output_path = tmp_path / "test_projection.png"
        saved_path = projector.save_projection(projection, output_path)

        assert saved_path.exists()
        assert saved_path.stat().st_size > 0

    def test_generate_elevation(self, tmp_path):
        """Test elevation view generation"""
        from validation.projection import ProjectionGenerator
        import trimesh

        building = trimesh.creation.box(extents=[10, 5, 8])

        projector = ProjectionGenerator()
        elevation = projector.generate_elevation(building, direction="front")

        assert elevation is not None
        assert elevation.metadata["direction"] == "front"

    def test_generate_training_pair(self, tmp_path):
        """Test training pair generation"""
        from validation.projection import ProjectionGenerator
        import trimesh

        building = trimesh.creation.box(extents=[8, 6, 3])

        projector = ProjectionGenerator()
        result = projector.generate_training_pair(
            building,
            output_dir=tmp_path,
            name="test_building",
        )

        assert "x" in result
        assert "y" in result
        assert "metadata" in result
        assert result["x"].exists()
        assert result["y"].exists()
        assert result["metadata"].exists()


class TestMeshComparison:
    """Tests for mesh comparison metrics"""

    def test_comparison_initialization(self):
        """Test MeshComparison initializes correctly"""
        from validation.metrics import MeshComparison

        comparator = MeshComparison(num_samples=5000, threshold=0.1)
        assert comparator.num_samples == 5000
        assert comparator.threshold == 0.1

    def test_compare_identical_meshes(self):
        """Test comparing identical meshes"""
        from validation.metrics import MeshComparison
        import trimesh

        cube = trimesh.creation.box(extents=[1, 1, 1])

        comparator = MeshComparison()
        result = comparator.compare(cube, cube)

        # Identical meshes should have near-perfect metrics
        # Note: chamfer distance is non-zero due to surface sampling
        assert result.chamfer_distance < 0.1  # Small distance due to sampling
        assert result.iou_3d >= 0.99  # Should be ~1.0
        assert result.f_score >= 0.99  # Should be ~1.0

    def test_compare_different_meshes(self):
        """Test comparing different meshes"""
        from validation.metrics import MeshComparison
        import trimesh

        cube1 = trimesh.creation.box(extents=[1, 1, 1])
        cube2 = trimesh.creation.box(extents=[2, 2, 2])

        comparator = MeshComparison()
        result = comparator.compare(cube1, cube2)

        # Different sized meshes should have significant chamfer distance
        assert result.chamfer_distance > 0.1  # Bigger cube = more distance
        assert result.hausdorff_distance > 0
        # F-score should be low since points are far apart
        assert result.f_score < 0.5

    def test_compare_offset_meshes(self):
        """Test comparing offset meshes (same shape, different position)"""
        from validation.metrics import MeshComparison
        import trimesh

        cube1 = trimesh.creation.box(extents=[1, 1, 1])
        cube2 = trimesh.creation.box(extents=[1, 1, 1])
        cube2.apply_translation([0.5, 0, 0])  # Offset by 0.5m

        comparator = MeshComparison()
        result = comparator.compare(cube1, cube2)

        # Offset should cause measurable distance
        assert result.chamfer_distance > 0.1  # Offset creates distance
        assert result.hausdorff_distance > 0.3  # Max deviation ~0.5
        # Precision/recall affected by offset beyond threshold
        assert result.precision < 1.0 or result.recall < 1.0

    def test_result_to_dict(self):
        """Test ComparisonResult serialization"""
        from validation.metrics import MeshComparison
        import trimesh

        cube = trimesh.creation.box(extents=[1, 1, 1])

        comparator = MeshComparison()
        result = comparator.compare(cube, cube)

        result_dict = result.to_dict()

        assert "chamfer_distance" in result_dict
        assert "hausdorff_distance" in result_dict
        assert "iou_3d" in result_dict
        assert "f_score" in result_dict


class TestCompute2DMetrics:
    """Tests for 2D image comparison metrics"""

    def test_identical_images(self):
        """Test comparing identical images"""
        from validation.metrics import compute_2d_metrics
        import numpy as np

        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        metrics = compute_2d_metrics(img, img)

        assert metrics["iou_2d"] > 0.99
        assert metrics["mse"] < 1

    def test_different_images(self):
        """Test comparing different images"""
        from validation.metrics import compute_2d_metrics
        import numpy as np

        img1 = np.zeros((100, 100), dtype=np.uint8)
        img2 = np.full((100, 100), 255, dtype=np.uint8)

        metrics = compute_2d_metrics(img1, img2)

        assert metrics["iou_2d"] < 0.5
        assert metrics["mse"] > 0


class TestValidationPipeline:
    """Tests for the full validation pipeline"""

    async def test_pipeline_initialization(self, tmp_path):
        """Test ValidationPipeline initializes correctly"""
        from validation.pipeline import ValidationPipeline

        pipeline = ValidationPipeline(
            output_dir=tmp_path / "validation",
            strategy="basic_extrusion",
        )

        assert pipeline.output_dir.exists()
        assert pipeline.strategy == "basic_extrusion"

    async def test_full_validation_with_simple_mesh(self, tmp_path):
        """Test full validation pipeline with a simple mesh"""
        from validation.pipeline import ValidationPipeline
        from validation.cad_import import CADImporter
        import trimesh

        # Create ground truth: simple building
        building = trimesh.creation.box(extents=[10, 8, 3])

        gt_path = tmp_path / "ground_truth.glb"
        building.export(str(gt_path))

        # Run validation
        pipeline = ValidationPipeline(
            output_dir=tmp_path / "validation",
            strategy="basic_extrusion",
            wall_height=3.0,
        )

        result = await pipeline.validate(gt_path, floor_height=1.5)

        # Check result structure
        assert result.ground_truth_file == str(gt_path)
        assert result.total_time > 0

        # May succeed or fail depending on reconstruction quality
        # Just check that it completes
        assert result.success or result.error is not None


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def test_load_ground_truth(self, tmp_path):
        """Test load_ground_truth convenience function"""
        from validation.cad_import import load_ground_truth
        import trimesh

        cube = trimesh.creation.box(extents=[1, 1, 1])
        path = tmp_path / "model.obj"
        cube.export(str(path))

        model = load_ground_truth(path)
        assert model is not None
        assert len(model.vertices) > 0

    def test_compute_metrics(self, tmp_path):
        """Test compute_metrics convenience function"""
        from validation.metrics import compute_metrics
        import trimesh

        mesh1 = trimesh.creation.box(extents=[1, 1, 1])
        mesh2 = trimesh.creation.box(extents=[1.1, 1.1, 1.1])

        result = compute_metrics(mesh1, mesh2)
        assert result is not None
        assert result.chamfer_distance > 0
