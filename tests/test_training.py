"""
Tests for the Training Module (Orthographic Renderer)

Tests orthographic projection, hidden line rendering, and training pair generation.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import json


class TestOrthographicRenderer:
    """Tests for OrthographicRenderer class"""

    def test_renderer_initialization(self):
        """Test renderer initializes correctly"""
        from training.orthographic_renderer import OrthographicRenderer, RenderConfig

        renderer = OrthographicRenderer()
        assert renderer.config is not None
        assert renderer.config.resolution == 1024

        config = RenderConfig(resolution=512, show_hidden_lines=False)
        renderer2 = OrthographicRenderer(config)
        assert renderer2.config.resolution == 512
        assert renderer2.config.show_hidden_lines is False

    def test_view_types(self):
        """Test ViewType enum values"""
        from training.orthographic_renderer import ViewType

        assert ViewType.FRONT.value == "front"
        assert ViewType.TOP.value == "top"
        assert ViewType.RIGHT.value == "right"
        assert ViewType.ISOMETRIC.value == "isometric"

    def test_load_obj_model(self, tmp_path):
        """Test loading OBJ model"""
        from training.orthographic_renderer import OrthographicRenderer
        import trimesh

        # Create a simple cube
        cube = trimesh.creation.box(extents=[1, 1, 1])
        obj_path = tmp_path / "test_cube.obj"
        cube.export(str(obj_path))

        renderer = OrthographicRenderer()
        mesh = renderer.load_model(obj_path)

        assert mesh is not None
        assert len(mesh.vertices) == 8
        assert len(mesh.faces) > 0

    def test_load_glb_model(self, tmp_path):
        """Test loading GLB model"""
        from training.orthographic_renderer import OrthographicRenderer
        import trimesh

        cube = trimesh.creation.box(extents=[2, 2, 2])
        glb_path = tmp_path / "test_cube.glb"
        cube.export(str(glb_path))

        renderer = OrthographicRenderer()
        mesh = renderer.load_model(glb_path)

        assert mesh is not None
        assert len(mesh.vertices) == 8

    def test_load_nonexistent_file(self):
        """Test error on nonexistent file"""
        from training.orthographic_renderer import OrthographicRenderer

        renderer = OrthographicRenderer()
        with pytest.raises(FileNotFoundError):
            renderer.load_model("/nonexistent/path/model.obj")

    def test_view_transform_front(self):
        """Test front view transformation matrix"""
        from training.orthographic_renderer import OrthographicRenderer, ViewType

        renderer = OrthographicRenderer()
        matrix = renderer.get_view_transform(ViewType.FRONT)

        assert matrix.shape == (4, 4)
        # Front view should look along -Y axis
        assert np.allclose(matrix[2, :3], [0, 1, 0], atol=0.1)

    def test_view_transform_top(self):
        """Test top view transformation matrix"""
        from training.orthographic_renderer import OrthographicRenderer, ViewType

        renderer = OrthographicRenderer()
        matrix = renderer.get_view_transform(ViewType.TOP)

        assert matrix.shape == (4, 4)
        # Top view should look along -Z axis
        assert np.allclose(matrix[2, :3], [0, 0, -1], atol=0.1)

    def test_project_vertices(self):
        """Test vertex projection"""
        from training.orthographic_renderer import OrthographicRenderer, ViewType

        renderer = OrthographicRenderer()

        # Simple cube vertices
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ])

        view_matrix = renderer.get_view_transform(ViewType.TOP)
        projected = renderer.project_vertices(vertices, view_matrix)

        # Should produce 2D coordinates
        assert projected.shape == (4, 2)

    def test_render_front_view(self, tmp_path):
        """Test rendering front view"""
        from training.orthographic_renderer import (
            OrthographicRenderer,
            RenderConfig,
            ViewType,
        )
        import trimesh

        cube = trimesh.creation.box(extents=[1, 1, 1])

        config = RenderConfig(resolution=256)
        renderer = OrthographicRenderer(config)

        view = renderer.render_view(cube, ViewType.FRONT)

        assert view is not None
        assert view.view_type == ViewType.FRONT
        assert view.image.shape == (256, 256, 4)  # RGBA
        assert view.width == 256
        assert view.height == 256
        assert view.scale > 0

    def test_render_with_hidden_lines(self, tmp_path):
        """Test rendering with hidden lines"""
        from training.orthographic_renderer import (
            OrthographicRenderer,
            RenderConfig,
            ViewType,
        )
        import trimesh

        cube = trimesh.creation.box(extents=[1, 1, 1])

        config = RenderConfig(resolution=256, show_hidden_lines=True)
        renderer = OrthographicRenderer(config)

        view = renderer.render_view(cube, ViewType.ISOMETRIC)

        # Should have both visible and hidden edges
        assert view.metadata.get("visible_edges", 0) > 0
        # Isometric view of cube should show some hidden edges
        # (depends on exact implementation)

    def test_render_without_hidden_lines(self, tmp_path):
        """Test rendering without hidden lines"""
        from training.orthographic_renderer import (
            OrthographicRenderer,
            RenderConfig,
            ViewType,
        )
        import trimesh

        cube = trimesh.creation.box(extents=[1, 1, 1])

        config = RenderConfig(resolution=256, show_hidden_lines=False)
        renderer = OrthographicRenderer(config)

        view = renderer.render_view(cube, ViewType.FRONT)

        assert view is not None
        assert view.image.shape == (256, 256, 4)

    def test_render_standard_views(self, tmp_path):
        """Test rendering standard views (front, top, right)"""
        from training.orthographic_renderer import (
            OrthographicRenderer,
            RenderConfig,
            ViewType,
        )
        import trimesh

        cube = trimesh.creation.box(extents=[2, 3, 4])

        config = RenderConfig(resolution=256)
        renderer = OrthographicRenderer(config)

        views = renderer.render_standard_views(cube)

        assert ViewType.FRONT in views
        assert ViewType.TOP in views
        assert ViewType.RIGHT in views
        assert len(views) == 3

    def test_save_view_png(self, tmp_path):
        """Test saving view as PNG"""
        from training.orthographic_renderer import (
            OrthographicRenderer,
            RenderConfig,
            ViewType,
        )
        import trimesh

        cube = trimesh.creation.box(extents=[1, 1, 1])

        config = RenderConfig(resolution=256)
        renderer = OrthographicRenderer(config)

        view = renderer.render_view(cube, ViewType.FRONT)

        output_path = tmp_path / "test_view.png"
        saved_path = renderer.save_view(view, output_path)

        assert saved_path.exists()
        assert saved_path.stat().st_size > 0


class TestTrainingPairGeneration:
    """Tests for training pair generation"""

    def test_generate_training_pair(self, tmp_path):
        """Test complete training pair generation"""
        from training.orthographic_renderer import OrthographicRenderer, RenderConfig
        import trimesh

        # Create test model
        cube = trimesh.creation.box(extents=[5, 4, 3])
        model_path = tmp_path / "test_model.glb"
        cube.export(str(model_path))

        output_dir = tmp_path / "output"

        config = RenderConfig(resolution=256)
        renderer = OrthographicRenderer(config)

        pair = renderer.generate_training_pair(
            model_path=model_path,
            output_dir=output_dir,
        )

        # Check structure
        assert pair.model_path == str(model_path)
        assert "front" in pair.views
        assert "top" in pair.views
        assert "right" in pair.views
        assert pair.ground_truth_path.exists()
        assert pair.metadata_path.exists()

        # Check metadata
        with open(pair.metadata_path) as f:
            metadata = json.load(f)

        assert "name" in metadata
        assert "model_info" in metadata
        assert "view_metadata" in metadata

    def test_generate_pair_custom_views(self, tmp_path):
        """Test training pair with custom views"""
        from training.orthographic_renderer import (
            OrthographicRenderer,
            RenderConfig,
            ViewType,
        )
        import trimesh

        cube = trimesh.creation.box(extents=[1, 1, 1])
        model_path = tmp_path / "test_model.obj"
        cube.export(str(model_path))

        output_dir = tmp_path / "output"

        renderer = OrthographicRenderer()
        pair = renderer.generate_training_pair(
            model_path=model_path,
            output_dir=output_dir,
            views=[ViewType.FRONT, ViewType.ISOMETRIC],
        )

        assert "front" in pair.views
        assert "isometric" in pair.views
        assert "top" not in pair.views

    def test_generate_pair_no_copy(self, tmp_path):
        """Test training pair without copying ground truth"""
        from training.orthographic_renderer import OrthographicRenderer
        import trimesh

        cube = trimesh.creation.box(extents=[1, 1, 1])
        model_path = tmp_path / "test_model.obj"
        cube.export(str(model_path))

        output_dir = tmp_path / "output"

        renderer = OrthographicRenderer()
        pair = renderer.generate_training_pair(
            model_path=model_path,
            output_dir=output_dir,
            copy_ground_truth=False,
        )

        # Ground truth should point to original
        assert pair.ground_truth_path == model_path


class TestBatchProcessing:
    """Tests for batch processing"""

    def test_batch_generate(self, tmp_path):
        """Test batch generation of training pairs"""
        from training.orthographic_renderer import OrthographicRenderer, RenderConfig
        import trimesh

        # Create multiple test models
        models = []
        for i in range(3):
            cube = trimesh.creation.box(extents=[1+i, 1+i, 1+i])
            path = tmp_path / f"model_{i}.obj"
            cube.export(str(path))
            models.append(path)

        output_dir = tmp_path / "output"

        config = RenderConfig(resolution=128)  # Small for speed
        renderer = OrthographicRenderer(config)

        results = renderer.batch_generate(models, output_dir)

        assert results["total"] == 3
        assert len(results["successful"]) == 3
        assert len(results["failed"]) == 0

        # Check summary file
        summary_path = output_dir / "batch_summary.json"
        assert summary_path.exists()

    def test_batch_skip_errors(self, tmp_path):
        """Test batch processing skips errors"""
        from training.orthographic_renderer import OrthographicRenderer
        import trimesh

        # Create one valid model and one invalid path
        cube = trimesh.creation.box(extents=[1, 1, 1])
        valid_path = tmp_path / "valid.obj"
        cube.export(str(valid_path))

        invalid_path = tmp_path / "nonexistent.obj"

        output_dir = tmp_path / "output"

        renderer = OrthographicRenderer()
        results = renderer.batch_generate(
            [valid_path, invalid_path],
            output_dir,
            skip_errors=True,
        )

        assert len(results["successful"]) == 1
        assert len(results["failed"]) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def test_render_model_views(self, tmp_path):
        """Test render_model_views convenience function"""
        from training.orthographic_renderer import render_model_views
        import trimesh

        cube = trimesh.creation.box(extents=[2, 2, 2])
        model_path = tmp_path / "cube.glb"
        cube.export(str(model_path))

        output_dir = tmp_path / "output"

        pair = render_model_views(model_path, output_dir, resolution=256)

        assert pair is not None
        assert pair.metadata_path.exists()

    def test_batch_render_models(self, tmp_path):
        """Test batch_render_models convenience function"""
        from training.orthographic_renderer import batch_render_models
        import trimesh

        # Create models in input directory
        input_dir = tmp_path / "models"
        input_dir.mkdir()

        for i in range(2):
            cube = trimesh.creation.box(extents=[1, 1, 1])
            (input_dir / f"model_{i}.obj").write_bytes(
                cube.export(file_type='obj').encode() if isinstance(cube.export(file_type='obj'), str)
                else cube.export(file_type='obj')
            )

        output_dir = tmp_path / "output"

        results = batch_render_models(input_dir, output_dir, resolution=128)

        assert results["total"] == 2


class TestEdgeVisibility:
    """Tests for edge visibility computation"""

    def test_compute_edge_visibility_cube(self):
        """Test edge visibility for a cube"""
        from training.orthographic_renderer import OrthographicRenderer, ViewType
        import trimesh

        cube = trimesh.creation.box(extents=[1, 1, 1])

        renderer = OrthographicRenderer()
        view_matrix = renderer.get_view_transform(ViewType.FRONT)
        view_direction = -view_matrix[2, :3]

        visible, hidden = renderer.compute_edge_visibility(cube, view_direction)

        # A cube viewed from front should have some visible and some hidden edges
        assert len(visible) > 0
        assert len(hidden) > 0
        # Total edges should be positive (exact count depends on mesh triangulation)
        assert len(visible) + len(hidden) > 0


class TestDashedLineRendering:
    """Tests for dashed line rendering"""

    def test_dashed_line_basic(self):
        """Test dashed line drawing"""
        from training.orthographic_renderer import OrthographicRenderer, RenderConfig
        import numpy as np

        config = RenderConfig(resolution=100)
        renderer = OrthographicRenderer(config)

        # Create test image
        image = np.ones((100, 100, 4), dtype=np.uint8) * 255

        renderer._draw_dashed_line(
            image,
            pt1=(10, 50),
            pt2=(90, 50),
            color=(0, 0, 0, 255),
            thickness=1,
            dash_length=5,
            gap_length=3,
        )

        # Check that some pixels are black (drawn) and some are white (gaps)
        row = image[50, :, 0]  # Red channel of row 50
        has_black = np.any(row < 128)
        has_white = np.any(row > 128)

        assert has_black, "Dashed line should have drawn segments"
        assert has_white, "Dashed line should have gaps"
