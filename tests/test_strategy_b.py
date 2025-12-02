"""
Tests for Strategy B: Basic Extrusion Reconstruction
"""
import pytest
import numpy as np
from pathlib import Path

pytestmark = pytest.mark.asyncio


class TestBasicExtrusionStrategy:
    """Test suite for BasicExtrusionStrategy"""

    def test_strategy_is_available(self):
        """Test that strategy reports availability correctly"""
        from reconstruction.strategies import BasicExtrusionStrategy

        strategy = BasicExtrusionStrategy()
        # Should be available since we have trimesh installed
        assert strategy.is_available() is True

    def test_strategy_properties(self):
        """Test strategy name and type"""
        from reconstruction.strategies import BasicExtrusionStrategy, StrategyType

        strategy = BasicExtrusionStrategy()
        assert strategy.name == "Basic Extrusion (Built-in)"
        assert strategy.strategy_type == StrategyType.BASIC_EXTRUSION

    async def test_reconstruct_sample_floorplan(self, sample_reconstruction_input, output_dir):
        """Test reconstruction of sample floor plan"""
        from reconstruction.strategies import BasicExtrusionStrategy

        strategy = BasicExtrusionStrategy()
        result = await strategy.reconstruct(sample_reconstruction_input)

        assert result.success is True
        assert result.model_data is not None
        assert result.format == "glb"
        assert result.strategy_used == "Basic Extrusion (Built-in)"
        assert result.metadata is not None

        # Save output for inspection
        output_path = output_dir / "test_output.glb"
        with open(output_path, "wb") as f:
            f.write(result.model_data)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    async def test_reconstruct_with_custom_wall_height(self, sample_floorplan_path, output_dir):
        """Test reconstruction with custom wall height"""
        from reconstruction.strategies import BasicExtrusionStrategy, ReconstructionInput

        strategy = BasicExtrusionStrategy()
        input_data = ReconstructionInput(
            primary_image=sample_floorplan_path,
            wall_height=3.5,  # Custom height
        )

        result = await strategy.reconstruct(input_data)

        assert result.success is True
        assert result.metadata["wall_height"] == 3.5


class TestWallDetection:
    """Test wall detection functionality"""

    def test_detect_walls_simple(self, sample_floorplan_path):
        """Test simple wall detection"""
        from reconstruction.strategies import BasicExtrusionStrategy
        from PIL import Image
        import numpy as np

        strategy = BasicExtrusionStrategy()
        image = Image.open(sample_floorplan_path).convert('L')
        img_array = np.array(image)

        walls = strategy._detect_walls_simple(img_array)

        assert isinstance(walls, list)
        # Should detect at least the outer boundary
        assert len(walls) >= 1

        # Each wall should have points and area
        for wall in walls:
            assert "points" in wall
            assert "area" in wall
            assert len(wall["points"]) >= 4  # At least a rectangle

    def test_detect_walls_filters_noise(self):
        """Test that small contours (noise) are filtered out"""
        from reconstruction.strategies import BasicExtrusionStrategy
        import numpy as np

        strategy = BasicExtrusionStrategy()

        # Create test image with small noise and large rectangle
        img = np.ones((500, 500), dtype=np.uint8) * 255
        # Large rectangle (should be detected)
        img[100:400, 100:400] = 0
        # Small noise (should be filtered)
        img[10:15, 10:15] = 0

        walls = strategy._detect_walls_simple(img)

        # Should detect only the large rectangle, not the noise
        assert len(walls) >= 1
        # All detected walls should have significant area
        min_area = 500 * 500 * 0.001
        for wall in walls:
            assert wall["area"] > min_area


class TestWallExtrusion:
    """Test wall extrusion functionality"""

    def test_extrude_walls(self, sample_reconstruction_input):
        """Test wall extrusion creates valid mesh"""
        from reconstruction.strategies import BasicExtrusionStrategy
        import trimesh

        strategy = BasicExtrusionStrategy()

        # Create simple test walls
        walls = [
            {
                "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "area": 10000,
            }
        ]

        mesh = strategy._extrude_walls(walls, sample_reconstruction_input)

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        assert mesh.is_watertight or mesh.is_volume  # Should be a solid

    def test_extrude_walls_respects_scale(self, sample_floorplan_path):
        """Test that scale parameter is applied correctly"""
        from reconstruction.strategies import BasicExtrusionStrategy, ReconstructionInput
        import trimesh

        strategy = BasicExtrusionStrategy()

        walls = [
            {
                "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "area": 10000,
            }
        ]

        # Scale 0.01 = 1 pixel = 1cm = 0.01m
        input_data = ReconstructionInput(
            primary_image=sample_floorplan_path,
            wall_height=2.8,
            scale=0.01,
        )

        mesh = strategy._extrude_walls(walls, input_data)

        # 100 pixels * 0.01 = 1.0 meters
        bounds = mesh.bounds
        x_size = bounds[1][0] - bounds[0][0]
        y_size = bounds[1][1] - bounds[0][1]

        assert 0.9 < x_size < 1.1  # ~1.0m
        assert 0.9 < y_size < 1.1  # ~1.0m


class TestPlaceholderModel:
    """Test placeholder model creation"""

    def test_create_placeholder_model(self, sample_reconstruction_input):
        """Test placeholder model creation"""
        from reconstruction.strategies import BasicExtrusionStrategy
        import trimesh

        strategy = BasicExtrusionStrategy()
        mesh = strategy._create_placeholder_model((500, 800), sample_reconstruction_input)

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


class TestDoorWindowCutouts:
    """Tests for door and window cutout functionality"""

    def test_detect_openings(self, sample_floorplan_path):
        """Test detection of door/window openings in walls"""
        from reconstruction.strategies import BasicExtrusionStrategy
        from PIL import Image
        import numpy as np

        strategy = BasicExtrusionStrategy()
        image = Image.open(sample_floorplan_path).convert('L')
        img_array = np.array(image)

        # This should detect gaps in walls
        openings = strategy._detect_openings(img_array)

        assert isinstance(openings, list)
        # Sample floor plan has doors
        assert len(openings) >= 1

        for opening in openings:
            assert "type" in opening  # 'door' or 'window'
            assert "position" in opening
            assert "width" in opening

    def test_apply_cutouts_to_mesh(self, sample_reconstruction_input):
        """Test applying door/window cutouts to extruded walls"""
        from reconstruction.strategies import BasicExtrusionStrategy
        import trimesh

        strategy = BasicExtrusionStrategy()

        # Create a simple wall mesh
        walls = [{"points": [[0, 0], [400, 0], [400, 20], [0, 20]], "area": 8000}]
        wall_mesh = strategy._extrude_walls(walls, sample_reconstruction_input)

        # Define a door opening
        openings = [
            {
                "type": "door",
                "position": (100, 0),  # x, y in pixels
                "width": 90,  # ~0.9m door
                "height": 210,  # ~2.1m door height
            }
        ]

        mesh_with_cutouts = strategy._apply_cutouts(wall_mesh, openings, sample_reconstruction_input)

        assert isinstance(mesh_with_cutouts, trimesh.Trimesh)
        # Mesh with cutouts should have different vertex count
        # (actual check depends on implementation)


class TestFloorCeilingSlab:
    """Tests for floor and ceiling slab generation"""

    def test_create_floor_slab(self, sample_reconstruction_input):
        """Test floor slab creation from detected walls"""
        from reconstruction.strategies import BasicExtrusionStrategy
        import trimesh

        strategy = BasicExtrusionStrategy()

        # Room boundary points
        room_boundary = [[0, 0], [500, 0], [500, 400], [0, 400]]

        floor_mesh = strategy._create_floor_slab(room_boundary, sample_reconstruction_input)

        assert isinstance(floor_mesh, trimesh.Trimesh)
        assert len(floor_mesh.vertices) > 0

        # Floor top should be at z=0, bottom at -thickness
        thickness = sample_reconstruction_input.floor_thickness
        assert floor_mesh.bounds[0][2] >= -thickness - 0.01  # Bottom
        assert floor_mesh.bounds[1][2] <= 0.01  # Top near z=0

    def test_create_ceiling_slab(self, sample_reconstruction_input):
        """Test ceiling slab creation"""
        from reconstruction.strategies import BasicExtrusionStrategy
        import trimesh

        strategy = BasicExtrusionStrategy()

        room_boundary = [[0, 0], [500, 0], [500, 400], [0, 400]]

        ceiling_mesh = strategy._create_ceiling_slab(room_boundary, sample_reconstruction_input)

        assert isinstance(ceiling_mesh, trimesh.Trimesh)

        # Ceiling should be at wall_height
        expected_height = sample_reconstruction_input.wall_height
        assert ceiling_mesh.bounds[0][2] >= expected_height - 0.1


class TestRoomSeparation:
    """Tests for room detection and separation"""

    def test_detect_rooms(self, sample_floorplan_path):
        """Test room detection from floor plan"""
        from reconstruction.strategies import BasicExtrusionStrategy
        from PIL import Image
        import numpy as np

        strategy = BasicExtrusionStrategy()
        image = Image.open(sample_floorplan_path).convert('L')
        img_array = np.array(image)

        rooms = strategy._detect_rooms(img_array)

        assert isinstance(rooms, list)
        # Should detect at least one room (the outer boundary or internal rooms)
        assert len(rooms) >= 1

        for room in rooms:
            assert "boundary" in room
            assert "area" in room
            assert "label" in room  # Estimated room label

    def test_separate_room_walls(self, sample_floorplan_path):
        """Test that rooms get individual wall meshes"""
        from reconstruction.strategies import BasicExtrusionStrategy, ReconstructionInput
        from PIL import Image
        import numpy as np

        strategy = BasicExtrusionStrategy()

        input_data = ReconstructionInput(
            primary_image=sample_floorplan_path,
            wall_height=2.8,
        )

        image = Image.open(sample_floorplan_path).convert('L')
        img_array = np.array(image)

        rooms = strategy._detect_rooms(img_array)
        room_meshes = strategy._create_room_meshes(rooms, input_data)

        assert isinstance(room_meshes, list)
        assert len(room_meshes) == len(rooms)
