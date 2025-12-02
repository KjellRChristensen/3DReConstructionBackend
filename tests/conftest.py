"""
Pytest configuration and fixtures for reconstruction tests
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_floorplan_path():
    """Path to sample floor plan image"""
    return Path(__file__).parent.parent / "data" / "input" / "sample_floorplan.png"

@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory for tests"""
    output = tmp_path / "output"
    output.mkdir()
    return output

@pytest.fixture
def sample_reconstruction_input(sample_floorplan_path):
    """Create a ReconstructionInput for testing"""
    from reconstruction.strategies import ReconstructionInput
    return ReconstructionInput(
        primary_image=sample_floorplan_path,
        wall_height=2.8,
        floor_thickness=0.3,
        scale=0.01,  # 1 pixel = 1cm
    )
