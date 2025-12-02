"""
2D Projection Generator - Create floor plans from 3D CAD models

Generates orthographic projections (top-down views) from 3D models
to create synthetic training data pairs (2D input, 3D ground truth).
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Projection2D:
    """Result of projecting a 3D model to 2D"""
    image: np.ndarray  # 2D image array (H, W) or (H, W, C)
    width: int
    height: int
    scale: float  # meters per pixel
    origin: Tuple[float, float]  # World origin (x, y) at image (0, 0)
    floor_height: float  # Z-level of the slice
    metadata: Dict[str, Any]


class ProjectionGenerator:
    """
    Generate 2D projections from 3D models.

    Supports:
    - Top-down orthographic projection (floor plan)
    - Cross-section slices at different heights
    - Elevation views (front, side)
    """

    def __init__(
        self,
        resolution: int = 1024,
        padding: float = 0.1,  # 10% padding around model
        line_width: int = 2,
        background_color: int = 255,
        line_color: int = 0,
    ):
        self.resolution = resolution
        self.padding = padding
        self.line_width = line_width
        self.background_color = background_color
        self.line_color = line_color

    def generate_floor_plan(
        self,
        mesh,
        floor_height: float = 1.0,
        slice_thickness: float = 0.5,
    ) -> Projection2D:
        """
        Generate a floor plan (top-down view) from a 3D mesh.

        Args:
            mesh: trimesh.Trimesh object
            floor_height: Z-level to slice at (meters from ground)
            slice_thickness: Thickness of the slice (for capturing walls)

        Returns:
            Projection2D with floor plan image
        """
        import trimesh
        from PIL import Image, ImageDraw

        # Get mesh bounds
        bounds = mesh.bounds
        min_pt = bounds[0]
        max_pt = bounds[1]

        # Calculate world dimensions
        world_width = max_pt[0] - min_pt[0]
        world_height = max_pt[1] - min_pt[1]

        # Add padding
        pad_x = world_width * self.padding
        pad_y = world_height * self.padding

        # Calculate scale (meters per pixel)
        aspect = world_width / world_height if world_height > 0 else 1.0

        if aspect > 1:
            img_width = self.resolution
            img_height = int(self.resolution / aspect)
        else:
            img_height = self.resolution
            img_width = int(self.resolution * aspect)

        scale = (world_width + 2 * pad_x) / img_width

        # Create image
        img = Image.new('L', (img_width, img_height), self.background_color)
        draw = ImageDraw.Draw(img)

        # Slice mesh at floor height
        try:
            # Create a horizontal plane at floor_height
            plane_origin = [0, 0, floor_height]
            plane_normal = [0, 0, 1]

            # Get cross-section
            slice_2d = mesh.section(
                plane_origin=plane_origin,
                plane_normal=plane_normal
            )

            if slice_2d is not None:
                # Get 2D path from slice
                slice_path, _ = slice_2d.to_planar()

                # Draw each polygon in the slice
                for entity in slice_path.entities:
                    if hasattr(entity, 'points'):
                        points = slice_path.vertices[entity.points]
                    else:
                        continue

                    # Transform to image coordinates
                    img_points = []
                    for pt in points:
                        x = int((pt[0] - min_pt[0] + pad_x) / scale)
                        y = int(img_height - (pt[1] - min_pt[1] + pad_y) / scale)
                        img_points.append((x, y))

                    if len(img_points) >= 2:
                        draw.line(img_points, fill=self.line_color, width=self.line_width)

        except Exception as e:
            logger.warning(f"Section slice failed: {e}, using vertex projection")
            # Fallback: project all vertices
            self._project_vertices(mesh, draw, min_pt, pad_x, pad_y, scale, img_height)

        # Also try edge projection for better results
        self._project_edges(mesh, draw, min_pt, pad_x, pad_y, scale, img_height, floor_height, slice_thickness)

        return Projection2D(
            image=np.array(img),
            width=img_width,
            height=img_height,
            scale=scale,
            origin=(min_pt[0] - pad_x, min_pt[1] - pad_y),
            floor_height=floor_height,
            metadata={
                "world_bounds": bounds.tolist(),
                "slice_thickness": slice_thickness,
            }
        )

    def _project_edges(
        self,
        mesh,
        draw,
        min_pt: np.ndarray,
        pad_x: float,
        pad_y: float,
        scale: float,
        img_height: int,
        floor_height: float,
        slice_thickness: float,
    ):
        """Project mesh edges that intersect the floor plane"""
        import trimesh

        # Get edges
        edges = mesh.edges_unique
        vertices = mesh.vertices

        for edge in edges:
            v0 = vertices[edge[0]]
            v1 = vertices[edge[1]]

            # Check if edge is near the floor height (for walls)
            z_min = min(v0[2], v1[2])
            z_max = max(v0[2], v1[2])

            # Include edge if it spans the floor height
            if z_min <= floor_height + slice_thickness and z_max >= floor_height - slice_thickness:
                # Project to 2D (ignore Z)
                x0 = int((v0[0] - min_pt[0] + pad_x) / scale)
                y0 = int(img_height - (v0[1] - min_pt[1] + pad_y) / scale)
                x1 = int((v1[0] - min_pt[0] + pad_x) / scale)
                y1 = int(img_height - (v1[1] - min_pt[1] + pad_y) / scale)

                draw.line([(x0, y0), (x1, y1)], fill=self.line_color, width=self.line_width)

    def _project_vertices(
        self,
        mesh,
        draw,
        min_pt: np.ndarray,
        pad_x: float,
        pad_y: float,
        scale: float,
        img_height: int,
    ):
        """Simple vertex projection (fallback)"""
        for face in mesh.faces:
            points = []
            for vertex_idx in face:
                v = mesh.vertices[vertex_idx]
                x = int((v[0] - min_pt[0] + pad_x) / scale)
                y = int(img_height - (v[1] - min_pt[1] + pad_y) / scale)
                points.append((x, y))

            if len(points) >= 3:
                draw.polygon(points, outline=self.line_color)

    def generate_elevation(
        self,
        mesh,
        direction: str = "front",  # front, back, left, right
    ) -> Projection2D:
        """
        Generate an elevation view (side view) from a 3D mesh.

        Args:
            mesh: trimesh.Trimesh object
            direction: View direction (front, back, left, right)

        Returns:
            Projection2D with elevation image
        """
        from PIL import Image, ImageDraw

        bounds = mesh.bounds
        min_pt = bounds[0]
        max_pt = bounds[1]

        # Determine view axes based on direction
        if direction in ("front", "back"):
            # X-Z plane
            world_width = max_pt[0] - min_pt[0]
            world_height = max_pt[2] - min_pt[2]
            x_idx, z_idx = 0, 2
            y_offset = min_pt[0]
            flip_x = direction == "back"
        else:
            # Y-Z plane
            world_width = max_pt[1] - min_pt[1]
            world_height = max_pt[2] - min_pt[2]
            x_idx, z_idx = 1, 2
            y_offset = min_pt[1]
            flip_x = direction == "right"

        pad_x = world_width * self.padding
        pad_z = world_height * self.padding

        aspect = world_width / world_height if world_height > 0 else 1.0

        if aspect > 1:
            img_width = self.resolution
            img_height = int(self.resolution / aspect)
        else:
            img_height = self.resolution
            img_width = int(self.resolution * aspect)

        scale = (world_width + 2 * pad_x) / img_width

        img = Image.new('L', (img_width, img_height), self.background_color)
        draw = ImageDraw.Draw(img)

        # Project edges
        edges = mesh.edges_unique
        vertices = mesh.vertices

        for edge in edges:
            v0 = vertices[edge[0]]
            v1 = vertices[edge[1]]

            x0 = v0[x_idx] - y_offset + pad_x
            z0 = v0[z_idx] - min_pt[2] + pad_z
            x1 = v1[x_idx] - y_offset + pad_x
            z1 = v1[z_idx] - min_pt[2] + pad_z

            if flip_x:
                x0 = world_width + 2 * pad_x - x0
                x1 = world_width + 2 * pad_x - x1

            px0 = int(x0 / scale)
            py0 = int(img_height - z0 / scale)
            px1 = int(x1 / scale)
            py1 = int(img_height - z1 / scale)

            draw.line([(px0, py0), (px1, py1)], fill=self.line_color, width=self.line_width)

        return Projection2D(
            image=np.array(img),
            width=img_width,
            height=img_height,
            scale=scale,
            origin=(y_offset - pad_x, min_pt[2] - pad_z),
            floor_height=0,
            metadata={
                "direction": direction,
                "world_bounds": bounds.tolist(),
            }
        )

    def generate_multi_floor(
        self,
        mesh,
        floor_heights: List[float],
    ) -> List[Projection2D]:
        """
        Generate floor plans for multiple floors.

        Args:
            mesh: trimesh.Trimesh object
            floor_heights: List of Z-levels for each floor

        Returns:
            List of Projection2D, one per floor
        """
        projections = []
        for height in floor_heights:
            proj = self.generate_floor_plan(mesh, floor_height=height)
            projections.append(proj)
        return projections

    def save_projection(
        self,
        projection: Projection2D,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Save projection as image file.

        Args:
            projection: Projection2D to save
            output_path: Output file path (.png, .jpg)

        Returns:
            Path to saved file
        """
        from PIL import Image

        output_path = Path(output_path)
        img = Image.fromarray(projection.image)
        img.save(str(output_path))

        logger.info(f"Saved projection to {output_path}")
        return output_path

    def generate_training_pair(
        self,
        mesh,
        output_dir: Union[str, Path],
        name: str,
        floor_height: float = 1.0,
    ) -> Dict[str, Path]:
        """
        Generate a training pair: 2D floor plan (X) and save 3D model info (Y metadata).

        Args:
            mesh: trimesh.Trimesh object (ground truth 3D)
            output_dir: Directory to save files
            name: Base name for files
            floor_height: Height for floor plan slice

        Returns:
            Dict with paths to generated files
        """
        import trimesh
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate 2D projection (X)
        projection = self.generate_floor_plan(mesh, floor_height=floor_height)
        x_path = output_dir / f"{name}_2d.png"
        self.save_projection(projection, x_path)

        # Save 3D model (Y)
        y_path = output_dir / f"{name}_3d.glb"
        mesh.export(str(y_path))

        # Save metadata
        meta_path = output_dir / f"{name}_meta.json"
        metadata = {
            "name": name,
            "x_file": str(x_path.name),
            "y_file": str(y_path.name),
            "projection": {
                "scale": projection.scale,
                "origin": projection.origin,
                "floor_height": projection.floor_height,
                "width": projection.width,
                "height": projection.height,
            },
            "mesh": {
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "bounds": mesh.bounds.tolist(),
            }
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return {
            "x": x_path,
            "y": y_path,
            "metadata": meta_path,
        }
