"""
Orthographic Renderer for Training Data Generation

Generates proper 2D orthographic views from 3D CAD models with:
- Front, Top, Right, and Isometric views
- Dashed hidden line support (engineering drawing style)
- Proper scale and dimensions
- SVG and PNG output formats

Compatible with CAD2Program training requirements.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json

logger = logging.getLogger(__name__)


class ViewType(Enum):
    """Standard orthographic view types"""
    FRONT = "front"
    TOP = "top"
    RIGHT = "right"
    LEFT = "left"
    BACK = "back"
    BOTTOM = "bottom"
    ISOMETRIC = "isometric"


@dataclass
class RenderConfig:
    """Configuration for orthographic rendering"""
    resolution: int = 1024
    line_width: float = 2.0
    hidden_line_width: float = 1.0
    hidden_line_dash: Tuple[int, int] = (10, 5)  # dash, gap
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    line_color: Tuple[int, int, int, int] = (0, 0, 0, 255)
    hidden_line_color: Tuple[int, int, int, int] = (128, 128, 128, 255)
    margin: float = 0.1  # 10% margin around the model
    show_hidden_lines: bool = True
    output_format: str = "png"  # png or svg


@dataclass
class OrthographicView:
    """Single orthographic view result"""
    view_type: ViewType
    image: np.ndarray
    width: int
    height: int
    scale: float  # units per pixel
    origin: Tuple[float, float]  # world coordinates of image origin
    bounds: Dict[str, float]  # min/max in view coordinates
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingPair:
    """Complete training pair for CAD2Program"""
    model_path: str
    views: Dict[str, Path]  # view_type -> image path
    ground_truth_path: Path
    metadata_path: Path
    metadata: Dict[str, Any]


class OrthographicRenderer:
    """
    Renders 3D models to 2D orthographic projections.

    Produces engineering-style drawings with visible and hidden lines,
    suitable for training CAD2Program models.
    """

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self._check_dependencies()

    def _check_dependencies(self):
        """Check that required libraries are available"""
        try:
            import trimesh
            self.has_trimesh = True
        except ImportError:
            self.has_trimesh = False
            logger.warning("trimesh not available - limited functionality")

        try:
            import cv2
            self.has_opencv = True
        except ImportError:
            self.has_opencv = False
            logger.warning("OpenCV not available - no image processing")

    def load_model(self, model_path: Union[str, Path]) -> Any:
        """Load a 3D model from file"""
        import trimesh

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        mesh = trimesh.load(str(model_path))

        # Handle scene vs single mesh
        if isinstance(mesh, trimesh.Scene):
            # Combine all meshes in scene
            meshes = []
            for name, geometry in mesh.geometry.items():
                if isinstance(geometry, trimesh.Trimesh):
                    meshes.append(geometry)
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
            else:
                raise ValueError(f"No valid meshes in {model_path}")

        return mesh

    def get_view_transform(self, view_type: ViewType) -> np.ndarray:
        """Get the transformation matrix for a specific view"""
        # Standard orthographic view directions
        # Each view is defined by (camera_direction, up_vector)
        views = {
            ViewType.FRONT: (np.array([0, -1, 0]), np.array([0, 0, 1])),
            ViewType.BACK: (np.array([0, 1, 0]), np.array([0, 0, 1])),
            ViewType.RIGHT: (np.array([1, 0, 0]), np.array([0, 0, 1])),
            ViewType.LEFT: (np.array([-1, 0, 0]), np.array([0, 0, 1])),
            ViewType.TOP: (np.array([0, 0, 1]), np.array([0, 1, 0])),
            ViewType.BOTTOM: (np.array([0, 0, -1]), np.array([0, -1, 0])),
            ViewType.ISOMETRIC: (
                np.array([1, 1, 1]) / np.sqrt(3),
                np.array([-1, -1, 2]) / np.sqrt(6)
            ),
        }

        direction, up = views[view_type]

        # Build view matrix (camera looking at origin)
        z_axis = -direction / np.linalg.norm(direction)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # 4x4 view matrix
        view_matrix = np.eye(4)
        view_matrix[0, :3] = x_axis
        view_matrix[1, :3] = y_axis
        view_matrix[2, :3] = z_axis

        return view_matrix

    def project_vertices(
        self,
        vertices: np.ndarray,
        view_matrix: np.ndarray
    ) -> np.ndarray:
        """Project 3D vertices to 2D using orthographic projection"""
        # Add homogeneous coordinate
        ones = np.ones((vertices.shape[0], 1))
        vertices_h = np.hstack([vertices, ones])

        # Transform to view space
        transformed = (view_matrix @ vertices_h.T).T

        # Orthographic projection (just drop Z)
        return transformed[:, :2]

    def compute_edge_visibility(
        self,
        mesh: Any,
        view_direction: np.ndarray
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Compute which edges are visible vs hidden.

        Returns:
            (visible_edges, hidden_edges) as lists of (v1_idx, v2_idx) tuples
        """
        import trimesh

        # Get face normals
        face_normals = mesh.face_normals

        # Determine front-facing faces (facing the camera)
        dot_products = np.dot(face_normals, -view_direction)
        front_facing = dot_products > 0

        # Get edges from faces
        edges_from_faces = {}  # edge -> list of face indices

        for face_idx, face in enumerate(mesh.faces):
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge = tuple(sorted([v1, v2]))
                if edge not in edges_from_faces:
                    edges_from_faces[edge] = []
                edges_from_faces[edge].append(face_idx)

        visible_edges = []
        hidden_edges = []

        for edge, face_indices in edges_from_faces.items():
            # Check if any adjacent face is front-facing
            any_front = any(front_facing[fi] for fi in face_indices)
            all_front = all(front_facing[fi] for fi in face_indices)

            if all_front:
                # Both faces visible - this is a silhouette or internal edge
                visible_edges.append(edge)
            elif any_front:
                # One face visible, one hidden - this is a visible contour edge
                visible_edges.append(edge)
            else:
                # Both faces hidden - this edge is hidden
                hidden_edges.append(edge)

        return visible_edges, hidden_edges

    def render_view(
        self,
        mesh: Any,
        view_type: ViewType,
    ) -> OrthographicView:
        """
        Render a single orthographic view of the mesh.

        Args:
            mesh: trimesh mesh object
            view_type: which view to render

        Returns:
            OrthographicView with rendered image and metadata
        """
        import cv2

        config = self.config

        # Get view transform
        view_matrix = self.get_view_transform(view_type)
        view_direction = -view_matrix[2, :3]  # Camera looks along -Z

        # Project all vertices
        projected = self.project_vertices(mesh.vertices, view_matrix)

        # Compute bounds
        min_xy = projected.min(axis=0)
        max_xy = projected.max(axis=0)
        size = max_xy - min_xy

        # Add margin
        margin = size * config.margin
        min_xy -= margin
        max_xy += margin
        size = max_xy - min_xy

        # Compute scale (units per pixel)
        scale = max(size) / (config.resolution * (1 - 2 * config.margin))

        # Create image
        img_size = config.resolution
        image = np.ones((img_size, img_size, 4), dtype=np.uint8)
        image[:, :] = config.background_color

        # Transform to image coordinates
        def to_image_coords(pts):
            # Center in image
            center = (min_xy + max_xy) / 2
            pts_centered = pts - center
            pts_scaled = pts_centered / scale
            pts_image = pts_scaled + img_size / 2
            # Flip Y axis (image Y is down)
            pts_image[:, 1] = img_size - pts_image[:, 1]
            return pts_image.astype(np.int32)

        projected_image = to_image_coords(projected)

        # Compute edge visibility
        visible_edges, hidden_edges = self.compute_edge_visibility(mesh, view_direction)

        # Draw hidden edges first (if enabled)
        if config.show_hidden_lines and hidden_edges:
            for v1_idx, v2_idx in hidden_edges:
                pt1 = tuple(projected_image[v1_idx])
                pt2 = tuple(projected_image[v2_idx])

                # Draw dashed line
                self._draw_dashed_line(
                    image, pt1, pt2,
                    color=config.hidden_line_color,
                    thickness=int(config.hidden_line_width),
                    dash_length=config.hidden_line_dash[0],
                    gap_length=config.hidden_line_dash[1]
                )

        # Draw visible edges
        for v1_idx, v2_idx in visible_edges:
            pt1 = tuple(projected_image[v1_idx])
            pt2 = tuple(projected_image[v2_idx])
            cv2.line(
                image, pt1, pt2,
                color=config.line_color,
                thickness=int(config.line_width),
                lineType=cv2.LINE_AA
            )

        return OrthographicView(
            view_type=view_type,
            image=image,
            width=img_size,
            height=img_size,
            scale=scale,
            origin=(min_xy[0], min_xy[1]),
            bounds={
                "min_x": float(min_xy[0]),
                "min_y": float(min_xy[1]),
                "max_x": float(max_xy[0]),
                "max_y": float(max_xy[1]),
            },
            metadata={
                "view_type": view_type.value,
                "visible_edges": len(visible_edges),
                "hidden_edges": len(hidden_edges),
            }
        )

    def _draw_dashed_line(
        self,
        image: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int, int],
        thickness: int,
        dash_length: int,
        gap_length: int
    ):
        """Draw a dashed line between two points"""
        import cv2

        # Calculate line parameters
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = np.sqrt(dx*dx + dy*dy)

        if length < 1:
            return

        # Unit vector along line
        ux = dx / length
        uy = dy / length

        # Draw dashes
        pos = 0
        drawing = True

        while pos < length:
            if drawing:
                # Calculate dash end
                end_pos = min(pos + dash_length, length)

                start_pt = (
                    int(pt1[0] + pos * ux),
                    int(pt1[1] + pos * uy)
                )
                end_pt = (
                    int(pt1[0] + end_pos * ux),
                    int(pt1[1] + end_pos * uy)
                )

                cv2.line(image, start_pt, end_pt, color, thickness, cv2.LINE_AA)
                pos = end_pos + gap_length
            else:
                pos += gap_length

            drawing = not drawing

    def render_standard_views(
        self,
        mesh: Any,
        views: Optional[List[ViewType]] = None
    ) -> Dict[ViewType, OrthographicView]:
        """
        Render standard engineering views.

        Args:
            mesh: trimesh mesh object
            views: list of views to render (default: front, top, right)

        Returns:
            Dictionary mapping ViewType to OrthographicView
        """
        if views is None:
            views = [ViewType.FRONT, ViewType.TOP, ViewType.RIGHT]

        results = {}
        for view_type in views:
            logger.info(f"Rendering {view_type.value} view...")
            results[view_type] = self.render_view(mesh, view_type)

        return results

    def save_view(
        self,
        view: OrthographicView,
        output_path: Union[str, Path],
        format: Optional[str] = None
    ) -> Path:
        """Save a rendered view to file"""
        import cv2

        output_path = Path(output_path)
        format = format or self.config.output_format

        if format == "png":
            # Convert RGBA to BGRA for OpenCV
            image_bgra = cv2.cvtColor(view.image, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(str(output_path), image_bgra)
        elif format == "svg":
            # SVG output would require different rendering approach
            # For now, save as PNG
            logger.warning("SVG output not yet implemented, saving as PNG")
            output_path = output_path.with_suffix(".png")
            image_bgra = cv2.cvtColor(view.image, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(str(output_path), image_bgra)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return output_path

    def generate_training_pair(
        self,
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        name: Optional[str] = None,
        views: Optional[List[ViewType]] = None,
        copy_ground_truth: bool = True
    ) -> TrainingPair:
        """
        Generate a complete training pair for CAD2Program.

        Creates:
        - Orthographic view images (front, top, right)
        - Copy of ground truth 3D model
        - Metadata JSON file

        Args:
            model_path: Path to source 3D model
            output_dir: Directory to save outputs
            name: Base name for output files (default: model filename)
            views: Which views to render
            copy_ground_truth: Whether to copy the 3D model

        Returns:
            TrainingPair with paths to all generated files
        """
        import shutil

        model_path = Path(model_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name = name or model_path.stem

        # Load model
        logger.info(f"Loading model: {model_path}")
        mesh = self.load_model(model_path)

        # Get model info
        bounds = mesh.bounds
        extents = mesh.extents

        # Render views
        if views is None:
            views = [ViewType.FRONT, ViewType.TOP, ViewType.RIGHT]

        rendered_views = self.render_standard_views(mesh, views)

        # Save views
        view_paths = {}
        for view_type, view in rendered_views.items():
            view_filename = f"{name}_{view_type.value}.png"
            view_path = output_dir / view_filename
            self.save_view(view, view_path)
            view_paths[view_type.value] = view_path
            logger.info(f"Saved {view_type.value} view: {view_path}")

        # Copy ground truth
        if copy_ground_truth:
            gt_path = output_dir / f"{name}_3d{model_path.suffix}"
            shutil.copy2(model_path, gt_path)
        else:
            gt_path = model_path

        # Create metadata
        metadata = {
            "name": name,
            "source_model": str(model_path),
            "ground_truth": str(gt_path),
            "views": {k: str(v) for k, v in view_paths.items()},
            "model_info": {
                "bounds_min": bounds[0].tolist(),
                "bounds_max": bounds[1].tolist(),
                "extents": extents.tolist(),
                "num_vertices": len(mesh.vertices),
                "num_faces": len(mesh.faces),
            },
            "render_config": {
                "resolution": self.config.resolution,
                "show_hidden_lines": self.config.show_hidden_lines,
            },
            "view_metadata": {
                vt.value: {
                    "scale": rv.scale,
                    "origin": rv.origin,
                    "bounds": rv.bounds,
                    "visible_edges": rv.metadata.get("visible_edges", 0),
                    "hidden_edges": rv.metadata.get("hidden_edges", 0),
                }
                for vt, rv in rendered_views.items()
            }
        }

        # Save metadata
        metadata_path = output_dir / f"{name}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return TrainingPair(
            model_path=str(model_path),
            views={k: Path(v) for k, v in view_paths.items()},
            ground_truth_path=gt_path,
            metadata_path=metadata_path,
            metadata=metadata
        )

    def batch_generate(
        self,
        model_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        views: Optional[List[ViewType]] = None,
        skip_errors: bool = True
    ) -> Dict[str, Any]:
        """
        Generate training pairs for multiple models.

        Args:
            model_paths: List of paths to 3D models
            output_dir: Directory to save all outputs
            views: Which views to render for each model
            skip_errors: Continue on individual model errors

        Returns:
            Summary dict with success/failure counts and paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "successful": [],
            "failed": [],
            "total": len(model_paths),
        }

        for i, model_path in enumerate(model_paths):
            model_path = Path(model_path)
            logger.info(f"Processing {i+1}/{len(model_paths)}: {model_path.name}")

            try:
                pair = self.generate_training_pair(
                    model_path=model_path,
                    output_dir=output_dir,
                    views=views
                )
                results["successful"].append({
                    "model": str(model_path),
                    "metadata": str(pair.metadata_path),
                })
            except Exception as e:
                logger.error(f"Failed to process {model_path}: {e}")
                results["failed"].append({
                    "model": str(model_path),
                    "error": str(e),
                })
                if not skip_errors:
                    raise

        # Save batch summary
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"Batch complete: {len(results['successful'])} successful, "
            f"{len(results['failed'])} failed"
        )

        return results


# Convenience functions
def render_model_views(
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    resolution: int = 1024,
    show_hidden_lines: bool = True
) -> TrainingPair:
    """
    Convenience function to render standard views of a model.

    Args:
        model_path: Path to 3D model file
        output_dir: Directory to save outputs
        resolution: Image resolution in pixels
        show_hidden_lines: Whether to show dashed hidden lines

    Returns:
        TrainingPair with all generated files
    """
    config = RenderConfig(
        resolution=resolution,
        show_hidden_lines=show_hidden_lines
    )
    renderer = OrthographicRenderer(config)
    return renderer.generate_training_pair(model_path, output_dir)


def batch_render_models(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    extensions: List[str] = None,
    resolution: int = 1024
) -> Dict[str, Any]:
    """
    Render all models in a directory.

    Args:
        input_dir: Directory containing 3D models
        output_dir: Directory to save outputs
        extensions: File extensions to process (default: .obj, .glb, .stl)
        resolution: Image resolution in pixels

    Returns:
        Batch summary dict
    """
    if extensions is None:
        extensions = [".obj", ".glb", ".gltf", ".stl", ".ply"]

    input_dir = Path(input_dir)
    model_paths = []

    for ext in extensions:
        model_paths.extend(input_dir.glob(f"*{ext}"))
        model_paths.extend(input_dir.glob(f"*{ext.upper()}"))

    if not model_paths:
        logger.warning(f"No models found in {input_dir}")
        return {"successful": [], "failed": [], "total": 0}

    config = RenderConfig(resolution=resolution)
    renderer = OrthographicRenderer(config)

    return renderer.batch_generate(model_paths, output_dir)
