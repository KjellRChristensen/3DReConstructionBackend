"""
Render orthographic views from DeepCAD JSON files

This script processes DeepCAD CAD JSON files and generates 2D orthographic
projection views (front, top, right) for VLM training.

Usage:
    python scripts/render_deepcad_views.py --input data/training/deepcad_1k --resolution 512
    python scripts/render_deepcad_views.py --input data/training/deepcad_1k --views front,top,right --resolution 1024
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepCADRenderer:
    """Renders orthographic views from DeepCAD JSON files"""

    def __init__(self, resolution: int = 512):
        self.resolution = resolution
        self.margin = 0.1  # 10% margin around the model

    def parse_cad_json(self, json_path: Path) -> Dict:
        """Load and parse DeepCAD JSON file"""
        with open(json_path) as f:
            return json.load(f)

    def extract_points(self, cad_data: Dict) -> List[Tuple[float, float, float]]:
        """Extract all 3D points from CAD data"""
        points = []

        entities = cad_data.get('entities', {})
        for entity_id, entity in entities.items():
            # Extract points from sketches
            if entity.get('type') == 'Sketch':
                profiles = entity.get('profiles', {})
                for profile_id, profile in profiles.items():
                    loops = profile.get('loops', [])
                    for loop in loops:
                        curves = loop.get('profile_curves', [])
                        for curve in curves:
                            # Get start and end points
                            start = curve.get('start_point')
                            end = curve.get('end_point')

                            if start:
                                points.append((start['x'], start['y'], start['z']))
                            if end:
                                points.append((end['x'], end['y'], end['z']))

            # Extract points from extrusions
            elif entity.get('type') == 'ExtrudeFeature':
                # Extrusions reference sketches, so points are already captured
                pass

        return points

    def extract_edges(self, cad_data: Dict) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Extract all edges (line segments) from CAD data"""
        edges = []

        entities = cad_data.get('entities', {})
        for entity_id, entity in entities.items():
            if entity.get('type') == 'Sketch':
                profiles = entity.get('profiles', {})
                for profile_id, profile in profiles.items():
                    loops = profile.get('loops', [])
                    for loop in loops:
                        curves = loop.get('profile_curves', [])
                        for curve in curves:
                            start = curve.get('start_point')
                            end = curve.get('end_point')

                            if start and end:
                                p1 = (start['x'], start['y'], start['z'])
                                p2 = (end['x'], end['y'], end['z'])
                                edges.append((p1, p2))

        return edges

    def project_points(self, points: List[Tuple[float, float, float]], view: str) -> np.ndarray:
        """Project 3D points to 2D based on view direction"""
        if not points:
            return np.array([[0, 0]])

        points_array = np.array(points)

        # Orthographic projections
        if view == 'front':  # X-Y plane (looking along Z axis)
            projected = points_array[:, [0, 1]]
        elif view == 'top':  # X-Z plane (looking along Y axis)
            projected = points_array[:, [0, 2]]
        elif view == 'right':  # Y-Z plane (looking along X axis)
            projected = points_array[:, [1, 2]]
        else:
            raise ValueError(f"Unknown view: {view}")

        return projected

    def normalize_to_image(self, points_2d: np.ndarray, resolution: int) -> np.ndarray:
        """Normalize 2D points to image coordinates"""
        if len(points_2d) == 0 or (len(points_2d) == 1 and np.all(points_2d[0] == 0)):
            # Empty model, return center point
            return np.array([[resolution // 2, resolution // 2]])

        # Find bounding box
        min_vals = np.min(points_2d, axis=0)
        max_vals = np.max(points_2d, axis=0)

        # Add margin
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)  # Avoid division by zero
        margin_size = range_vals * self.margin
        min_vals -= margin_size
        max_vals += margin_size
        range_vals = max_vals - min_vals

        # Normalize to [0, resolution]
        normalized = (points_2d - min_vals) / range_vals * (resolution - 1)

        # Flip Y axis (image coordinates start from top-left)
        normalized[:, 1] = resolution - 1 - normalized[:, 1]

        return normalized

    def render_view_edges(self, edges: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
                         view: str, resolution: int) -> Image.Image:
        """Render a single view by drawing edges"""
        # Create white background
        img = Image.new('RGB', (resolution, resolution), 'white')
        draw = ImageDraw.Draw(img)

        if not edges:
            return img

        # Project edges to 2D
        projected_edges = []
        for p1, p2 in edges:
            proj_p1 = self.project_points([p1], view)[0]
            proj_p2 = self.project_points([p2], view)[0]
            projected_edges.append((proj_p1, proj_p2))

        # Normalize all points
        all_points = []
        for p1, p2 in projected_edges:
            all_points.append(p1)
            all_points.append(p2)

        if not all_points:
            return img

        all_points = np.array(all_points)
        normalized_points = self.normalize_to_image(all_points, resolution)

        # Draw edges
        for i in range(0, len(normalized_points), 2):
            p1 = tuple(normalized_points[i])
            p2 = tuple(normalized_points[i + 1])
            draw.line([p1, p2], fill='black', width=2)

        return img

    def render_cad_file(self, json_path: Path, output_dir: Path, views: List[str]):
        """Render all specified views for a single CAD file"""
        try:
            # Parse CAD data
            cad_data = self.parse_cad_json(json_path)

            # Extract geometry
            edges = self.extract_edges(cad_data)

            if not edges:
                logger.warning(f"No geometry found in {json_path.name}, creating blank images")

            # Render each view
            for view in views:
                img = self.render_view_edges(edges, view, self.resolution)

                # Save image
                output_file = output_dir / f"{json_path.stem}_{view}.png"
                img.save(output_file)

            return True

        except Exception as e:
            logger.error(f"Error rendering {json_path.name}: {e}")
            return False


def process_dataset(input_dir: Path, output_dir: Path, views: List[str], resolution: int = 512):
    """Process all CAD JSON files in the dataset"""

    # Find all JSON files
    cad_json_dir = input_dir / "cad_json"
    if not cad_json_dir.exists():
        logger.error(f"CAD JSON directory not found: {cad_json_dir}")
        return

    json_files = list(cad_json_dir.rglob("*.json"))
    logger.info(f"Found {len(json_files)} CAD JSON files")

    # Create output directory
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Create renderer
    renderer = DeepCADRenderer(resolution=resolution)

    # Process files
    success_count = 0
    fail_count = 0

    for i, json_file in enumerate(json_files):
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(json_files)} files...")

        if renderer.render_cad_file(json_file, output_images_dir, views):
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"✓ Rendering complete: {success_count} succeeded, {fail_count} failed")

    # Create metadata
    metadata = {
        "total_models": len(json_files),
        "successful": success_count,
        "failed": fail_count,
        "views": views,
        "resolution": resolution,
        "output_dir": str(output_images_dir)
    }

    metadata_path = output_dir / "rendering_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Saved metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render orthographic views from DeepCAD JSON files"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing cad_json subdirectory"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for rendered images (default: same as input)"
    )
    parser.add_argument(
        "--views",
        type=str,
        default="front,top,right",
        help="Comma-separated list of views to render (default: front,top,right)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution in pixels (default: 512)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    views = [v.strip() for v in args.views.split(',')]

    logger.info("DeepCAD Orthographic View Renderer")
    logger.info("=" * 60)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Views: {views}")
    logger.info(f"Resolution: {args.resolution}x{args.resolution}")
    logger.info("=" * 60)

    process_dataset(input_dir, output_dir, views, args.resolution)

    logger.info("\n✅ Rendering pipeline complete!")
    logger.info("Next step: Convert to TinyLLaVA format for VLM training")


if __name__ == "__main__":
    main()
