"""
Reconstruction Strategies - Multiple approaches for 2D to 3D conversion

Strategy A: External DNN APIs (Kaedim, Replicate, Meshy, etc.)
Strategy B: Built-in basic reconstruction (single-view wall extrusion)
Strategy C: Multi-view DNN reconstruction (GaussianCAD, custom models)
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
import httpx
import asyncio
import numpy as np

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    EXTERNAL_API = "external_api"
    BASIC_EXTRUSION = "basic_extrusion"
    MULTI_VIEW_DNN = "multi_view_dnn"
    VLM_CAD = "vlm_cad"  # Vision-Language Model for CAD reconstruction


@dataclass
class ReconstructionInput:
    """Input for reconstruction"""
    primary_image: Path  # Main floor plan / view
    additional_views: List[Path] = None  # Elevations, sections
    detected_elements: Optional[Dict] = None  # Walls, doors, windows from recognition
    vector_data: Optional[Dict] = None  # Vectorized line data

    # Parameters
    wall_height: float = 2.8
    floor_thickness: float = 0.3
    scale: Optional[float] = None  # meters per pixel

    def __post_init__(self):
        if self.additional_views is None:
            self.additional_views = []


@dataclass
class ReconstructionOutput:
    """Output from reconstruction"""
    success: bool
    model_path: Optional[Path] = None  # Path to generated 3D model
    model_data: Optional[bytes] = None  # Raw model data (GLB, OBJ, etc.)
    format: str = "glb"
    metadata: Optional[Dict] = None
    error: Optional[str] = None
    strategy_used: Optional[str] = None
    processing_time: Optional[float] = None


class ReconstructionStrategy(ABC):
    """Base class for reconstruction strategies"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass

    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        """Strategy type"""
        pass

    @abstractmethod
    async def reconstruct(self, input_data: ReconstructionInput) -> ReconstructionOutput:
        """Perform 3D reconstruction"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if strategy is available (API keys set, models loaded, etc.)"""
        pass


# =============================================================================
# Strategy A: External DNN API Integration
# =============================================================================

class ExternalAPIStrategy(ReconstructionStrategy):
    """
    Strategy A: Use external DNN APIs for reconstruction

    Supported services:
    - Kaedim: 2D sketch to 3D model
    - Meshy: AI 3D generation
    - Replicate: Various 3D models
    - Tripo3D: Image to 3D
    """

    def __init__(
        self,
        service: str = "kaedim",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.service = service.lower()
        self.api_key = api_key
        self.base_url = base_url or self._get_default_url()
        self._client = None

    @property
    def name(self) -> str:
        return f"External API ({self.service})"

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.EXTERNAL_API

    def _get_default_url(self) -> str:
        urls = {
            "kaedim": "https://api.kaedim3d.com/api/v1",
            "meshy": "https://api.meshy.ai/v1",
            "replicate": "https://api.replicate.com/v1",
            "tripo3d": "https://api.tripo3d.ai/v1",
        }
        return urls.get(self.service, "")

    def is_available(self) -> bool:
        return bool(self.api_key and self.base_url)

    async def reconstruct(self, input_data: ReconstructionInput) -> ReconstructionOutput:
        """Call external API for reconstruction"""
        if not self.is_available():
            return ReconstructionOutput(
                success=False,
                error=f"API key not configured for {self.service}",
                strategy_used=self.name,
            )

        try:
            if self.service == "kaedim":
                return await self._reconstruct_kaedim(input_data)
            elif self.service == "meshy":
                return await self._reconstruct_meshy(input_data)
            elif self.service == "replicate":
                return await self._reconstruct_replicate(input_data)
            elif self.service == "tripo3d":
                return await self._reconstruct_tripo3d(input_data)
            else:
                return ReconstructionOutput(
                    success=False,
                    error=f"Unknown service: {self.service}",
                    strategy_used=self.name,
                )
        except Exception as e:
            logger.error(f"External API error: {e}")
            return ReconstructionOutput(
                success=False,
                error=str(e),
                strategy_used=self.name,
            )

    async def _reconstruct_kaedim(self, input_data: ReconstructionInput) -> ReconstructionOutput:
        """Kaedim API integration"""
        async with httpx.AsyncClient() as client:
            # Upload image
            with open(input_data.primary_image, "rb") as f:
                image_data = f.read()

            # Create generation request
            response = await client.post(
                f"{self.base_url}/generate",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"image": (input_data.primary_image.name, image_data)},
                data={
                    "output_format": "glb",
                    "quality": "high",
                },
                timeout=300,
            )

            if response.status_code != 200:
                return ReconstructionOutput(
                    success=False,
                    error=f"Kaedim API error: {response.text}",
                    strategy_used=self.name,
                )

            result = response.json()

            # Poll for completion
            generation_id = result.get("id")
            model_url = await self._poll_kaedim_status(client, generation_id)

            if model_url:
                # Download model
                model_response = await client.get(model_url)
                return ReconstructionOutput(
                    success=True,
                    model_data=model_response.content,
                    format="glb",
                    strategy_used=self.name,
                    metadata={"generation_id": generation_id},
                )
            else:
                return ReconstructionOutput(
                    success=False,
                    error="Generation timed out",
                    strategy_used=self.name,
                )

    async def _poll_kaedim_status(self, client: httpx.AsyncClient, generation_id: str, timeout: int = 300) -> Optional[str]:
        """Poll Kaedim for generation completion"""
        import time
        start = time.time()

        while time.time() - start < timeout:
            response = await client.get(
                f"{self.base_url}/generations/{generation_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "completed":
                    return data.get("model_url")
                elif data.get("status") == "failed":
                    return None

            await asyncio.sleep(5)

        return None

    async def _reconstruct_meshy(self, input_data: ReconstructionInput) -> ReconstructionOutput:
        """Meshy API integration"""
        async with httpx.AsyncClient() as client:
            with open(input_data.primary_image, "rb") as f:
                image_data = f.read()

            # Image to 3D endpoint
            response = await client.post(
                f"{self.base_url}/image-to-3d",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"image": (input_data.primary_image.name, image_data)},
                data={"output_format": "glb"},
                timeout=300,
            )

            if response.status_code in [200, 201]:
                result = response.json()
                # Meshy returns task ID, need to poll
                task_id = result.get("result")

                # Poll for completion
                for _ in range(60):  # 5 minutes max
                    status_response = await client.get(
                        f"{self.base_url}/image-to-3d/{task_id}",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                    status_data = status_response.json()

                    if status_data.get("status") == "SUCCEEDED":
                        model_url = status_data.get("model_urls", {}).get("glb")
                        if model_url:
                            model_response = await client.get(model_url)
                            return ReconstructionOutput(
                                success=True,
                                model_data=model_response.content,
                                format="glb",
                                strategy_used=self.name,
                            )
                    elif status_data.get("status") == "FAILED":
                        return ReconstructionOutput(
                            success=False,
                            error="Meshy generation failed",
                            strategy_used=self.name,
                        )

                    await asyncio.sleep(5)

            return ReconstructionOutput(
                success=False,
                error=f"Meshy API error: {response.text}",
                strategy_used=self.name,
            )

    async def _reconstruct_replicate(self, input_data: ReconstructionInput) -> ReconstructionOutput:
        """Replicate API integration (various models)"""
        async with httpx.AsyncClient() as client:
            import base64

            with open(input_data.primary_image, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()

            # Use a 3D reconstruction model (e.g., TripoSR)
            response = await client.post(
                f"{self.base_url}/predictions",
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "version": "stability-ai/triposr",  # Example model
                    "input": {
                        "image": f"data:image/png;base64,{image_b64}",
                        "output_format": "glb",
                    },
                },
                timeout=300,
            )

            if response.status_code == 201:
                result = response.json()
                prediction_url = result.get("urls", {}).get("get")

                # Poll for completion
                for _ in range(60):
                    status_response = await client.get(
                        prediction_url,
                        headers={"Authorization": f"Token {self.api_key}"},
                    )
                    status_data = status_response.json()

                    if status_data.get("status") == "succeeded":
                        output = status_data.get("output")
                        if output:
                            model_response = await client.get(output)
                            return ReconstructionOutput(
                                success=True,
                                model_data=model_response.content,
                                format="glb",
                                strategy_used=self.name,
                            )
                    elif status_data.get("status") == "failed":
                        return ReconstructionOutput(
                            success=False,
                            error=status_data.get("error", "Replicate generation failed"),
                            strategy_used=self.name,
                        )

                    await asyncio.sleep(5)

            return ReconstructionOutput(
                success=False,
                error=f"Replicate API error: {response.text}",
                strategy_used=self.name,
            )

    async def _reconstruct_tripo3d(self, input_data: ReconstructionInput) -> ReconstructionOutput:
        """Tripo3D API integration"""
        # Similar pattern to others
        return ReconstructionOutput(
            success=False,
            error="Tripo3D integration not yet implemented",
            strategy_used=self.name,
        )


# =============================================================================
# Strategy B: Built-in Basic Reconstruction (Single View Extrusion)
# =============================================================================

class BasicExtrusionStrategy(ReconstructionStrategy):
    """
    Strategy B: Built-in basic reconstruction using wall extrusion

    Process:
    1. Take detected walls from floor plan
    2. Extrude walls to specified height
    3. Cut openings for doors/windows
    4. Generate floor/ceiling slabs
    5. Export as GLB/OBJ
    """

    def __init__(self):
        self._trimesh_available = None

    @property
    def name(self) -> str:
        return "Basic Extrusion (Built-in)"

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.BASIC_EXTRUSION

    def is_available(self) -> bool:
        if self._trimesh_available is None:
            try:
                import trimesh
                import numpy as np
                self._trimesh_available = True
            except ImportError:
                self._trimesh_available = False
        return self._trimesh_available

    async def reconstruct(self, input_data: ReconstructionInput) -> ReconstructionOutput:
        """Perform basic extrusion-based reconstruction"""
        if not self.is_available():
            return ReconstructionOutput(
                success=False,
                error="trimesh library not installed. Run: pip install trimesh",
                strategy_used=self.name,
            )

        try:
            import trimesh
            import numpy as np
            from PIL import Image

            # Load and process image
            image = Image.open(input_data.primary_image).convert('L')
            img_array = np.array(image)

            # Simple edge detection for walls
            walls = self._detect_walls_simple(img_array)

            if not walls:
                # Fallback: create a simple box from image bounds
                logger.warning("No walls detected, creating placeholder model")
                mesh = self._create_placeholder_model(img_array.shape, input_data)
            else:
                # Extrude walls
                mesh = self._extrude_walls(walls, input_data)

            # Export to GLB
            glb_data = mesh.export(file_type='glb')

            return ReconstructionOutput(
                success=True,
                model_data=glb_data,
                format="glb",
                strategy_used=self.name,
                metadata={
                    "walls_detected": len(walls) if walls else 0,
                    "wall_height": input_data.wall_height,
                },
            )

        except Exception as e:
            logger.error(f"Basic reconstruction error: {e}")
            return ReconstructionOutput(
                success=False,
                error=str(e),
                strategy_used=self.name,
            )

    def _detect_walls_simple(self, img_array) -> List[Dict]:
        """Simple wall detection using edge detection"""
        try:
            import cv2
            import numpy as np

            # Threshold to binary
            _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            walls = []
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) >= 4:  # At least a rectangle
                    points = approx.reshape(-1, 2).tolist()
                    walls.append({
                        "points": points,
                        "area": cv2.contourArea(contour),
                    })

            # Filter small contours (noise)
            min_area = img_array.shape[0] * img_array.shape[1] * 0.001
            walls = [w for w in walls if w["area"] > min_area]

            return walls

        except ImportError:
            logger.warning("OpenCV not available for wall detection")
            return []

    def _extrude_walls(self, walls: List[Dict], input_data: ReconstructionInput):
        """Extrude detected wall polygons to 3D"""
        import trimesh
        import numpy as np

        meshes = []
        scale = input_data.scale or 0.01  # Default: 1 pixel = 1cm
        height = input_data.wall_height

        for wall in walls:
            points = np.array(wall["points"]) * scale

            # Create 2D polygon
            try:
                from shapely.geometry import Polygon
                poly = Polygon(points)

                if poly.is_valid and poly.area > 0:
                    # Extrude to 3D
                    mesh = trimesh.creation.extrude_polygon(poly, height)
                    meshes.append(mesh)
            except Exception as e:
                logger.warning(f"Failed to extrude wall: {e}")
                continue

        if meshes:
            combined = trimesh.util.concatenate(meshes)
            return combined
        else:
            return self._create_placeholder_model((100, 100), input_data)

    def _create_placeholder_model(self, shape, input_data: ReconstructionInput):
        """Create a simple placeholder 3D model"""
        import trimesh

        scale = input_data.scale or 0.01
        width = shape[1] * scale
        depth = shape[0] * scale
        height = input_data.wall_height

        # Create a simple box representing the building footprint
        mesh = trimesh.creation.box(extents=[width, depth, height])
        mesh.apply_translation([width/2, depth/2, height/2])

        return mesh

    # =========================================================================
    # Door/Window Detection and Cutouts
    # =========================================================================

    def _detect_openings(self, img_array: np.ndarray) -> List[Dict]:
        """
        Detect door and window openings in walls.

        Looks for gaps in wall lines that indicate doors/windows.
        """
        try:
            import cv2

            # Threshold to binary (walls are dark)
            _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)

            # Detect horizontal and vertical lines (walls)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

            # Combine walls
            walls = cv2.bitwise_or(horizontal_lines, vertical_lines)

            # Find gaps in walls (potential openings)
            # Dilate walls slightly then find difference
            dilated = cv2.dilate(walls, np.ones((5, 5), np.uint8), iterations=2)

            # Find contours in the original binary that are NOT walls
            # These are potential openings
            openings = []

            # Use Hough Line Transform to find wall segments
            lines = cv2.HoughLinesP(walls, 1, np.pi/180, threshold=50,
                                    minLineLength=30, maxLineGap=10)

            if lines is not None:
                # Group lines by orientation and find gaps
                horizontal = []
                vertical = []

                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                    if abs(angle) < 20 or abs(angle) > 160:  # Horizontal
                        horizontal.append((x1, y1, x2, y2))
                    elif 70 < abs(angle) < 110:  # Vertical
                        vertical.append((x1, y1, x2, y2))

                # Find gaps in horizontal walls (doors on horizontal walls)
                openings.extend(self._find_gaps_in_lines(horizontal, "horizontal", img_array.shape))
                # Find gaps in vertical walls
                openings.extend(self._find_gaps_in_lines(vertical, "vertical", img_array.shape))

            return openings

        except ImportError:
            logger.warning("OpenCV not available for opening detection")
            return []

    def _find_gaps_in_lines(self, lines: List, orientation: str, img_shape: tuple) -> List[Dict]:
        """Find gaps between line segments that could be door/window openings"""
        if not lines:
            return []

        openings = []

        # Sort lines by position
        if orientation == "horizontal":
            # Group by y-coordinate (same wall)
            lines_sorted = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)
        else:
            # Group by x-coordinate
            lines_sorted = sorted(lines, key=lambda l: (l[0] + l[2]) / 2)

        # Find gaps between consecutive segments on same wall
        for i in range(len(lines_sorted) - 1):
            l1 = lines_sorted[i]
            l2 = lines_sorted[i + 1]

            if orientation == "horizontal":
                # Check if on same horizontal level
                y1_avg = (l1[1] + l1[3]) / 2
                y2_avg = (l2[1] + l2[3]) / 2

                if abs(y1_avg - y2_avg) < 20:  # Same wall
                    # Gap between end of l1 and start of l2
                    gap_start = max(l1[0], l1[2])
                    gap_end = min(l2[0], l2[2])
                    gap_width = gap_end - gap_start

                    if 50 < gap_width < 200:  # Reasonable door/window width (pixels)
                        opening_type = "door" if gap_width > 80 else "window"
                        openings.append({
                            "type": opening_type,
                            "position": (gap_start, int(y1_avg)),
                            "width": gap_width,
                            "height": 210 if opening_type == "door" else 120,
                            "orientation": orientation,
                        })
            else:
                # Vertical wall
                x1_avg = (l1[0] + l1[2]) / 2
                x2_avg = (l2[0] + l2[2]) / 2

                if abs(x1_avg - x2_avg) < 20:  # Same wall
                    gap_start = max(l1[1], l1[3])
                    gap_end = min(l2[1], l2[3])
                    gap_width = gap_end - gap_start

                    if 50 < gap_width < 200:
                        opening_type = "door" if gap_width > 80 else "window"
                        openings.append({
                            "type": opening_type,
                            "position": (int(x1_avg), gap_start),
                            "width": gap_width,
                            "height": 210 if opening_type == "door" else 120,
                            "orientation": orientation,
                        })

        return openings

    def _apply_cutouts(self, wall_mesh, openings: List[Dict], input_data: ReconstructionInput):
        """
        Apply door/window cutouts to wall mesh using boolean operations.
        """
        import trimesh

        if not openings:
            return wall_mesh

        scale = input_data.scale or 0.01
        result_mesh = wall_mesh

        for opening in openings:
            try:
                # Create cutout box
                width = opening["width"] * scale
                height_m = opening["height"] * scale / 100  # Convert cm to m
                depth = 0.5  # Wall thickness buffer

                cutout = trimesh.creation.box(extents=[width, depth, height_m])

                # Position the cutout
                x = opening["position"][0] * scale
                y = opening["position"][1] * scale

                if opening["type"] == "door":
                    z = height_m / 2  # Door starts at floor
                else:
                    z = 1.0 + height_m / 2  # Window at ~1m height

                cutout.apply_translation([x + width/2, y, z])

                # Boolean difference (requires manifold3d)
                try:
                    result_mesh = result_mesh.difference(cutout)
                except Exception as e:
                    logger.warning(f"Boolean operation failed: {e}")
                    continue

            except Exception as e:
                logger.warning(f"Failed to create cutout: {e}")
                continue

        return result_mesh

    # =========================================================================
    # Floor and Ceiling Slabs
    # =========================================================================

    def _create_floor_slab(self, room_boundary: List, input_data: ReconstructionInput):
        """
        Create a floor slab mesh from room boundary points.
        """
        import trimesh
        from shapely.geometry import Polygon

        scale = input_data.scale or 0.01
        thickness = input_data.floor_thickness

        # Scale boundary points
        points = np.array(room_boundary) * scale

        # Create polygon
        poly = Polygon(points)

        if not poly.is_valid:
            poly = poly.buffer(0)  # Fix invalid polygon

        if poly.area <= 0:
            logger.warning("Invalid floor polygon")
            return trimesh.Trimesh()

        # Extrude floor slab (thin, at z=0)
        floor_mesh = trimesh.creation.extrude_polygon(poly, thickness)

        # Move to z=0 (floor level) - extrusion starts at z=0 by default
        # Shift down by thickness so top of floor is at z=0
        floor_mesh.apply_translation([0, 0, -thickness])

        return floor_mesh

    def _create_ceiling_slab(self, room_boundary: List, input_data: ReconstructionInput):
        """
        Create a ceiling slab mesh from room boundary points.
        """
        import trimesh
        from shapely.geometry import Polygon

        scale = input_data.scale or 0.01
        thickness = input_data.floor_thickness
        wall_height = input_data.wall_height

        # Scale boundary points
        points = np.array(room_boundary) * scale

        # Create polygon
        poly = Polygon(points)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.area <= 0:
            logger.warning("Invalid ceiling polygon")
            return trimesh.Trimesh()

        # Extrude ceiling slab
        ceiling_mesh = trimesh.creation.extrude_polygon(poly, thickness)

        # Move to wall height
        ceiling_mesh.apply_translation([0, 0, wall_height])

        return ceiling_mesh

    # =========================================================================
    # Room Detection and Separation
    # =========================================================================

    def _detect_rooms(self, img_array: np.ndarray) -> List[Dict]:
        """
        Detect individual rooms from floor plan.

        Uses flood fill to find enclosed spaces.
        """
        try:
            import cv2

            # Threshold to binary
            _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)

            # Close small gaps in walls
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Find contours (rooms are white enclosed areas)
            contours, hierarchy = cv2.findContours(
                closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            rooms = []
            min_room_area = img_array.shape[0] * img_array.shape[1] * 0.005  # 0.5% of image

            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                # Filter by area (not too small, not the outer boundary)
                if area < min_room_area:
                    continue

                # Check if it's an inner contour (room) not outer boundary
                if hierarchy is not None and hierarchy[0][i][3] != -1:
                    # Has a parent, so it's an inner contour
                    continue

                # Approximate polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) >= 4:
                    boundary = approx.reshape(-1, 2).tolist()

                    # Try to detect room label using simple heuristics
                    # (Full OCR would require pytesseract)
                    label = self._estimate_room_label(contour, img_array)

                    rooms.append({
                        "boundary": boundary,
                        "area": area,
                        "label": label,
                        "centroid": self._get_centroid(boundary),
                    })

            # Sort rooms by area (largest first)
            rooms.sort(key=lambda r: r["area"], reverse=True)

            return rooms

        except ImportError:
            logger.warning("OpenCV not available for room detection")
            return []

    def _estimate_room_label(self, contour, img_array: np.ndarray) -> str:
        """
        Estimate room label based on size and position.

        For full OCR, would need pytesseract.
        """
        import cv2

        area = cv2.contourArea(contour)
        total_area = img_array.shape[0] * img_array.shape[1]
        ratio = area / total_area

        # Simple heuristics based on relative size
        if ratio > 0.15:
            return "Living Room"
        elif ratio > 0.08:
            return "Bedroom"
        elif ratio > 0.04:
            return "Kitchen"
        elif ratio > 0.02:
            return "Bathroom"
        else:
            return "Room"

    def _get_centroid(self, boundary: List) -> tuple:
        """Calculate centroid of a polygon"""
        points = np.array(boundary)
        return (int(np.mean(points[:, 0])), int(np.mean(points[:, 1])))

    def _create_room_meshes(self, rooms: List[Dict], input_data: ReconstructionInput) -> List:
        """
        Create individual mesh for each room (walls + floor + ceiling).
        """
        import trimesh

        room_meshes = []

        for room in rooms:
            try:
                boundary = room["boundary"]

                # Create walls for this room
                walls = [{"points": boundary, "area": room["area"]}]
                wall_mesh = self._extrude_walls(walls, input_data)

                # Create floor and ceiling
                floor_mesh = self._create_floor_slab(boundary, input_data)
                ceiling_mesh = self._create_ceiling_slab(boundary, input_data)

                # Combine into single room mesh
                meshes_to_combine = [m for m in [wall_mesh, floor_mesh, ceiling_mesh]
                                     if m is not None and len(m.vertices) > 0]

                if meshes_to_combine:
                    room_mesh = trimesh.util.concatenate(meshes_to_combine)
                    room_mesh.metadata["label"] = room.get("label", "Room")
                    room_meshes.append(room_mesh)

            except Exception as e:
                logger.warning(f"Failed to create room mesh: {e}")
                continue

        return room_meshes


# =============================================================================
# Strategy C: Multi-View DNN Reconstruction
# =============================================================================

class MultiViewDNNStrategy(ReconstructionStrategy):
    """
    Strategy C: Multi-view DNN reconstruction

    Approaches:
    - GaussianCAD: 3D Gaussian splatting from multiple views
    - CAD2Program: Vision-language model for parametric reconstruction
    - Custom CNN: Trained encoder-decoder for 2D to 3D
    """

    def __init__(
        self,
        model_type: str = "gaussian_cad",
        model_path: Optional[Path] = None,
        use_gpu: bool = True,
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._model = None
        self._torch_available = None

    @property
    def name(self) -> str:
        return f"Multi-View DNN ({self.model_type})"

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.MULTI_VIEW_DNN

    def is_available(self) -> bool:
        if self._torch_available is None:
            try:
                import torch
                self._torch_available = True
            except ImportError:
                self._torch_available = False
        return self._torch_available

    async def reconstruct(self, input_data: ReconstructionInput) -> ReconstructionOutput:
        """Perform multi-view DNN reconstruction"""
        if not self.is_available():
            return ReconstructionOutput(
                success=False,
                error="PyTorch not installed. Run: pip install torch torchvision",
                strategy_used=self.name,
            )

        try:
            if self.model_type == "gaussian_cad":
                return await self._reconstruct_gaussian_cad(input_data)
            elif self.model_type == "depth_estimation":
                return await self._reconstruct_depth_estimation(input_data)
            elif self.model_type == "cad2program":
                return await self._reconstruct_cad2program(input_data)
            else:
                return ReconstructionOutput(
                    success=False,
                    error=f"Unknown model type: {self.model_type}",
                    strategy_used=self.name,
                )
        except Exception as e:
            logger.error(f"Multi-view DNN error: {e}")
            return ReconstructionOutput(
                success=False,
                error=str(e),
                strategy_used=self.name,
            )

    async def _reconstruct_gaussian_cad(self, input_data: ReconstructionInput) -> ReconstructionOutput:
        """
        GaussianCAD-style reconstruction using 3D Gaussian Splatting

        This is a simplified implementation. Full GaussianCAD requires:
        - Multiple calibrated views
        - Gaussian splatting optimization
        - Mesh extraction
        """
        import torch
        import numpy as np
        from PIL import Image

        # Check for multiple views
        all_views = [input_data.primary_image] + input_data.additional_views

        if len(all_views) < 2:
            logger.warning("GaussianCAD works best with 3+ views. Using single-view fallback.")
            # Fall back to depth estimation
            return await self._reconstruct_depth_estimation(input_data)

        # Load images
        images = []
        for view_path in all_views:
            img = Image.open(view_path).convert('RGB')
            images.append(np.array(img))

        # TODO: Full implementation would include:
        # 1. Camera pose estimation (SfM or manual calibration)
        # 2. Initialize 3D Gaussians
        # 3. Optimize Gaussian parameters
        # 4. Extract mesh from Gaussians

        # For now, use depth estimation as fallback
        logger.info("Full GaussianCAD not implemented, using depth estimation")
        return await self._reconstruct_depth_estimation(input_data)

    async def _reconstruct_depth_estimation(self, input_data: ReconstructionInput) -> ReconstructionOutput:
        """
        Depth estimation based reconstruction

        Uses monocular depth estimation (MiDaS, Depth Anything, etc.)
        to create a 2.5D relief model from a single image.
        """
        import torch
        import numpy as np
        from PIL import Image
        import trimesh

        # Load image
        img = Image.open(input_data.primary_image).convert('RGB')
        img_array = np.array(img)

        # Try to use a depth estimation model
        depth_map = await self._estimate_depth(img_array)

        if depth_map is None:
            # Fallback: use simple edge-based depth
            depth_map = self._simple_depth_from_edges(img_array)

        # Create mesh from depth map
        mesh = self._depth_to_mesh(depth_map, input_data)

        # Export
        glb_data = mesh.export(file_type='glb')

        return ReconstructionOutput(
            success=True,
            model_data=glb_data,
            format="glb",
            strategy_used=self.name,
            metadata={
                "depth_method": "estimated" if depth_map is not None else "edge_based",
                "image_size": img_array.shape[:2],
            },
        )

    def _get_device(self):
        """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)"""
        import torch

        if self.use_gpu:
            if torch.backends.mps.is_available():
                logger.info("Using MPS (Apple Silicon GPU)")
                return torch.device("mps")
            elif torch.cuda.is_available():
                logger.info("Using CUDA GPU")
                return torch.device("cuda")

        logger.info("Using CPU")
        return torch.device("cpu")

    async def _estimate_depth(self, img_array: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth using a pretrained model"""
        try:
            import torch
            from torchvision import transforms

            device = self._get_device()

            # Try MiDaS
            try:
                model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
                model.eval()
                model = model.to(device)

                # Preprocess
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(384),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                input_tensor = transform(img_array).unsqueeze(0).to(device)

                with torch.no_grad():
                    depth = model(input_tensor)

                depth = depth.squeeze().cpu().numpy()

                # Resize to original
                from PIL import Image
                depth_img = Image.fromarray(depth)
                depth_img = depth_img.resize((img_array.shape[1], img_array.shape[0]))

                return np.array(depth_img)

            except Exception as e:
                logger.warning(f"MiDaS not available: {e}")
                return None

        except Exception as e:
            logger.warning(f"Depth estimation failed: {e}")
            return None

    def _simple_depth_from_edges(self, img_array: np.ndarray) -> np.ndarray:
        """Create simple depth map from edge detection"""
        import numpy as np

        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Invert (dark lines = walls = higher)
        depth = 255 - gray

        # Normalize to 0-1
        depth = depth / 255.0

        # Apply some smoothing
        try:
            import cv2
            depth = cv2.GaussianBlur(depth, (5, 5), 0)
        except ImportError:
            pass

        return depth

    def _depth_to_mesh(self, depth_map: np.ndarray, input_data: ReconstructionInput):
        """Convert depth map to 3D mesh"""
        import trimesh
        import numpy as np

        scale = input_data.scale or 0.01
        height_scale = input_data.wall_height

        h, w = depth_map.shape

        # Create vertex grid
        x = np.arange(w) * scale
        y = np.arange(h) * scale
        xx, yy = np.meshgrid(x, y)

        # Z from depth
        zz = depth_map * height_scale

        # Flatten to vertices
        vertices = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

        # Create faces (triangles)
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                idx = i * w + j
                # Two triangles per quad
                faces.append([idx, idx + 1, idx + w])
                faces.append([idx + 1, idx + w + 1, idx + w])

        faces = np.array(faces)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Clean up
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()

        return mesh

    async def _reconstruct_cad2program(self, input_data: ReconstructionInput) -> ReconstructionOutput:
        """
        CAD2Program style reconstruction using vision-language model

        This would use a VLM to generate parametric CAD commands.
        Full implementation requires fine-tuned model.
        """
        return ReconstructionOutput(
            success=False,
            error="CAD2Program model not yet implemented. Use depth_estimation or gaussian_cad.",
            strategy_used=self.name,
        )


# =============================================================================
# Strategy D: VLM-based CAD Reconstruction (OpenECAD, InternVL)
# =============================================================================

class VLMCADStrategy(ReconstructionStrategy):
    """
    Strategy D: Vision-Language Model for CAD reconstruction

    Uses VLMs fine-tuned for CAD generation:
    - OpenECAD (0.55B, 0.89B, 2.4B, 3.1B)
    - InternVL2 (1B, 2B, 4B, 8B)

    These models generate parametric CAD code from 2D images,
    which is then executed to create 3D geometry.
    """

    def __init__(
        self,
        model_id: str = "openecad-0.89b",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        auto_download: bool = True,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device
        self.auto_download = auto_download
        self._inference = None
        self._model_manager = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization to avoid loading heavy dependencies on import"""
        if self._initialized:
            return

        try:
            from .vlm_cad_strategy import ModelManager, VLMCADInference
            self._model_manager = ModelManager(cache_dir=self.cache_dir)
            self._inference = VLMCADInference(self._model_manager)
            self._initialized = True
        except ImportError as e:
            logger.warning(f"VLM CAD dependencies not available: {e}")
            self._initialized = False

    @property
    def name(self) -> str:
        return f"VLM CAD ({self.model_id})"

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.VLM_CAD

    def is_available(self) -> bool:
        """Check if VLM strategy is available"""
        self._lazy_init()
        if not self._initialized or not self._model_manager:
            return False

        # Check if required packages are installed
        return self._model_manager.has_torch and self._model_manager.has_transformers

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available VLM models"""
        self._lazy_init()
        if not self._model_manager:
            return []
        return self._model_manager.list_available_models()

    def get_device_info(self) -> Dict[str, Any]:
        """Get device and capability information"""
        self._lazy_init()
        if not self._model_manager:
            return {"device": "unknown", "available": False}

        return {
            "device": self._model_manager.device,
            "has_torch": self._model_manager.has_torch,
            "has_transformers": self._model_manager.has_transformers,
            "cuda_available": self._model_manager.device == "cuda",
            "mps_available": self._model_manager.device == "mps",
        }

    async def download_model(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Download a model"""
        self._lazy_init()
        if not self._model_manager:
            return {"success": False, "error": "Model manager not initialized"}

        target_model = model_id or self.model_id
        try:
            path = self._model_manager.download_model(target_model)
            return {"success": True, "model_id": target_model, "path": str(path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def reconstruct(self, input_data: ReconstructionInput) -> ReconstructionOutput:
        """
        Perform reconstruction using VLM

        The VLM analyzes the 2D CAD drawing and generates parametric
        CAD code that can be executed to create the 3D model.
        """
        import time
        start_time = time.time()

        self._lazy_init()

        if not self.is_available():
            return ReconstructionOutput(
                success=False,
                error="VLM CAD strategy not available. Install: pip install torch transformers",
                strategy_used=self.name,
            )

        try:
            # Ensure model is loaded
            if self._inference.current_model_id != self.model_id:
                # Download if needed and auto_download is enabled
                if self.auto_download:
                    config = self._model_manager.get_model_config(self.model_id)
                    if config and not self._model_manager._is_model_cached(self.model_id):
                        logger.info(f"Auto-downloading model {self.model_id}...")
                        self._model_manager.download_model(self.model_id)

                self._inference.load_model(self.model_id)

            # Generate CAD code from image
            result = self._inference.generate_cad_code(
                image_path=input_data.primary_image,
                max_tokens=2048,
                temperature=0.1,
            )

            # Convert generated code to mesh
            mesh_data = self._inference.code_to_mesh(
                code=result["code"],
                output_format="glb",
            )

            processing_time = time.time() - start_time

            return ReconstructionOutput(
                success=True,
                model_data=mesh_data,
                format="glb",
                metadata={
                    "model_id": result["model_id"],
                    "model_name": result["model_name"],
                    "source_image": result["image_path"],
                    "generated_code": result["code"],
                    "device": self._model_manager.device,
                },
                strategy_used=self.name,
                processing_time=processing_time,
            )

        except FileNotFoundError as e:
            return ReconstructionOutput(
                success=False,
                error=f"Model not found: {e}. Try downloading with download_model() first.",
                strategy_used=self.name,
            )
        except Exception as e:
            logger.exception(f"VLM reconstruction failed: {e}")
            return ReconstructionOutput(
                success=False,
                error=str(e),
                strategy_used=self.name,
            )


# =============================================================================
# Factory function to get strategy
# =============================================================================

def get_strategy(
    strategy_type: Union[str, StrategyType],
    **kwargs
) -> ReconstructionStrategy:
    """
    Factory function to get a reconstruction strategy

    Args:
        strategy_type: Type of strategy (external_api, basic_extrusion, multi_view_dnn)
        **kwargs: Strategy-specific parameters

    Returns:
        ReconstructionStrategy instance
    """
    if isinstance(strategy_type, str):
        strategy_type = StrategyType(strategy_type.lower())

    if strategy_type == StrategyType.EXTERNAL_API:
        return ExternalAPIStrategy(
            service=kwargs.get("service", "kaedim"),
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
        )
    elif strategy_type == StrategyType.BASIC_EXTRUSION:
        return BasicExtrusionStrategy()
    elif strategy_type == StrategyType.MULTI_VIEW_DNN:
        return MultiViewDNNStrategy(
            model_type=kwargs.get("model_type", "depth_estimation"),
            model_path=kwargs.get("model_path"),
            use_gpu=kwargs.get("use_gpu", True),
        )
    elif strategy_type == StrategyType.VLM_CAD:
        return VLMCADStrategy(
            model_id=kwargs.get("model_id", "openecad-0.89b"),
            cache_dir=kwargs.get("cache_dir"),
            device=kwargs.get("device"),
            auto_download=kwargs.get("auto_download", True),
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


# =============================================================================
# Reconstruction Manager - orchestrates strategies
# =============================================================================

class ReconstructionManager:
    """
    Manages multiple reconstruction strategies and provides fallback logic
    """

    def __init__(self):
        self.strategies: Dict[StrategyType, ReconstructionStrategy] = {}
        self._default_order = [
            StrategyType.VLM_CAD,  # Prefer VLM CAD for CAD drawings
            StrategyType.MULTI_VIEW_DNN,
            StrategyType.EXTERNAL_API,
            StrategyType.BASIC_EXTRUSION,
        ]

    def register_strategy(self, strategy: ReconstructionStrategy):
        """Register a strategy"""
        self.strategies[strategy.strategy_type] = strategy

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies"""
        return [
            s.name for s in self.strategies.values()
            if s.is_available()
        ]

    async def reconstruct(
        self,
        input_data: ReconstructionInput,
        preferred_strategy: Optional[StrategyType] = None,
        fallback: bool = True,
    ) -> ReconstructionOutput:
        """
        Perform reconstruction with optional fallback

        Args:
            input_data: Input data for reconstruction
            preferred_strategy: Preferred strategy to use
            fallback: Whether to try other strategies if preferred fails

        Returns:
            ReconstructionOutput
        """
        # Determine order of strategies to try
        if preferred_strategy:
            order = [preferred_strategy] + [s for s in self._default_order if s != preferred_strategy]
        else:
            order = self._default_order

        last_error = None

        for strategy_type in order:
            strategy = self.strategies.get(strategy_type)

            if not strategy or not strategy.is_available():
                continue

            logger.info(f"Trying reconstruction with {strategy.name}")
            result = await strategy.reconstruct(input_data)

            if result.success:
                return result

            last_error = result.error

            if not fallback:
                break

        return ReconstructionOutput(
            success=False,
            error=last_error or "No available reconstruction strategy",
            strategy_used="none",
        )
