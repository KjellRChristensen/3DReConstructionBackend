"""
CAD Import Module - Load 3D ground truth models from various CAD formats

Supported formats:
- IFC (Industry Foundation Classes) - BIM standard
- DXF (Drawing Exchange Format) - AutoCAD 2D/3D
- OBJ (Wavefront) - 3D mesh
- GLB/GLTF - GL Transmission Format
- STL - Stereolithography
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CADModel:
    """Represents a loaded CAD model"""
    mesh: Any  # trimesh.Trimesh or Scene
    format: str
    file_path: Path
    metadata: Dict[str, Any]

    # Extracted geometry info
    vertices: np.ndarray
    faces: np.ndarray
    bounds: np.ndarray  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]

    # Building-specific data (from IFC/BIM)
    floors: Optional[List[Dict]] = None
    walls: Optional[List[Dict]] = None
    rooms: Optional[List[Dict]] = None
    openings: Optional[List[Dict]] = None  # Doors, windows


class CADImporter:
    """
    Import CAD files as ground truth 3D models.

    Supports: IFC, DXF, OBJ, GLB, STL
    """

    SUPPORTED_FORMATS = {
        '.ifc': 'ifc',
        '.dxf': 'dxf',
        '.dwg': 'dwg',
        '.obj': 'obj',
        '.glb': 'glb',
        '.gltf': 'gltf',
        '.stl': 'stl',
        '.step': 'step',
        '.stp': 'step',
    }

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self):
        """Check which import libraries are available"""
        self.has_ifcopenshell = False
        self.has_ezdxf = False
        self.has_trimesh = False

        try:
            import ifcopenshell
            self.has_ifcopenshell = True
        except ImportError:
            logger.warning("ifcopenshell not installed - IFC import disabled")

        try:
            import ezdxf
            self.has_ezdxf = True
        except ImportError:
            logger.warning("ezdxf not installed - DXF import disabled")

        try:
            import trimesh
            self.has_trimesh = True
        except ImportError:
            logger.warning("trimesh not installed - OBJ/GLB/STL import disabled")

    def load(self, file_path: Union[str, Path]) -> CADModel:
        """
        Load a CAD file and return standardized CADModel.

        Args:
            file_path: Path to CAD file

        Returns:
            CADModel with mesh data and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"CAD file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}. Supported: {list(self.SUPPORTED_FORMATS.keys())}")

        format_type = self.SUPPORTED_FORMATS[suffix]

        if format_type == 'ifc':
            return self._load_ifc(file_path)
        elif format_type == 'dxf':
            return self._load_dxf(file_path)
        elif format_type in ('obj', 'glb', 'gltf', 'stl'):
            return self._load_mesh(file_path, format_type)
        elif format_type == 'step':
            return self._load_step(file_path)
        else:
            raise ValueError(f"Format {format_type} not yet implemented")

    def _load_ifc(self, file_path: Path) -> CADModel:
        """
        Load IFC (BIM) file with full building information.

        Extracts:
        - 3D geometry (mesh)
        - Floor plans
        - Walls, doors, windows
        - Room information
        """
        if not self.has_ifcopenshell:
            raise ImportError("ifcopenshell required for IFC import. Run: pip install ifcopenshell")

        import ifcopenshell
        import ifcopenshell.geom
        import trimesh

        logger.info(f"Loading IFC file: {file_path}")

        ifc = ifcopenshell.open(str(file_path))

        # Get project info
        project = ifc.by_type("IfcProject")[0] if ifc.by_type("IfcProject") else None

        metadata = {
            "schema": ifc.schema,
            "project_name": project.Name if project else None,
            "project_description": project.Description if project else None,
        }

        # Extract geometry using ifcopenshell.geom
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        # Extract building elements
        walls = []
        floors_data = []
        rooms = []
        openings = []

        # Process walls
        for wall in ifc.by_type("IfcWall"):
            try:
                shape = ifcopenshell.geom.create_shape(settings, wall)
                verts = np.array(shape.geometry.verts).reshape(-1, 3)
                faces = np.array(shape.geometry.faces).reshape(-1, 3)

                all_vertices.append(verts)
                all_faces.append(faces + vertex_offset)
                vertex_offset += len(verts)

                walls.append({
                    "id": wall.GlobalId,
                    "name": wall.Name,
                    "type": wall.is_a(),
                })
            except Exception as e:
                logger.warning(f"Failed to process wall {wall.GlobalId}: {e}")

        # Process slabs (floors/ceilings)
        for slab in ifc.by_type("IfcSlab"):
            try:
                shape = ifcopenshell.geom.create_shape(settings, slab)
                verts = np.array(shape.geometry.verts).reshape(-1, 3)
                faces = np.array(shape.geometry.faces).reshape(-1, 3)

                all_vertices.append(verts)
                all_faces.append(faces + vertex_offset)
                vertex_offset += len(verts)

                floors_data.append({
                    "id": slab.GlobalId,
                    "name": slab.Name,
                    "type": slab.PredefinedType if hasattr(slab, 'PredefinedType') else None,
                })
            except Exception as e:
                logger.warning(f"Failed to process slab {slab.GlobalId}: {e}")

        # Process doors and windows
        for opening_type in ["IfcDoor", "IfcWindow"]:
            for element in ifc.by_type(opening_type):
                try:
                    shape = ifcopenshell.geom.create_shape(settings, element)
                    verts = np.array(shape.geometry.verts).reshape(-1, 3)
                    faces = np.array(shape.geometry.faces).reshape(-1, 3)

                    all_vertices.append(verts)
                    all_faces.append(faces + vertex_offset)
                    vertex_offset += len(verts)

                    openings.append({
                        "id": element.GlobalId,
                        "name": element.Name,
                        "type": opening_type.replace("Ifc", "").lower(),
                        "width": element.OverallWidth if hasattr(element, 'OverallWidth') else None,
                        "height": element.OverallHeight if hasattr(element, 'OverallHeight') else None,
                    })
                except Exception as e:
                    logger.warning(f"Failed to process {opening_type} {element.GlobalId}: {e}")

        # Process spaces (rooms)
        for space in ifc.by_type("IfcSpace"):
            rooms.append({
                "id": space.GlobalId,
                "name": space.Name,
                "long_name": space.LongName if hasattr(space, 'LongName') else None,
            })

        # Combine all geometry
        if all_vertices:
            vertices = np.vstack(all_vertices)
            faces = np.vstack(all_faces)
        else:
            vertices = np.array([]).reshape(0, 3)
            faces = np.array([]).reshape(0, 3)

        # Create trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        return CADModel(
            mesh=mesh,
            format='ifc',
            file_path=file_path,
            metadata=metadata,
            vertices=vertices,
            faces=faces,
            bounds=mesh.bounds if len(vertices) > 0 else np.array([[0, 0, 0], [0, 0, 0]]),
            floors=floors_data,
            walls=walls,
            rooms=rooms,
            openings=openings,
        )

    def _load_dxf(self, file_path: Path) -> CADModel:
        """
        Load DXF file (AutoCAD format).

        Can contain 2D or 3D geometry.
        """
        if not self.has_ezdxf:
            raise ImportError("ezdxf required for DXF import. Run: pip install ezdxf")

        import ezdxf
        import trimesh

        logger.info(f"Loading DXF file: {file_path}")

        doc = ezdxf.readfile(str(file_path))
        msp = doc.modelspace()

        metadata = {
            "dxf_version": doc.dxfversion,
            "units": doc.units,
        }

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        walls = []

        # Process 3D entities
        for entity in msp:
            try:
                if entity.dxftype() == '3DFACE':
                    # 3D face - direct triangle/quad
                    pts = [entity.dxf.vtx0, entity.dxf.vtx1, entity.dxf.vtx2]
                    if entity.dxf.vtx3 != entity.dxf.vtx2:
                        pts.append(entity.dxf.vtx3)

                    verts = np.array([[p.x, p.y, p.z] for p in pts])
                    if len(verts) == 3:
                        faces = np.array([[0, 1, 2]])
                    else:
                        faces = np.array([[0, 1, 2], [0, 2, 3]])

                    all_vertices.append(verts)
                    all_faces.append(faces + vertex_offset)
                    vertex_offset += len(verts)

                elif entity.dxftype() == 'MESH':
                    # DXF mesh entity
                    verts = np.array([[v.x, v.y, v.z] for v in entity.vertices])
                    faces = np.array(list(entity.faces))

                    all_vertices.append(verts)
                    all_faces.append(faces + vertex_offset)
                    vertex_offset += len(verts)

                elif entity.dxftype() == 'LINE':
                    # 2D/3D line - store as wall info
                    start = entity.dxf.start
                    end = entity.dxf.end
                    walls.append({
                        "start": [start.x, start.y, getattr(start, 'z', 0)],
                        "end": [end.x, end.y, getattr(end, 'z', 0)],
                        "layer": entity.dxf.layer,
                    })

                elif entity.dxftype() == 'POLYLINE':
                    # Polyline - could be wall outline
                    points = [[v.dxf.location.x, v.dxf.location.y,
                              getattr(v.dxf.location, 'z', 0)]
                             for v in entity.vertices]
                    walls.append({
                        "points": points,
                        "layer": entity.dxf.layer,
                        "closed": entity.is_closed,
                    })

            except Exception as e:
                logger.warning(f"Failed to process DXF entity {entity.dxftype()}: {e}")

        # Combine geometry
        if all_vertices:
            vertices = np.vstack(all_vertices)
            faces = np.vstack(all_faces)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            # No 3D geometry - create empty mesh
            vertices = np.array([]).reshape(0, 3)
            faces = np.array([]).reshape(0, 3)
            mesh = trimesh.Trimesh()

        return CADModel(
            mesh=mesh,
            format='dxf',
            file_path=file_path,
            metadata=metadata,
            vertices=vertices,
            faces=faces,
            bounds=mesh.bounds if len(vertices) > 0 else np.array([[0, 0, 0], [0, 0, 0]]),
            walls=walls if walls else None,
        )

    def _load_mesh(self, file_path: Path, format_type: str) -> CADModel:
        """
        Load standard mesh formats (OBJ, GLB, GLTF, STL).
        """
        if not self.has_trimesh:
            raise ImportError("trimesh required for mesh import. Run: pip install trimesh")

        import trimesh

        logger.info(f"Loading {format_type.upper()} file: {file_path}")

        mesh = trimesh.load(str(file_path))

        # Handle scene (multiple meshes) vs single mesh
        if isinstance(mesh, trimesh.Scene):
            # Combine all meshes in scene
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
            else:
                mesh = trimesh.Trimesh()

        metadata = {
            "format": format_type,
            "vertex_count": len(mesh.vertices) if hasattr(mesh, 'vertices') else 0,
            "face_count": len(mesh.faces) if hasattr(mesh, 'faces') else 0,
        }

        vertices = np.array(mesh.vertices) if len(mesh.vertices) > 0 else np.array([]).reshape(0, 3)
        faces = np.array(mesh.faces) if len(mesh.faces) > 0 else np.array([]).reshape(0, 3)

        return CADModel(
            mesh=mesh,
            format=format_type,
            file_path=file_path,
            metadata=metadata,
            vertices=vertices,
            faces=faces,
            bounds=mesh.bounds if len(vertices) > 0 else np.array([[0, 0, 0], [0, 0, 0]]),
        )

    def _load_step(self, file_path: Path) -> CADModel:
        """
        Load STEP file (CAD exchange format).

        Requires OpenCASCADE or FreeCAD for full support.
        Falls back to trimesh if available.
        """
        # Try trimesh first (limited STEP support)
        if self.has_trimesh:
            try:
                return self._load_mesh(file_path, 'step')
            except Exception as e:
                logger.warning(f"trimesh STEP load failed: {e}")

        raise NotImplementedError(
            "Full STEP support requires OpenCASCADE. "
            "Try converting to OBJ/GLB first, or install: pip install pythonocc-core"
        )


def load_ground_truth(file_path: Union[str, Path]) -> CADModel:
    """
    Convenience function to load a ground truth CAD file.

    Args:
        file_path: Path to CAD file (IFC, DXF, OBJ, GLB, STL)

    Returns:
        CADModel with mesh and metadata
    """
    importer = CADImporter()
    return importer.load(file_path)
