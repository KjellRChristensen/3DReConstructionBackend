"""
Vision-Language Model CAD Strategy

Uses vision-language models (OpenECAD, InternVL, CAD-Coder) to convert 2D CAD drawings
to 3D parametric models.

Supported models:
- OpenECAD (0.55B, 0.89B, 2.4B, 3.1B) - Fine-tuned for CAD code generation
- InternVL2 (1B, 2B, 4B, 8B) - Base model for fine-tuning
- CAD-Coder - Outputs CadQuery Python code

Output formats supported:
- OpenECAD format: Custom CAD command sequences → PythonOCC
- CadQuery: Python code → OCC geometry
- Build123d: Python code → OCC geometry
"""

import logging
import re
import math
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class CADCodeFormat(Enum):
    """Supported CAD code output formats"""
    OPENECAD = "openecad"      # OpenECAD command sequences
    CADQUERY = "cadquery"      # CadQuery Python code
    BUILD123D = "build123d"    # Build123d Python code
    PYTHON_OCC = "pythonocc"   # Raw PythonOCC code
    UNKNOWN = "unknown"


@dataclass
class VLMModelConfig:
    """Configuration for a VLM model"""
    model_id: str
    name: str
    size: str  # e.g., "0.89B"
    hf_repo: str
    model_type: str  # "openecad", "internvl", "cadcoder", "custom"
    requires_gpu: bool = True
    min_vram_gb: float = 4.0
    supports_mps: bool = True
    output_format: CADCodeFormat = CADCodeFormat.OPENECAD
    default_prompt: str = ""

    def __post_init__(self):
        if not self.default_prompt:
            self.default_prompt = CAD_PROMPTS.get(self.model_type, CAD_PROMPTS["default"])


@dataclass
class CADCodeResult:
    """Result of CAD code parsing/validation"""
    valid: bool
    format: CADCodeFormat
    code: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    detected_operations: List[str] = field(default_factory=list)


# =============================================================================
# Prompt Templates
# =============================================================================

CAD_PROMPTS = {
    "openecad": """Analyze this engineering drawing and generate the CAD construction sequence.
Output the complete code to reconstruct this 3D model using the following commands:
- add_sketchplane(origin, normal, x_axis)
- add_line(start_point, end_point)
- add_arc(start_point, end_point, mid_point)
- add_circle(center_point, radius)
- add_profile(loops_list)
- add_sketch(sketchplane, profile)
- add_extrude(sketch, operation, type, extent)

Generate only the code, no explanations.""",

    "cadquery": """Analyze this 2D engineering drawing and generate CadQuery Python code to create the corresponding 3D model.

Requirements:
1. Import cadquery as cq
2. Create a Workplane and build the geometry
3. Use standard CadQuery operations: box, cylinder, extrude, cut, fillet, etc.
4. Assign the final result to a variable named 'result'
5. Output only valid Python code, no markdown or explanations

Example format:
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 5)""",

    "build123d": """Analyze this 2D engineering drawing and generate Build123d Python code to create the corresponding 3D model.

Requirements:
1. Use build123d's builder syntax
2. Create sketches and extrude them into 3D
3. Assign the final Part to a variable named 'result'
4. Output only valid Python code

Example format:
from build123d import *
with BuildPart() as part:
    with BuildSketch():
        Rectangle(10, 10)
    extrude(amount=5)
result = part.part""",

    "internvl": """You are a CAD expert. Analyze this 2D engineering drawing and generate Python code using CadQuery to create the corresponding 3D solid model.

The code must:
1. Import cadquery as cq
2. Create the geometry step by step
3. Handle all visible features in the drawing
4. Assign the final result to 'result'

Output ONLY the Python code, no explanations or markdown.""",

    "default": """Generate CAD code for this engineering drawing. Output Python code that creates the 3D model.""",
}


# =============================================================================
# Available Models Registry
# =============================================================================

AVAILABLE_MODELS = {
    # OpenECAD v2 models (latest, from Yuan-Che)
    # Uses TinyLLaVA architecture with SigLIP/CLIP vision encoder
    "openecad-0.55b": VLMModelConfig(
        model_id="openecad-0.55b",
        name="OpenECAD v2 CLIP 0.55B",
        size="0.55B",
        hf_repo="Yuan-Che/OpenECADv2-CLIP-0.55B",
        model_type="openecad",
        min_vram_gb=2.0,
        output_format=CADCodeFormat.OPENECAD,
    ),
    "openecad-0.89b": VLMModelConfig(
        model_id="openecad-0.89b",
        name="OpenECAD v2 SigLIP 0.89B",
        size="0.89B",
        hf_repo="Yuan-Che/OpenECADv2-SigLIP-0.89B",
        model_type="openecad",
        min_vram_gb=3.0,
        output_format=CADCodeFormat.OPENECAD,
    ),
    "openecad-2.4b": VLMModelConfig(
        model_id="openecad-2.4b",
        name="OpenECAD v2 SigLIP 2.4B",
        size="2.4B",
        hf_repo="Yuan-Che/OpenECADv2-SigLIP-2.4B",
        model_type="openecad",
        min_vram_gb=6.0,
        output_format=CADCodeFormat.OPENECAD,
    ),
    "openecad-3.1b": VLMModelConfig(
        model_id="openecad-3.1b",
        name="OpenECAD v1 SigLIP 3.1B",
        size="3.1B",
        hf_repo="Yuan-Che/OpenECAD-SigLIP-3.1B",  # v2 not available for 3.1B
        model_type="openecad",
        min_vram_gb=8.0,
        output_format=CADCodeFormat.OPENECAD,
    ),

    # InternVL2 models (for fine-tuning or general use)
    "internvl2-1b": VLMModelConfig(
        model_id="internvl2-1b",
        name="InternVL2 1B",
        size="1B",
        hf_repo="OpenGVLab/InternVL2-1B",
        model_type="internvl",
        min_vram_gb=4.0,
        output_format=CADCodeFormat.CADQUERY,
    ),
    "internvl2-2b": VLMModelConfig(
        model_id="internvl2-2b",
        name="InternVL2 2B",
        size="2B",
        hf_repo="OpenGVLab/InternVL2-2B",
        model_type="internvl",
        min_vram_gb=6.0,
        output_format=CADCodeFormat.CADQUERY,
    ),
    "internvl2-4b": VLMModelConfig(
        model_id="internvl2-4b",
        name="InternVL2 4B",
        size="4B",
        hf_repo="OpenGVLab/InternVL2-4B",
        model_type="internvl",
        min_vram_gb=12.0,
        output_format=CADCodeFormat.CADQUERY,
    ),
    "internvl2-8b": VLMModelConfig(
        model_id="internvl2-8b",
        name="InternVL2 8B",
        size="8B",
        hf_repo="OpenGVLab/InternVL2-8B",
        model_type="internvl",
        min_vram_gb=16.0,
        output_format=CADCodeFormat.CADQUERY,
    ),
}


# =============================================================================
# CAD Code Validator
# =============================================================================

class CADCodeValidator:
    """Validates and analyzes generated CAD code"""

    # OpenECAD command patterns
    OPENECAD_COMMANDS = {
        "add_sketchplane": r"add_sketchplane\s*\(",
        "add_sketchplane_ref": r"add_sketchplane_ref\s*\(",
        "add_line": r"add_line\s*\(",
        "add_arc": r"add_arc\s*\(",
        "add_circle": r"add_circle\s*\(",
        "add_profile": r"add_profile\s*\(",
        "add_sketch": r"add_sketch\s*\(",
        "add_extrude": r"add_extrude\s*\(",
    }

    # CadQuery patterns
    CADQUERY_PATTERNS = {
        "import": r"import\s+cadquery|from\s+cadquery",
        "workplane": r"Workplane\s*\(",
        "box": r"\.box\s*\(",
        "cylinder": r"\.cylinder\s*\(",
        "extrude": r"\.extrude\s*\(",
        "cut": r"\.cut\s*\(",
        "union": r"\.union\s*\(",
        "fillet": r"\.fillet\s*\(",
        "chamfer": r"\.chamfer\s*\(",
    }

    # Build123d patterns
    BUILD123D_PATTERNS = {
        "import": r"from\s+build123d\s+import|import\s+build123d",
        "buildpart": r"BuildPart\s*\(",
        "buildsketch": r"BuildSketch\s*\(",
        "extrude": r"extrude\s*\(",
        "rectangle": r"Rectangle\s*\(",
        "circle": r"Circle\s*\(",
    }

    # Dangerous patterns to reject
    DANGEROUS_PATTERNS = [
        r"import\s+os",
        r"import\s+subprocess",
        r"import\s+sys",
        r"__import__",
        r"eval\s*\(",
        r"exec\s*\(",
        r"open\s*\([^)]*['\"][wax]",  # file writing
        r"compile\s*\(",
        r"globals\s*\(",
        r"locals\s*\(",
        r"getattr\s*\(",
        r"setattr\s*\(",
        r"delattr\s*\(",
    ]

    @classmethod
    def detect_format(cls, code: str) -> CADCodeFormat:
        """Detect the format of generated CAD code"""
        code_lower = code.lower()

        # Check for OpenECAD commands
        openecad_count = sum(
            1 for pattern in cls.OPENECAD_COMMANDS.values()
            if re.search(pattern, code)
        )
        if openecad_count >= 2:
            return CADCodeFormat.OPENECAD

        # Check for CadQuery
        if re.search(cls.CADQUERY_PATTERNS["import"], code):
            return CADCodeFormat.CADQUERY
        if re.search(cls.CADQUERY_PATTERNS["workplane"], code):
            return CADCodeFormat.CADQUERY

        # Check for Build123d
        if re.search(cls.BUILD123D_PATTERNS["import"], code):
            return CADCodeFormat.BUILD123D
        if re.search(cls.BUILD123D_PATTERNS["buildpart"], code):
            return CADCodeFormat.BUILD123D

        # Check for PythonOCC
        if "OCC." in code or "from OCC" in code:
            return CADCodeFormat.PYTHON_OCC

        return CADCodeFormat.UNKNOWN

    @classmethod
    def validate(cls, code: str) -> CADCodeResult:
        """Validate CAD code for safety and correctness"""
        errors = []
        warnings = []
        operations = []

        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(f"Dangerous pattern detected: {pattern}")

        # Detect format
        code_format = cls.detect_format(code)

        if code_format == CADCodeFormat.UNKNOWN:
            warnings.append("Could not detect CAD code format")

        # Validate based on format
        if code_format == CADCodeFormat.OPENECAD:
            for name, pattern in cls.OPENECAD_COMMANDS.items():
                if re.search(pattern, code):
                    operations.append(name)

            # Check for required operations
            if "add_sketchplane" not in operations and "add_sketchplane_ref" not in operations:
                warnings.append("No sketchplane definition found")
            if "add_extrude" not in operations:
                warnings.append("No extrusion operation found")

        elif code_format == CADCodeFormat.CADQUERY:
            for name, pattern in cls.CADQUERY_PATTERNS.items():
                if re.search(pattern, code):
                    operations.append(name)

            # Check for result assignment
            if not re.search(r"result\s*=", code):
                warnings.append("No 'result' variable assignment found")

        elif code_format == CADCodeFormat.BUILD123D:
            for name, pattern in cls.BUILD123D_PATTERNS.items():
                if re.search(pattern, code):
                    operations.append(name)

        # Basic syntax check
        try:
            compile(code, "<cad_code>", "exec")
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")

        return CADCodeResult(
            valid=len(errors) == 0,
            format=code_format,
            code=code,
            errors=errors,
            warnings=warnings,
            detected_operations=operations,
        )

    @classmethod
    def extract_code_block(cls, text: str) -> str:
        """Extract code from markdown code blocks or raw text"""
        # Try to extract from markdown code block
        code_block_pattern = r"```(?:python|cad)?\s*\n(.*?)```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)

        if matches:
            # Return the longest code block (likely the main code)
            return max(matches, key=len).strip()

        # Check if it looks like raw code (starts with import or function call)
        lines = text.strip().split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            stripped = line.strip()
            # Detect start of code
            if not in_code:
                if (stripped.startswith('import ') or
                    stripped.startswith('from ') or
                    stripped.startswith('add_') or
                    stripped.startswith('result ') or
                    re.match(r'^[a-z_]+\s*=', stripped)):
                    in_code = True

            if in_code:
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)

        return text.strip()


# =============================================================================
# CAD Code Executors
# =============================================================================

class OpenECADExecutor:
    """
    Executes OpenECAD command sequences using PythonOCC.

    OpenECAD format:
    - add_sketchplane(origin, normal, x_axis)
    - add_line(start, end)
    - add_arc(start, end, mid)
    - add_circle(center, radius)
    - add_profile(loops)
    - add_sketch(plane, profile)
    - add_extrude(sketch, operation, type, extent)
    """

    def __init__(self):
        self.shapes = []
        self.sketchplanes = {}
        self.curves = []
        self.profiles = {}
        self.sketches = {}
        self._check_occ()

    def _check_occ(self):
        """Check if PythonOCC is available"""
        self.has_occ = False
        try:
            from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
            self.has_occ = True
        except ImportError:
            logger.warning("PythonOCC not available. Install: pip install pythonocc-core")

    def execute(self, code: str) -> Any:
        """Execute OpenECAD code and return the resulting shape"""
        if not self.has_occ:
            raise RuntimeError("PythonOCC required for OpenECAD execution")

        # Parse and execute commands
        namespace = self._create_namespace()

        try:
            exec(code, namespace)
        except Exception as e:
            raise RuntimeError(f"OpenECAD execution error: {e}")

        # Get the result
        if 'result' in namespace:
            return namespace['result']
        elif self.shapes:
            return self.shapes[-1]
        else:
            raise RuntimeError("No shape created by OpenECAD code")

    def _create_namespace(self) -> Dict[str, Any]:
        """Create execution namespace with OpenECAD functions"""
        from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax2, gp_Pln
        from OCC.Core.BRepBuilderAPI import (
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeWire,
            BRepBuilderAPI_MakeFace,
        )
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
        from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeCircle
        from OCC.Core.TopoDS import TopoDS_Shape

        sketchplanes = {}
        curves = {}
        profiles = {}
        sketches = {}
        shapes = []
        curve_counter = [0]

        def add_sketchplane(origin, normal, x_axis):
            """Create a sketch plane"""
            plane_id = f"plane_{len(sketchplanes)}"
            pnt = gp_Pnt(*origin)
            dir_n = gp_Dir(*normal)
            dir_x = gp_Dir(*x_axis)
            ax2 = gp_Ax2(pnt, dir_n, dir_x)
            sketchplanes[plane_id] = gp_Pln(ax2)
            return plane_id

        def add_line(start, end):
            """Create a line edge"""
            curve_id = f"curve_{curve_counter[0]}"
            curve_counter[0] += 1
            p1 = gp_Pnt(*start)
            p2 = gp_Pnt(*end)
            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            curves[curve_id] = edge
            return curve_id

        def add_arc(start, end, mid):
            """Create an arc edge"""
            curve_id = f"curve_{curve_counter[0]}"
            curve_counter[0] += 1
            p1 = gp_Pnt(*start)
            p2 = gp_Pnt(*end)
            p3 = gp_Pnt(*mid)
            arc = GC_MakeArcOfCircle(p1, p3, p2).Value()
            edge = BRepBuilderAPI_MakeEdge(arc).Edge()
            curves[curve_id] = edge
            return curve_id

        def add_circle(center, radius):
            """Create a circle edge"""
            curve_id = f"curve_{curve_counter[0]}"
            curve_counter[0] += 1
            pnt = gp_Pnt(*center) if len(center) == 3 else gp_Pnt(center[0], center[1], 0)
            circ = GC_MakeCircle(pnt, gp_Dir(0, 0, 1), radius).Value()
            edge = BRepBuilderAPI_MakeEdge(circ).Edge()
            curves[curve_id] = edge
            return curve_id

        def add_profile(loops):
            """Create a profile from curve loops"""
            profile_id = f"profile_{len(profiles)}"
            wires = []
            for loop in loops:
                wire_builder = BRepBuilderAPI_MakeWire()
                for curve_id in loop:
                    if curve_id in curves:
                        wire_builder.Add(curves[curve_id])
                if wire_builder.IsDone():
                    wires.append(wire_builder.Wire())
            profiles[profile_id] = wires
            return profile_id

        def add_sketch(sketchplane_id, profile_id):
            """Combine plane and profile into a sketch"""
            sketch_id = f"sketch_{len(sketches)}"
            plane = sketchplanes.get(sketchplane_id)
            profile_wires = profiles.get(profile_id, [])

            if plane and profile_wires:
                # Create face from first wire
                if profile_wires:
                    face = BRepBuilderAPI_MakeFace(plane, profile_wires[0]).Face()
                    sketches[sketch_id] = {"plane": plane, "face": face}
            return sketch_id

        def add_extrude(sketch_id, operation, extrude_type, extent):
            """Extrude a sketch to create 3D geometry"""
            sketch_data = sketches.get(sketch_id)
            if not sketch_data:
                return None

            face = sketch_data["face"]
            plane = sketch_data["plane"]

            # Get extrusion direction from plane normal
            normal = plane.Axis().Direction()
            vec = gp_Vec(normal).Multiplied(extent)

            # Create prism
            prism = BRepPrimAPI_MakePrism(face, vec).Shape()

            # Handle boolean operations
            if operation == "NewBodyFeatureOperation" or not shapes:
                shapes.append(prism)
            elif operation == "JoinFeatureOperation":
                if shapes:
                    fused = BRepAlgoAPI_Fuse(shapes[-1], prism).Shape()
                    shapes[-1] = fused
            elif operation == "CutFeatureOperation":
                if shapes:
                    cut = BRepAlgoAPI_Cut(shapes[-1], prism).Shape()
                    shapes[-1] = cut

            return prism

        return {
            "add_sketchplane": add_sketchplane,
            "add_sketchplane_ref": add_sketchplane,  # Alias
            "add_line": add_line,
            "add_arc": add_arc,
            "add_circle": add_circle,
            "add_profile": add_profile,
            "add_sketch": add_sketch,
            "add_extrude": add_extrude,
            "shapes": shapes,
            "result": None,
        }


class CadQueryExecutor:
    """Executes CadQuery Python code safely"""

    def __init__(self):
        self._check_cadquery()

    def _check_cadquery(self):
        """Check if CadQuery is available"""
        self.has_cadquery = False
        try:
            import cadquery as cq
            self.has_cadquery = True
        except ImportError:
            logger.warning("CadQuery not available. Install: pip install cadquery")

    def execute(self, code: str) -> Any:
        """Execute CadQuery code and return the resulting shape"""
        if not self.has_cadquery:
            raise RuntimeError("CadQuery required. Install: pip install cadquery")

        import cadquery as cq

        # Create safe namespace
        namespace = {
            "cq": cq,
            "Workplane": cq.Workplane,
            "Vector": cq.Vector,
            "result": None,
            "math": math,
        }

        try:
            exec(code, namespace)
        except Exception as e:
            raise RuntimeError(f"CadQuery execution error: {e}")

        result = namespace.get("result")
        if result is None:
            raise RuntimeError("No 'result' variable found in CadQuery code")

        return result


class Build123dExecutor:
    """Executes Build123d Python code safely"""

    def __init__(self):
        self._check_build123d()

    def _check_build123d(self):
        """Check if Build123d is available"""
        self.has_build123d = False
        try:
            import build123d
            self.has_build123d = True
        except ImportError:
            logger.warning("Build123d not available. Install: pip install build123d")

    def execute(self, code: str) -> Any:
        """Execute Build123d code and return the resulting shape"""
        if not self.has_build123d:
            raise RuntimeError("Build123d required. Install: pip install build123d")

        import build123d as bd

        # Create namespace with build123d imports
        namespace = {
            "bd": bd,
            "math": math,
            "result": None,
        }

        # Add common build123d classes
        for name in dir(bd):
            if not name.startswith('_'):
                namespace[name] = getattr(bd, name)

        try:
            exec(code, namespace)
        except Exception as e:
            raise RuntimeError(f"Build123d execution error: {e}")

        result = namespace.get("result")
        if result is None:
            # Try to find a Part in the namespace
            for name, value in namespace.items():
                if hasattr(value, 'wrapped'):  # Build123d objects have .wrapped
                    result = value
                    break

        if result is None:
            raise RuntimeError("No 'result' variable found in Build123d code")

        return result


# =============================================================================
# Shape to Mesh Converter
# =============================================================================

class ShapeToMeshConverter:
    """Converts OCC shapes to trimesh"""

    @staticmethod
    def occ_to_trimesh(shape, linear_deflection: float = 0.1, angular_deflection: float = 0.5):
        """Convert OCC shape to trimesh"""
        try:
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopLoc import TopLoc_Location
            import numpy as np
            import trimesh

            # Mesh the shape
            mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
            mesh.Perform()

            all_vertices = []
            all_faces = []
            vertex_offset = 0

            # Extract triangulation from each face
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                face = explorer.Current()
                location = TopLoc_Location()
                triangulation = BRep_Tool.Triangulation(face, location)

                if triangulation is not None:
                    # Get vertices
                    nb_nodes = triangulation.NbNodes()
                    vertices = []
                    for i in range(1, nb_nodes + 1):
                        node = triangulation.Node(i)
                        # Apply location transformation
                        if not location.IsIdentity():
                            node = node.Transformed(location.Transformation())
                        vertices.append([node.X(), node.Y(), node.Z()])

                    # Get faces (triangles)
                    nb_triangles = triangulation.NbTriangles()
                    faces = []
                    for i in range(1, nb_triangles + 1):
                        tri = triangulation.Triangle(i)
                        n1, n2, n3 = tri.Get()
                        faces.append([
                            n1 - 1 + vertex_offset,
                            n2 - 1 + vertex_offset,
                            n3 - 1 + vertex_offset
                        ])

                    all_vertices.extend(vertices)
                    all_faces.extend(faces)
                    vertex_offset += nb_nodes

                explorer.Next()

            if not all_vertices:
                raise RuntimeError("No triangulation data extracted from shape")

            return trimesh.Trimesh(
                vertices=np.array(all_vertices),
                faces=np.array(all_faces)
            )

        except ImportError as e:
            raise RuntimeError(f"OCC conversion requires pythonocc-core: {e}")

    @staticmethod
    def cadquery_to_trimesh(cq_object, tolerance: float = 0.1):
        """Convert CadQuery object to trimesh"""
        try:
            import trimesh

            # CadQuery can export to STL directly
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
                cq_object.val().exportStl(f.name, tolerance=tolerance)
                mesh = trimesh.load(f.name)
                os.unlink(f.name)
                return mesh

        except Exception as e:
            raise RuntimeError(f"CadQuery to mesh conversion failed: {e}")

    @staticmethod
    def build123d_to_trimesh(bd_object, tolerance: float = 0.1):
        """Convert Build123d object to trimesh"""
        try:
            import trimesh

            # Build123d can export to STL
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
                if hasattr(bd_object, 'export_stl'):
                    bd_object.export_stl(f.name)
                elif hasattr(bd_object, 'wrapped'):
                    # Access the underlying OCC shape
                    from OCC.Core.StlAPI import StlAPI_Writer
                    writer = StlAPI_Writer()
                    writer.Write(bd_object.wrapped, f.name)
                else:
                    raise RuntimeError("Cannot export Build123d object")

                mesh = trimesh.load(f.name)
                os.unlink(f.name)
                return mesh

        except Exception as e:
            raise RuntimeError(f"Build123d to mesh conversion failed: {e}")


# =============================================================================
# Model Manager
# =============================================================================

class ModelManager:
    """
    Manages VLM model downloading, caching, and loading.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/3d_reconstruction/models"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, Any] = {}
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if required dependencies are available"""
        self.has_torch = False
        self.has_transformers = False
        self.has_cadquery = False
        self.has_build123d = False
        self.has_occ = False
        self.device = "cpu"

        try:
            import torch
            self.has_torch = True
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            logger.info(f"PyTorch available, device: {self.device}")
        except ImportError:
            logger.warning("PyTorch not available")

        try:
            import transformers
            self.has_transformers = True
            logger.info("Transformers available")
        except ImportError:
            logger.warning("Transformers not available")

        try:
            import cadquery
            self.has_cadquery = True
            logger.info("CadQuery available")
        except ImportError:
            logger.debug("CadQuery not available")

        try:
            import build123d
            self.has_build123d = True
            logger.info("Build123d available")
        except ImportError:
            logger.debug("Build123d not available")

        try:
            from OCC.Core.gp import gp_Pnt
            self.has_occ = True
            logger.info("PythonOCC available")
        except ImportError:
            logger.debug("PythonOCC not available")

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        models = []
        for model_id, config in AVAILABLE_MODELS.items():
            models.append({
                "id": model_id,
                "name": config.name,
                "size": config.size,
                "type": config.model_type,
                "requires_gpu": config.requires_gpu,
                "min_vram_gb": config.min_vram_gb,
                "output_format": config.output_format.value,
                "downloaded": self._is_model_cached(model_id),
            })
        return models

    def _is_model_cached(self, model_id: str) -> bool:
        """Check if a model is already downloaded"""
        model_dir = self.cache_dir / model_id
        return model_dir.exists() and any(model_dir.iterdir())

    def get_model_config(self, model_id: str) -> Optional[VLMModelConfig]:
        """Get configuration for a model"""
        return AVAILABLE_MODELS.get(model_id)

    def download_model(self, model_id: str, force: bool = False) -> Path:
        """
        Download a model from Hugging Face.

        Args:
            model_id: ID of the model to download
            force: Force re-download even if cached

        Returns:
            Path to the downloaded model
        """
        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(AVAILABLE_MODELS.keys())}")

        config = AVAILABLE_MODELS[model_id]
        model_dir = self.cache_dir / model_id

        if model_dir.exists() and not force:
            logger.info(f"Model {model_id} already cached at {model_dir}")
            return model_dir

        if not self.has_transformers:
            raise RuntimeError("transformers library required. Install: pip install transformers")

        logger.info(f"Downloading {config.name} from {config.hf_repo}...")

        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=config.hf_repo,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )

        logger.info(f"Model downloaded to {model_dir}")
        return model_dir

    def load_model(self, model_id: str, device: Optional[str] = None):
        """
        Load a model into memory.

        Args:
            model_id: ID of the model to load
            device: Device to load on (cuda, mps, cpu)

        Returns:
            Loaded model and processor/tokenizer
        """
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_id}")

        config = AVAILABLE_MODELS[model_id]
        model_dir = self.cache_dir / model_id

        if not model_dir.exists():
            self.download_model(model_id)

        device = device or self.device
        logger.info(f"Loading {config.name} on {device}...")

        if config.model_type == "openecad":
            model, processor = self._load_openecad(model_dir, device)
        elif config.model_type == "internvl":
            model, processor = self._load_internvl(model_dir, device)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

        self.loaded_models[model_id] = {
            "model": model,
            "processor": processor,
            "config": config,
            "device": device,
        }

        return self.loaded_models[model_id]

    def _load_openecad(self, model_dir: Path, device: str):
        """
        Load OpenECAD model (TinyLLaVA-based architecture).

        OpenECAD uses TinyLLaVA with SigLIP/CLIP vision encoder.
        Models have a custom .chat() method for inference.

        IMPORTANT: Requires TinyLLaVA library installed from source:
            git clone https://github.com/TinyLLaVA/TinyLLaVA_Factory.git
            cd TinyLLaVA_Factory && pip install -e .
        """
        import torch

        # Check if TinyLLaVA is installed
        try:
            from tinyllava.model.builder import load_pretrained_model
            from tinyllava.mm_utils import get_model_name_from_path
            has_tinyllava = True
        except ImportError:
            has_tinyllava = False
            logger.warning(
                "TinyLLaVA not installed. OpenECAD models require TinyLLaVA. "
                "Install from: https://github.com/TinyLLaVA/TinyLLaVA_Factory"
            )

        # Get the HuggingFace repo ID for this model
        config = self.get_model_config(model_dir.name)
        hf_repo = config.hf_repo if config else str(model_dir)

        if has_tinyllava:
            # Use TinyLLaVA's native loading
            model_name = get_model_name_from_path(hf_repo)
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=hf_repo,
                model_base=None,
                model_name=model_name,
                device=device,
            )
            # Store image_processor for later use
            model._image_processor = image_processor
            model._context_len = context_len
            return model, tokenizer
        else:
            # Fallback: try loading with transformers (may not work for all models)
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                model = AutoModelForCausalLM.from_pretrained(
                    hf_repo,
                    dtype=torch.float16 if device != "cpu" else torch.float32,  # Changed from torch_dtype (deprecated)
                    trust_remote_code=True,
                    device_map="auto" if device == "cuda" else None,
                    low_cpu_mem_usage=True,
                )

                if device == "mps":
                    model = model.to(device)

                tokenizer = AutoTokenizer.from_pretrained(
                    hf_repo,
                    trust_remote_code=True,
                    use_fast=False,
                )

                return model, tokenizer

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load OpenECAD model. TinyLLaVA library required. "
                    f"Install from: https://github.com/TinyLLaVA/TinyLLaVA_Factory\n"
                    f"Error: {e}"
                )

    def _load_internvl(self, model_dir: Path, device: str):
        """Load InternVL model"""
        import torch
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(
            str(model_dir),
            dtype=torch.bfloat16 if device != "cpu" else torch.float32,  # Changed from torch_dtype (deprecated)
            trust_remote_code=True,
            device_map=device if device == "cuda" else None,
        )

        if device == "mps":
            model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )

        return model, tokenizer

    def unload_model(self, model_id: str):
        """Unload a model from memory"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            import gc
            gc.collect()
            if self.has_torch:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # MPS doesn't have empty_cache, but gc.collect helps
                    pass


# =============================================================================
# VLM CAD Inference
# =============================================================================

class VLMCADInference:
    """
    Inference pipeline for VLM-based CAD reconstruction.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.current_model_id: Optional[str] = None

        # Initialize executors lazily
        self._openecad_executor: Optional[OpenECADExecutor] = None
        self._cadquery_executor: Optional[CadQueryExecutor] = None
        self._build123d_executor: Optional[Build123dExecutor] = None

    def load_model(self, model_id: str = "openecad-0.89b"):
        """Load a model for inference"""
        self.model_manager.load_model(model_id)
        self.current_model_id = model_id

    def generate_cad_code(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Generate CAD code from an image.

        Args:
            image_path: Path to the 2D CAD drawing image
            prompt: Custom prompt (uses model default if None)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with generated code, validation result, and metadata
        """
        if not self.current_model_id:
            raise RuntimeError("No model loaded. Call load_model() first.")

        model_data = self.model_manager.loaded_models[self.current_model_id]
        model = model_data["model"]
        processor = model_data["processor"]
        config = model_data["config"]

        # Load image
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        # Use model-specific prompt if not provided
        if prompt is None:
            prompt = config.default_prompt

        # Generate based on model type
        if config.model_type == "openecad":
            raw_output = self._generate_openecad(model, processor, image, prompt, max_tokens, temperature)
        elif config.model_type == "internvl":
            raw_output = self._generate_internvl(model, processor, image, prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

        # Extract and validate code
        code = CADCodeValidator.extract_code_block(raw_output)
        validation = CADCodeValidator.validate(code)

        return {
            "code": code,
            "raw_output": raw_output,
            "validation": {
                "valid": validation.valid,
                "format": validation.format.value,
                "errors": validation.errors,
                "warnings": validation.warnings,
                "operations": validation.detected_operations,
            },
            "model_id": self.current_model_id,
            "model_name": config.name,
            "image_path": str(image_path),
        }

    def _generate_openecad(self, model, tokenizer, image, prompt, max_tokens, temperature):
        """
        Generate using OpenECAD model (TinyLLaVA-based).

        Supports both TinyLLaVA native loading and transformers fallback.
        """
        import torch

        # Check if this is a TinyLLaVA model with native eval
        try:
            from tinyllava.eval.run_tiny_llava import eval_model
            has_tinyllava_eval = True
        except ImportError:
            has_tinyllava_eval = False

        # Save image to temp file
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            image.save(f.name)
            image_path = f.name

        try:
            if has_tinyllava_eval and hasattr(model, '_image_processor'):
                # Use TinyLLaVA's native eval
                from argparse import Namespace
                args = Namespace(
                    model_path=None,  # Already loaded
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=model._image_processor,
                    context_len=getattr(model, '_context_len', 2048),
                    query=prompt,
                    conv_mode="phi",  # Default conversation mode
                    image_file=image_path,
                    temperature=temperature if temperature > 0 else 0,
                    num_beams=1,
                    max_new_tokens=max_tokens,
                )
                output_text = eval_model(args)
                return output_text

            elif hasattr(model, 'chat'):
                # Use model's chat method
                output_text, generation_time = model.chat(
                    prompt=prompt,
                    image=image_path,
                    tokenizer=tokenizer,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                )
                return output_text

            else:
                # Fallback to standard generation
                logger.warning("OpenECAD model: using fallback generation (may not include image)")
                return self._generate_openecad_fallback(model, tokenizer, image, prompt, max_tokens, temperature)

        finally:
            os.unlink(image_path)

    def _generate_openecad_fallback(self, model, tokenizer, image, prompt, max_tokens, temperature):
        """Fallback generation for OpenECAD models without TinyLLaVA"""
        import torch

        device = next(model.parameters()).device

        # This fallback doesn't process the image - just generates text
        # It's here to prevent complete failure but won't produce good CAD output
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated

    def _generate_internvl(self, model, tokenizer, image, prompt, max_tokens, temperature):
        """Generate using InternVL model"""
        import torch

        # InternVL uses a different interface
        pixel_values = self._process_image_internvl(image, model)

        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
        }

        with torch.no_grad():
            response = model.chat(
                tokenizer,
                pixel_values,
                prompt,
                generation_config,
            )

        return response

    def _process_image_internvl(self, image, model):
        """Process image for InternVL"""
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        pixel_values = transform(image).unsqueeze(0)
        device = next(model.parameters()).device
        return pixel_values.to(device, dtype=torch.bfloat16)

    def code_to_mesh(
        self,
        code: str,
        output_format: str = "glb",
        code_format: Optional[CADCodeFormat] = None,
    ) -> bytes:
        """
        Convert generated CAD code to a mesh.

        Args:
            code: Generated CAD code
            output_format: Output format (glb, obj, stl)
            code_format: Force a specific code format (auto-detect if None)

        Returns:
            Mesh data as bytes
        """
        import trimesh

        # Validate and detect format
        validation = CADCodeValidator.validate(code)

        if not validation.valid:
            error_msg = "; ".join(validation.errors)
            raise ValueError(f"Invalid CAD code: {error_msg}")

        detected_format = code_format or validation.format

        # Execute code based on format
        mesh = None

        if detected_format == CADCodeFormat.OPENECAD:
            mesh = self._execute_openecad(code)
        elif detected_format == CADCodeFormat.CADQUERY:
            mesh = self._execute_cadquery(code)
        elif detected_format == CADCodeFormat.BUILD123D:
            mesh = self._execute_build123d(code)
        elif detected_format == CADCodeFormat.PYTHON_OCC:
            mesh = self._execute_pythonocc(code)
        else:
            # Fallback: try each executor
            mesh = self._try_all_executors(code)

        if mesh is None:
            raise RuntimeError(f"Failed to execute CAD code (format: {detected_format.value})")

        # Export to requested format
        return self._export_mesh(mesh, output_format)

    def _execute_openecad(self, code: str):
        """Execute OpenECAD format code"""
        if self._openecad_executor is None:
            self._openecad_executor = OpenECADExecutor()

        if not self._openecad_executor.has_occ:
            logger.warning("PythonOCC not available, falling back to CadQuery")
            return self._try_cadquery_fallback(code)

        try:
            shape = self._openecad_executor.execute(code)
            return ShapeToMeshConverter.occ_to_trimesh(shape)
        except Exception as e:
            logger.error(f"OpenECAD execution failed: {e}")
            raise

    def _execute_cadquery(self, code: str):
        """Execute CadQuery format code"""
        if self._cadquery_executor is None:
            self._cadquery_executor = CadQueryExecutor()

        if not self._cadquery_executor.has_cadquery:
            raise RuntimeError("CadQuery not available. Install: pip install cadquery")

        try:
            result = self._cadquery_executor.execute(code)
            return ShapeToMeshConverter.cadquery_to_trimesh(result)
        except Exception as e:
            logger.error(f"CadQuery execution failed: {e}")
            raise

    def _execute_build123d(self, code: str):
        """Execute Build123d format code"""
        if self._build123d_executor is None:
            self._build123d_executor = Build123dExecutor()

        if not self._build123d_executor.has_build123d:
            raise RuntimeError("Build123d not available. Install: pip install build123d")

        try:
            result = self._build123d_executor.execute(code)
            return ShapeToMeshConverter.build123d_to_trimesh(result)
        except Exception as e:
            logger.error(f"Build123d execution failed: {e}")
            raise

    def _execute_pythonocc(self, code: str):
        """Execute raw PythonOCC code"""
        if not self.model_manager.has_occ:
            raise RuntimeError("PythonOCC not available. Install: pip install pythonocc-core")

        # Create execution namespace with OCC imports
        namespace = {"result": None}

        try:
            exec(code, namespace)
            shape = namespace.get("result")
            if shape is None:
                raise RuntimeError("No 'result' variable found")
            return ShapeToMeshConverter.occ_to_trimesh(shape)
        except Exception as e:
            logger.error(f"PythonOCC execution failed: {e}")
            raise

    def _try_all_executors(self, code: str):
        """Try all available executors"""
        errors = []

        # Try CadQuery first (most common)
        if self.model_manager.has_cadquery:
            try:
                return self._execute_cadquery(code)
            except Exception as e:
                errors.append(f"CadQuery: {e}")

        # Try Build123d
        if self.model_manager.has_build123d:
            try:
                return self._execute_build123d(code)
            except Exception as e:
                errors.append(f"Build123d: {e}")

        # Try OpenECAD/PythonOCC
        if self.model_manager.has_occ:
            try:
                return self._execute_openecad(code)
            except Exception as e:
                errors.append(f"OpenECAD: {e}")

        raise RuntimeError(f"All executors failed: {'; '.join(errors)}")

    def _try_cadquery_fallback(self, code: str):
        """Convert OpenECAD code to CadQuery and execute"""
        # This is a simplified conversion - real implementation would be more complex
        logger.warning("OpenECAD to CadQuery conversion not fully implemented")
        raise NotImplementedError("OpenECAD to CadQuery conversion not yet implemented")

    def _export_mesh(self, mesh, output_format: str) -> bytes:
        """Export trimesh to bytes in requested format"""
        if output_format == "glb":
            return mesh.export(file_type='glb')
        elif output_format == "obj":
            result = mesh.export(file_type='obj')
            return result.encode() if isinstance(result, str) else result
        elif output_format == "stl":
            return mesh.export(file_type='stl')
        elif output_format == "ply":
            return mesh.export(file_type='ply')
        else:
            raise ValueError(f"Unsupported format: {output_format}")


# =============================================================================
# Convenience Functions
# =============================================================================

def list_models() -> List[Dict[str, Any]]:
    """List all available VLM models"""
    manager = ModelManager()
    return manager.list_available_models()


def download_model(model_id: str) -> Path:
    """Download a model"""
    manager = ModelManager()
    return manager.download_model(model_id)


def generate_cad(
    image_path: Union[str, Path],
    model_id: str = "openecad-0.89b",
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate CAD code from an image using a VLM.

    Args:
        image_path: Path to 2D CAD drawing
        model_id: Model to use
        prompt: Custom prompt (optional)

    Returns:
        Generated code, validation result, and metadata
    """
    inference = VLMCADInference()
    inference.load_model(model_id)
    return inference.generate_cad_code(image_path, prompt=prompt)


def reconstruct_from_image(
    image_path: Union[str, Path],
    model_id: str = "openecad-0.89b",
    output_format: str = "glb",
) -> bytes:
    """
    Full pipeline: image → VLM → CAD code → mesh

    Args:
        image_path: Path to 2D CAD drawing
        model_id: Model to use
        output_format: Output mesh format (glb, obj, stl)

    Returns:
        Mesh data as bytes
    """
    inference = VLMCADInference()
    inference.load_model(model_id)

    result = inference.generate_cad_code(image_path)

    if not result["validation"]["valid"]:
        raise ValueError(f"Generated code is invalid: {result['validation']['errors']}")

    return inference.code_to_mesh(result["code"], output_format=output_format)


def validate_cad_code(code: str) -> Dict[str, Any]:
    """
    Validate CAD code without executing it.

    Args:
        code: CAD code to validate

    Returns:
        Validation result dict
    """
    result = CADCodeValidator.validate(code)
    return {
        "valid": result.valid,
        "format": result.format.value,
        "errors": result.errors,
        "warnings": result.warnings,
        "operations": result.detected_operations,
    }


def get_available_executors() -> Dict[str, bool]:
    """Get availability status of CAD code executors"""
    manager = ModelManager()
    return {
        "cadquery": manager.has_cadquery,
        "build123d": manager.has_build123d,
        "pythonocc": manager.has_occ,
        "torch": manager.has_torch,
        "transformers": manager.has_transformers,
    }
