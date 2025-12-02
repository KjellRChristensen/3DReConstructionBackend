"""
FastAPI Server - REST API for iOS frontend communication
"""
from pathlib import Path
from typing import Optional
from enum import Enum
import uuid
import asyncio
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

# Will be imported when config is properly set up
# from config import settings


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExportFormat(str, Enum):
    OBJ = "obj"
    GLTF = "gltf"
    GLB = "glb"
    STL = "stl"
    USDZ = "usdz"
    IFC = "ifc"


class ReconStrategy(str, Enum):
    """Available reconstruction strategies"""
    EXTERNAL_API = "external_api"      # Strategy A: Kaedim, Meshy, Replicate
    BASIC_EXTRUSION = "basic_extrusion"  # Strategy B: Built-in wall extrusion
    MULTI_VIEW_DNN = "multi_view_dnn"  # Strategy C: GaussianCAD, depth estimation
    AUTO = "auto"  # Auto-select best available


class JobCreate(BaseModel):
    """Request to create a new reconstruction job"""
    export_formats: list[ExportFormat] = [ExportFormat.GLB, ExportFormat.USDZ]
    wall_height: Optional[float] = None
    num_floors: int = 1
    options: Optional[dict] = None


class JobResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: JobStatus
    progress: float  # 0.0 to 1.0
    current_stage: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    output_files: list[str] = []


class PipelineProgress(BaseModel):
    """Detailed pipeline progress"""
    stage: str
    progress: float
    message: str


# In-memory job storage (replace with Redis/DB in production)
jobs: dict[str, dict] = {}


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title="3D Reconstruction API",
        description="Convert 2D floor plans to 3D models",
        version="0.1.0",
    )

    # CORS for iOS app
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "version": "0.1.0"}

    @app.post("/jobs", response_model=JobResponse)
    async def create_job(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        export_formats: str = "glb,usdz",  # Comma-separated
        wall_height: Optional[float] = None,
        num_floors: int = 1,
    ):
        """
        Upload a floor plan and start reconstruction job.

        - **file**: PDF or image file of floor plan
        - **export_formats**: Comma-separated output formats (obj, gltf, glb, stl, usdz, ifc)
        - **wall_height**: Override default wall height (meters)
        - **num_floors**: Number of floors to generate
        """
        # Validate file type
        allowed_types = {
            "application/pdf",
            "image/png",
            "image/jpeg",
            "image/tiff",
        }
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )

        # Create job
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Save uploaded file
        input_dir = Path("data/input") / job_id
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / file.filename

        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # Parse export formats
        formats = [f.strip() for f in export_formats.split(",")]

        # Store job info
        jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "progress": 0.0,
            "current_stage": None,
            "created_at": now,
            "completed_at": None,
            "error": None,
            "output_files": [],
            "input_path": str(input_path),
            "export_formats": formats,
            "wall_height": wall_height,
            "num_floors": num_floors,
        }

        # Start processing in background
        background_tasks.add_task(process_job, job_id)

        return JobResponse(**jobs[job_id])

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: str):
        """Get job status and progress"""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobResponse(**jobs[job_id])

    @app.get("/jobs/{job_id}/progress", response_model=PipelineProgress)
    async def get_progress(job_id: str):
        """Get detailed pipeline progress"""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]
        return PipelineProgress(
            stage=job.get("current_stage") or "waiting",
            progress=job["progress"],
            message=f"Processing: {job.get('current_stage', 'queued')}"
        )

    @app.get("/jobs/{job_id}/download/{filename}")
    async def download_output(job_id: str, filename: str):
        """Download a completed output file"""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]
        if job["status"] != JobStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Job not completed")

        output_path = Path("data/output") / job_id / filename
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            output_path,
            filename=filename,
            media_type="application/octet-stream"
        )

    @app.delete("/jobs/{job_id}")
    async def delete_job(job_id: str):
        """Delete a job and its files"""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        # Clean up files
        import shutil
        input_dir = Path("data/input") / job_id
        output_dir = Path("data/output") / job_id
        if input_dir.exists():
            shutil.rmtree(input_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)

        del jobs[job_id]
        return {"status": "deleted"}

    @app.get("/jobs")
    async def list_jobs(limit: int = 50):
        """List recent jobs"""
        sorted_jobs = sorted(
            jobs.values(),
            key=lambda j: j["created_at"],
            reverse=True
        )[:limit]
        return [JobResponse(**j) for j in sorted_jobs]

    # === Input Files Browser Endpoints ===

    @app.get("/files/input")
    async def list_input_files():
        """
        List all files in the input folder.
        Returns file names, sizes, and types for frontend browsing.
        """
        input_dir = Path("data/input")
        if not input_dir.exists():
            input_dir.mkdir(parents=True, exist_ok=True)
            return {"files": [], "total": 0}

        files = []
        for item in input_dir.iterdir():
            if item.is_file():
                stat = item.stat()
                files.append({
                    "name": item.name,
                    "path": str(item),
                    "size": stat.st_size,
                    "size_human": _format_size(stat.st_size),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": _get_file_type(item.suffix),
                    "extension": item.suffix.lower(),
                })

        # Sort by modified date, newest first
        files.sort(key=lambda x: x["modified"], reverse=True)

        return {
            "files": files,
            "total": len(files),
            "directory": str(input_dir.absolute())
        }

    @app.get("/files/input/{filename}")
    async def get_input_file(filename: str):
        """
        Get a specific input file (for preview/download).
        """
        input_path = Path("data/input") / filename

        if not input_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Determine media type
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".dxf": "application/dxf",
            ".dwg": "application/dwg",
        }
        media_type = media_types.get(input_path.suffix.lower(), "application/octet-stream")

        return FileResponse(
            input_path,
            filename=filename,
            media_type=media_type
        )

    @app.delete("/files/input/{filename}")
    async def delete_input_file(filename: str):
        """Delete an input file"""
        input_path = Path("data/input") / filename

        if not input_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        input_path.unlink()
        return {"status": "deleted", "filename": filename}

    @app.post("/jobs/from-file")
    async def create_job_from_file(
        background_tasks: BackgroundTasks,
        filename: str,
        export_formats: str = "glb,usdz",
        wall_height: Optional[float] = None,
        num_floors: int = 1,
    ):
        """
        Start a reconstruction job from an existing file in the input folder.

        - **filename**: Name of file in input folder
        - **export_formats**: Comma-separated output formats
        - **wall_height**: Override default wall height (meters)
        - **num_floors**: Number of floors to generate
        """
        input_path = Path("data/input") / filename

        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        # Create job
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Parse export formats
        formats = [f.strip() for f in export_formats.split(",")]

        # Store job info
        jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "progress": 0.0,
            "current_stage": None,
            "created_at": now,
            "completed_at": None,
            "error": None,
            "output_files": [],
            "input_path": str(input_path),
            "input_filename": filename,
            "export_formats": formats,
            "wall_height": wall_height,
            "num_floors": num_floors,
        }

        # Start processing in background
        background_tasks.add_task(process_job, job_id)

        return JobResponse(**jobs[job_id])

    @app.post("/files/input")
    async def upload_input_file(file: UploadFile = File(...)):
        """
        Upload a new file to the input folder.
        """
        input_dir = Path("data/input")
        input_dir.mkdir(parents=True, exist_ok=True)

        input_path = input_dir / file.filename

        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        stat = input_path.stat()
        return {
            "status": "uploaded",
            "file": {
                "name": file.filename,
                "path": str(input_path),
                "size": stat.st_size,
                "size_human": _format_size(stat.st_size),
                "type": _get_file_type(input_path.suffix),
            }
        }

    # === Output Files Browser Endpoints ===

    @app.get("/files/output")
    async def list_output_files():
        """List all files in the output folder"""
        output_dir = Path("data/output")
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            return {"files": [], "total": 0}

        files = []
        for item in output_dir.rglob("*"):
            if item.is_file():
                stat = item.stat()
                rel_path = item.relative_to(output_dir)
                files.append({
                    "name": item.name,
                    "path": str(rel_path),
                    "full_path": str(item),
                    "size": stat.st_size,
                    "size_human": _format_size(stat.st_size),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": _get_file_type(item.suffix),
                    "extension": item.suffix.lower(),
                })

        files.sort(key=lambda x: x["modified"], reverse=True)

        return {
            "files": files,
            "total": len(files),
            "directory": str(output_dir.absolute())
        }

    @app.get("/files/output/{filepath:path}")
    async def get_output_file(filepath: str):
        """Get a specific output file"""
        output_path = Path("data/output") / filepath

        if not output_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        media_types = {
            ".obj": "model/obj",
            ".glb": "model/gltf-binary",
            ".gltf": "model/gltf+json",
            ".stl": "model/stl",
            ".usdz": "model/vnd.usdz+zip",
            ".ifc": "application/x-step",
            ".png": "image/png",
            ".jpg": "image/jpeg",
        }
        media_type = media_types.get(output_path.suffix.lower(), "application/octet-stream")

        return FileResponse(
            output_path,
            filename=output_path.name,
            media_type=media_type
        )

    # === Reconstruction Strategy Endpoints ===

    @app.get("/system/gpu")
    async def check_gpu():
        """
        Check GPU availability (MPS for Apple Silicon, CUDA, or CPU).
        """
        result = {
            "pytorch_installed": False,
            "device": "cpu",
            "mps_available": False,
            "mps_built": False,
            "cuda_available": False,
            "device_name": "CPU",
        }

        try:
            import torch
            result["pytorch_installed"] = True
            result["pytorch_version"] = torch.__version__

            # Check MPS (Apple Silicon)
            result["mps_available"] = torch.backends.mps.is_available()
            result["mps_built"] = torch.backends.mps.is_built()

            # Check CUDA
            result["cuda_available"] = torch.cuda.is_available()
            if result["cuda_available"]:
                result["cuda_device_count"] = torch.cuda.device_count()
                result["cuda_device_name"] = torch.cuda.get_device_name(0)

            # Determine best device
            if result["mps_available"]:
                result["device"] = "mps"
                result["device_name"] = "Apple Silicon GPU (MPS)"
            elif result["cuda_available"]:
                result["device"] = "cuda"
                result["device_name"] = result["cuda_device_name"]
            else:
                result["device"] = "cpu"
                result["device_name"] = "CPU"

        except ImportError:
            result["error"] = "PyTorch not installed"

        return result

    @app.get("/strategies")
    async def list_strategies():
        """
        List available reconstruction strategies and their status.
        """
        strategies = [
            {
                "id": "external_api",
                "name": "External DNN API",
                "description": "Use external AI services (Kaedim, Meshy, Replicate, Tripo3D)",
                "services": ["kaedim", "meshy", "replicate", "tripo3d"],
                "requires_api_key": True,
                "best_for": "High-quality results, production use",
                "accuracy": "85-95%",
                "speed": "1-5 minutes",
            },
            {
                "id": "basic_extrusion",
                "name": "Basic Extrusion",
                "description": "Built-in wall detection and extrusion (no external dependencies)",
                "services": [],
                "requires_api_key": False,
                "best_for": "Quick previews, simple floor plans",
                "accuracy": "60-80%",
                "speed": "Seconds",
            },
            {
                "id": "multi_view_dnn",
                "name": "Multi-View DNN",
                "description": "Deep learning reconstruction (depth estimation, GaussianCAD)",
                "services": ["depth_estimation", "gaussian_cad", "cad2program"],
                "requires_api_key": False,
                "best_for": "Multiple views, complex buildings",
                "accuracy": "80-95%",
                "speed": "30 seconds - 2 minutes",
            },
        ]

        # Check availability
        for strategy in strategies:
            strategy["available"] = _check_strategy_available(strategy["id"])

        return {
            "strategies": strategies,
            "default": "basic_extrusion",
            "recommended": "multi_view_dnn" if _check_strategy_available("multi_view_dnn") else "basic_extrusion",
        }

    @app.post("/reconstruct")
    async def reconstruct_file(
        background_tasks: BackgroundTasks,
        filename: str,
        strategy: ReconStrategy = ReconStrategy.AUTO,
        service: Optional[str] = None,
        model_type: Optional[str] = None,
        wall_height: float = 2.8,
        export_format: str = "glb",
        additional_views: Optional[str] = None,  # Comma-separated filenames
    ):
        """
        Reconstruct a 3D model from an input file.

        - **filename**: Name of file in input folder
        - **strategy**: Reconstruction strategy (external_api, basic_extrusion, multi_view_dnn, auto)
        - **service**: For external_api: kaedim, meshy, replicate, tripo3d
        - **model_type**: For multi_view_dnn: depth_estimation, gaussian_cad
        - **wall_height**: Wall height in meters (default 2.8)
        - **export_format**: Output format (glb, obj, stl)
        - **additional_views**: Comma-separated list of additional view files (for multi-view)
        """
        input_path = Path("data/input") / filename

        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        # Parse additional views
        extra_views = []
        if additional_views:
            for view_name in additional_views.split(","):
                view_path = Path("data/input") / view_name.strip()
                if view_path.exists():
                    extra_views.append(view_path)

        # Create job
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()

        jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "progress": 0.0,
            "current_stage": None,
            "created_at": now,
            "completed_at": None,
            "error": None,
            "output_files": [],
            "input_path": str(input_path),
            "input_filename": filename,
            "strategy": strategy.value,
            "service": service,
            "model_type": model_type,
            "wall_height": wall_height,
            "export_format": export_format,
            "additional_views": [str(v) for v in extra_views],
        }

        # Start processing
        background_tasks.add_task(process_reconstruction_job, job_id)

        return {
            "job_id": job_id,
            "status": "pending",
            "strategy": strategy.value,
            "message": f"Reconstruction started with {strategy.value} strategy",
        }

    @app.post("/reconstruct/preview")
    async def reconstruct_preview(
        filename: str,
        wall_height: float = 2.8,
    ):
        """
        Quick preview reconstruction (synchronous, basic strategy only).
        Returns immediately with a simple 3D model.
        """
        input_path = Path("data/input") / filename

        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        try:
            from ..reconstruction.strategies import BasicExtrusionStrategy, ReconstructionInput

            strategy = BasicExtrusionStrategy()

            if not strategy.is_available():
                raise HTTPException(
                    status_code=503,
                    detail="Basic reconstruction not available. Install: pip install trimesh numpy"
                )

            input_data = ReconstructionInput(
                primary_image=input_path,
                wall_height=wall_height,
            )

            # Run reconstruction (already in async context)
            result = await strategy.reconstruct(input_data)

            if result.success:
                # Save to output
                output_dir = Path("data/output") / "previews"
                output_dir.mkdir(parents=True, exist_ok=True)

                output_filename = f"{input_path.stem}_preview.{result.format}"
                output_path = output_dir / output_filename

                with open(output_path, "wb") as f:
                    f.write(result.model_data)

                return {
                    "success": True,
                    "output_file": f"previews/{output_filename}",
                    "download_url": f"/files/output/previews/{output_filename}",
                    "format": result.format,
                    "strategy": result.strategy_used,
                }
            else:
                raise HTTPException(status_code=500, detail=result.error)

        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Missing dependencies: {e}. Install: pip install trimesh numpy opencv-python"
            )

    # === Validation Endpoints ===

    @app.get("/validation/formats")
    async def list_validation_formats():
        """List supported ground truth formats for validation"""
        return {
            "supported_formats": [
                {"extension": ".ifc", "name": "IFC (BIM)", "description": "Industry Foundation Classes - full building information"},
                {"extension": ".obj", "name": "OBJ", "description": "Wavefront 3D mesh"},
                {"extension": ".glb", "name": "GLB", "description": "GL Transmission Format (binary)"},
                {"extension": ".gltf", "name": "GLTF", "description": "GL Transmission Format (JSON)"},
                {"extension": ".stl", "name": "STL", "description": "Stereolithography mesh"},
                {"extension": ".dxf", "name": "DXF", "description": "AutoCAD Drawing Exchange Format"},
            ],
            "metrics_computed": [
                {"name": "chamfer_distance", "description": "Average nearest-neighbor distance (lower is better)"},
                {"name": "hausdorff_distance", "description": "Maximum deviation (lower is better)"},
                {"name": "iou_3d", "description": "3D Intersection over Union (higher is better, 0-1)"},
                {"name": "f_score", "description": "F-score at distance threshold (higher is better, 0-1)"},
            ]
        }

    @app.post("/validation/run")
    async def run_validation(
        background_tasks: BackgroundTasks,
        ground_truth_file: str,
        strategy: str = "auto",
        wall_height: float = 2.8,
        floor_height: float = 1.0,
    ):
        """
        Run validation against a ground truth 3D model.

        Pipeline:
        1. Load ground truth CAD file
        2. Generate 2D floor plan projection
        3. Run reconstruction on the projection
        4. Compare reconstructed 3D to ground truth
        5. Return metrics

        Args:
            ground_truth_file: Filename in input folder (IFC, OBJ, GLB, etc.)
            strategy: Reconstruction strategy (auto, basic_extrusion, multi_view_dnn)
            wall_height: Wall height in meters
            floor_height: Height for floor plan slice
        """
        gt_path = Path("data/input") / ground_truth_file

        if not gt_path.exists():
            raise HTTPException(status_code=404, detail=f"Ground truth file not found: {ground_truth_file}")

        # Create validation job
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()

        jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "progress": 0.0,
            "current_stage": "initializing",
            "created_at": now,
            "completed_at": None,
            "error": None,
            "output_files": [],
            "job_type": "validation",
            "ground_truth_file": ground_truth_file,
            "strategy": strategy,
            "wall_height": wall_height,
            "floor_height": floor_height,
        }

        background_tasks.add_task(process_validation_job, job_id)

        return {
            "job_id": job_id,
            "status": "pending",
            "message": f"Validation started for {ground_truth_file}",
        }

    @app.post("/validation/compare")
    async def compare_meshes(
        predicted_file: str,
        ground_truth_file: str,
    ):
        """
        Compare two 3D mesh files directly.

        Returns comparison metrics without running reconstruction.
        """
        pred_path = Path("data/output") / predicted_file
        gt_path = Path("data/input") / ground_truth_file

        if not pred_path.exists():
            # Also check in input
            pred_path = Path("data/input") / predicted_file
            if not pred_path.exists():
                raise HTTPException(status_code=404, detail=f"Predicted file not found: {predicted_file}")

        if not gt_path.exists():
            raise HTTPException(status_code=404, detail=f"Ground truth file not found: {ground_truth_file}")

        try:
            from ..validation.cad_import import CADImporter
            from ..validation.metrics import MeshComparison

            importer = CADImporter()
            pred_model = importer.load(pred_path)
            gt_model = importer.load(gt_path)

            comparator = MeshComparison()
            result = comparator.compare(pred_model.mesh, gt_model.mesh)

            return {
                "success": True,
                "metrics": result.to_dict(),
                "summary": result.summary(),
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/validation/project")
    async def generate_projection(
        model_file: str,
        floor_height: float = 1.0,
        resolution: int = 1024,
    ):
        """
        Generate a 2D floor plan projection from a 3D model.

        Useful for creating training data pairs.
        """
        model_path = Path("data/input") / model_file

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_file}")

        try:
            from ..validation.cad_import import CADImporter
            from ..validation.projection import ProjectionGenerator

            importer = CADImporter()
            model = importer.load(model_path)

            projector = ProjectionGenerator(resolution=resolution)
            projection = projector.generate_floor_plan(model.mesh, floor_height=floor_height)

            # Save projection
            output_dir = Path("data/output") / "projections"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_name = f"{model_path.stem}_projection.png"
            output_path = output_dir / output_name
            projector.save_projection(projection, output_path)

            return {
                "success": True,
                "output_file": f"projections/{output_name}",
                "download_url": f"/files/output/projections/{output_name}",
                "projection_info": {
                    "width": projection.width,
                    "height": projection.height,
                    "scale": projection.scale,
                    "floor_height": projection.floor_height,
                }
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/validation/training-pair")
    async def generate_training_pair(
        model_file: str,
        floor_height: float = 1.0,
    ):
        """
        Generate a training data pair (2D projection + 3D model metadata).

        Creates:
        - {name}_2d.png: Floor plan image (X)
        - {name}_3d.glb: 3D model copy (Y)
        - {name}_meta.json: Metadata linking the pair
        """
        model_path = Path("data/input") / model_file

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_file}")

        try:
            from ..validation.cad_import import CADImporter
            from ..validation.projection import ProjectionGenerator

            importer = CADImporter()
            model = importer.load(model_path)

            projector = ProjectionGenerator()
            output_dir = Path("data/output") / "training_pairs"

            result = projector.generate_training_pair(
                model.mesh,
                output_dir=output_dir,
                name=model_path.stem,
                floor_height=floor_height,
            )

            return {
                "success": True,
                "files": {
                    "x_2d": str(result["x"].relative_to(Path("data/output"))),
                    "y_3d": str(result["y"].relative_to(Path("data/output"))),
                    "metadata": str(result["metadata"].relative_to(Path("data/output"))),
                },
                "download_urls": {
                    "x_2d": f"/files/output/{result['x'].relative_to(Path('data/output'))}",
                    "y_3d": f"/files/output/{result['y'].relative_to(Path('data/output'))}",
                    "metadata": f"/files/output/{result['metadata'].relative_to(Path('data/output'))}",
                }
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # === Training Data Generation Endpoints ===

    @app.get("/training/info")
    async def training_info():
        """Get information about training data generation capabilities"""
        return {
            "description": "Generate training pairs for CAD2Program fine-tuning",
            "supported_input_formats": [".obj", ".glb", ".gltf", ".stl", ".ply", ".ifc"],
            "output_views": ["front", "top", "right", "left", "back", "bottom", "isometric"],
            "default_views": ["front", "top", "right"],
            "output_formats": ["png", "svg"],
            "features": {
                "hidden_lines": "Dashed hidden line rendering (engineering style)",
                "batch_processing": "Process multiple models at once",
                "metadata": "JSON metadata with scale, bounds, and model info",
            },
            "endpoints": {
                "render_views": "POST /training/render-views - Render orthographic views",
                "batch_render": "POST /training/batch-render - Batch process directory",
                "generate_pair": "POST /training/generate-pair - Create complete training pair",
            }
        }

    @app.post("/training/render-views")
    async def render_orthographic_views(
        model_file: str,
        resolution: int = 1024,
        show_hidden_lines: bool = True,
        views: str = "front,top,right",  # Comma-separated
    ):
        """
        Render orthographic views of a 3D model.

        - **model_file**: Name of 3D model in input folder
        - **resolution**: Image resolution in pixels (default 1024)
        - **show_hidden_lines**: Include dashed hidden lines (default True)
        - **views**: Comma-separated list of views (front,top,right,left,back,bottom,isometric)
        """
        model_path = Path("data/input") / model_file

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_file}")

        try:
            from ..training.orthographic_renderer import (
                OrthographicRenderer,
                RenderConfig,
                ViewType,
            )

            # Parse views
            view_list = []
            for v in views.split(","):
                v = v.strip().lower()
                try:
                    view_list.append(ViewType(v))
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid view type: {v}. Valid: front,top,right,left,back,bottom,isometric"
                    )

            # Configure renderer
            config = RenderConfig(
                resolution=resolution,
                show_hidden_lines=show_hidden_lines,
            )
            renderer = OrthographicRenderer(config)

            # Load and render
            mesh = renderer.load_model(model_path)
            rendered = renderer.render_standard_views(mesh, view_list)

            # Save views
            output_dir = Path("data/output") / "orthographic_views"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_files = {}
            for view_type, view in rendered.items():
                filename = f"{model_path.stem}_{view_type.value}.png"
                output_path = output_dir / filename
                renderer.save_view(view, output_path)
                output_files[view_type.value] = {
                    "file": f"orthographic_views/{filename}",
                    "download_url": f"/files/output/orthographic_views/{filename}",
                    "scale": view.scale,
                    "visible_edges": view.metadata.get("visible_edges", 0),
                    "hidden_edges": view.metadata.get("hidden_edges", 0),
                }

            return {
                "success": True,
                "model": model_file,
                "views": output_files,
                "config": {
                    "resolution": resolution,
                    "show_hidden_lines": show_hidden_lines,
                }
            }

        except Exception as e:
            import traceback
            raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")

    @app.post("/training/generate-pair")
    async def generate_cad2program_training_pair(
        model_file: str,
        resolution: int = 1024,
        show_hidden_lines: bool = True,
        views: str = "front,top,right",
        copy_ground_truth: bool = True,
    ):
        """
        Generate a complete CAD2Program training pair.

        Creates:
        - Orthographic view images (PNG)
        - Copy of ground truth 3D model
        - Metadata JSON with all relevant information

        - **model_file**: Name of 3D model in input folder
        - **resolution**: Image resolution in pixels
        - **show_hidden_lines**: Include dashed hidden lines
        - **views**: Comma-separated list of views
        - **copy_ground_truth**: Copy 3D model to output (default True)
        """
        model_path = Path("data/input") / model_file

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_file}")

        try:
            from ..training.orthographic_renderer import (
                OrthographicRenderer,
                RenderConfig,
                ViewType,
            )

            # Parse views
            view_list = []
            for v in views.split(","):
                v = v.strip().lower()
                try:
                    view_list.append(ViewType(v))
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid view type: {v}"
                    )

            config = RenderConfig(
                resolution=resolution,
                show_hidden_lines=show_hidden_lines,
            )
            renderer = OrthographicRenderer(config)

            output_dir = Path("data/output") / "cad2program_training"

            pair = renderer.generate_training_pair(
                model_path=model_path,
                output_dir=output_dir,
                views=view_list,
                copy_ground_truth=copy_ground_truth,
            )

            # Build response
            view_urls = {}
            for view_name, view_path in pair.views.items():
                relative = view_path.relative_to(Path("data/output"))
                view_urls[view_name] = {
                    "file": str(relative),
                    "download_url": f"/files/output/{relative}",
                }

            gt_relative = pair.ground_truth_path.relative_to(Path("data/output"))
            meta_relative = pair.metadata_path.relative_to(Path("data/output"))

            return {
                "success": True,
                "training_pair": {
                    "views": view_urls,
                    "ground_truth": {
                        "file": str(gt_relative),
                        "download_url": f"/files/output/{gt_relative}",
                    },
                    "metadata": {
                        "file": str(meta_relative),
                        "download_url": f"/files/output/{meta_relative}",
                    },
                },
                "model_info": pair.metadata.get("model_info", {}),
            }

        except Exception as e:
            import traceback
            raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")

    @app.post("/training/batch-render")
    async def batch_render_training_data(
        background_tasks: BackgroundTasks,
        input_folder: str = "training_input",
        resolution: int = 1024,
        show_hidden_lines: bool = True,
    ):
        """
        Batch process all 3D models in a folder to generate training data.

        - **input_folder**: Subfolder in data/input containing 3D models
        - **resolution**: Image resolution in pixels
        - **show_hidden_lines**: Include dashed hidden lines

        Returns job_id for tracking progress.
        """
        input_dir = Path("data/input") / input_folder

        if not input_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Input folder not found: {input_folder}. Create data/input/{input_folder}/"
            )

        # Find all model files
        extensions = [".obj", ".glb", ".gltf", ".stl", ".ply"]
        model_files = []
        for ext in extensions:
            model_files.extend(input_dir.glob(f"*{ext}"))
            model_files.extend(input_dir.glob(f"*{ext.upper()}"))

        if not model_files:
            raise HTTPException(
                status_code=400,
                detail=f"No 3D models found in {input_folder}. Supported: {extensions}"
            )

        # Create job
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()

        jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "progress": 0.0,
            "current_stage": "initializing",
            "created_at": now,
            "completed_at": None,
            "error": None,
            "output_files": [],
            "job_type": "batch_training",
            "input_folder": input_folder,
            "model_count": len(model_files),
            "resolution": resolution,
            "show_hidden_lines": show_hidden_lines,
            "processed": 0,
            "successful": [],
            "failed": [],
        }

        # Start background processing
        background_tasks.add_task(process_batch_training_job, job_id, model_files)

        return {
            "job_id": job_id,
            "status": "pending",
            "model_count": len(model_files),
            "message": f"Batch training data generation started for {len(model_files)} models",
        }

    # =========================================================================
    # VLM CAD Strategy Endpoints
    # =========================================================================

    @app.get("/vlm-cad/info")
    async def vlm_cad_info():
        """
        Get VLM CAD strategy information and available models.

        Returns:
            Dict with device info, available models, and strategy status
        """
        try:
            from ..reconstruction.strategies import VLMCADStrategy

            strategy = VLMCADStrategy()

            return {
                "available": strategy.is_available(),
                "device_info": strategy.get_device_info(),
                "models": strategy.get_available_models(),
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "models": [],
            }

    @app.post("/vlm-cad/download-model")
    async def vlm_cad_download_model(
        model_id: str = "openecad-0.89b",
    ):
        """
        Download a VLM model from Hugging Face.

        Args:
            model_id: Model ID to download (openecad-0.89b, internvl2-2b, internvl2-8b)

        Returns:
            Download status and model path
        """
        try:
            from ..reconstruction.strategies import VLMCADStrategy

            strategy = VLMCADStrategy(model_id=model_id)
            result = await strategy.download_model(model_id)

            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/vlm-cad/reconstruct")
    async def vlm_cad_reconstruct(
        file: UploadFile = File(...),
        model_id: str = "openecad-0.89b",
        output_format: str = "glb",
    ):
        """
        Reconstruct 3D model from 2D CAD drawing using VLM.

        Args:
            file: 2D CAD drawing image (PNG, JPG)
            model_id: VLM model to use
            output_format: Output format (glb, obj, stl)

        Returns:
            Generated 3D model file
        """
        try:
            from ..reconstruction.strategies import VLMCADStrategy, ReconstructionInput
            import tempfile
            import shutil

            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = Path(tmp.name)

            try:
                strategy = VLMCADStrategy(model_id=model_id)

                input_data = ReconstructionInput(
                    primary_image=tmp_path,
                )

                result = await strategy.reconstruct(input_data)

                if not result.success:
                    raise HTTPException(status_code=500, detail=result.error)

                # Return the model data
                media_types = {
                    "glb": "model/gltf-binary",
                    "obj": "text/plain",
                    "stl": "application/octet-stream",
                }

                return Response(
                    content=result.model_data,
                    media_type=media_types.get(output_format, "application/octet-stream"),
                    headers={
                        "Content-Disposition": f'attachment; filename="reconstructed.{output_format}"',
                        "X-Strategy-Used": result.strategy_used or "unknown",
                        "X-Processing-Time": str(result.processing_time or 0),
                    },
                )
            finally:
                # Clean up temp file
                tmp_path.unlink(missing_ok=True)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/vlm-cad/generate-code")
    async def vlm_cad_generate_code(
        file: UploadFile = File(...),
        model_id: str = "openecad-0.89b",
        prompt: Optional[str] = None,
    ):
        """
        Generate CAD code from 2D drawing without converting to mesh.

        This endpoint returns the raw generated CAD code for inspection
        or custom processing.

        Args:
            file: 2D CAD drawing image
            model_id: VLM model to use
            prompt: Optional custom prompt

        Returns:
            Generated CAD code and metadata
        """
        try:
            from ..reconstruction.vlm_cad_strategy import VLMCADInference, ModelManager
            import tempfile
            import shutil

            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = Path(tmp.name)

            try:
                model_manager = ModelManager()
                inference = VLMCADInference(model_manager)
                inference.load_model(model_id)

                result = inference.generate_cad_code(
                    image_path=tmp_path,
                    prompt=prompt,
                )

                return {
                    "success": True,
                    "code": result["code"],
                    "model_id": result["model_id"],
                    "model_name": result["model_name"],
                }
            finally:
                tmp_path.unlink(missing_ok=True)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


async def process_batch_training_job(job_id: str, model_files: list):
    """Background task to process batch training data generation"""
    job = jobs[job_id]

    try:
        from ..training.orthographic_renderer import OrthographicRenderer, RenderConfig

        job["status"] = JobStatus.PROCESSING
        job["current_stage"] = "processing"

        config = RenderConfig(
            resolution=job["resolution"],
            show_hidden_lines=job["show_hidden_lines"],
        )
        renderer = OrthographicRenderer(config)

        output_dir = Path("data/output") / "cad2program_training"

        total = len(model_files)
        for i, model_path in enumerate(model_files):
            job["current_stage"] = f"processing {model_path.name}"
            job["progress"] = i / total

            try:
                pair = renderer.generate_training_pair(
                    model_path=model_path,
                    output_dir=output_dir,
                )
                job["successful"].append({
                    "model": model_path.name,
                    "metadata": str(pair.metadata_path),
                })
                job["output_files"].append(str(pair.metadata_path))
            except Exception as e:
                job["failed"].append({
                    "model": model_path.name,
                    "error": str(e),
                })

            job["processed"] = i + 1

        job["status"] = JobStatus.COMPLETED
        job["progress"] = 1.0
        job["current_stage"] = "complete"
        job["completed_at"] = datetime.utcnow()

    except Exception as e:
        import traceback
        job["status"] = JobStatus.FAILED
        job["error"] = f"{str(e)}\n{traceback.format_exc()}"
        job["current_stage"] = "failed"


async def process_validation_job(job_id: str):
    """Background task to process a validation job"""
    job = jobs[job_id]

    try:
        job["status"] = JobStatus.PROCESSING
        job["current_stage"] = "loading_ground_truth"
        job["progress"] = 0.1

        from ..validation.pipeline import ValidationPipeline

        pipeline = ValidationPipeline(
            strategy=job["strategy"],
            wall_height=job["wall_height"],
        )

        gt_path = Path("data/input") / job["ground_truth_file"]

        job["current_stage"] = "running_validation"
        job["progress"] = 0.3

        result = await pipeline.validate(
            gt_path,
            floor_height=job["floor_height"],
        )

        job["progress"] = 0.9

        if result.success:
            job["status"] = JobStatus.COMPLETED
            job["progress"] = 1.0
            job["current_stage"] = "complete"
            job["completed_at"] = datetime.utcnow()
            job["validation_result"] = result.to_dict()
            job["output_files"] = [
                result.generated_2d_file,
                result.reconstructed_3d_file,
            ]
        else:
            job["status"] = JobStatus.FAILED
            job["error"] = result.error
            job["current_stage"] = "failed"

    except Exception as e:
        import traceback
        job["status"] = JobStatus.FAILED
        job["error"] = f"{str(e)}\n{traceback.format_exc()}"
        job["current_stage"] = "failed"


def _check_strategy_available(strategy_id: str) -> bool:
    """Check if a strategy is available"""
    if strategy_id == "external_api":
        # Check for API keys in environment
        import os
        return any([
            os.getenv("KAEDIM_API_KEY"),
            os.getenv("MESHY_API_KEY"),
            os.getenv("REPLICATE_API_TOKEN"),
        ])
    elif strategy_id == "basic_extrusion":
        try:
            import trimesh
            import numpy
            return True
        except ImportError:
            return False
    elif strategy_id == "multi_view_dnn":
        try:
            import torch
            return True
        except ImportError:
            return False
    return False


def _format_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _get_file_type(extension: str) -> str:
    """Get file type category from extension"""
    ext = extension.lower().lstrip('.')
    type_map = {
        'png': 'image',
        'jpg': 'image',
        'jpeg': 'image',
        'svg': 'image',
        'tiff': 'image',
        'tif': 'image',
        'pdf': 'document',
        'dxf': 'cad',
        'dwg': 'cad',
        'obj': '3d-model',
        'glb': '3d-model',
        'gltf': '3d-model',
        'stl': '3d-model',
        'usdz': '3d-model',
        'ifc': '3d-model',
    }
    return type_map.get(ext, 'unknown')


async def process_job(job_id: str):
    """
    Background task to process a reconstruction job.
    This orchestrates the full pipeline.
    """
    job = jobs[job_id]

    try:
        job["status"] = JobStatus.PROCESSING

        # Stage 1: Load and preprocess
        job["current_stage"] = "ingestion"
        job["progress"] = 0.1
        await asyncio.sleep(0.5)  # Placeholder for actual processing

        # Stage 2: Vectorize
        job["current_stage"] = "vectorization"
        job["progress"] = 0.3
        await asyncio.sleep(0.5)

        # Stage 3: Detect elements
        job["current_stage"] = "recognition"
        job["progress"] = 0.5
        await asyncio.sleep(0.5)

        # Stage 4: Build 3D model
        job["current_stage"] = "reconstruction"
        job["progress"] = 0.7
        await asyncio.sleep(0.5)

        # Stage 5: Export
        job["current_stage"] = "export"
        job["progress"] = 0.9

        # Create output directory
        output_dir = Path("data/output") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Actual export - for now create placeholder files
        output_files = []
        for fmt in job["export_formats"]:
            output_file = f"model.{fmt}"
            output_path = output_dir / output_file
            output_path.write_text(f"Placeholder {fmt} file")
            output_files.append(output_file)

        job["output_files"] = output_files
        job["status"] = JobStatus.COMPLETED
        job["progress"] = 1.0
        job["current_stage"] = "complete"
        job["completed_at"] = datetime.utcnow()

    except Exception as e:
        job["status"] = JobStatus.FAILED
        job["error"] = str(e)


async def process_reconstruction_job(job_id: str):
    """
    Background task to process a reconstruction job using selected strategy.
    """
    job = jobs[job_id]

    try:
        job["status"] = JobStatus.PROCESSING
        job["current_stage"] = "initializing"
        job["progress"] = 0.1

        # Import reconstruction modules
        from ..reconstruction.strategies import (
            ReconstructionInput,
            ReconstructionManager,
            ExternalAPIStrategy,
            BasicExtrusionStrategy,
            MultiViewDNNStrategy,
            StrategyType,
        )

        # Set up reconstruction manager
        manager = ReconstructionManager()

        # Register strategies based on job config
        strategy_type = job.get("strategy", "auto")

        if strategy_type in ["external_api", "auto"]:
            service = job.get("service", "kaedim")
            api_key = os.getenv(f"{service.upper()}_API_KEY")
            if api_key:
                manager.register_strategy(ExternalAPIStrategy(
                    service=service,
                    api_key=api_key,
                ))

        if strategy_type in ["basic_extrusion", "auto"]:
            manager.register_strategy(BasicExtrusionStrategy())

        if strategy_type in ["multi_view_dnn", "auto"]:
            model_type = job.get("model_type", "depth_estimation")
            manager.register_strategy(MultiViewDNNStrategy(
                model_type=model_type,
            ))

        job["current_stage"] = "preparing_input"
        job["progress"] = 0.2

        # Prepare input
        input_path = Path(job["input_path"])
        additional_views = [Path(v) for v in job.get("additional_views", [])]

        input_data = ReconstructionInput(
            primary_image=input_path,
            additional_views=additional_views,
            wall_height=job.get("wall_height", 2.8),
        )

        job["current_stage"] = "reconstruction"
        job["progress"] = 0.4

        # Determine preferred strategy
        preferred = None
        if strategy_type != "auto":
            preferred = StrategyType(strategy_type)

        # Run reconstruction
        result = await manager.reconstruct(
            input_data,
            preferred_strategy=preferred,
            fallback=(strategy_type == "auto"),
        )

        job["progress"] = 0.8

        if result.success:
            job["current_stage"] = "saving_output"

            # Save output
            output_dir = Path("data/output") / job_id
            output_dir.mkdir(parents=True, exist_ok=True)

            output_filename = f"model.{result.format}"
            output_path = output_dir / output_filename

            with open(output_path, "wb") as f:
                f.write(result.model_data)

            job["output_files"] = [output_filename]
            job["status"] = JobStatus.COMPLETED
            job["progress"] = 1.0
            job["current_stage"] = "complete"
            job["completed_at"] = datetime.utcnow()
            job["strategy_used"] = result.strategy_used
            job["metadata"] = result.metadata

        else:
            job["status"] = JobStatus.FAILED
            job["error"] = result.error
            job["current_stage"] = "failed"

    except Exception as e:
        import traceback
        job["status"] = JobStatus.FAILED
        job["error"] = f"{str(e)}\n{traceback.format_exc()}"
        job["current_stage"] = "failed"


# Create app instance for uvicorn
app = create_app()
