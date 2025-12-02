# 3D Reconstruction Strategies

Research notes and implementation strategies for converting 2D floor plans/blueprints to 3D models.

---

## Overview

Three main approaches implemented:

| Strategy | Type | Speed | Accuracy | Requirements |
|----------|------|-------|----------|--------------|
| **A: External API** | Cloud DNN | 1-5 min | 85-95% | API key |
| **B: Basic Extrusion** | Local geometry | Seconds | 60-80% | trimesh, opencv |
| **C: Multi-View DNN** | Local ML | 30s-2min | 80-95% | PyTorch + MPS |

---

## Strategy A: External DNN APIs

Cloud-based AI services for image-to-3D conversion.

### Services

| Service | URL | Strengths |
|---------|-----|-----------|
| **Kaedim** | api.kaedim3d.com | Sketch/drawing to 3D, good for floor plans |
| **Meshy** | api.meshy.ai | Fast image-to-3D, texture generation |
| **Replicate** | api.replicate.com | Multiple models (TripoSR, etc.) |
| **Tripo3D** | api.tripo3d.ai | Single image reconstruction |

### Workflow
1. Upload image to API
2. Poll for completion (async job)
3. Download GLB/OBJ result

### Environment Variables
```bash
KAEDIM_API_KEY=xxx
MESHY_API_KEY=xxx
REPLICATE_API_TOKEN=xxx
```

---

## Strategy B: Basic Extrusion (Built-in)

Local geometric reconstruction using wall detection and extrusion.

### Process
1. **Edge Detection**: OpenCV threshold + contour detection
2. **Wall Identification**: Filter contours by area, approximate to polygons
3. **Extrusion**: Shapely polygon → trimesh 3D extrusion
4. **Export**: GLB/OBJ via trimesh

### Dependencies
```bash
pip install trimesh numpy opencv-python shapely pillow
```

### Parameters
- `wall_height`: Default 2.8m
- `floor_thickness`: Default 0.3m
- `scale`: Meters per pixel (default 0.01 = 1cm/pixel)

### Limitations
- Works best with clean, high-contrast floor plans
- No furniture/fixture recognition
- Single-floor only (no multi-story)

---

## Strategy C: Multi-View DNN Reconstruction

Deep learning approaches using PyTorch with MPS (Apple Silicon GPU).

### Models

#### 1. Depth Estimation (MiDaS)
- **Model**: `intel-isl/MiDaS` (MiDaS_small)
- **Input**: Single RGB image
- **Output**: Depth map → 2.5D relief mesh
- **Best for**: Quick single-view reconstruction

```python
import torch
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("mps")  # Apple Silicon
model.to(device)
```

#### 2. GaussianCAD (Multi-View)
- **Concept**: 3D Gaussian Splatting from multiple calibrated views
- **Input**: 3+ views (floor plan, elevations, sections)
- **Output**: Full 3D mesh with accurate geometry
- **Status**: Framework implemented, needs full Gaussian splatting optimization

**Process**:
1. Camera pose estimation (SfM or manual calibration)
2. Initialize 3D Gaussians
3. Optimize Gaussian parameters (differentiable rendering)
4. Extract mesh from Gaussians

#### 3. CAD2Program (Vision-Language)
- **Concept**: VLM generates parametric CAD commands
- **Status**: Placeholder - requires fine-tuned model
- **Research**: Similar to OpenAI's approach for code generation

### MPS (Metal Performance Shaders) Support

Apple Silicon GPU acceleration:

```python
import torch

# Device selection priority: MPS > CUDA > CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

### Dependencies
```bash
pip install torch torchvision
# PyTorch 2.9.1 includes MPS support for Apple Silicon
```

---

## Input Formats Supported

| Format | Type | Notes |
|--------|------|-------|
| PNG/JPG | Raster image | Most common |
| PDF | Document | Multi-page support planned |
| SVG | Vector | Better for clean CAD exports |
| DXF/DWG | CAD | Requires ezdxf library |
| TIFF | Raster | High-res scans |

---

## Output Formats Supported

| Format | Use Case |
|--------|----------|
| **GLB** | Web viewers, Three.js, general 3D |
| **OBJ** | Legacy compatibility, Blender |
| **USDZ** | iOS AR Quick Look |
| **STL** | 3D printing |
| **IFC** | BIM software (Revit, ArchiCAD) |
| **GLTF** | Web, with separate textures |

---

## Research References

### Papers & Projects
- **GaussianCAD**: 3D Gaussian splatting for CAD reconstruction
- **CAD2Program**: Vision-language model for parametric CAD
- **MiDaS**: Monocular depth estimation (Intel ISL)
- **Depth Anything**: Alternative depth model (TikTok)
- **TripoSR**: Stability AI's single-image 3D

### Tools
- **trimesh**: Python 3D mesh library
- **Open3D**: Point cloud and mesh processing
- **PyTorch3D**: Differentiable 3D rendering
- **Kaolin**: NVIDIA's 3D deep learning library

---

## API Endpoints

```bash
# Check GPU status
curl http://localhost:7001/system/gpu

# List strategies
curl http://localhost:7001/strategies

# Quick preview (basic extrusion)
curl -X POST "http://localhost:7001/reconstruct/preview?filename=sample_floorplan.png"

# Full reconstruction with strategy
curl -X POST "http://localhost:7001/reconstruct?filename=sample_floorplan.png&strategy=multi_view_dnn&model_type=depth_estimation"

# List input files
curl http://localhost:7001/files/input

# List output files
curl http://localhost:7001/files/output
```

---

## Future Improvements

### Short-term
- [ ] Implement full GaussianCAD pipeline
- [ ] Add Depth Anything V2 as alternative to MiDaS
- [ ] Door/window detection and cutouts
- [ ] Room labeling from OCR

### Medium-term
- [ ] Multi-floor reconstruction
- [ ] Furniture placement from symbols
- [ ] Texture generation for walls/floors
- [ ] IFC export with proper BIM metadata

### Long-term
- [ ] Fine-tune CAD2Program on floor plan dataset
- [ ] Real-time reconstruction feedback
- [ ] AR preview in iOS app
- [ ] Integration with BIM software

---

## File Structure

```
Backend/
├── src/
│   ├── api/server.py           # FastAPI endpoints
│   ├── reconstruction/
│   │   ├── strategies.py       # Strategy A, B, C implementations
│   │   └── builder.py          # Model building utilities
│   ├── ingestion/              # File loading
│   ├── vectorization/          # Image to vector
│   └── recognition/            # Element detection
├── data/
│   ├── input/                  # Source files
│   └── output/                 # Generated 3D models
└── venv/                       # Python virtual environment
```
