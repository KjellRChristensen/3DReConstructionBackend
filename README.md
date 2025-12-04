# 3D Reconstruction Backend

Backend API for 3D CAD model reconstruction using Vision-Language Models (VLMs).

## Overview

This backend provides a complete pipeline for:
- Processing CAD models (STEP, IFC, OBJ, STL formats)
- Training VLMs on CAD reconstruction tasks
- Generating construction sequences from 3D models
- Orthographic view rendering
- Fine-tuning models with LoRA

## Features

- **Multi-format CAD Support**: STEP, IFC, OBJ, STL
- **Training Pipeline**: Complete dataset preparation and VLM fine-tuning
- **Orthographic Rendering**: Automated view generation for training
- **DeepCAD Integration**: Support for DeepCAD dataset (178K models)
- **REST API**: FastAPI-based endpoints for frontend integration
- **Apple Silicon Optimized**: MPS acceleration support

## Architecture

```
Backend/
├── src/                    # Core application code
│   ├── models/            # Data models and schemas
│   ├── services/          # Business logic services
│   ├── routes/            # API route handlers
│   └── training/          # VLM training modules
├── scripts/               # Standalone utility scripts
├── config/                # Configuration files
├── data/                  # Training datasets
│   ├── input/            # Raw input data
│   └── training/         # Processed training data
└── docs/                  # Documentation

```

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Development Server

```bash
# Start FastAPI server (recommended - includes startup logging)
python3 main.py server --reload

# Or specify custom host/port
python3 main.py server --host 0.0.0.0 --port 7001 --reload

# Alternative: Direct uvicorn (not recommended)
# uvicorn src.api.server:app --reload --host 0.0.0.0 --port 7001
```

The API will be available at `http://localhost:7001`

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:7001/docs`
- ReDoc: `http://localhost:7001/redoc`

## Training Pipeline

### Datasets

Three dataset sizes are available for testing and production:

| Dataset | Models | Images | Conversations | Training Time | Use Case |
|---------|--------|--------|---------------|---------------|----------|
| **1K** | 1,100 | 3,300 | 3,300 | ~2-4 hours | Quick testing |
| **10K** | 11,000 | 33,000 | 33,000 | ~6-9 hours | Pipeline validation |
| **100K** | 108,946 | 326,838 | 326,838 | ~60-90 hours | Production |

### Training Results

**Loss Convergence** (100 samples, 5 epochs):

| Dataset | Initial Loss | Final Loss | Reduction |
|---------|-------------|------------|-----------|
| 1K | 5.17 | 1.63 | 68.5% |
| 10K | 2.15 | 0.08 | 96.2% |
| 100K | 2.12 | 0.08 | 96.2% |

### Pipeline Stages

1. **Extract Dataset**
```bash
python3 scripts/extract_deepcad_subset.py --train-size 10000 --val-size 1000
```

2. **Render Orthographic Views**
```bash
python3 scripts/render_deepcad_views.py --input data/training/deepcad_10k
```

3. **Convert to Training Format**
```bash
python3 scripts/convert_to_tinyllava.py --input data/training/deepcad_10k
```

4. **Validate Dataset**
```bash
python3 scripts/test_training_pipeline.py --config config/deepcad_10k_test.yaml
```

5. **Run Training Test**
```bash
python3 scripts/minimal_training_test.py --data data/training/deepcad_10k/train.json --images data/training/deepcad_10k/images
```

## API Endpoints

### Training Management

- `POST /training/datasets/extract` - Extract dataset subset
- `POST /training/datasets/{name}/render` - Render orthographic views
- `POST /training/datasets/{name}/convert` - Convert to training format
- `POST /training/datasets/{name}/validate` - Validate dataset
- `GET /training/datasets` - List all datasets
- `POST /training/test-run` - Run training test
- `POST /training/start` - Start full VLM fine-tuning
- `GET /training/jobs/{id}/progress` - Monitor job progress

See `docs/TRAINING_API_ENDPOINTS.md` for complete API documentation.

## Configuration

Training configurations are stored in `config/`:

- `finetune_config.yaml` - Production training settings
- `deepcad_10k_test.yaml` - 10K dataset validation config
- `deepcad_100k_test.yaml` - 100K dataset validation config

## Results

Detailed test results are available:

- `DEEPCAD_10K_RESULTS.md` - 10K dataset analysis
- `DEEPCAD_100K_RESULTS.md` - 100K dataset analysis with comparisons

## Development

### Release Process

**IMPORTANT**: This project follows a strict release-based workflow:

1. **No intermediate commits**: We do NOT add/commit files between releases
2. **Release = Complete deployment**: A release must be committed, tagged, AND pushed to the remote GitHub repository
3. **Version tags**: All releases use semantic versioning (v1.0.0, v1.0.1, etc.)

### Creating a Release

```bash
# 1. Stage all changes
git add .

# 2. Create commit with release message
git commit -m "Release v1.0.3: Training pipeline with 100K dataset support

- Added 100K dataset extraction and processing
- Complete training pipeline validation
- API endpoint documentation
- Comprehensive test results and comparison reports"

# 3. Create and push tag
git tag -a v1.0.3 -m "Release v1.0.3"
git push origin main
git push origin v1.0.3
```

### Release History

- **v1.0.0** - Initial release with basic CAD processing
- **v1.0.1** - Added rendering and dataset management
- **v1.0.2** - 1K and 10K dataset support with training tests
- **v1.0.3** - 100K dataset support with complete pipeline validation
- **v1.1.4** - Critical fix for MPS/Metal crash on macOS 26 (Tahoe)

## Technical Stack

- **Framework**: FastAPI
- **ML/AI**: PyTorch, Transformers, LoRA
- **CAD Processing**: OCP (Open Cascade), ifcopenshell, trimesh
- **Rendering**: Matplotlib, PIL
- **Device Support**: CUDA, MPS (Apple Silicon - macOS 15 and earlier), CPU

## Requirements

- Python 3.11+
- 16+ GB RAM
- GPU recommended (NVIDIA with CUDA or Apple Silicon with MPS)
- 10+ GB disk space for datasets

## Performance

**Dataset Processing** (100K dataset):
- Extraction: ~2-3 minutes (~50,000 files/min)
- Rendering: ~10 minutes (~10,933 models/min)
- Conversion: ~3-4 minutes (~25,000 samples/min)
- Total: ~20 minutes for complete pipeline

**Training** (Apple M1/M2, batch size 2):
- Throughput: ~22 samples/second
- 10K dataset: ~6-9 hours
- 100K dataset: ~60-90 hours

## License

[Your License Here]

## Contact

[Your Contact Information]

## macOS 26 (Tahoe) Compatibility

⚠️ **Known Issue**: PyTorch's MPS backend crashes on macOS 26 due to Metal API changes.

The training pipeline automatically detects macOS 26+ and falls back to CPU mode.
For GPU acceleration, use:
- macOS 15 (Sequoia) or earlier with MPS
- NVIDIA GPU with CUDA

See `CHANGELOG.md` for details and workarounds.

---

**Last Updated**: December 4, 2025
**Version**: v1.1.4
