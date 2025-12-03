# CAD VLM Training Strategies and Approaches

This document outlines strategies for training and fine-tuning Vision-Language Models (VLMs) for 2D-to-3D CAD reconstruction, specifically targeting the OpenECAD family of models.

---

## Table of Contents

1. [Overview](#overview)
2. [Available Datasets](#available-datasets)
3. [Training Strategies](#training-strategies)
4. [Implementation Approaches](#implementation-approaches)
5. [Convergence Testing Strategy](#convergence-testing-strategy)
6. [Hardware Requirements](#hardware-requirements)
7. [Evaluation Metrics](#evaluation-metrics)

---

## Overview

### Goal
Train VLM models to understand 2D engineering drawings and generate parametric CAD construction code.

### Pipeline
```
2D Engineering Drawing (Input)
         ↓
    VLM Model (OpenECAD)
         ↓
CAD Construction Code (Output)
         ↓
    Execute Code
         ↓
    3D CAD Model
```

### Model Architecture
- **Base Models**: TinyLLaVA architecture with SigLIP/CLIP vision encoder
- **Training Method**: LoRA fine-tuning (rank 128, alpha 256)
- **Code Format**: OpenECAD command sequences (Python-like syntax)

---

## Available Datasets

> **⚠️ IMPORTANT NOTE (Updated Dec 2025):**
> The OpenECAD dataset on HuggingFace only contains metadata (conversation text + image filenames), **not actual image files**. For immediate testing, we recommend:
> 1. **DeepCAD Dataset** - Complete dataset with 178K models (requires image rendering)
> 2. **Fusion 360 Dataset** - 8.6K models with thumbnails (smaller but complete)
> 3. **Contact authors** - Request complete OpenECAD dataset at yuanzhe1999@outlook.com
> 4. **Synthetic data** - Generate small test dataset for convergence testing (see below)

### 1. OpenECAD Dataset ⚠️ LIMITED AVAILABILITY

**Status**: HuggingFace version contains metadata only (no images)

| Property | Details |
|----------|---------|
| **Source** | [HuggingFace: Yuan-Che/OpenECAD-Dataset](https://huggingface.co/datasets/Yuan-Che/OpenECAD-Dataset) |
| **Size** | 918,719 image-code pairs |
| **Download Size** | 1.73 GB parquet (metadata only) |
| **Format** | TinyLLaVA conversation format |
| **Images** | ⚠️ **Filenames only, not actual images** |
| **Code** | OpenECAD Python commands (available) |
| **Train/Val** | Single train split (needs splitting) |
| **Token Length** | 1024-3072 tokens per sequence |

**Current Limitations:**
- ⚠️ **HuggingFace version is incomplete** - only contains image filenames (e.g., "00000022.jpg"), not actual image files
- ⚠️ Images must be requested separately from authors (yuanzhe1999@outlook.com)
- ⚠️ Not directly usable for VLM training without image files

**Advantages (when complete):**
- ✅ Already in correct format for TinyLLaVA
- ✅ Pre-tokenized and filtered
- ✅ Directly compatible with OpenECAD models
- ✅ Quick to download metadata

**Download:**
```python
from datasets import load_dataset
dataset = load_dataset("Yuan-Che/OpenECAD-Dataset")
```

**Data Structure:**
```json
{
  "image": "<PIL.Image>",
  "conversations": [
    {"from": "human", "value": "<image>\nGenerate CAD code for this design."},
    {"from": "gpt", "value": "plane_0 = add_sketchplane([0,0,0], [0,0,1], [1,0,0])\nline_0 = add_line(plane_0, [0,0], [10,0])\n..."}
  ]
}
```

### 2. DeepCAD Dataset

**Best for**: Research, understanding raw data format, custom preprocessing

| Property | Details |
|----------|---------|
| **Source** | [GitHub: ChrisWu1997/DeepCAD](https://github.com/ChrisWu1997/DeepCAD) |
| **Size** | 178,238 CAD models |
| **Download** | http://www.cs.columbia.edu/cg/deepcad/data.tar (backup: [Google Drive](https://drive.google.com/drive/folders/1mSJBZjKC-Z5I7pLPTgb4b5ZP-Y6itvGG)) |
| **Format** | JSON construction sequences from Onshape |
| **Content** | `cad_json/` (raw), `cad_vec/` (vectorized) |
| **Train/Val/Test** | Provided split in `train_val_test_split.json` |

**Advantages:**
- ✅ Original source dataset
- ✅ Large number of models
- ✅ Includes train/val/test splits
- ✅ Well-documented

**Disadvantages:**
- ❌ Requires rendering 2D views from 3D models
- ❌ Needs conversion to image-code pairs
- ❌ More preprocessing required

**Preprocessing Steps:**
```bash
# Extract dataset
tar -xf data.tar

# Convert to vectorized format
cd dataset
python json2vec.py

# Generate point clouds for evaluation
python json2pc.py --only_test
```

### 3. Text2CAD Dataset

**Best for**: Text-to-CAD research, multi-level instruction following

| Property | Details |
|----------|---------|
| **Source** | [HuggingFace: SadilKhan/Text2CAD](https://huggingface.co/datasets/SadilKhan/Text2CAD) |
| **Size** | 1.3 GB (v1.1), 605 GB with images |
| **Format** | CSV/JSON with text annotations |
| **Content** | Text descriptions (4 levels) + CAD sequences + RGB views |
| **License** | CC BY-NC-SA 4.0 (non-commercial) |
| **Base** | Derived from DeepCAD |

**Text Annotation Levels:**
- **Level 0**: Abstract ("a mechanical part")
- **Level 1**: Beginner ("rectangular base with circular hole")
- **Level 2**: Intermediate (dimensions and operations specified)
- **Level 3**: Expert (complete parametric description)

**Advantages:**
- ✅ Multi-level text annotations
- ✅ Useful for text-to-CAD generation
- ✅ RGB multi-view images included

**Disadvantages:**
- ❌ Requires conversion for image-to-CAD task
- ❌ Large download size with images
- ❌ Non-commercial license

### 4. Fusion 360 Gallery Dataset

**Best for**: Real-world human designs, sketch-extrude operations

| Property | Details |
|----------|---------|
| **Source** | [GitHub: AutodeskAILab/Fusion360GalleryDataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset) |
| **Models** | 8,625 design sequences |
| **Size** | 2.0 GB (reconstruction) |
| **Download** | [r1.0.1.zip](https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/reconstruction/r1.0.1/r1.0.1.zip) |
| **Format** | JSON sequences + B-Rep (.smt, .step) + thumbnails |
| **Operations** | Sketch + Extrude only (simpler than DeepCAD) |

**Additional Subsets:**
- **Segmentation**: [s2.0.1 (3.1 GB)](https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1.zip) - 35,680 parts
- **Assembly**: [j1.0.0 (2.8 GB)](https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/assembly/j1.0.0/j1.0.0.7z) - 32,148 joints

**Advantages:**
- ✅ Real-world human designs
- ✅ Simpler operation set (easier to learn)
- ✅ Multiple representations (B-Rep, mesh, JSON)
- ✅ Includes thumbnails

**Disadvantages:**
- ❌ Smaller dataset (8.6K vs 178K)
- ❌ Requires rendering orthographic views
- ❌ Needs conversion to OpenECAD format

### 5. ABC Dataset

**Best for**: Geometric deep learning, NOT construction sequences

| Property | Details |
|----------|---------|
| **Source** | [ABC Dataset Website](https://deep-geometry.github.io/abc-dataset/) |
| **Size** | 1 million CAD models |
| **Format** | OBJ meshes with differential properties |
| **Content** | Mechanical parts from Onshape |

**⚠️ Not Recommended**: While massive, this dataset lacks parametric construction sequences. Only useful for geometric learning tasks, not generative CAD modeling.

---

## Training Strategies

### Strategy 1: Quick Start with OpenECAD Dataset ⭐

**Objective**: Verify infrastructure and test convergence quickly

**Dataset**: OpenECAD subset (10K-50K samples)

**Steps:**
1. Download OpenECAD dataset from HuggingFace
2. Split into train (90%) and val (10%)
3. Fine-tune with LoRA (rank 128)
4. Monitor convergence on validation set

**Expected Timeline**: 1-2 hours on Apple Silicon M2/M3

**Code:**
```python
from datasets import load_dataset

# Load subset
dataset = load_dataset("Yuan-Che/OpenECAD-Dataset", split="train[:10000]")

# Use via API
POST /training/datasets/create
{
  "name": "openecad_10k_test",
  "source_folder": "openecad_subset",
  "resolution": 512,
  "views": ["front", "isometric"],
  "train_split": 0.9
}

# Start training
POST /training/finetune/start
{
  "dataset_name": "openecad_10k_test",
  "base_model": "Yuan-Che/OpenECADv2-SigLIP-0.89B",
  "epochs": 3,
  "batch_size": 2,
  "learning_rate": 0.0001,
  "lora_rank": 128
}
```

**Success Criteria:**
- ✅ Training loss decreases consistently
- ✅ Validation loss converges
- ✅ Generated code is syntactically valid
- ✅ Simple shapes can be reconstructed

### Strategy 2: Full-Scale Training

**Objective**: Train production-ready model

**Dataset**: Full OpenECAD dataset (919K samples)

**Steps:**
1. Download complete OpenECAD dataset
2. Perform 90/10 train/val split
3. Train for 3 epochs with LoRA
4. Evaluate on held-out test set
5. Fine-tune on domain-specific data if available

**Expected Timeline**: 24-48 hours on Apple Silicon

**Configuration:**
```yaml
# config/finetune_config.yaml
base_model: "Yuan-Che/OpenECADv2-SigLIP-0.89B"
train_data: "data/training/openecad_full/train/train.json"
val_data: "data/training/openecad_full/val/val.json"

use_lora: true
lora:
  r: 128
  alpha: 256
  dropout: 0.05

training:
  num_epochs: 3
  per_device_batch_size: 2
  gradient_accumulation_steps: 2
  learning_rate: 0.0001
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
```

**Success Criteria:**
- ✅ Validation loss < 0.5
- ✅ Code execution success rate > 80%
- ✅ Chamfer distance < 0.1 on test set
- ✅ Generalizes to unseen designs

### Strategy 3: Domain Adaptation

**Objective**: Adapt model to client-specific CAD style

**Dataset**: Client CAD models (when available)

**Approach**: Two-stage training
1. **Stage 1**: Pre-train on OpenECAD (919K samples)
2. **Stage 2**: Fine-tune on client data (500-5000 samples)

**Steps:**
1. Complete Strategy 2 (full-scale training)
2. Generate training pairs from client CAD models
3. Annotate with client-specific CAD code
4. Fine-tune pre-trained checkpoint with lower learning rate
5. Validate on client test set

**Configuration for Stage 2:**
```yaml
base_model: "checkpoints/openecad_full_checkpoint"  # From Stage 1
train_data: "data/training/client_parts/train/train.json"
val_data: "data/training/client_parts/val/val.json"

use_lora: true
lora:
  r: 64  # Lower rank for fine-tuning
  alpha: 128

training:
  num_epochs: 5  # More epochs for smaller dataset
  learning_rate: 0.00005  # Lower LR for fine-tuning
```

**Success Criteria:**
- ✅ Matches client CAD conventions
- ✅ Generates client-style parametric features
- ✅ Maintains general capabilities from Stage 1

### Strategy 4: Multi-Dataset Training

**Objective**: Maximize diversity and robustness

**Datasets**:
- OpenECAD (919K)
- Fusion 360 Gallery (8.6K)
- DeepCAD subset (50K)

**Steps:**
1. Preprocess all datasets to common format
2. Mix datasets with sampling weights
3. Train with curriculum learning (simple → complex)
4. Evaluate on each dataset separately

**Dataset Mixing:**
```python
# Sampling weights
openecad_weight = 0.7   # 70% OpenECAD
fusion360_weight = 0.2  # 20% Fusion 360
deepcad_weight = 0.1    # 10% DeepCAD

# Total training samples per epoch: ~650K
```

**Success Criteria:**
- ✅ Good performance across all test sets
- ✅ Handles diverse CAD styles
- ✅ Robust to different input types

---

## Implementation Approaches

### Approach 1: Using HuggingFace Datasets (Recommended)

**Best for**: Quick prototyping, standard workflows

```python
from datasets import load_dataset
from pathlib import Path
import json

def prepare_openecad_dataset(output_dir="data/training/openecad"):
    """Download and prepare OpenECAD dataset"""

    # Load from HuggingFace
    dataset = load_dataset("Yuan-Che/OpenECAD-Dataset")

    # Create directories
    output_path = Path(output_dir)
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "train").mkdir(exist_ok=True)
    (output_path / "val").mkdir(exist_ok=True)

    # Split 90/10
    split_idx = int(len(dataset["train"]) * 0.9)
    train_data = dataset["train"].select(range(split_idx))
    val_data = dataset["train"].select(range(split_idx, len(dataset["train"])))

    # Save images and create JSON
    train_samples = []
    for i, sample in enumerate(train_data):
        img_path = output_path / "images" / f"train_{i:06d}.jpg"
        sample["image"].save(img_path)
        train_samples.append({
            "id": f"train_{i:06d}",
            "image": f"train_{i:06d}.jpg",
            "conversations": sample["conversations"]
        })

    val_samples = []
    for i, sample in enumerate(val_data):
        img_path = output_path / "images" / f"val_{i:06d}.jpg"
        sample["image"].save(img_path)
        val_samples.append({
            "id": f"val_{i:06d}",
            "image": f"val_{i:06d}.jpg",
            "conversations": sample["conversations"]
        })

    # Save JSON files
    with open(output_path / "train" / "train.json", 'w') as f:
        json.dump(train_samples, f, indent=2)
    with open(output_path / "val" / "val.json", 'w') as f:
        json.dump(val_samples, f, indent=2)

    print(f"✅ Dataset prepared: {len(train_samples)} train, {len(val_samples)} val")

# Run
prepare_openecad_dataset()
```

### Approach 2: Using Training API

**Best for**: Production workflows, monitoring, job management

```python
import requests

BASE_URL = "http://localhost:7001"

def train_with_api():
    # Check fine-tuning status
    status = requests.get(f"{BASE_URL}/training/finetune/status").json()
    print(f"Training ready: {status['ready']}")
    print(f"Device: {status['device']}")

    # Create dataset (if using raw CAD files)
    dataset_response = requests.post(
        f"{BASE_URL}/training/datasets/create",
        params={
            "name": "openecad_test",
            "source_folder": "cad_models",
            "resolution": 512,
            "views": "front,isometric",
            "train_split": 0.9
        }
    ).json()

    job_id = dataset_response["job_id"]

    # Monitor dataset creation
    while True:
        job = requests.get(f"{BASE_URL}/jobs/{job_id}").json()
        print(f"Dataset creation: {job['progress']:.1%} - {job['current_stage']}")
        if job["status"] in ["completed", "failed"]:
            break
        time.sleep(5)

    # Start fine-tuning
    finetune_response = requests.post(
        f"{BASE_URL}/training/finetune/start",
        params={
            "dataset_name": "openecad_test",
            "base_model": "Yuan-Che/OpenECADv2-SigLIP-0.89B",
            "epochs": 3,
            "batch_size": 2,
            "learning_rate": 0.0001,
            "lora_rank": 128
        }
    ).json()

    finetune_job_id = finetune_response["job_id"]

    # Monitor training
    while True:
        job = requests.get(f"{BASE_URL}/jobs/{finetune_job_id}").json()
        if "metrics" in job:
            print(f"Epoch {job['metrics']['epoch']}, "
                  f"Step {job['metrics']['step']}, "
                  f"Loss: {job['metrics']['loss']:.4f}")
        if job["status"] in ["completed", "failed"]:
            break
        time.sleep(10)

    # List checkpoints
    checkpoints = requests.get(f"{BASE_URL}/training/finetune/checkpoints").json()
    print(f"Training complete! Checkpoints: {checkpoints['total']}")

train_with_api()
```

### Approach 3: Command-Line Training

**Best for**: Server/cluster training, scripting, automation

```bash
#!/bin/bash
# scripts/train_openecad.sh

# Download dataset
python3 scripts/download_openecad_dataset.py

# Train with config file
python3 -m training.finetune \
    --config config/finetune_config.yaml

# Or with command-line args
python3 -m training.finetune \
    --base-model "Yuan-Che/OpenECADv2-SigLIP-0.89B" \
    --train-data "data/training/openecad/train/train.json" \
    --image-folder "data/training/openecad/images" \
    --output-dir "checkpoints" \
    --epochs 3 \
    --batch-size 2 \
    --learning-rate 0.0001 \
    --lora-r 128
```

---

## Convergence Testing Strategy

### Phase 1: Smoke Test (1-2 hours)

**Goal**: Verify infrastructure works

**Dataset**: 1,000 samples
**Epochs**: 1
**Expected Loss**: Should decrease from ~2.0 to ~1.0

```python
# Quick test
dataset = load_dataset("Yuan-Che/OpenECAD-Dataset", split="train[:1000]")
```

**Success**: Training completes without errors, loss decreases

### Phase 2: Small-Scale Convergence (2-4 hours)

**Goal**: Verify model can learn

**Dataset**: 10,000 samples
**Epochs**: 3
**Expected Loss**: Train ~0.5, Val ~0.8

**Metrics to Track:**
- Training loss curve
- Validation loss curve
- Code syntax validity rate
- Execution success rate

**Success**: Validation loss converges, can generate valid CAD code

### Phase 3: Medium-Scale Training (12-24 hours)

**Goal**: Achieve good performance

**Dataset**: 100,000 samples
**Epochs**: 3
**Expected Loss**: Train ~0.3, Val ~0.5

**Additional Metrics:**
- Chamfer distance on reconstructed meshes
- F-score at different thresholds
- Visual inspection of generated models

**Success**: Generated models match ground truth reasonably well

### Phase 4: Full-Scale Training (24-48 hours)

**Goal**: Production-ready model

**Dataset**: 919,000 samples
**Epochs**: 3
**Expected Loss**: Train ~0.2, Val ~0.4

**Evaluation Protocol:**
1. Quantitative metrics on test set
2. Qualitative assessment of 100 random samples
3. Edge case testing (complex/simple/unusual designs)
4. Cross-dataset validation (test on Fusion 360)

**Success**:
- Chamfer distance < 0.1
- Execution success > 85%
- Generalizes to unseen designs

---

## Hardware Requirements

### Minimum (Testing Only)

| Component | Spec |
|-----------|------|
| **GPU** | Apple M1 or NVIDIA RTX 3060 (8GB VRAM) |
| **RAM** | 16 GB |
| **Storage** | 50 GB free |
| **Model** | OpenECAD 0.55B (smallest) |
| **Dataset** | 10K samples |
| **Training Time** | ~4 hours |

### Recommended (Full Training)

| Component | Spec |
|-----------|------|
| **GPU** | Apple M2/M3 Pro or NVIDIA RTX 4090 (24GB VRAM) |
| **RAM** | 32 GB |
| **Storage** | 200 GB free |
| **Model** | OpenECAD 0.89B (recommended) |
| **Dataset** | Full 919K samples |
| **Training Time** | ~24 hours |

### Optimal (Production)

| Component | Spec |
|-----------|------|
| **GPU** | Apple M3 Max or NVIDIA A100 (40GB VRAM) |
| **RAM** | 64 GB |
| **Storage** | 500 GB free |
| **Model** | OpenECAD 2.4B or 3.1B |
| **Dataset** | Multi-dataset mix |
| **Training Time** | ~48 hours |

### Multi-GPU Training

For faster training, use multiple GPUs:

```yaml
# Enable DeepSpeed
deepspeed_config: "config/deepspeed_config.json"

# Or use PyTorch DDP
# Run with: torchrun --nproc_per_node=4 -m training.finetune
```

---

## Evaluation Metrics

### 1. Loss Metrics

**Training Loss**: Should decrease consistently
- Good: < 0.3 after 3 epochs
- Excellent: < 0.2 after 3 epochs

**Validation Loss**: Should track training loss
- Good convergence: Val loss within 0.2 of train loss
- Overfitting: Val loss > train loss + 0.5

### 2. Code Quality Metrics

**Syntax Validity Rate**: Percentage of generated code that parses
- Minimum: > 90%
- Good: > 95%
- Excellent: > 98%

**Execution Success Rate**: Percentage of code that executes without errors
- Minimum: > 70%
- Good: > 85%
- Excellent: > 95%

### 3. Geometric Metrics

**Chamfer Distance**: Average point-to-surface distance
- Good: < 0.1
- Excellent: < 0.05

**Hausdorff Distance**: Maximum point-to-surface distance
- Good: < 0.3
- Excellent: < 0.15

**3D IoU**: Intersection over Union of voxelized models
- Good: > 0.7
- Excellent: > 0.85

**F-Score**: Precision-recall harmonic mean at distance threshold
- Good: > 0.8
- Excellent: > 0.9

### 4. Qualitative Metrics

**Visual Similarity**: Human assessment of generated vs ground truth
- Rate on scale 1-5
- Sample 100 random test cases

**Feature Completeness**: Are all design features captured?
- Holes, fillets, chamfers, etc.
- Boolean operations (unions, cuts)

**Parametric Validity**: Are constraints properly applied?
- Parallel lines, perpendicular edges
- Symmetry, patterns

---

## Best Practices

### Data Preparation

1. **Quality over Quantity**: 10K high-quality samples > 100K noisy samples
2. **Balanced Distribution**: Include simple and complex designs
3. **Data Augmentation**: Rotate views, vary rendering styles
4. **Validation Set**: Always hold out 10% for validation

### Training

1. **Start Small**: Test with 1K samples before full training
2. **Monitor Metrics**: Track loss, learning rate, gradient norms
3. **Use Checkpointing**: Save every 500 steps
4. **Early Stopping**: Stop if val loss increases for 3 evaluations
5. **Learning Rate**: Use cosine schedule with warmup

### Debugging

**High Training Loss**:
- Reduce learning rate (try 5e-5)
- Increase batch size (if memory allows)
- Check data quality

**High Validation Loss**:
- More data augmentation
- Increase dropout (try 0.1)
- Reduce model complexity

**Code Execution Failures**:
- Validate training data annotations
- Add code syntax checking during training
- Filter out invalid sequences

### Hyperparameter Tuning

**Learning Rate**: Most important
- Start: 1e-4 (OpenECAD default)
- Try: 5e-5, 2e-4
- Use learning rate finder

**LoRA Rank**: Balance performance vs memory
- Low (32-64): Faster, less memory, may underfit
- Medium (128): Recommended default
- High (256): Better performance, more memory

**Batch Size**: Maximize GPU utilization
- Small GPU: 1-2
- Medium GPU: 2-4
- Large GPU: 4-8
- Use gradient accumulation for larger effective batch

---

## Troubleshooting

### Common Issues

**OOM (Out of Memory)**:
```yaml
# Solution 1: Reduce batch size
per_device_batch_size: 1

# Solution 2: Enable gradient checkpointing
gradient_checkpointing: true

# Solution 3: Use smaller model
base_model: "Yuan-Che/OpenECADv2-CLIP-0.55B"
```

**Slow Training**:
```yaml
# Solution 1: Reduce workers if I/O bound
dataloader_num_workers: 2

# Solution 2: Use FP16
fp16: true

# Solution 3: Increase batch size with gradient accumulation
per_device_batch_size: 1
gradient_accumulation_steps: 4  # Effective batch = 4
```

**Poor Convergence**:
```yaml
# Solution 1: Adjust learning rate
learning_rate: 0.00005  # Lower

# Solution 2: Use warmup
warmup_ratio: 0.05  # Increase

# Solution 3: Check data quality
# Inspect samples manually, filter invalid annotations
```

---

## Next Steps

1. **Download OpenECAD Dataset** (Start here!)
   ```bash
   python3 scripts/download_openecad_dataset.py
   ```

2. **Run Smoke Test** (1K samples, 1 epoch)
   ```bash
   python3 -m training.finetune --config config/test_config.yaml
   ```

3. **Scale Up** (10K → 100K → Full)

4. **Evaluate** on test set

5. **Fine-tune** on client data when available

---

## References

- [OpenECAD Paper](https://arxiv.org/abs/2406.09913)
- [DeepCAD Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_DeepCAD_A_Deep_Generative_Network_for_Computer-Aided_Design_Models_ICCV_2021_paper.pdf)
- [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Fusion 360 Gallery Paper](https://arxiv.org/abs/2010.02392)

---

**Last Updated**: December 2, 2025
