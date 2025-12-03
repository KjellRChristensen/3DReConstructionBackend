# Training Time Estimates - DeepCAD Dataset

## Full Dataset Training Time

### Dataset Size: 161,240 Training Models

#### Phase 1: Image Rendering (One-time Cost)
Rendering orthographic views from CAD JSON files:

| Views per Model | Total Images | Est. Time/Model | Total Time | Storage |
|----------------|--------------|-----------------|------------|---------|
| 1 view (front) | 161,240 | 2-5 sec | **90-224 hours** | ~50 GB |
| 3 views (front/top/right) | 483,720 | 6-15 sec | **270-672 hours** | ~150 GB |
| 6 views (all orthographic) | 967,440 | 12-30 sec | **540-1,344 hours** | ~300 GB |

**Realistic estimate with parallelization (8 cores):**
- 1 view: **11-28 hours**
- 3 views: **34-84 hours**
- 6 views: **68-168 hours**

#### Phase 2: Model Training (Per Run)
Fine-tuning OpenECAD VLM with LoRA on rendered images:

**Hardware Assumptions:**
- GPU: Apple Silicon (MPS) or NVIDIA with 8-16GB VRAM
- Batch size: 2 per device
- Gradient accumulation: 4 steps
- Effective batch size: 8

**Training Time Calculations:**

| Dataset Size | Steps/Epoch | Time/Step | Epoch Time | 3 Epochs | 5 Epochs |
|--------------|-------------|-----------|------------|----------|----------|
| **161K (Full)** | 20,155 | 1.5s | 8.4 hours | **25 hours** | **42 hours** |
| **100K** | 12,500 | 1.5s | 5.2 hours | **16 hours** | **26 hours** |
| **10K** | 1,250 | 1.5s | 31 min | **1.5 hours** | **2.5 hours** |
| **1K (Test)** | 125 | 1.5s | 3 min | **9 min** | **15 min** |

**Total Pipeline Time (Rendering + 3 Epochs Training):**
- **Full (161K)**: 36-112 hours (1.5-4.5 days)
- **10K subset**: 2-6 hours
- **1K subset**: 15-30 minutes

---

## Recommended Training Strategy

### ⚡ Quick Convergence Testing (Recommended)

Start with progressively larger subsets to test convergence:

#### Phase 1: Smoke Test (1K samples)
```bash
Purpose: Verify pipeline works end-to-end
Time: 15-30 minutes
Cost: Minimal
Result: Confirm setup is correct
```

#### Phase 2: Small Scale (10K samples)
```bash
Purpose: Test convergence on diverse data
Time: 2-6 hours
Cost: ~3 GB storage
Result: Initial quality assessment
```

#### Phase 3: Medium Scale (50K samples)
```bash
Purpose: Evaluate scaling behavior
Time: 8-20 hours
Cost: ~15 GB storage
Result: Near-production quality
```

#### Phase 4: Full Scale (161K samples) - Only if needed
```bash
Purpose: Maximum performance
Time: 36-112 hours (1.5-4.5 days)
Cost: ~50-150 GB storage
Result: Best possible model
```

---

## Subset Recommendations

### Option 1: Small Test (1K models) ⚡ FASTEST
**Best for:** Pipeline testing, quick iteration
```
Training samples: 1,000
Validation samples: 100
Rendering time: 20-45 minutes (3 views, 8 cores)
Training time (3 epochs): 9 minutes
Total time: ~1 hour
Storage: ~300 MB
```

### Option 2: Development (10K models) 🎯 RECOMMENDED
**Best for:** Model development, convergence testing
```
Training samples: 10,000
Validation samples: 1,000
Rendering time: 3-7 hours (3 views, 8 cores)
Training time (3 epochs): 1.5 hours
Total time: ~5-9 hours
Storage: ~3 GB
```

### Option 3: Production Lite (50K models)
**Best for:** High-quality results without full dataset
```
Training samples: 50,000
Validation samples: 5,000
Rendering time: 17-42 hours (3 views, 8 cores)
Training time (3 epochs): 8 hours
Total time: ~25-50 hours (1-2 days)
Storage: ~15 GB
```

### Option 4: Full Dataset (161K models)
**Best for:** Maximum performance, research
```
Training samples: 161,240
Validation samples: 8,946
Rendering time: 34-84 hours (3 views, 8 cores)
Training time (3 epochs): 25 hours
Total time: ~60-110 hours (2.5-4.5 days)
Storage: ~50-150 GB
```

---

## Hardware-Specific Estimates

### Apple Silicon (M1/M2/M3)
- **Training Speed**: 1.0-1.5 sec/step (MPS)
- **VRAM**: 8-16 GB unified memory
- **Rendering**: 8-10 CPU cores
- **Recommended**: 10K-50K subsets

### NVIDIA GPU (RTX 3090 / 4090)
- **Training Speed**: 0.8-1.2 sec/step (CUDA)
- **VRAM**: 24 GB
- **Rendering**: 16-24 CPU cores
- **Recommended**: 50K-161K (full dataset)

### Cloud (AWS/GCP)
- **Instance**: p3.2xlarge (V100) or g5.xlarge (A10G)
- **Training Speed**: 0.6-1.0 sec/step
- **Cost**: $3-8/hour
- **Recommended**: Use spot instances for rendering, on-demand for training

---

## Cost-Benefit Analysis

### Convergence vs Dataset Size

Based on typical VLM fine-tuning patterns:

| Dataset Size | Expected Quality | Diminishing Returns |
|--------------|------------------|---------------------|
| 1K | Baseline | - |
| 10K | +40% improvement | High value |
| 50K | +25% improvement | Good value |
| 100K | +10% improvement | Moderate value |
| 161K (Full) | +5% improvement | Low value |

**Recommendation:** Start with **10K subset** for best time/quality tradeoff.

---

## Practical Implementation

### Extract 10K Subset from DeepCAD

```bash
# Create subset directory
mkdir -p data/training/deepcad_10k

# Copy first 10K from training split
python scripts/create_subset.py \
    --input data/input/deepcad/data/train_val_test_split.json \
    --output data/training/deepcad_10k \
    --train-size 10000 \
    --val-size 1000
```

### Render Orthographic Views

```bash
# Option 1: Use API (recommended)
POST /training/datasets/create \
    name=deepcad_10k \
    source_folder=deepcad/data/cad_json \
    resolution=512 \
    views=front,top,right

# Option 2: Use script
python scripts/render_cad_dataset.py \
    --input data/input/deepcad/data/cad_json \
    --output data/training/deepcad_10k \
    --subset 10000 \
    --views front,top,right \
    --resolution 512
```

### Start Training

```bash
# Quick test (1 epoch)
POST /training/finetune/start \
    dataset_name=deepcad_10k \
    base_model=Yuan-Che/OpenECADv2-SigLIP-0.89B \
    epochs=1 \
    batch_size=2

# Full training (3 epochs)
POST /training/finetune/start \
    dataset_name=deepcad_10k \
    epochs=3 \
    batch_size=2 \
    learning_rate=0.0001 \
    lora_rank=128
```

---

## Summary

### Time Investment Ladder

```
1K subset:   1 hour    → Quick validation ⚡
10K subset:  8 hours   → Recommended starting point 🎯
50K subset:  2 days    → High quality
161K full:   4 days    → Maximum quality
```

### Recommended Approach

1. **Day 1**: Test with 1K subset (1 hour)
2. **Day 2**: Train on 10K subset (8 hours)
3. **Evaluate**: Check if quality is sufficient
4. **Optional**: Scale to 50K or full dataset if needed

**Most users will be satisfied with 10K-50K subsets.**

---

**Last Updated:** December 3, 2025
**Status:** Ready for subset extraction and training
