# Training Pipeline API Endpoints

Frontend API calls to run the complete DeepCAD dataset preparation and training pipeline.

---

## Current Status

### ✅ Existing Endpoints

1. **GET /training/info** - Get training capabilities info
2. **POST /training/render-views** - Render orthographic views (single model)
3. **POST /training/batch-render** - Batch render multiple models
4. **POST /training/generate-pair** - Generate training pair
5. **POST /validation/training-pair** - Validate training pair

### ⏳ Needed Endpoints (To Be Implemented)

The following endpoints are needed to run the complete pipeline from the frontend:

---

## 1. Dataset Management

### **POST /training/datasets/extract**
Extract a subset from DeepCAD dataset.

**Request**:
```json
{
  "source": "deepcad",
  "train_size": 10000,
  "val_size": 1000,
  "output_name": "deepcad_10k"
}
```

**Response**:
```json
{
  "success": true,
  "job_id": "extract_12345",
  "status": "started",
  "estimated_time": "20 minutes",
  "output_path": "data/training/deepcad_10k"
}
```

**Backend Script**: `scripts/extract_deepcad_subset.py`

---

### **GET /training/datasets/{dataset_name}/status**
Check dataset preparation status.

**Response**:
```json
{
  "dataset": "deepcad_10k",
  "status": "extracting",
  "progress": {
    "current": 5000,
    "total": 11000,
    "percentage": 45.5
  },
  "stages": {
    "extraction": "completed",
    "rendering": "in_progress",
    "conversion": "pending",
    "validation": "pending"
  }
}
```

---

### **GET /training/datasets**
List all available datasets.

**Response**:
```json
{
  "datasets": [
    {
      "name": "deepcad_1k",
      "models": 1100,
      "images": 3300,
      "conversations": 3300,
      "size": "27 MB",
      "status": "ready"
    },
    {
      "name": "deepcad_10k",
      "models": 11000,
      "images": 33000,
      "conversations": 33000,
      "size": "270 MB",
      "status": "ready"
    }
  ]
}
```

---

## 2. Rendering Pipeline

### **POST /training/datasets/{dataset_name}/render**
Render orthographic views for entire dataset.

**Request**:
```json
{
  "views": ["front", "top", "right"],
  "resolution": 512,
  "batch_size": 100
}
```

**Response**:
```json
{
  "success": true,
  "job_id": "render_12346",
  "status": "started",
  "estimated_time": "90 minutes",
  "models_to_process": 11000,
  "images_to_create": 33000
}
```

**Backend Script**: `scripts/render_deepcad_views.py`

---

### **GET /training/jobs/{job_id}/progress**
Monitor rendering/processing job progress.

**Response**:
```json
{
  "job_id": "render_12346",
  "status": "running",
  "progress": {
    "current": 7500,
    "total": 11000,
    "percentage": 68.2,
    "images_rendered": 22500,
    "estimated_remaining": "30 minutes"
  },
  "stats": {
    "succeeded": 7500,
    "failed": 0,
    "warnings": 15
  }
}
```

---

## 3. Format Conversion

### **POST /training/datasets/{dataset_name}/convert**
Convert to TinyLLaVA training format.

**Request**:
```json
{
  "format": "tinyllava",
  "randomize_templates": true
}
```

**Response**:
```json
{
  "success": true,
  "job_id": "convert_12347",
  "status": "completed",
  "output": {
    "train_file": "data/training/deepcad_10k/train.json",
    "val_file": "data/training/deepcad_10k/val.json",
    "train_conversations": 30000,
    "val_conversations": 3000
  }
}
```

**Backend Script**: `scripts/convert_to_tinyllava.py`

---

## 4. Validation

### **POST /training/datasets/{dataset_name}/validate**
Run validation tests on dataset.

**Response**:
```json
{
  "success": true,
  "validation": {
    "format_valid": true,
    "all_images_exist": true,
    "images_checked": 20,
    "total_images": 33000,
    "conversations_valid": true,
    "no_missing_files": true
  },
  "stats": {
    "train_samples": 30000,
    "val_samples": 3000,
    "avg_text_length": 461,
    "image_resolution": "512x512"
  },
  "warnings": []
}
```

**Backend Script**: `scripts/test_training_pipeline.py`

---

## 5. Training Test

### **POST /training/test-run**
Run a quick training test to verify convergence.

**Request**:
```json
{
  "dataset": "deepcad_10k",
  "train_samples": 100,
  "val_samples": 20,
  "epochs": 5,
  "device": "mps"
}
```

**Response**:
```json
{
  "success": true,
  "job_id": "training_test_12348",
  "status": "started",
  "estimated_time": "5 minutes"
}
```

---

### **GET /training/test-run/{job_id}/results**
Get training test results.

**Response**:
```json
{
  "job_id": "training_test_12348",
  "status": "completed",
  "results": {
    "epochs": 5,
    "loss_progression": [
      {"epoch": 1, "train_loss": 2.15, "val_loss": 0.67},
      {"epoch": 2, "train_loss": 0.40, "val_loss": 0.25},
      {"epoch": 3, "train_loss": 0.19, "val_loss": 0.16},
      {"epoch": 4, "train_loss": 0.13, "val_loss": 0.11},
      {"epoch": 5, "train_loss": 0.08, "val_loss": 0.07}
    ],
    "total_improvement": 96.2,
    "convergence": "excellent",
    "overfitting": false,
    "performance": {
      "throughput": 22,
      "avg_batch_time": 0.18
    }
  }
}
```

**Backend Script**: `scripts/minimal_training_test.py`

---

## 6. Full Training

### **POST /training/start**
Start full VLM fine-tuning.

**Request**:
```json
{
  "dataset": "deepcad_10k",
  "config": {
    "base_model": "Yuan-Che/OpenECADv2-SigLIP-0.89B",
    "use_lora": true,
    "lora_r": 128,
    "lora_alpha": 256,
    "epochs": 5,
    "batch_size": 4,
    "learning_rate": 0.0001
  }
}
```

**Response**:
```json
{
  "success": true,
  "job_id": "training_12349",
  "status": "started",
  "estimated_time": "6-9 hours",
  "checkpoint_dir": "checkpoints/deepcad_10k"
}
```

**Backend Module**: `src/training/finetune.py`

---

### **GET /training/{job_id}/status**
Monitor training progress.

**Response**:
```json
{
  "job_id": "training_12349",
  "status": "training",
  "progress": {
    "current_epoch": 3,
    "total_epochs": 5,
    "current_step": 1500,
    "total_steps": 2500,
    "percentage": 60.0
  },
  "metrics": {
    "current_loss": 0.45,
    "best_loss": 0.42,
    "val_loss": 0.48,
    "learning_rate": 0.00008
  },
  "estimated_remaining": "2.5 hours"
}
```

---

## Example Frontend Workflow

### Complete Pipeline (1K Dataset - Quick Test)

```javascript
// 1. Extract 1K dataset
const extract = await fetch('/training/datasets/extract', {
  method: 'POST',
  body: JSON.stringify({
    source: 'deepcad',
    train_size: 1000,
    val_size: 100,
    output_name: 'deepcad_1k'
  })
});
const { job_id: extractJob } = await extract.json();

// 2. Poll until extraction complete
await pollJobStatus(extractJob);

// 3. Start rendering
const render = await fetch('/training/datasets/deepcad_1k/render', {
  method: 'POST',
  body: JSON.stringify({
    views: ['front', 'top', 'right'],
    resolution: 512
  })
});
const { job_id: renderJob } = await render.json();

// 4. Poll until rendering complete
await pollJobStatus(renderJob);

// 5. Convert to TinyLLaVA format
const convert = await fetch('/training/datasets/deepcad_1k/convert', {
  method: 'POST',
  body: JSON.stringify({
    format: 'tinyllava',
    randomize_templates: true
  })
});

// 6. Validate dataset
const validate = await fetch('/training/datasets/deepcad_1k/validate', {
  method: 'POST'
});
const validationResults = await validate.json();

// 7. Run training test
const test = await fetch('/training/test-run', {
  method: 'POST',
  body: JSON.stringify({
    dataset: 'deepcad_1k',
    train_samples: 100,
    val_samples: 20,
    epochs: 5
  })
});
const { job_id: testJob } = await test.json();

// 8. Get test results
await pollJobStatus(testJob);
const results = await fetch(`/training/test-run/${testJob}/results`);
const testResults = await results.json();

// 9. If test successful, start full training
if (testResults.results.convergence === 'excellent') {
  const training = await fetch('/training/start', {
    method: 'POST',
    body: JSON.stringify({
      dataset: 'deepcad_1k',
      config: {
        epochs: 3,
        batch_size: 4
      }
    })
  });
}

// Helper function to poll job status
async function pollJobStatus(jobId) {
  while (true) {
    const response = await fetch(`/training/jobs/${jobId}/progress`);
    const status = await response.json();

    if (status.status === 'completed') break;
    if (status.status === 'failed') throw new Error('Job failed');

    // Update UI with progress
    console.log(`Progress: ${status.progress.percentage}%`);

    await sleep(5000); // Poll every 5 seconds
  }
}
```

---

## Summary of Implementation Needed

### New Endpoints to Implement:

1. **POST /training/datasets/extract** - Dataset extraction
2. **GET /training/datasets/{name}/status** - Dataset status
3. **GET /training/datasets** - List datasets
4. **POST /training/datasets/{name}/render** - Batch rendering
5. **GET /training/jobs/{id}/progress** - Job progress tracking
6. **POST /training/datasets/{name}/convert** - Format conversion
7. **POST /training/datasets/{name}/validate** - Validation
8. **POST /training/test-run** - Training test
9. **GET /training/test-run/{id}/results** - Test results
10. **POST /training/start** - Full training
11. **GET /training/{id}/status** - Training status

### Existing Endpoints to Keep:

- **GET /training/info** - Training info
- **POST /training/render-views** - Single model rendering
- **POST /validation/training-pair** - Pair validation

---

## Time Estimates (Based on 10K Dataset)

| Stage | Time | API Endpoint |
|-------|------|--------------|
| Extraction | ~2 min | POST /training/datasets/extract |
| Rendering | ~10 min | POST /training/datasets/{name}/render |
| Conversion | ~30 sec | POST /training/datasets/{name}/convert |
| Validation | ~10 sec | POST /training/datasets/{name}/validate |
| Training Test | ~5 min | POST /training/test-run |
| **Total** | **~18 min** | **Full pipeline** |

---

**Generated**: December 3, 2025
**For**: Frontend Integration
**Pipeline Version**: v1.0.2
