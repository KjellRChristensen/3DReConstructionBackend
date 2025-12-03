# Training Scripts

Utility scripts for downloading and preparing CAD training datasets.

## Available Scripts

### download_openecad_dataset.py

Downloads the OpenECAD dataset from HuggingFace and prepares it for training.

**Quick Start (1K samples for testing):**
```bash
python scripts/download_openecad_dataset.py \
    --subset 1000 \
    --output data/training/openecad_test
```

**Full Dataset (919K samples):**
```bash
python scripts/download_openecad_dataset.py \
    --output data/training/openecad_full
```

**Custom split ratio:**
```bash
python scripts/download_openecad_dataset.py \
    --subset 10000 \
    --train-split 0.85 \
    --output data/training/openecad_10k
```

**Arguments:**
- `--output`: Output directory (default: `data/training/openecad_full`)
- `--subset`: Only download first N samples (optional, for testing)
- `--train-split`: Train/val split ratio (default: 0.9)

**Output Structure:**
```
data/training/openecad_test/
├── images/
│   ├── train_000000.jpg
│   ├── train_000001.jpg
│   └── ...
├── train/
│   └── train.json
├── val/
│   └── val.json
└── dataset_info.json
```

## Quick Training Test

After downloading a test dataset:

```bash
# 1. Download test dataset (1K samples)
python scripts/download_openecad_dataset.py --subset 1000 --output data/training/openecad_test

# 2. Run smoke test (1 epoch, ~10 minutes)
python -m training.finetune --config config/test_config.yaml

# 3. Check results
ls checkpoints/test_smoke/
```

## Full Training Pipeline

```bash
# 1. Download full dataset (takes ~30 minutes)
python scripts/download_openecad_dataset.py --output data/training/openecad_full

# 2. Update config/finetune_config.yaml with paths from output

# 3. Start training (~24 hours)
python -m training.finetune --config config/finetune_config.yaml

# 4. Monitor with tensorboard
tensorboard --logdir checkpoints/
```

## Requirements

```bash
pip install datasets pillow
```

## Troubleshooting

**Import Error:**
```
ImportError: No module named 'datasets'
```
**Solution:** Install required packages:
```bash
pip install datasets pillow
```

**Slow Download:**
The HuggingFace dataset is ~1.7GB. On slow connections:
- Use `--subset 1000` for initial testing
- Download continues from where it left off if interrupted

**Out of Space:**
Full dataset with images: ~2.5GB
- 1K samples: ~3 MB
- 10K samples: ~30 MB
- 100K samples: ~300 MB
- 919K samples: ~2.5 GB
