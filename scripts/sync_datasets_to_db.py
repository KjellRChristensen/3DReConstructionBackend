#!/usr/bin/env python3
"""
Sync existing datasets from filesystem to database

This script scans the data/training directory and populates the database
with metadata from existing datasets.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import init_db, get_db, Dataset


def sync_datasets():
    """Scan filesystem and sync datasets to database"""

    # Initialize database
    print("Initializing database...")
    init_db()

    datasets_dir = Path("data/training")

    if not datasets_dir.exists():
        print(f"❌ Directory not found: {datasets_dir}")
        return

    print(f"\n📂 Scanning datasets in: {datasets_dir}")
    print("=" * 70)

    synced_count = 0
    skipped_count = 0
    error_count = 0

    with get_db() as db:
        for dataset_path in sorted(datasets_dir.iterdir()):
            if not dataset_path.is_dir() or dataset_path.name.startswith('.'):
                continue

            dataset_name = dataset_path.name
            print(f"\n📦 Processing: {dataset_name}")

            # Check if already exists in database
            existing = db.query(Dataset).filter(Dataset.name == dataset_name).first()
            if existing:
                print(f"   ⏭️  Already in database (ID: {existing.id})")
                skipped_count += 1
                continue

            # Look for metadata files
            dataset_info_file = dataset_path / "dataset_info.json"
            tinyllava_metadata = dataset_path / "tinyllava_metadata.json"
            rendering_metadata = dataset_path / "rendering_metadata.json"

            # Skip if no metadata (incomplete dataset)
            if not dataset_info_file.exists():
                print(f"   ⚠️  No dataset_info.json found - skipping")
                skipped_count += 1
                continue

            try:
                # Load dataset_info.json
                with open(dataset_info_file) as f:
                    info = json.load(f)

                train_samples = info.get("train_samples", 0)
                val_samples = info.get("val_samples", 0)
                total_models = info.get("total_models", train_samples + val_samples)
                description = info.get("description", "")

                # Load TinyLLaVA metadata for conversation counts
                conversations = 0
                if tinyllava_metadata.exists():
                    try:
                        with open(tinyllava_metadata) as f:
                            meta = json.load(f)
                            conversations = meta.get("total_conversations", 0)
                    except:
                        pass

                # Estimate if not found
                if conversations == 0:
                    conversations = (train_samples + val_samples) * 3

                # Load rendering metadata for image counts
                images = 0
                if rendering_metadata.exists():
                    try:
                        with open(rendering_metadata) as f:
                            meta = json.load(f)
                            images = meta.get("total_images", 0)
                    except:
                        pass

                # Estimate if not found
                if images == 0:
                    images = total_models * 3

                # Estimate size (25 KB per model on average)
                size_bytes = total_models * 25000

                # Get creation time from directory
                created_at = datetime.fromtimestamp(dataset_path.stat().st_mtime)

                # Create database entry
                dataset = Dataset(
                    name=dataset_name,
                    path=str(dataset_path),
                    description=description,
                    total_models=total_models,
                    train_samples=train_samples,
                    val_samples=val_samples,
                    images=images,
                    conversations=conversations,
                    size_bytes=size_bytes,
                    status="ready",
                    created_at=created_at,
                    updated_at=created_at,
                )

                db.add(dataset)
                db.flush()  # Get the ID

                print(f"   ✅ Added to database (ID: {dataset.id})")
                print(f"      - Models: {total_models:,}")
                print(f"      - Train samples: {train_samples:,}")
                print(f"      - Val samples: {val_samples:,}")
                print(f"      - Images: {images:,}")
                print(f"      - Conversations: {conversations:,}")
                print(f"      - Size: {dataset._format_size(size_bytes)}")

                synced_count += 1

            except Exception as e:
                print(f"   ❌ Error: {e}")
                error_count += 1
                continue

    print("\n" + "=" * 70)
    print(f"📊 Summary:")
    print(f"   ✅ Synced: {synced_count}")
    print(f"   ⏭️  Skipped: {skipped_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📚 Total: {synced_count + skipped_count}")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Sync Datasets to Database")
    print("=" * 70)

    try:
        sync_datasets()
        print("\n✅ Sync completed successfully!\n")
    except Exception as e:
        print(f"\n❌ Sync failed: {e}\n")
        sys.exit(1)
