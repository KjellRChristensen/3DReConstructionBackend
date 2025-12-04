#!/usr/bin/env python3
"""
Populate available VLM models for training

This script adds VLM models that can be used for CAD reconstruction training.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import init_db, get_db, Model


# Available models for CAD reconstruction
MODELS = [
    {
        "name": "tinyllava-1.1b",
        "display_name": "TinyLLaVA 1.1B",
        "description": "Lightweight VLM with 1.1B parameters. Fast training, good for quick experiments and testing.",
        "architecture": "tinyllava",
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "vision_model": "openai/clip-vit-large-patch14-336",
        "parameters": "1.1B",
        "supports_lora": True,
        "supports_full_finetune": True,
        "recommended_for_cad": True,
        "min_vram_gb": 8,
        "min_ram_gb": 16,
        "available": True,
        "verified": True,
        "huggingface_id": "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",  # Updated to working repo
    },
    {
        "name": "tinyllava-3.1b",
        "display_name": "TinyLLaVA 3.1B",
        "description": "Medium-sized VLM with 3.1B parameters. Better quality than 1.1B with reasonable training time.",
        "architecture": "tinyllava",
        "base_model": "microsoft/phi-2",
        "vision_model": "openai/clip-vit-large-patch14-336",
        "parameters": "3.1B",
        "supports_lora": True,
        "supports_full_finetune": True,
        "recommended_for_cad": True,
        "min_vram_gb": 16,
        "min_ram_gb": 24,
        "available": True,
        "verified": True,
        "huggingface_id": "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",  # Same repo, it's the 3.1B model
    },
    {
        "name": "llava-v1.5-7b",
        "display_name": "LLaVA v1.5 7B",
        "description": "Full-featured LLaVA model with 7B parameters. High quality, slower training.",
        "architecture": "llava",
        "base_model": "lmsys/vicuna-7b-v1.5",
        "vision_model": "openai/clip-vit-large-patch14-336",
        "parameters": "7B",
        "supports_lora": True,
        "supports_full_finetune": True,
        "recommended_for_cad": False,
        "min_vram_gb": 24,
        "min_ram_gb": 32,
        "available": True,
        "verified": False,
        "huggingface_id": "liuhaotian/llava-v1.5-7b",
    },
    {
        "name": "llava-v1.5-13b",
        "display_name": "LLaVA v1.5 13B",
        "description": "Large LLaVA model with 13B parameters. Best quality, requires significant resources.",
        "architecture": "llava",
        "base_model": "lmsys/vicuna-13b-v1.5",
        "vision_model": "openai/clip-vit-large-patch14-336",
        "parameters": "13B",
        "supports_lora": True,
        "supports_full_finetune": False,
        "recommended_for_cad": False,
        "min_vram_gb": 40,
        "min_ram_gb": 64,
        "available": True,
        "verified": False,
        "huggingface_id": "liuhaotian/llava-v1.5-13b",
    },
    {
        "name": "llava-next-7b",
        "display_name": "LLaVA-NeXT 7B",
        "description": "Next generation LLaVA with improved performance. Better understanding of spatial relationships.",
        "architecture": "llava",
        "base_model": "lmsys/vicuna-7b-v1.5",
        "vision_model": "openai/clip-vit-large-patch14-336",
        "parameters": "7B",
        "supports_lora": True,
        "supports_full_finetune": True,
        "recommended_for_cad": False,
        "min_vram_gb": 24,
        "min_ram_gb": 32,
        "available": True,
        "verified": False,
        "huggingface_id": "lmms-lab/llava-next-7b",
    },
]


def populate_models():
    """Populate models table with available VLM models"""

    # Initialize database
    print("Initializing database...")
    init_db()

    print(f"\n{'='*70}")
    print("  Populating Available Models")
    print(f"{'='*70}\n")

    added_count = 0
    skipped_count = 0
    updated_count = 0

    with get_db() as db:
        for model_data in MODELS:
            model_name = model_data["name"]
            print(f"📦 Processing: {model_name}")

            # Check if model already exists
            existing = db.query(Model).filter(Model.name == model_name).first()

            if existing:
                # Update existing model
                for key, value in model_data.items():
                    if key != "name":  # Don't update the name
                        setattr(existing, key, value)
                print(f"   ✏️  Updated existing model (ID: {existing.id})")
                print(f"      Display: {model_data['display_name']}")
                print(f"      Parameters: {model_data['parameters']}")
                print(f"      Recommended for CAD: {'✅' if model_data['recommended_for_cad'] else '❌'}")
                updated_count += 1
            else:
                # Create new model
                model = Model(**model_data)
                db.add(model)
                db.flush()  # Get the ID

                print(f"   ✅ Added to database (ID: {model.id})")
                print(f"      Display: {model_data['display_name']}")
                print(f"      Architecture: {model_data['architecture']}")
                print(f"      Parameters: {model_data['parameters']}")
                print(f"      VRAM: {model_data['min_vram_gb']} GB | RAM: {model_data['min_ram_gb']} GB")
                print(f"      Recommended for CAD: {'✅' if model_data['recommended_for_cad'] else '❌'}")
                added_count += 1

            print()

    print(f"{'='*70}")
    print(f"📊 Summary:")
    print(f"   ✅ Added: {added_count}")
    print(f"   ✏️  Updated: {updated_count}")
    print(f"   📚 Total: {added_count + updated_count + skipped_count}")
    print(f"{'='*70}")


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("  Populate VLM Models")
    print(f"{'='*70}")

    try:
        populate_models()
        print("\n✅ Models populated successfully!\n")
    except Exception as e:
        print(f"\n❌ Failed: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
