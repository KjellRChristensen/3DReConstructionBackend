#!/usr/bin/env python3
"""
Quick test script to verify training loss is non-zero
"""
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import VLMTrainer, CADDataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_loss():
    """Test that loss computation works and is non-zero"""

    print("=" * 80)
    print("TESTING TRAINING LOSS")
    print("=" * 80)

    # Configuration
    dataset_path = "data/training/deepcad_1k"
    model_name = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"

    print(f"\n📁 Dataset: {dataset_path}")
    print(f"🤖 Model: {model_name}")

    # Create a minimal trainer instance
    print("\n🔧 Initializing trainer...")
    trainer = VLMTrainer(
        job_id="test-local",
        dataset_path=dataset_path,
        model_name=model_name,
        output_dir="data/training/outputs/test-local",
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        use_lora=True,
    )

    # Load model and processor
    print("\n📥 Loading model and processor...")
    trainer._load_model_and_processor()

    # Load dataset
    print("\n📊 Loading dataset...")
    trainer._prepare_datasets()

    # Get a small batch
    print(f"\n🎲 Getting test batch (batch_size=2)...")
    batch_data = [trainer.train_dataset[i] for i in range(2)]

    # Run through collate function
    print("\n🔄 Running collate function...")
    batch = trainer._collate_fn(batch_data)

    # Move to device
    device = trainer.device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    print(f"\n📊 Batch information:")
    print(f"  - input_ids shape: {batch['input_ids'].shape}")
    print(f"  - images shape: {batch['images'].shape}")
    print(f"  - labels shape: {batch['labels'].shape}")

    # Check label statistics
    labels = batch['labels']
    total_tokens = labels.numel()
    training_tokens = (labels != -100).sum().item()
    masked_tokens = total_tokens - training_tokens

    print(f"\n🏷️  Label statistics:")
    print(f"  - Total tokens: {total_tokens}")
    print(f"  - Training tokens: {training_tokens} ({100*training_tokens/total_tokens:.1f}%)")
    print(f"  - Masked tokens: {masked_tokens} ({100*masked_tokens/total_tokens:.1f}%)")

    if training_tokens == 0:
        print("\n❌ ERROR: No training tokens! All labels are masked.")
        return False

    print(f"\n✅ Good! {training_tokens} tokens will be trained on")

    # Forward pass
    print("\n⚡ Running forward pass (train mode)...")
    trainer.model.train()

    # Don't use no_grad - we need gradients for loss computation
    outputs = trainer.model(**batch)

    # Extract loss
    if isinstance(outputs, dict):
        loss = outputs.get("loss")
        logits = outputs.get("logits")
        print(f"\n📊 Model outputs:")
        print(f"  - Has 'loss': {loss is not None}")
        print(f"  - Has 'logits': {logits is not None}")
        if logits is not None:
            print(f"  - Logits shape: {logits.shape}")
            print(f"  - Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
            print(f"  - Logits has NaN: {torch.isnan(logits).any().item()}")
    else:
        loss = outputs

    if loss is None:
        print("\n⚠️  Model didn't return loss, computing manually...")
        logits = outputs.get("logits")
        labels = batch["labels"]

        # Compute cross-entropy loss manually
        import torch.nn.functional as F
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        print(f"  - Manually computed loss: {loss.item():.6f}")

    print(f"\n📈 LOSS RESULTS:")
    print(f"  - Loss value: {loss.item() if loss is not None else 'None':.6f}")
    print(f"  - Is finite: {torch.isfinite(loss).item() if loss is not None else False}")
    print(f"  - Is zero: {loss.item() == 0.0 if loss is not None else 'N/A'}")

    # Check if loss is valid
    if loss.item() == 0.0:
        print("\n❌ FAILED: Loss is still 0.0!")
        return False
    elif not torch.isfinite(loss).item():
        print("\n❌ FAILED: Loss is NaN or Inf!")
        return False
    else:
        print(f"\n✅ SUCCESS: Loss is {loss.item():.6f} (non-zero and finite!)")
        return True

if __name__ == "__main__":
    try:
        success = test_loss()
        print("\n" + "=" * 80)
        if success:
            print("✅ TEST PASSED: Training loss is working correctly!")
        else:
            print("❌ TEST FAILED: Training loss is still broken")
        print("=" * 80)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
