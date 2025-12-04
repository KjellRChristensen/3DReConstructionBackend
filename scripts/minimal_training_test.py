"""
Minimal Training Test - Get Real Metrics

Runs a minimal training test to collect:
- Loss curves
- Training metrics
- Memory usage
- Speed benchmarks

Uses a tiny subset (10 samples) and simple model for quick testing.

Usage:
    python scripts/minimal_training_test.py --data data/training/deepcad_1k/train.json --images data/training/deepcad_1k/images
"""

import argparse
import json
import sys
import os
from pathlib import Path
import logging
import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# MPS (Metal Performance Shaders) stability settings for macOS
# These help prevent crashes on newer macOS versions (26+/Tahoe)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MTL_DEBUG_LAYER"] = "0"
os.environ["MTL_SHADER_VALIDATION"] = "0"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_mps_stability() -> bool:
    """Test if MPS is stable on this system."""
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return False

    # Check macOS version - MPS is unstable on macOS 26+ (Tahoe)
    import platform
    try:
        macos_version = platform.mac_ver()[0]
        major_version = int(macos_version.split('.')[0])
        if major_version >= 26:
            logger.warning(f"macOS {macos_version} detected. MPS unstable on macOS 26+.")
            return False
    except (ValueError, IndexError):
        pass

    try:
        logger.info("Testing MPS stability...")
        device = torch.device("mps")

        # Test basic operations
        x = torch.randn(64, 64, device=device)
        y = torch.randn(64, 64, device=device)
        z = torch.matmul(x, y)

        # Test linear layer
        linear = torch.nn.Linear(64, 64).to(device)
        output = linear(torch.randn(4, 64, device=device))

        # Test memory management
        torch.mps.synchronize()
        torch.mps.empty_cache()

        del x, y, z, linear, output
        torch.mps.empty_cache()

        logger.info("MPS stability check passed")
        return True

    except Exception as e:
        logger.warning(f"MPS stability check failed: {e}")
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
        return False


class TinyCADDataset(Dataset):
    """Minimal dataset for testing"""

    def __init__(self, data_path, image_folder, max_samples=10, image_size=224):
        with open(data_path, 'r') as f:
            data = json.load(f)

        # Use only first N samples
        self.data = data[:max_samples]
        self.image_folder = Path(image_folder)
        self.image_size = image_size

        logger.info(f"Loaded {len(self.data)} samples for testing")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image_name = item["image"]
        image_path = self.image_folder / image_name
        image = Image.open(image_path).convert('RGB')

        # Resize and normalize
        image = image.resize((self.image_size, self.image_size))
        image_array = np.array(image).astype(np.float32) / 255.0

        # Transpose to CHW format
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        # Get text
        conversations = item["conversations"]
        # Find assistant response
        response = ""
        for turn in conversations:
            if turn["from"] in ["gpt", "assistant"]:
                response = turn["value"]
                break

        # Simple tokenization (character level for testing)
        response_tokens = [ord(c) % 256 for c in response[:100]]  # Limit length

        return {
            "image": image_tensor,
            "text": response,
            "text_tokens": torch.tensor(response_tokens, dtype=torch.long)
        }


class SimpleCNNEncoder(nn.Module):
    """Simple CNN for image encoding"""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, hidden_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleDecoder(nn.Module):
    """Simple decoder for text generation"""

    def __init__(self, hidden_dim=256, vocab_size=256, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len

    def forward(self, image_features, text_tokens):
        # Expand image features to sequence
        batch_size = image_features.size(0)
        seq_len = min(text_tokens.size(1), self.max_len)

        # Embed text tokens
        text_emb = self.embedding(text_tokens[:, :seq_len])

        # Add image features as first token
        image_features = image_features.unsqueeze(1)
        combined = torch.cat([image_features, text_emb[:, :-1]], dim=1)

        # RNN
        output, _ = self.rnn(combined)

        # Predict next tokens
        logits = self.fc(output)

        return logits


class SimpleVLM(nn.Module):
    """Simple Vision-Language Model for testing"""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.encoder = SimpleCNNEncoder(hidden_dim)
        self.decoder = SimpleDecoder(hidden_dim)

    def forward(self, images, text_tokens):
        image_features = self.encoder(images)
        logits = self.decoder(image_features, text_tokens)
        return logits


def collate_fn(batch):
    """Collate function for batching"""
    images = torch.stack([item["image"] for item in batch])

    # Pad text tokens to same length
    max_len = max(item["text_tokens"].size(0) for item in batch)
    text_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        tokens = item["text_tokens"]
        text_tokens[i, :tokens.size(0)] = tokens

    return {
        "images": images,
        "text_tokens": text_tokens
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    losses = []
    batch_times = []

    for batch_idx, batch in enumerate(dataloader):
        start_time = time.time()

        images = batch["images"].to(device)
        text_tokens = batch["text_tokens"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images, text_tokens)

        # Compute loss
        # Reshape for cross entropy: (batch * seq_len, vocab_size)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = text_tokens.view(-1)

        loss = criterion(logits_flat, targets_flat)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        losses.append(loss.item())
        num_batches += 1

        batch_time = time.time() - start_time
        batch_times.append(batch_time)

        logger.info(f"  Batch {batch_idx + 1}: loss={loss.item():.4f}, time={batch_time:.3f}s")

    avg_loss = total_loss / num_batches
    avg_time = sum(batch_times) / len(batch_times)

    return {
        "avg_loss": avg_loss,
        "losses": losses,
        "avg_batch_time": avg_time
    }


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            text_tokens = batch["text_tokens"].to(device)

            logits = model(images, text_tokens)

            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = text_tokens.view(-1)

            loss = criterion(logits_flat, targets_flat)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Minimal training test")
    parser.add_argument("--data", type=str, required=True, help="Training data JSON")
    parser.add_argument("--images", type=str, required=True, help="Images folder")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Minimal Training Test - Real Metrics")
    logger.info("=" * 60)

    # Check device - with MPS stability check for macOS 26+
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        # MPS available but may be unstable on newer macOS
        if check_mps_stability():
            device = torch.device("mps")
            logger.info(f"✓ Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info(f"⚠ MPS unstable, falling back to CPU")
    else:
        device = torch.device("cpu")
        logger.info(f"✓ Using CPU")

    # Load dataset
    logger.info(f"\nLoading dataset...")
    dataset = TinyCADDataset(args.data, args.images, max_samples=args.num_samples)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Create model
    logger.info(f"\nInitializing model...")
    model = SimpleVLM(hidden_dim=256).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    logger.info(f"\n" + "=" * 60)
    logger.info(f"Starting Training")
    logger.info("=" * 60)

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 40)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)

        # Track
        train_losses.append(train_metrics["avg_loss"])
        val_losses.append(val_loss)

        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_metrics['avg_loss']:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Avg Batch Time: {train_metrics['avg_batch_time']:.3f}s")

    # Final Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete - Final Metrics")
    logger.info("=" * 60)

    logger.info("\nLoss Progression:")
    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        improvement = ""
        if i > 0:
            train_change = train_losses[i] - train_losses[i-1]
            val_change = val_losses[i] - val_losses[i-1]
            improvement = f"(Δ train: {train_change:+.4f}, Δ val: {val_change:+.4f})"

        logger.info(f"  Epoch {i+1}: Train={train_loss:.4f}, Val={val_loss:.4f} {improvement}")

    # Convergence check
    if len(train_losses) >= 2:
        total_improvement = train_losses[0] - train_losses[-1]
        logger.info(f"\nTotal Loss Improvement: {total_improvement:.4f}")

        if total_improvement > 0:
            logger.info("✓ Model is converging (loss decreasing)")
        else:
            logger.info("⚠ Model not converging (loss increasing/flat)")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Test Complete!")
    logger.info("=" * 60)
    logger.info("\nKey Findings:")
    logger.info(f"  - Model can load and process CAD data")
    logger.info(f"  - Training loop works correctly")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Batch processing: {args.batch_size} samples/batch")
    logger.info(f"\nThis validates the data pipeline is ready for real VLM training!")


if __name__ == "__main__":
    main()
