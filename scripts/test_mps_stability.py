#!/usr/bin/env python3
"""
Test MPS Stability

Quick test script to verify that MPS (Metal Performance Shaders) is stable
on this system before running training.

Usage:
    python scripts/test_mps_stability.py
"""

import os
import sys

# Set MPS stability environment variables BEFORE importing torch
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MTL_DEBUG_LAYER"] = "0"
os.environ["MTL_SHADER_VALIDATION"] = "0"

import torch
import torch.nn as nn
import platform


def get_system_info():
    """Get system information."""
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"macOS version: {platform.mac_ver()[0]}")
    print()


def check_mps_available():
    """Check if MPS is available."""
    print("=" * 60)
    print("MPS Availability Check")
    print("=" * 60)

    has_mps = hasattr(torch.backends, 'mps')
    print(f"torch.backends.mps exists: {has_mps}")

    if has_mps:
        is_available = torch.backends.mps.is_available()
        is_built = torch.backends.mps.is_built()
        print(f"MPS is available: {is_available}")
        print(f"MPS is built: {is_built}")

        # Check macOS version
        macos_version = platform.mac_ver()[0]
        try:
            major_version = int(macos_version.split('.')[0])
            if major_version >= 26:
                print()
                print("⚠️  WARNING: macOS 26+ (Tahoe) detected!")
                print("   MPS has known crashes with Metal heap allocator on this version.")
                print("   Training will automatically use CPU instead of MPS.")
                print("   This is a PyTorch/Metal compatibility issue, not a bug in this code.")
                return False
        except (ValueError, IndexError):
            pass

        return is_available

    return False


def run_stability_tests():
    """Run MPS stability tests."""
    print()
    print("=" * 60)
    print("MPS Stability Tests")
    print("=" * 60)

    device = torch.device("mps")
    tests_passed = 0
    tests_failed = 0

    # Test 1: Basic tensor creation
    print("\n[Test 1] Basic tensor creation...")
    try:
        x = torch.randn(64, 64, device=device)
        print(f"  ✓ Created tensor on MPS: shape={x.shape}, dtype={x.dtype}")
        del x
        torch.mps.synchronize()
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
        return False

    # Test 2: Matrix multiplication
    print("\n[Test 2] Matrix multiplication...")
    try:
        a = torch.randn(128, 128, device=device)
        b = torch.randn(128, 128, device=device)
        c = torch.matmul(a, b)
        print(f"  ✓ Matrix multiplication successful: result shape={c.shape}")
        del a, b, c
        torch.mps.synchronize()
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
        return False

    # Test 3: Linear layer (where crashes typically occur)
    print("\n[Test 3] Linear layer forward pass...")
    try:
        linear = nn.Linear(256, 512).to(device)
        x = torch.randn(16, 256, device=device)
        y = linear(x)
        print(f"  ✓ Linear layer forward: input={x.shape}, output={y.shape}")
        del linear, x, y
        torch.mps.synchronize()
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
        return False

    # Test 4: Memory allocation/deallocation stress test
    print("\n[Test 4] Memory allocation stress test...")
    try:
        tensors = []
        for i in range(20):
            t = torch.randn(256, 256, device=device)
            tensors.append(t)

        # Force synchronization
        torch.mps.synchronize()

        # Delete in reverse order
        for t in reversed(tensors):
            del t
        tensors.clear()

        torch.mps.empty_cache()
        print(f"  ✓ Allocated and freed 20 tensors successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
        return False

    # Test 5: Simple neural network with backward pass
    print("\n[Test 5] Neural network with backward pass...")
    try:
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).to(device)

        x = torch.randn(8, 128, device=device)
        y = model(x)
        loss = y.mean()
        loss.backward()

        print(f"  ✓ Forward + backward pass successful")
        print(f"    Input: {x.shape}, Output: {y.shape}, Loss: {loss.item():.4f}")

        del model, x, y, loss
        torch.mps.synchronize()
        torch.mps.empty_cache()
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
        return False

    # Test 6: Larger model similar to LoRA training
    print("\n[Test 6] Larger model (similar to LoRA training)...")
    try:
        # Simulate a transformer-like layer
        batch_size = 4
        seq_len = 128
        hidden_dim = 512

        q_proj = nn.Linear(hidden_dim, hidden_dim).to(device)
        k_proj = nn.Linear(hidden_dim, hidden_dim).to(device)
        v_proj = nn.Linear(hidden_dim, hidden_dim).to(device)
        out_proj = nn.Linear(hidden_dim, hidden_dim).to(device)

        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (hidden_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        output = out_proj(attn_output)

        loss = output.mean()
        loss.backward()

        print(f"  ✓ Transformer-like layer successful")
        print(f"    Input: {x.shape}, Output: {output.shape}")

        del q_proj, k_proj, v_proj, out_proj, x, q, k, v, attn_weights, attn_output, output, loss
        torch.mps.synchronize()
        torch.mps.empty_cache()
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
        return False

    print()
    print("=" * 60)
    print(f"Test Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)

    return tests_failed == 0


def main():
    get_system_info()

    if not check_mps_available():
        print("\n⚠ MPS is not available on this system.")
        print("Training will use CPU instead.")
        return 1

    print("\nMPS is available. Running stability tests...")

    try:
        if run_stability_tests():
            print("\n✅ All MPS stability tests PASSED!")
            print("   MPS can be used for training on this system.")
            return 0
        else:
            print("\n❌ Some MPS stability tests FAILED!")
            print("   Training will automatically fall back to CPU.")
            return 1
    except Exception as e:
        print(f"\n❌ MPS stability test crashed: {e}")
        print("   This confirms MPS is unstable on this system.")
        print("   Training will automatically fall back to CPU.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
