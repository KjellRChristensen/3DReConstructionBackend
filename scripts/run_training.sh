#!/bin/bash
#
# run_training.sh - Launch training with MPS stability settings
#
# This script sets Metal/MPS environment variables BEFORE Python starts,
# which is required for them to take effect. Setting them in Python code
# is too late since Metal is already loaded by then.
#
# Usage:
#   ./scripts/run_training.sh [python_script] [args...]
#
# Examples:
#   ./scripts/run_training.sh scripts/test_mps_stability.py
#   ./scripts/run_training.sh scripts/minimal_training_test.py --data data/training/deepcad_1k/train.json --images data/training/deepcad_1k/images
#   ./scripts/run_training.sh -m src.training.trainer  # Run as module
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"

# Change to backend directory
cd "$BACKEND_DIR"

# ============================================================================
# MPS/Metal Stability Settings
# These MUST be set before Python/PyTorch loads the Metal framework
# ============================================================================

# Disable Metal debug/validation layers (prevents assertion crashes)
export MTL_DEBUG_LAYER=0
export MTL_SHADER_VALIDATION=0
export METAL_DEVICE_WRAPPER_TYPE=0
export METAL_DEBUG_ERROR_MODE=0

# PyTorch MPS settings
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Disable aggressive memory caching
export PYTORCH_ENABLE_MPS_FALLBACK=1          # Enable CPU fallback for unsupported ops

# Disable parallelism that can cause issues
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================================
# Activate virtual environment if it exists
# ============================================================================

if [ -f "venv/bin/activate" ]; then
    echo "[run_training.sh] Activating virtual environment..."
    source venv/bin/activate
fi

# ============================================================================
# Run Python with the provided arguments
# ============================================================================

echo "[run_training.sh] Metal/MPS environment configured:"
echo "  MTL_DEBUG_LAYER=$MTL_DEBUG_LAYER"
echo "  MTL_SHADER_VALIDATION=$MTL_SHADER_VALIDATION"
echo "  PYTORCH_MPS_HIGH_WATERMARK_RATIO=$PYTORCH_MPS_HIGH_WATERMARK_RATIO"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=$PYTORCH_ENABLE_MPS_FALLBACK"
echo ""

if [ $# -eq 0 ]; then
    echo "Usage: $0 [python_script] [args...]"
    echo ""
    echo "Examples:"
    echo "  $0 scripts/test_mps_stability.py"
    echo "  $0 scripts/minimal_training_test.py --data data/training/deepcad_1k/train.json"
    echo "  $0 main.py server --port 7001"
    exit 1
fi

echo "[run_training.sh] Running: python3 $@"
echo "============================================================"
exec python3 "$@"
