#!/bin/bash
# Monitor 10K dataset rendering progress

echo "=== DeepCAD 10K Rendering Progress ==="
echo ""

LOG_FILE="data/training/deepcad_10k/rendering.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

# Get latest progress
LATEST_PROGRESS=$(grep "Processed" "$LOG_FILE" | tail -1)
TOTAL=11000

if [ -n "$LATEST_PROGRESS" ]; then
    # Extract number (e.g., "3000/11000")
    CURRENT=$(echo "$LATEST_PROGRESS" | grep -oE '[0-9]+/[0-9]+' | cut -d'/' -f1)
    PERCENT=$((CURRENT * 100 / TOTAL))

    echo "📊 Progress: $CURRENT / $TOTAL files ($PERCENT%)"
    echo ""

    # Calculate ETA
    if [ -f "data/training/deepcad_10k/images" ]; then
        IMAGES=$(find data/training/deepcad_10k/images -name "*.png" 2>/dev/null | wc -l)
        echo "🖼️  Images rendered: $IMAGES / 33000"
    fi

    # Show recent warnings
    echo ""
    echo "Recent activity:"
    tail -5 "$LOG_FILE"
else
    echo "⏳ Starting..."
fi

echo ""
echo "To view full log:"
echo "  tail -f $LOG_FILE"
