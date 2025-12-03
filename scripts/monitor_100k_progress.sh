#!/bin/bash
# Monitor 100K dataset processing progress

echo "=== DeepCAD 100K Dataset Progress ==="
echo ""

# Extraction Progress
EXTRACTION_LOG="data/training/deepcad_100k_extraction.log"
if [ -f "$EXTRACTION_LOG" ]; then
    echo "📦 Extraction Progress:"
    LATEST=$(grep "Copied.*files" "$EXTRACTION_LOG" | tail -1)
    if [ -n "$LATEST" ]; then
        echo "  $LATEST"
    else
        echo "  Starting..."
    fi
    echo ""
fi

# Rendering Progress
RENDERING_LOG="data/training/deepcad_100k/rendering.log"
if [ -f "$RENDERING_LOG" ]; then
    echo "🎨 Rendering Progress:"
    LATEST=$(grep "Processed" "$RENDERING_LOG" | tail -1)
    if [ -n "$LATEST" ]; then
        CURRENT=$(echo "$LATEST" | grep -oE '[0-9]+/[0-9]+' | cut -d'/' -f1)
        TOTAL=108946
        PERCENT=$((CURRENT * 100 / TOTAL))
        echo "  $CURRENT / $TOTAL files ($PERCENT%)"

        # Show images rendered
        if [ -d "data/training/deepcad_100k/images" ]; then
            IMAGES=$(find data/training/deepcad_100k/images -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
            echo "  🖼️  Images: $IMAGES / 326838"
        fi
    else
        echo "  Waiting to start..."
    fi
    echo ""
fi

# Conversion Progress
CONVERSION_LOG="data/training/deepcad_100k_conversion.log"
if [ -f "$CONVERSION_LOG" ]; then
    echo "🔄 Conversion Progress:"
    LATEST=$(grep "Processed.*samples" "$CONVERSION_LOG" | tail -1)
    if [ -n "$LATEST" ]; then
        echo "  $LATEST"
    else
        echo "  Starting..."
    fi
    echo ""
fi

# Summary
echo "Dataset Info:"
if [ -f "data/training/deepcad_100k/dataset_info.json" ]; then
    echo "  ✓ Dataset created"
fi
if [ -d "data/training/deepcad_100k/cad_json" ]; then
    JSON_COUNT=$(find data/training/deepcad_100k/cad_json -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "  📄 JSON files: $JSON_COUNT / 108946"
fi

echo ""
echo "To view full logs:"
echo "  tail -f $EXTRACTION_LOG"
echo "  tail -f $RENDERING_LOG"
echo "  tail -f $CONVERSION_LOG"
