#!/bin/bash
# =============================================================================
# Moondream 3 Anomaly Detection Diagnostic
# Run this script on an H100 or similar GPU (24GB+ VRAM required)
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Moondream 3 Anomaly Detection Diagnostic"
echo "=============================================="

# Configuration
DATA_DIR="${DATA_DIR:-data/mmad}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # Empty = all samples

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Check GPU
echo ""
echo "GPU Check:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run diagnostic
echo "Starting diagnostic..."
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

if [ -z "$MAX_SAMPLES" ]; then
    python scripts/diagnostic_moondream.py \
        --data-dir "$DATA_DIR" \
        --output "$OUTPUT_DIR/diagnostic_results.json"
else
    echo "Running with max $MAX_SAMPLES samples per test (quick mode)"
    python scripts/diagnostic_moondream.py \
        --data-dir "$DATA_DIR" \
        --output "$OUTPUT_DIR/diagnostic_results.json" \
        --max-samples "$MAX_SAMPLES"
fi

echo ""
echo "=============================================="
echo "Diagnostic complete!"
echo "Results: $OUTPUT_DIR/diagnostic_results.json"
echo "=============================================="
