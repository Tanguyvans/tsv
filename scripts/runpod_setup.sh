#!/bin/bash
# RunPod setup script for Moondream 3 MVTec AD diagnostic
# Run this on a RunPod instance with RTX 4090 / A100 / H100

set -e

echo "=== Moondream 3 Anomaly Detection Diagnostic ==="

# Use /workspace for persistence on RunPod
WORK_DIR="${WORKSPACE:-/workspace}/tsv"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Install system dependencies
apt-get update && apt-get install -y git wget unzip

# Clone repo
git clone https://github.com/Tanguyvans/tsv.git . 2>/dev/null || git pull

# Install Python dependencies
pip install transformers accelerate pillow tqdm huggingface_hub einops pyvips

# Download MVTec AD dataset
echo "=== Downloading MVTec AD dataset ==="
python scripts/download_mvtec.py

# Run diagnostic
echo "=== Running diagnostic with 10 samples ==="
mkdir -p results
python scripts/diagnostic_mvtec.py --num-samples 10 --output results/mvtec_test.json

echo ""
echo "=== DONE ==="
echo "Results saved to results/mvtec_test.json"
