#!/bin/bash
# RunPod setup script for Moondream 3 MMAD diagnostic
# Run this on a RunPod instance with RTX 4090 / A100 / H100

set -e

echo "=== Setting up Moondream 3 MMAD Diagnostic ==="

# Create working directory (use /workspace for persistence on RunPod)
WORK_DIR="${WORKSPACE:-/workspace}/tsv"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Install system dependencies
apt-get update && apt-get install -y git wget unzip

# Clone the repo instead of embedding scripts
git clone https://github.com/Tanguyvans/tsv.git . 2>/dev/null || git pull

# Create the diagnostic script (simplified version for quick test)
cat > scripts/diagnostic_moondream.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Moondream 3 Diagnostic on MMAD Dataset
Tests VLM anomaly detection capabilities
"""

import json
import os
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_mmad_samples(data_dir: str, num_samples: int = None, seed: int = 42):
    """Load samples from MMAD dataset structure."""
    data_path = Path(data_dir)
    samples = []

    # Walk through all QA.json files
    for qa_file in data_path.rglob("QA.json"):
        category_dir = qa_file.parent
        with open(qa_file) as f:
            qa_data = json.load(f)

        for item in qa_data:
            image_rel_path = item.get("image", "")
            if image_rel_path:
                image_path = category_dir / image_rel_path
                if image_path.exists():
                    samples.append({
                        "image_path": str(image_path),
                        "question": item.get("question", "Is there any anomaly in this image?"),
                        "ground_truth": item.get("answer", ""),
                        "category": category_dir.name,
                        "options": item.get("options", [])
                    })

    if num_samples and num_samples < len(samples):
        random.seed(seed)
        samples = random.sample(samples, num_samples)

    print(f"Loaded {len(samples)} samples from MMAD dataset")
    return samples


def run_diagnostic(model, tokenizer, samples, device="cuda"):
    """Run Moondream 3 on MMAD samples."""
    results = []
    correct = 0

    for sample in tqdm(samples, desc="Processing samples"):
        try:
            image = Image.open(sample["image_path"]).convert("RGB")

            # Format question with options if available
            question = sample["question"]
            if sample["options"]:
                options_str = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(sample["options"])])
                question = f"{question}\n\nOptions:\n{options_str}\n\nAnswer with the letter only."

            # Encode and generate
            enc_image = model.encode_image(image)
            response = model.query(enc_image, question)["answer"]

            # Check correctness
            gt = sample["ground_truth"].strip().upper()
            pred = response.strip().upper()
            is_correct = gt in pred or pred in gt or (len(gt) == 1 and gt == pred[0] if pred else False)

            if is_correct:
                correct += 1

            results.append({
                "image_path": sample["image_path"],
                "category": sample["category"],
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "prediction": response,
                "correct": is_correct
            })

        except Exception as e:
            print(f"Error processing {sample['image_path']}: {e}")
            results.append({
                "image_path": sample["image_path"],
                "category": sample["category"],
                "error": str(e),
                "correct": False
            })

    accuracy = correct / len(samples) if samples else 0
    return results, accuracy


def main():
    parser = argparse.ArgumentParser(description="Moondream 3 MMAD Diagnostic")
    parser.add_argument("--data-dir", type=str, default="data/mmad/images",
                        help="Path to MMAD images directory")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="results/moondream3_diagnostic.json",
                        help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Check GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    print("\nLoading Moondream 3...")
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load samples
    print(f"\nLoading {args.num_samples} samples from MMAD...")
    samples = load_mmad_samples(args.data_dir, args.num_samples, args.seed)

    if not samples:
        print("ERROR: No samples found. Check data directory.")
        return

    # Run diagnostic
    print("\nRunning diagnostic...")
    results, accuracy = run_diagnostic(model, None, samples)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_data = {
        "model": "moondream2 (2025-01-09)",
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(samples),
        "accuracy": accuracy,
        "results": results
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    correct_count = sum(1 for r in results if r.get("correct"))
    print(f"\n{'='*50}")
    print(f"DIAGNOSTIC RESULTS")
    print(f"{'='*50}")
    print(f"Model: Moondream 3 (2025-01-09)")
    print(f"Samples: {len(samples)}")
    print(f"Accuracy: {accuracy:.1%} ({correct_count}/{len(samples)})")
    print(f"Results saved to: {args.output}")

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r.get("correct"):
            categories[cat]["correct"] += 1

    print(f"\nPer-category results:")
    for cat, stats in sorted(categories.items()):
        cat_acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"  {cat}: {cat_acc:.1%} ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    main()
PYTHON_SCRIPT

# Create dataset download script
cat > scripts/download_mmad.py << 'DOWNLOAD_SCRIPT'
#!/usr/bin/env python3
"""Download MMAD dataset from Hugging Face."""

import os

def main():
    print("Downloading MMAD dataset...")
    os.makedirs("data/mmad", exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="jiang-cc/MMAD",
        repo_type="dataset",
        local_dir="data/mmad",
    )
    print("Download complete")

if __name__ == "__main__":
    main()
DOWNLOAD_SCRIPT

# Install Python dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate pillow tqdm huggingface_hub einops

# Download dataset
echo "=== Downloading MMAD dataset ==="
python scripts/download_mmad.py

# Run quick test
echo "=== Running diagnostic with 10 samples ==="
mkdir -p results
python scripts/diagnostic_moondream.py --num-samples 10 --output results/test_10_samples.json

echo ""
echo "=== DONE ==="
echo "Results saved to results/test_10_samples.json"
