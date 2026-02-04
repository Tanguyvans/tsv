#!/usr/bin/env python3
"""
Moondream 3 Anomaly Detection Diagnostic on MVTec AD

Evaluates zero-shot defect detection capability.
"""

import argparse
import json
import os
import random
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def load_mvtec_samples(data_dir: str, num_samples: int = None, seed: int = 42):
    """
    Load samples from MVTec AD dataset (Voxel51/FiftyOne format).

    Parses samples.json which contains:
    - filepath: path to image
    - category: product category (bottle, grid, etc.)
    - defect: defect type ("good" = normal, anything else = anomaly)
    - split: train/test
    """
    data_path = Path(data_dir)
    samples_file = data_path / "samples.json"

    if not samples_file.exists():
        print(f"ERROR: {samples_file} not found")
        return []

    with open(samples_file) as f:
        data = json.load(f)

    samples = []
    for item in data.get("samples", []):
        # Only use test split
        if item.get("split") != "test":
            continue

        filepath = data_path / item["filepath"]
        if not filepath.exists():
            continue

        category = item.get("category", {}).get("label", "unknown")
        defect_label = item.get("defect", {}).get("label", "good")
        is_normal = defect_label == "good"

        samples.append({
            "image_path": str(filepath),
            "category": category,
            "defect_type": defect_label,
            "is_anomaly": not is_normal,
            "label": 0 if is_normal else 1,
        })

    print(f"Found {len(samples)} samples")
    print(f"  Normal: {sum(1 for s in samples if not s['is_anomaly'])}")
    print(f"  Anomaly: {sum(1 for s in samples if s['is_anomaly'])}")

    if num_samples and num_samples < len(samples):
        random.seed(seed)
        # Stratified sampling - keep ratio of normal/anomaly
        normal = [s for s in samples if not s["is_anomaly"]]
        anomaly = [s for s in samples if s["is_anomaly"]]

        n_normal = num_samples // 2
        n_anomaly = num_samples - n_normal

        samples = random.sample(normal, min(n_normal, len(normal))) + \
                  random.sample(anomaly, min(n_anomaly, len(anomaly)))
        random.shuffle(samples)

    return samples


def load_model():
    """Load Moondream 3 model."""
    print("Loading Moondream 3...")

    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to("cuda")

    print(f"Model loaded on {next(model.parameters()).device}")
    return model


def query_moondream(model, image, question):
    """Query Moondream with an image and question."""
    enc_image = model.encode_image(image)
    response = model.query(enc_image, question)["answer"]
    return response


def run_diagnostic(model, samples):
    """Run anomaly detection diagnostic."""
    results = []

    question = "Is there any defect or anomaly in this image? Answer only YES or NO."

    for sample in tqdm(samples, desc="Evaluating"):
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            response = query_moondream(model, image, question)

            # Parse response
            response_lower = response.strip().lower()
            predicted_anomaly = "yes" in response_lower

            correct = predicted_anomaly == sample["is_anomaly"]

            results.append({
                "image_path": sample["image_path"],
                "category": sample["category"],
                "defect_type": sample["defect_type"],
                "ground_truth": sample["is_anomaly"],
                "predicted": predicted_anomaly,
                "response": response,
                "correct": correct,
            })

        except Exception as e:
            print(f"Error: {sample['image_path']}: {e}")
            results.append({
                "image_path": sample["image_path"],
                "category": sample["category"],
                "error": str(e),
                "correct": False,
            })

    return results


def compute_metrics(results, category=None):
    """Compute evaluation metrics."""
    valid = [r for r in results if "error" not in r]
    if category:
        valid = [r for r in valid if r["category"] == category]

    if not valid:
        return {}

    tp = sum(1 for r in valid if r["predicted"] and r["ground_truth"])
    tn = sum(1 for r in valid if not r["predicted"] and not r["ground_truth"])
    fp = sum(1 for r in valid if r["predicted"] and not r["ground_truth"])
    fn = sum(1 for r in valid if not r["predicted"] and r["ground_truth"])

    accuracy = (tp + tn) / len(valid) if valid else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": len(valid),
    }


def compute_per_category_metrics(results):
    """Compute metrics per category."""
    categories = set(r["category"] for r in results if "error" not in r)
    return {cat: compute_metrics(results, cat) for cat in sorted(categories)}


def main():
    parser = argparse.ArgumentParser(description="Moondream 3 MVTec AD Diagnostic")
    parser.add_argument("--data-dir", type=str, default="data/mvtec",
                        help="Path to MVTec AD dataset")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--output", type=str, default="results/mvtec_diagnostic.json",
                        help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Check GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    model = load_model()

    # Load samples
    print(f"\nLoading samples from MVTec AD...")
    samples = load_mvtec_samples(args.data_dir, args.num_samples, args.seed)

    if not samples:
        print("ERROR: No samples found. Check --data-dir path.")
        return

    # Run diagnostic
    print(f"\nRunning diagnostic on {len(samples)} samples...")
    results = run_diagnostic(model, samples)

    # Compute metrics
    metrics = compute_metrics(results)
    per_category = compute_per_category_metrics(results)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_data = {
        "model": "vikhyatk/moondream2 (2025-01-09)",
        "dataset": "MVTec AD",
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(samples),
        "metrics": metrics,
        "per_category": per_category,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Samples: {len(samples)}")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall: {metrics['recall']:.1%}")
    print(f"F1 Score: {metrics['f1']:.1%}")
    print(f"\nConfusion Matrix:")
    print(f"  TP={metrics['tp']} FP={metrics['fp']}")
    print(f"  FN={metrics['fn']} TN={metrics['tn']}")

    # Per-category results
    print(f"\n{'='*50}")
    print("PER-CATEGORY RESULTS")
    print(f"{'='*50}")
    print(f"{'Category':<15} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'N':>5}")
    print("-" * 50)
    for cat, m in sorted(per_category.items()):
        print(f"{cat:<15} {m['accuracy']:>6.1%} {m['precision']:>6.1%} {m['recall']:>6.1%} {m['f1']:>6.1%} {m['total']:>5}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
