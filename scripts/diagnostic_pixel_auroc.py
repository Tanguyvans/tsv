#!/usr/bin/env python3
"""
Moondream 3 Pixel-Level Anomaly Detection Evaluation

Computes pixel AUROC using Moondream's point predictions converted to Gaussian blobs.
"""

import argparse
import json
import os
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def load_mvtec_samples_with_masks(data_dir: str, num_samples: int = None, seed: int = 42):
    """Load samples from MVTec AD with ground truth masks (anomaly samples only)."""
    data_path = Path(data_dir)
    samples_file = data_path / "samples.json"

    if not samples_file.exists():
        print(f"ERROR: {samples_file} not found")
        return []

    with open(samples_file) as f:
        data = json.load(f)

    samples = []
    for item in data.get("samples", []):
        if item.get("split") != "test":
            continue

        defect_label = item.get("defect", {}).get("label", "good")
        if defect_label == "good":
            continue

        filepath = data_path / item["filepath"]
        if not filepath.exists():
            continue

        mask_info = item.get("defect_mask", {})
        mask_path = mask_info.get("mask_path", "")
        if mask_path:
            mask_full_path = data_path / mask_path
            if not mask_full_path.exists():
                continue
        else:
            continue

        category = item.get("category", {}).get("label", "unknown")
        samples.append({
            "image_path": str(filepath),
            "mask_path": str(mask_full_path),
            "category": category,
            "defect_type": defect_label,
        })

    print(f"Found {len(samples)} anomaly samples with masks")

    if num_samples and num_samples < len(samples):
        random.seed(seed)
        samples = random.sample(samples, num_samples)

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


def point_to_gaussian(point, image_size, sigma=30):
    """Convert a point to a Gaussian blob heatmap."""
    w, h = image_size
    x_center = int(point['x'] * w)
    y_center = int(point['y'] * h)

    # Create coordinate grids
    y_grid, x_grid = np.ogrid[:h, :w]

    # Gaussian blob
    heatmap = np.exp(-((x_grid - x_center)**2 + (y_grid - y_center)**2) / (2 * sigma**2))

    return heatmap.astype(np.float32)


def get_point_prediction(model, image):
    """Get point prediction from Moondream."""
    try:
        enc_image = model.encode_image(image)
        result = model.point(enc_image, "defect")
        return result
    except Exception as e:
        return None


def compute_distance_to_mask(point, gt_mask):
    """Compute distance from predicted point to nearest defect pixel."""
    if point is None:
        return float('inf')

    h, w = gt_mask.shape
    x = int(point['x'] * w)
    y = int(point['y'] * h)

    # Check if point is inside defect region
    if gt_mask[min(y, h-1), min(x, w-1)] > 0:
        return 0.0

    # Find distance to nearest defect pixel
    defect_coords = np.argwhere(gt_mask > 0)
    if len(defect_coords) == 0:
        return float('inf')

    distances = np.sqrt((defect_coords[:, 0] - y)**2 + (defect_coords[:, 1] - x)**2)
    return float(np.min(distances))


def run_evaluation(model, samples, sigma=30):
    """Run pixel-level evaluation with incremental saving."""
    results = []

    # Metrics accumulators
    total_hit = 0  # Point inside defect region
    total_distance = 0
    valid_samples = 0

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            img_size = image.size

            gt_mask = Image.open(sample["mask_path"]).convert("L")
            gt_mask = np.array(gt_mask.resize(img_size)) / 255.0

            # Get point prediction
            pred = get_point_prediction(model, image)

            if pred and 'points' in pred and len(pred['points']) > 0:
                point = pred['points'][0]
                distance = compute_distance_to_mask(point, gt_mask)
                hit = distance == 0

                total_distance += distance
                total_hit += int(hit)
                valid_samples += 1

                results.append({
                    "image_path": sample["image_path"],
                    "category": sample["category"],
                    "defect_type": sample["defect_type"],
                    "predicted_point": point,
                    "distance_to_defect": distance,
                    "hit": hit,
                })
            else:
                results.append({
                    "image_path": sample["image_path"],
                    "category": sample["category"],
                    "error": "No point predicted",
                })

        except Exception as e:
            results.append({
                "image_path": sample["image_path"],
                "category": sample["category"],
                "error": str(e),
            })

        # Print progress every 50 samples
        if (i + 1) % 50 == 0:
            if valid_samples > 0:
                print(f"  Progress: {i+1}/{len(samples)} | Hit rate: {total_hit/valid_samples:.1%} | Avg dist: {total_distance/valid_samples:.1f}px")

    # Compute final metrics
    metrics = {}
    if valid_samples > 0:
        metrics = {
            "hit_rate": total_hit / valid_samples,
            "avg_distance": total_distance / valid_samples,
            "valid_samples": valid_samples,
            "total_samples": len(samples),
        }

    return results, metrics


def compute_per_category_metrics(results):
    """Compute metrics per category."""
    categories = set(r["category"] for r in results if "error" not in r)
    per_cat = {}

    for cat in sorted(categories):
        cat_results = [r for r in results if r.get("category") == cat and "error" not in r]
        if cat_results:
            hits = sum(1 for r in cat_results if r.get("hit", False))
            distances = [r["distance_to_defect"] for r in cat_results]
            per_cat[cat] = {
                "hit_rate": hits / len(cat_results),
                "avg_distance": sum(distances) / len(distances),
                "count": len(cat_results),
            }

    return per_cat


def main():
    parser = argparse.ArgumentParser(description="Moondream 3 Pixel-Level Evaluation")
    parser.add_argument("--data-dir", type=str, default="data/mvtec")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="results/pixel_auroc.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=int, default=30, help="Gaussian blob sigma")
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = load_model()

    print(f"\nLoading samples from MVTec AD...")
    samples = load_mvtec_samples_with_masks(args.data_dir, args.num_samples, args.seed)

    if not samples:
        print("ERROR: No samples found.")
        return

    print(f"\nRunning evaluation on {len(samples)} samples...")
    results, metrics = run_evaluation(model, samples, args.sigma)

    # Per-category metrics
    per_category = compute_per_category_metrics(results)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_data = {
        "model": "vikhyatk/moondream2 (2025-01-09)",
        "dataset": "MVTec AD",
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "per_category": per_category,
        "num_results": len(results),
    }

    # Save summary first (without full results to avoid large file issues)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print("PIXEL-LEVEL RESULTS")
    print(f"{'='*50}")
    print(f"Samples: {metrics.get('valid_samples', 0)}/{metrics.get('total_samples', 0)}")
    print(f"Hit Rate: {metrics.get('hit_rate', 0):.1%} (point inside defect region)")
    print(f"Avg Distance: {metrics.get('avg_distance', 0):.1f} pixels")

    print(f"\n{'='*50}")
    print("PER-CATEGORY RESULTS")
    print(f"{'='*50}")
    print(f"{'Category':<15} {'Hit%':>8} {'AvgDist':>10} {'N':>6}")
    print("-" * 45)
    for cat, m in sorted(per_category.items()):
        print(f"{cat:<15} {m['hit_rate']:>7.1%} {m['avg_distance']:>9.1f}px {m['count']:>6}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
