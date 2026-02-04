#!/usr/bin/env python3
"""
Moondream 3 Pixel-Level Anomaly Detection Evaluation

Computes pixel AUROC using Moondream's segmentation capabilities.
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
from sklearn.metrics import roc_auc_score


def load_mvtec_samples_with_masks(data_dir: str, num_samples: int = None, seed: int = 42):
    """
    Load samples from MVTec AD dataset with ground truth masks.

    Returns only anomaly samples (which have masks).
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
        # Only use test split anomalies (they have masks)
        if item.get("split") != "test":
            continue

        defect_label = item.get("defect", {}).get("label", "good")
        if defect_label == "good":
            continue  # Skip normal samples - no mask

        filepath = data_path / item["filepath"]
        if not filepath.exists():
            continue

        # Get mask path
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

    # Print available methods for debugging
    methods = [x for x in dir(model) if not x.startswith('_') and callable(getattr(model, x, None))]
    print(f"Available methods: {methods[:20]}...")

    print(f"Model loaded on {next(model.parameters()).device}")
    return model


def get_segmentation_mask(model, image, prompt="defect"):
    """
    Get segmentation mask from Moondream.

    Tries different methods based on what's available.
    """
    # Try different segmentation approaches

    # Method 1: Try segment method
    if hasattr(model, 'segment'):
        try:
            result = model.segment(image, prompt)
            return result
        except Exception as e:
            print(f"segment() failed: {e}")

    # Method 2: Try point/region method
    if hasattr(model, 'point'):
        try:
            result = model.point(image, prompt)
            return result
        except Exception as e:
            print(f"point() failed: {e}")

    # Method 3: Try detect method and convert to mask
    if hasattr(model, 'detect'):
        try:
            result = model.detect(image, prompt)
            return {"type": "detect", "result": result}
        except Exception as e:
            print(f"detect() failed: {e}")

    # Method 4: Use query to ask for defect location
    if hasattr(model, 'query') or hasattr(model, 'encode_image'):
        try:
            enc_image = model.encode_image(image)
            response = model.query(enc_image, f"Point to the {prompt} in this image. Output the coordinates.")
            return {"type": "query", "result": response}
        except Exception as e:
            print(f"query() failed: {e}")

    return None


def bbox_to_mask(bbox, image_size):
    """Convert bounding box to binary mask."""
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.float32)

    if bbox is None:
        return mask

    # bbox format: [x1, y1, x2, y2] normalized or absolute
    x1, y1, x2, y2 = bbox

    # If normalized (0-1), convert to pixels
    if max(x1, y1, x2, y2) <= 1.0:
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
    else:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Clamp to image bounds
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    mask[y1:y2, x1:x2] = 1.0
    return mask


def compute_pixel_auroc(pred_masks, gt_masks):
    """Compute pixel-level AUROC."""
    # Flatten all masks
    pred_flat = np.concatenate([m.flatten() for m in pred_masks])
    gt_flat = np.concatenate([m.flatten() for m in gt_masks])

    # Binarize ground truth
    gt_binary = (gt_flat > 0).astype(int)

    # Check if we have both classes
    if len(np.unique(gt_binary)) < 2:
        print("Warning: Only one class in ground truth")
        return 0.0

    return roc_auc_score(gt_binary, pred_flat)


def run_pixel_evaluation(model, samples, output_dir="results"):
    """Run pixel-level anomaly detection evaluation."""

    pred_masks = []
    gt_masks = []
    results = []

    for sample in tqdm(samples, desc="Evaluating pixel-level"):
        try:
            # Load image
            image = Image.open(sample["image_path"]).convert("RGB")
            img_size = image.size

            # Load ground truth mask
            gt_mask = Image.open(sample["mask_path"]).convert("L")
            gt_mask = np.array(gt_mask.resize(img_size)) / 255.0

            # Get prediction from Moondream
            seg_result = get_segmentation_mask(model, image, "defect")

            # Convert result to mask based on type
            if seg_result is None:
                pred_mask = np.zeros_like(gt_mask)
            elif isinstance(seg_result, dict):
                if seg_result.get("type") == "detect":
                    # Parse detection results
                    det = seg_result.get("result", {})
                    if isinstance(det, dict) and "objects" in det:
                        pred_mask = np.zeros_like(gt_mask)
                        for obj in det["objects"]:
                            bbox = obj.get("bbox", obj.get("box", None))
                            if bbox:
                                pred_mask = np.maximum(pred_mask, bbox_to_mask(bbox, img_size))
                    else:
                        pred_mask = np.zeros_like(gt_mask)
                elif seg_result.get("type") == "query":
                    # Can't easily convert text response to mask
                    pred_mask = np.zeros_like(gt_mask)
                else:
                    pred_mask = np.zeros_like(gt_mask)
            elif isinstance(seg_result, np.ndarray):
                pred_mask = seg_result
            else:
                pred_mask = np.zeros_like(gt_mask)

            # Ensure same shape
            if pred_mask.shape != gt_mask.shape:
                pred_mask = np.array(Image.fromarray(pred_mask).resize(img_size))

            pred_masks.append(pred_mask)
            gt_masks.append(gt_mask)

            # Per-sample IoU
            intersection = np.sum((pred_mask > 0.5) & (gt_mask > 0.5))
            union = np.sum((pred_mask > 0.5) | (gt_mask > 0.5))
            iou = intersection / union if union > 0 else 0

            results.append({
                "image_path": sample["image_path"],
                "category": sample["category"],
                "defect_type": sample["defect_type"],
                "iou": iou,
                "seg_result_type": type(seg_result).__name__ if seg_result else "None",
            })

        except Exception as e:
            print(f"Error processing {sample['image_path']}: {e}")
            results.append({
                "image_path": sample["image_path"],
                "category": sample["category"],
                "error": str(e),
            })

    # Compute metrics
    if pred_masks and gt_masks:
        pixel_auroc = compute_pixel_auroc(pred_masks, gt_masks)
    else:
        pixel_auroc = 0.0

    avg_iou = np.mean([r["iou"] for r in results if "iou" in r])

    return {
        "pixel_auroc": pixel_auroc,
        "avg_iou": avg_iou,
        "num_samples": len(results),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Moondream 3 Pixel-Level Evaluation")
    parser.add_argument("--data-dir", type=str, default="data/mvtec",
                        help="Path to MVTec AD dataset")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--output", type=str, default="results/pixel_auroc.json",
                        help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Check GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model = load_model()

    # Load samples (only anomalies with masks)
    print(f"\nLoading samples from MVTec AD...")
    samples = load_mvtec_samples_with_masks(args.data_dir, args.num_samples, args.seed)

    if not samples:
        print("ERROR: No samples found.")
        return

    # Test segmentation on first sample
    print("\nTesting segmentation on first sample...")
    test_image = Image.open(samples[0]["image_path"]).convert("RGB")
    test_result = get_segmentation_mask(model, test_image, "defect")
    print(f"Segmentation result type: {type(test_result)}")
    print(f"Segmentation result: {test_result}")

    # Run evaluation
    print(f"\nRunning pixel-level evaluation on {len(samples)} samples...")
    eval_results = run_pixel_evaluation(model, samples)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_data = {
        "model": "vikhyatk/moondream2 (2025-01-09)",
        "dataset": "MVTec AD",
        "timestamp": datetime.now().isoformat(),
        "pixel_auroc": eval_results["pixel_auroc"],
        "avg_iou": eval_results["avg_iou"],
        "num_samples": eval_results["num_samples"],
        "results": eval_results["results"],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print("PIXEL-LEVEL RESULTS")
    print(f"{'='*50}")
    print(f"Samples: {eval_results['num_samples']}")
    print(f"Pixel AUROC: {eval_results['pixel_auroc']:.1%}")
    print(f"Average IoU: {eval_results['avg_iou']:.1%}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
