"""
Moondream 3 Anomaly Detection Diagnostic Script

This script evaluates Moondream 3's capability to detect industrial anomalies
using the MMAD benchmark dataset.

Usage:
    python diagnostic_moondream.py --output results/diagnostic_results.json

Requirements:
    pip install torch transformers datasets pandas pillow tqdm

References:
    - Moondream 3: https://huggingface.co/moondream/moondream3-preview
    - MMAD Dataset: https://huggingface.co/datasets/jiang-cc/MMAD
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

MMAD_DATASET_ID = "jiang-cc/MMAD"
MOONDREAM_MODEL_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2025-01-09"  # Moondream 3

MVTEC_SUBSET_CATEGORIES = [
    "bottle", "cable", "capsule", "metal_nut",
    "screw", "carpet", "grid", "leather"
]

# Diagnostic test thresholds
THRESHOLDS = {
    "binary_accuracy": 0.80,      # >80% accuracy on defect detection
    "hallucination_rate": 0.10,   # <10% false positives on normal images
    "miss_rate": 0.15,            # <15% false negatives on anomaly images
    "description_match": 0.70,    # >70% correct defect type identification
}


# =============================================================================
# Data Download & Preparation
# =============================================================================

def download_mmad_dataset(output_dir: str = "data/mmad") -> pd.DataFrame:
    """Download MMAD dataset from HuggingFace."""
    from datasets import load_dataset

    output_path = Path(output_dir)
    parquet_file = output_path / "mmad_mvtec_subset.parquet"

    # Check if already downloaded
    if parquet_file.exists():
        print(f"Loading cached dataset from {parquet_file}")
        return pd.read_parquet(parquet_file)

    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading MMAD dataset from HuggingFace...")
    dataset = load_dataset(MMAD_DATASET_ID, split="train")
    df = dataset.to_pandas()

    # Extract category from path
    def extract_category(path):
        parts = path.split("/")
        return parts[1].lower() if len(parts) >= 2 else None

    df["category"] = df["query_image"].apply(extract_category)

    # Filter for MVTec sources and our subset
    df_mvtec = df[df["query_image"].str.contains("MVTec", case=False)]
    df_subset = df_mvtec[df_mvtec["category"].isin(MVTEC_SUBSET_CATEGORIES)]

    # Classify question types
    def classify_question(q):
        q_lower = q.lower()
        if "is there any defect" in q_lower:
            return "defect_detection"
        elif "type of" in q_lower:
            return "defect_type"
        elif "where" in q_lower:
            return "location"
        elif "appearance" in q_lower:
            return "appearance"
        elif "compare" in q_lower:
            return "comparison"
        else:
            return "other"

    df_subset = df_subset.copy()
    df_subset["question_type"] = df_subset["question"].apply(classify_question)

    # Save
    df_subset.to_parquet(parquet_file)
    print(f"Saved {len(df_subset)} samples to {parquet_file}")

    return df_subset


def download_mmad_images(output_dir: str = "data/mmad"):
    """Download MMAD images if not present."""
    images_dir = Path(output_dir) / "images"

    if (images_dir / "DS-MVTec").exists():
        print("Images already downloaded")
        return images_dir

    print("Downloading MMAD images (~28GB)...")
    print("This may take a while...")

    images_dir.mkdir(parents=True, exist_ok=True)

    zip_url = "https://huggingface.co/datasets/jiang-cc/MMAD/resolve/refs%2Fpr%2F1/ALL_DATA.zip?download=true"
    zip_path = Path(output_dir) / "ALL_DATA.zip"

    # Download
    import subprocess
    subprocess.run([
        "wget", "-O", str(zip_path), zip_url
    ], check=True)

    # Extract
    subprocess.run([
        "unzip", "-q", str(zip_path), "-d", str(images_dir)
    ], check=True)

    print(f"Images extracted to {images_dir}")
    return images_dir


def get_image_path(query_image: str, images_dir: Path) -> Optional[Path]:
    """Resolve image path from MMAD query_image field."""
    # query_image format: "DS-MVTec/bottle/image/broken_large/000.png"
    full_path = images_dir / query_image
    if full_path.exists():
        return full_path
    return None


# =============================================================================
# Model Loading
# =============================================================================

def load_moondream3(device: str = "cuda"):
    """Load Moondream 3 model."""
    from transformers import AutoModelForCausalLM

    print(f"Loading Moondream 3 from {MOONDREAM_MODEL_ID} (rev: {MOONDREAM_REVISION})...")

    model = AutoModelForCausalLM.from_pretrained(
        MOONDREAM_MODEL_ID,
        revision=MOONDREAM_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": device},
    )

    print("Moondream 3 loaded successfully")
    return model


# =============================================================================
# Diagnostic Tests
# =============================================================================

def parse_mcq_answer(response: str, options: str) -> str:
    """
    Parse model response to extract MCQ answer (A, B, C, or D).
    """
    response_upper = response.upper().strip()

    # Direct answer patterns
    patterns = [
        r'^([A-D])\b',           # Starts with A, B, C, D
        r'ANSWER[:\s]*([A-D])',  # "Answer: A"
        r'\b([A-D])\s*[:\.]',    # "A:" or "A."
        r'OPTION\s*([A-D])',     # "Option A"
    ]

    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match:
            return match.group(1)

    # Check if response contains option text
    option_lines = options.split("\n")
    for line in option_lines:
        if line.strip():
            letter = line[0].upper()
            option_text = line[2:].strip().lower()  # Remove "A: " prefix
            if option_text in response.lower():
                return letter

    # Default: try to infer from yes/no
    if "yes" in response.lower():
        # Find which option says "Yes"
        for line in option_lines:
            if "yes" in line.lower():
                return line[0].upper()
    if "no" in response.lower():
        for line in option_lines:
            if "no" in line.lower():
                return line[0].upper()

    return "UNKNOWN"


def test_defect_detection(
    model,
    df: pd.DataFrame,
    images_dir: Path,
    max_samples: int = None,
    save_details: bool = True
) -> dict:
    """
    Test 1: Binary classification - Is there a defect?

    Uses MMAD "defect_detection" questions.
    """
    print("\n" + "="*60)
    print("TEST 1: Defect Detection (Binary Classification)")
    print("="*60)

    df_detection = df[df["question_type"] == "defect_detection"].copy()

    if max_samples:
        df_detection = df_detection.head(max_samples)

    print(f"Running on {len(df_detection)} samples...")

    results = []
    correct = 0

    for idx, row in tqdm(df_detection.iterrows(), total=len(df_detection)):
        img_path = get_image_path(row["query_image"], images_dir)

        if img_path is None:
            continue

        try:
            image = Image.open(img_path).convert("RGB")

            # Query Moondream (encode image first, then query)
            prompt = row["question"] + "\n" + row["options"] + "\nAnswer with the letter only."
            enc_image = model.encode_image(image)
            response = model.query(enc_image, prompt)["answer"]

            predicted = parse_mcq_answer(response, row["options"])
            gt_answer = row["answer"].strip().upper()

            is_correct = (predicted == gt_answer)
            if is_correct:
                correct += 1

            results.append({
                "image": row["query_image"],
                "category": row["category"],
                "question": row["question"],
                "options": row["options"],
                "gt_answer": gt_answer,
                "predicted": predicted,
                "raw_response": response,
                "correct": is_correct
            })

        except Exception as e:
            print(f"Error processing {row['query_image']}: {e}")
            continue

    accuracy = correct / len(results) if results else 0

    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"Threshold: {THRESHOLDS['binary_accuracy']:.0%}")
    print(f"Status: {'PASS' if accuracy >= THRESHOLDS['binary_accuracy'] else 'FAIL'}")

    return {
        "test_name": "defect_detection",
        "metric": "accuracy",
        "value": accuracy,
        "correct": correct,
        "total": len(results),
        "threshold": THRESHOLDS["binary_accuracy"],
        "passed": accuracy >= THRESHOLDS["binary_accuracy"],
        "details": results if save_details else None
    }


def test_hallucination_rate(
    model,
    df: pd.DataFrame,
    images_dir: Path,
    max_samples: int = None,
    save_details: bool = True
) -> dict:
    """
    Test 2: Hallucination Rate - False positives on normal images.

    Uses MMAD "defect_detection" questions where answer is "No" (normal images).
    """
    print("\n" + "="*60)
    print("TEST 2: Hallucination Rate (False Positives)")
    print("="*60)

    df_detection = df[df["question_type"] == "defect_detection"].copy()

    # Filter for normal images (ground truth answer indicates no defect)
    # In MMAD, "No" answer means normal
    df_normal = df_detection[
        df_detection.apply(
            lambda row: "no" in row["options"].split("\n")[
                ord(row["answer"].upper()) - ord("A")
            ].lower(),
            axis=1
        )
    ]

    if max_samples:
        df_normal = df_normal.head(max_samples)

    print(f"Running on {len(df_normal)} normal image samples...")

    results = []
    false_positives = 0

    for idx, row in tqdm(df_normal.iterrows(), total=len(df_normal)):
        img_path = get_image_path(row["query_image"], images_dir)

        if img_path is None:
            continue

        try:
            image = Image.open(img_path).convert("RGB")

            prompt = row["question"] + "\n" + row["options"] + "\nAnswer with the letter only."
            enc_image = model.encode_image(image)
            response = model.query(enc_image, prompt)["answer"]

            predicted = parse_mcq_answer(response, row["options"])
            gt_answer = row["answer"].strip().upper()

            # Hallucination = predicted defect when there is none
            is_hallucination = (predicted != gt_answer)
            if is_hallucination:
                false_positives += 1

            results.append({
                "image": row["query_image"],
                "category": row["category"],
                "gt_answer": gt_answer,
                "predicted": predicted,
                "raw_response": response,
                "hallucination": is_hallucination
            })

        except Exception as e:
            print(f"Error processing {row['query_image']}: {e}")
            continue

    rate = false_positives / len(results) if results else 0

    print(f"\nHallucination Rate: {rate:.2%} ({false_positives}/{len(results)})")
    print(f"Threshold: <{THRESHOLDS['hallucination_rate']:.0%}")
    print(f"Status: {'PASS' if rate <= THRESHOLDS['hallucination_rate'] else 'FAIL'}")

    return {
        "test_name": "hallucination_rate",
        "metric": "false_positive_rate",
        "value": rate,
        "false_positives": false_positives,
        "total": len(results),
        "threshold": THRESHOLDS["hallucination_rate"],
        "passed": rate <= THRESHOLDS["hallucination_rate"],
        "details": results if save_details else None
    }


def test_miss_rate(
    model,
    df: pd.DataFrame,
    images_dir: Path,
    max_samples: int = None,
    save_details: bool = True
) -> dict:
    """
    Test 3: Miss Rate - False negatives on anomaly images.

    Uses MMAD "defect_detection" questions where answer is "Yes" (anomaly images).
    """
    print("\n" + "="*60)
    print("TEST 3: Miss Rate (False Negatives)")
    print("="*60)

    df_detection = df[df["question_type"] == "defect_detection"].copy()

    # Filter for anomaly images (ground truth answer indicates defect)
    df_anomaly = df_detection[
        df_detection.apply(
            lambda row: "yes" in row["options"].split("\n")[
                ord(row["answer"].upper()) - ord("A")
            ].lower(),
            axis=1
        )
    ]

    if max_samples:
        df_anomaly = df_anomaly.head(max_samples)

    print(f"Running on {len(df_anomaly)} anomaly image samples...")

    results = []
    misses = 0

    for idx, row in tqdm(df_anomaly.iterrows(), total=len(df_anomaly)):
        img_path = get_image_path(row["query_image"], images_dir)

        if img_path is None:
            continue

        try:
            image = Image.open(img_path).convert("RGB")

            prompt = row["question"] + "\n" + row["options"] + "\nAnswer with the letter only."
            enc_image = model.encode_image(image)
            response = model.query(enc_image, prompt)["answer"]

            predicted = parse_mcq_answer(response, row["options"])
            gt_answer = row["answer"].strip().upper()

            # Miss = predicted no defect when there is one
            is_miss = (predicted != gt_answer)
            if is_miss:
                misses += 1

            results.append({
                "image": row["query_image"],
                "category": row["category"],
                "gt_answer": gt_answer,
                "predicted": predicted,
                "raw_response": response,
                "missed": is_miss
            })

        except Exception as e:
            print(f"Error processing {row['query_image']}: {e}")
            continue

    rate = misses / len(results) if results else 0

    print(f"\nMiss Rate: {rate:.2%} ({misses}/{len(results)})")
    print(f"Threshold: <{THRESHOLDS['miss_rate']:.0%}")
    print(f"Status: {'PASS' if rate <= THRESHOLDS['miss_rate'] else 'FAIL'}")

    return {
        "test_name": "miss_rate",
        "metric": "false_negative_rate",
        "value": rate,
        "misses": misses,
        "total": len(results),
        "threshold": THRESHOLDS["miss_rate"],
        "passed": rate <= THRESHOLDS["miss_rate"],
        "details": results if save_details else None
    }


def test_defect_type_classification(
    model,
    df: pd.DataFrame,
    images_dir: Path,
    max_samples: int = None,
    save_details: bool = True
) -> dict:
    """
    Test 4: Defect Type Classification - Can it identify the type of defect?
    """
    print("\n" + "="*60)
    print("TEST 4: Defect Type Classification")
    print("="*60)

    df_type = df[df["question_type"] == "defect_type"].copy()

    if max_samples:
        df_type = df_type.head(max_samples)

    print(f"Running on {len(df_type)} samples...")

    results = []
    correct = 0

    for idx, row in tqdm(df_type.iterrows(), total=len(df_type)):
        img_path = get_image_path(row["query_image"], images_dir)

        if img_path is None:
            continue

        try:
            image = Image.open(img_path).convert("RGB")

            prompt = row["question"] + "\n" + row["options"] + "\nAnswer with the letter only."
            enc_image = model.encode_image(image)
            response = model.query(enc_image, prompt)["answer"]

            predicted = parse_mcq_answer(response, row["options"])
            gt_answer = row["answer"].strip().upper()

            is_correct = (predicted == gt_answer)
            if is_correct:
                correct += 1

            results.append({
                "image": row["query_image"],
                "category": row["category"],
                "question": row["question"],
                "options": row["options"],
                "gt_answer": gt_answer,
                "predicted": predicted,
                "raw_response": response,
                "correct": is_correct
            })

        except Exception as e:
            print(f"Error processing {row['query_image']}: {e}")
            continue

    accuracy = correct / len(results) if results else 0

    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"Threshold: {THRESHOLDS['description_match']:.0%}")
    print(f"Status: {'PASS' if accuracy >= THRESHOLDS['description_match'] else 'FAIL'}")

    return {
        "test_name": "defect_type_classification",
        "metric": "accuracy",
        "value": accuracy,
        "correct": correct,
        "total": len(results),
        "threshold": THRESHOLDS["description_match"],
        "passed": accuracy >= THRESHOLDS["description_match"],
        "details": results if save_details else None
    }


def test_defect_location(
    model,
    df: pd.DataFrame,
    images_dir: Path,
    max_samples: int = None,
    save_details: bool = True
) -> dict:
    """
    Test 5: Defect Location - Can it localize the defect?
    """
    print("\n" + "="*60)
    print("TEST 5: Defect Location")
    print("="*60)

    df_loc = df[df["question_type"] == "location"].copy()

    if max_samples:
        df_loc = df_loc.head(max_samples)

    print(f"Running on {len(df_loc)} samples...")

    results = []
    correct = 0

    for idx, row in tqdm(df_loc.iterrows(), total=len(df_loc)):
        img_path = get_image_path(row["query_image"], images_dir)

        if img_path is None:
            continue

        try:
            image = Image.open(img_path).convert("RGB")

            prompt = row["question"] + "\n" + row["options"] + "\nAnswer with the letter only."
            enc_image = model.encode_image(image)
            response = model.query(enc_image, prompt)["answer"]

            predicted = parse_mcq_answer(response, row["options"])
            gt_answer = row["answer"].strip().upper()

            is_correct = (predicted == gt_answer)
            if is_correct:
                correct += 1

            results.append({
                "image": row["query_image"],
                "category": row["category"],
                "gt_answer": gt_answer,
                "predicted": predicted,
                "raw_response": response,
                "correct": is_correct
            })

        except Exception as e:
            print(f"Error processing {row['query_image']}: {e}")
            continue

    accuracy = correct / len(results) if results else 0

    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"(No threshold for location - informational only)")

    return {
        "test_name": "defect_location",
        "metric": "accuracy",
        "value": accuracy,
        "correct": correct,
        "total": len(results),
        "threshold": None,
        "passed": None,  # Informational
        "details": results if save_details else None
    }


# =============================================================================
# Main
# =============================================================================

def run_diagnostic(
    data_dir: str = "data/mmad",
    output_file: str = "results/diagnostic_results.json",
    max_samples_per_test: int = None,
    save_details: bool = True,
    device: str = "cuda"
):
    """Run full diagnostic suite."""

    start_time = time.time()

    # Setup
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download data
    print("\n" + "="*60)
    print("PHASE 1: Data Preparation")
    print("="*60)

    df = download_mmad_dataset(data_dir)
    images_dir = download_mmad_images(data_dir)

    print(f"\nDataset: {len(df)} samples")
    print(f"Categories: {df['category'].unique().tolist()}")
    print(f"Question types: {df['question_type'].value_counts().to_dict()}")

    # Load model
    print("\n" + "="*60)
    print("PHASE 2: Model Loading")
    print("="*60)

    model = load_moondream3(device=device)

    # Run tests
    print("\n" + "="*60)
    print("PHASE 3: Diagnostic Tests")
    print("="*60)

    results = {
        "model": MOONDREAM_MODEL_ID,
        "dataset": "MMAD (MVTec subset)",
        "categories": MVTEC_SUBSET_CATEGORIES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_samples_per_test": max_samples_per_test,
        "tests": {}
    }

    # Test 1: Defect Detection
    results["tests"]["defect_detection"] = test_defect_detection(
        model, df, images_dir, max_samples_per_test, save_details
    )

    # Test 2: Hallucination Rate
    results["tests"]["hallucination_rate"] = test_hallucination_rate(
        model, df, images_dir, max_samples_per_test, save_details
    )

    # Test 3: Miss Rate
    results["tests"]["miss_rate"] = test_miss_rate(
        model, df, images_dir, max_samples_per_test, save_details
    )

    # Test 4: Defect Type Classification
    results["tests"]["defect_type"] = test_defect_type_classification(
        model, df, images_dir, max_samples_per_test, save_details
    )

    # Test 5: Defect Location
    results["tests"]["defect_location"] = test_defect_location(
        model, df, images_dir, max_samples_per_test, save_details
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed_tests = sum(
        1 for t in results["tests"].values()
        if t["passed"] is True
    )
    total_graded = sum(
        1 for t in results["tests"].values()
        if t["passed"] is not None
    )

    results["summary"] = {
        "passed": passed_tests,
        "total": total_graded,
        "pass_rate": passed_tests / total_graded if total_graded > 0 else 0,
        "recommendation": get_recommendation(passed_tests, total_graded),
        "elapsed_time": time.time() - start_time
    }

    print(f"\nTests Passed: {passed_tests}/{total_graded}")
    print(f"Recommendation: {results['summary']['recommendation']}")
    print(f"Elapsed Time: {results['summary']['elapsed_time']:.1f}s")

    # Per-test summary
    print("\nPer-test results:")
    for name, test in results["tests"].items():
        status = "PASS" if test["passed"] else ("FAIL" if test["passed"] is False else "INFO")
        print(f"  {name}: {test['value']:.2%} [{status}]")

    # Save results
    # Remove details for summary file (too large)
    results_summary = json.loads(json.dumps(results))
    for test in results_summary["tests"].values():
        if "details" in test:
            test["details"] = f"[{len(test.get('details', []))} items - see full results]"

    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Save full results with details
    if save_details:
        full_output = output_path.with_suffix(".full.json")
        with open(full_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Full results saved to: {full_output}")

    return results


def get_recommendation(passed: int, total: int) -> str:
    """Get recommendation based on test results."""
    if total == 0:
        return "ERROR: No tests completed"

    pass_rate = passed / total

    if pass_rate >= 0.8:
        return "PROCEED_WITH_FINETUNING - Moondream shows good anomaly awareness"
    elif pass_rate >= 0.5:
        return "INTENSIVE_FINETUNING_NEEDED - Moondream needs significant training"
    else:
        return "CONSIDER_AA_CLIP - Moondream may not be suitable for this task"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Moondream 3 anomaly detection diagnostic"
    )
    parser.add_argument(
        "--data-dir",
        default="data/mmad",
        help="Directory for MMAD dataset"
    )
    parser.add_argument(
        "--output",
        default="results/diagnostic_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per test (for quick testing)"
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Don't save per-sample details"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (cuda/cpu)"
    )

    args = parser.parse_args()

    run_diagnostic(
        data_dir=args.data_dir,
        output_file=args.output,
        max_samples_per_test=args.max_samples,
        save_details=not args.no_details,
        device=args.device
    )
