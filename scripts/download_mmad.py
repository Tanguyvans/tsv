"""
Download and prepare MMAD dataset for Moondream diagnostic.

MMAD: Multi-Modal Anomaly Detection benchmark
Source: https://huggingface.co/datasets/jiang-cc/MMAD
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path
import json

# Our target subset categories (MVTec AD)
MVTEC_SUBSET = [
    "bottle",
    "cable",
    "capsule",
    "metal_nut",
    "screw",
    "carpet",
    "grid",
    "leather",
]

def download_mmad(output_dir: str = "data/mmad", subset_only: bool = True):
    """
    Download MMAD dataset from HuggingFace.

    Args:
        output_dir: Directory to save the processed dataset
        subset_only: If True, only keep MVTec AD subset categories
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading MMAD dataset from HuggingFace...")
    dataset = load_dataset("jiang-cc/MMAD", split="train")

    print(f"Total samples: {len(dataset)}")

    # Convert to pandas for easier filtering
    df = dataset.to_pandas()

    # Extract category from query_image path
    # Format: "DS-MVTec/bottle/image/broken_large/000.png" or "MVTec-AD/bottle/train/good/001.png"
    def extract_category(path):
        parts = path.split("/")
        if len(parts) >= 2:
            return parts[1].lower()
        return None

    df["category"] = df["query_image"].apply(extract_category)

    # Filter for MVTec sources
    df_mvtec = df[df["query_image"].str.contains("MVTec", case=False)]

    print(f"\nMVTec samples: {len(df_mvtec)}")
    print(f"Categories found: {df_mvtec['category'].unique().tolist()}")

    if subset_only:
        # Filter for our subset categories
        df_subset = df_mvtec[df_mvtec["category"].isin(MVTEC_SUBSET)]
        print(f"\nSubset samples ({len(MVTEC_SUBSET)} categories): {len(df_subset)}")
    else:
        df_subset = df_mvtec

    # Analyze question types
    print("\n--- Question Analysis ---")

    # Identify question types by keywords
    def classify_question(q):
        q_lower = q.lower()
        if "is there any defect" in q_lower or "any defect" in q_lower:
            return "defect_detection"
        elif "type of" in q_lower or "what type" in q_lower:
            return "defect_type"
        elif "where" in q_lower or "location" in q_lower:
            return "location"
        elif "appearance" in q_lower or "describe" in q_lower or "look like" in q_lower:
            return "appearance"
        elif "compare" in q_lower or "difference" in q_lower:
            return "comparison"
        else:
            return "other"

    df_subset = df_subset.copy()
    df_subset["question_type"] = df_subset["question"].apply(classify_question)

    print("\nQuestion type distribution:")
    print(df_subset["question_type"].value_counts())

    print("\nSamples per category:")
    print(df_subset["category"].value_counts())

    # Save processed dataset
    output_file = output_path / "mmad_mvtec_subset.parquet"
    df_subset.to_parquet(output_file)
    print(f"\nSaved to: {output_file}")

    # Save metadata
    metadata = {
        "total_samples": len(df_subset),
        "categories": df_subset["category"].unique().tolist(),
        "question_types": df_subset["question_type"].value_counts().to_dict(),
        "samples_per_category": df_subset["category"].value_counts().to_dict(),
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_file}")

    # Save a few examples for inspection
    examples_file = output_path / "examples.json"
    examples = df_subset.head(10).to_dict(orient="records")
    with open(examples_file, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Examples saved to: {examples_file}")

    return df_subset


def check_image_availability(df, mmad_images_dir: str = None):
    """
    Check if MMAD images need to be downloaded separately.

    Note: MMAD dataset on HuggingFace may only contain paths, not actual images.
    Images might need to be downloaded from original sources (MVTec AD).
    """
    print("\n--- Image Availability Check ---")

    sample_paths = df["query_image"].head(5).tolist()
    print("Sample image paths:")
    for p in sample_paths:
        print(f"  {p}")

    print("\nNote: MMAD may require downloading images from original sources.")
    print("Options:")
    print("1. Download MVTec AD from https://www.mvtec.com/company/research/datasets/mvtec-ad")
    print("2. Check if MMAD GitHub has image download instructions")
    print("3. Use Voxel51/mvtec-ad HuggingFace dataset for images")

    return sample_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MMAD dataset")
    parser.add_argument("--output", default="data/mmad", help="Output directory")
    parser.add_argument("--full", action="store_true", help="Download full dataset (not just subset)")
    args = parser.parse_args()

    df = download_mmad(output_dir=args.output, subset_only=not args.full)
    check_image_availability(df)
