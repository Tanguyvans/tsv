#!/usr/bin/env python3
"""Download MVTec AD dataset from Hugging Face."""

from huggingface_hub import snapshot_download
import os

def main():
    os.makedirs("data/mvtec", exist_ok=True)

    print("Downloading MVTec AD dataset...")
    snapshot_download(
        repo_id="Voxel51/mvtec-ad",
        repo_type="dataset",
        local_dir="data/mvtec",
    )
    print("Download complete")

if __name__ == "__main__":
    main()
