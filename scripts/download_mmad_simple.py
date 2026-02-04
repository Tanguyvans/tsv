#!/usr/bin/env python3
from huggingface_hub import snapshot_download, list_repo_files
import os

os.makedirs("data/mmad", exist_ok=True)

print("Checking dataset contents...")
files = list_repo_files("jiang-cc/MMAD", repo_type="dataset")
print(f"Found {len(files)} files")
print("First 10:", files[:10])

print("\nDownloading...")
snapshot_download(
    repo_id="jiang-cc/MMAD",
    repo_type="dataset",
    local_dir="data/mmad",
)
print("Done")
