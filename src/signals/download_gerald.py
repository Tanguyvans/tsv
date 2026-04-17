"""Télécharge le dataset GERALD (German Railway Lightsignal Dataset).

Hébergé sur RWTH Aachen : https://publications.rwth-aachen.de/record/980030
Archive : GERALD.zip (~4.2 GB, 5000 images + annotations PASCAL VOC XML)

Usage:
  python src/signals/download_gerald.py
"""
from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

DATASET_URL = "https://publications.rwth-aachen.de/record/980030/files/GERALD.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="data/_raw/gerald_dataset")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    zip_path = dst / "GERALD.zip"

    if not zip_path.exists():
        print(f"Downloading {DATASET_URL} → {zip_path} (~4.2 GB)...")
        subprocess.run(["curl", "-L", "-o", str(zip_path), DATASET_URL], check=True)
    else:
        print(f"{zip_path} already exists, skipping download")

    if not args.no_extract:
        print(f"Extracting → {dst}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dst)

    images = list(dst.rglob("*.jpg")) + list(dst.rglob("*.png"))
    annots = list(dst.rglob("*.xml"))
    print(f"Found {len(images)} images, {len(annots)} XML annotations in {dst}")


if __name__ == "__main__":
    main()
