"""Télécharge le dataset GERALD (German Railway Signaling).

Repo : https://github.com/ifs-rwth-aachen/GERALD
Dataset hosted on the lab's data server (link in the repo README).

Si le clone ne contient pas les images (datasets externes courants), affiche
un message indiquant à l'utilisateur de les télécharger manuellement vers
data/gerald/ avec la structure attendue:
  data/gerald/JPEGImages/
  data/gerald/Annotations/  (PASCAL VOC XML)
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/ifs-rwth-aachen/GERALD.git"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="data/_raw/gerald")
    args = ap.parse_args()
    dst = Path(args.dst)
    if dst.exists():
        print(f"{dst} already exists")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(dst)], check=True)

    images = list((dst).rglob("*.jpg")) + list((dst).rglob("*.png"))
    annots = list((dst).rglob("*.xml"))
    print(f"Found {len(images)} images, {len(annots)} XML annotations")
    if not images:
        print("\n⚠ Pas d'images dans le repo cloné — GERALD héberge les images séparément.")
        print("  Suivre les instructions du README de https://github.com/ifs-rwth-aachen/GERALD")
        print("  puis placer JPEGImages/ et Annotations/ sous data/gerald/")


if __name__ == "__main__":
    main()
