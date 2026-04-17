"""Extrait les crops de mâts/poteaux annotés dans GERALD pour entraîner le stage 2."""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

MAST_NAMES = {"mast", "Mast", "pole", "Pole"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/_raw/gerald_dataset/GERALD/dataset")
    ap.add_argument("--out", default="data/gerald_augmented/has_panel")
    args = ap.parse_args()

    src = Path(args.src)
    images_dir = next((d for d in src.rglob("JPEGImages") if d.is_dir()), src)
    annots_dir = next((d for d in src.rglob("Annotations") if d.is_dir()), None)
    if annots_dir is None:
        print("No Annotations dir"); return

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    n = 0
    for xml in annots_dir.glob("*.xml"):
        tree = ET.parse(xml); root = tree.getroot()
        img_path = images_dir / f"{xml.stem}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"{xml.stem}.png"
            if not img_path.exists():
                continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for i, obj in enumerate(root.findall("object")):
            name = obj.find("name").text
            # heuristic: extract any object whose name contains "signal" or matches mast set
            if not (name in MAST_NAMES or "signal" in name.lower() or "sign" in name.lower()):
                continue
            b = obj.find("bndbox")
            x1, y1 = int(float(b.find("xmin").text)), int(float(b.find("ymin").text))
            x2, y2 = int(float(b.find("xmax").text)), int(float(b.find("ymax").text))
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            cv2.imwrite(str(out / f"{xml.stem}_{i}.jpg"), crop)
            n += 1
    print(f"Saved {n} crops to {out}")


if __name__ == "__main__":
    main()
