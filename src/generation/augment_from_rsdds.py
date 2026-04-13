"""Augmentation : transplante des défauts RSDDs sur des rails sains en vue de dessus.

Idée : RSDDs fournit des masques GT précis pour chaque défaut. On en extrait un
crop serré (bbox autour du masque) et on demande à fal.ai/nano-banana-2/edit de
poser ce défaut sur une image cible (vue de dessus type `data/surface/`).

Usage :
    python src/generation/augment_from_rsdds.py \
        --rsdds data/_raw/rsdds \
        --target-class Cracks \
        --target-dir data/surface/Cracks \
        --n 50

Le script :
  1. Liste les paires (img RSDDs, GT). Filtre celles dont la zone défaut est
     suffisamment grande.
  2. Pour chaque paire, extrait un crop du défaut + sauvegarde un masque brut.
  3. Tire au sort une image cible "propre" (depuis --target-dir).
  4. Upload les deux images sur fal et envoie un prompt d'edition.
  5. Sauvegarde le résultat dans data/rare_synthetic/<target_class>/.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from src.generation.fal_client import submit_and_wait, upload_file

MODEL_EDIT = "fal-ai/nano-banana-2/edit"

CLASS_PROMPT = {
    "Cracks": (
        "Apply the crack pattern from the first reference image onto the railway "
        "rail surface shown in the second reference image. Keep the rail's exact "
        "perspective, top-down view, lighting and background. The crack should "
        "look natural and integrated, photorealistic, no border, seamless."
    ),
    "Squats": (
        "Apply the squat-style surface defect from the first reference onto the "
        "railway rail in the second reference. Top-down view, photorealistic, "
        "preserve lighting, perspective and background, seamless integration."
    ),
    "Spallings": (
        "Apply the spalling/chipping defect from the first reference onto the "
        "railway rail in the second reference. Top-down view, photorealistic, "
        "preserve lighting and background."
    ),
    "Shellings": (
        "Apply the shelling defect pattern from the first reference onto the "
        "railway rail in the second reference. Top-down view, photorealistic, "
        "preserve lighting and background."
    ),
    "Flakings": (
        "Apply the flaking surface defect from the first reference onto the "
        "railway rail in the second reference. Top-down view, photorealistic, "
        "preserve lighting and background."
    ),
    "Joints": (
        "Add a railway rail joint inspired by the first reference onto the rail "
        "shown in the second reference. Top-down view, photorealistic, "
        "fishplate and bolts visible, preserve lighting and background."
    ),
    "Grooves": (
        "Apply the grooved wear pattern from the first reference along the "
        "railway rail shown in the second reference. Top-down view, "
        "photorealistic, preserve lighting and background."
    ),
}


def find_pairs(rsdds_root: Path) -> list[tuple[Path, Path]]:
    img_dirs = [p for p in rsdds_root.rglob("INPUT_IMG") if p.is_dir()]
    gt_dirs = [p for p in rsdds_root.rglob("Ground Truth") if p.is_dir()]
    if not img_dirs or not gt_dirs:
        return []
    img_dir, gt_dir = img_dirs[0], gt_dirs[0]
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".bmp", ".jpg", ".png"}:
            continue
        gt = gt_dir / f"{img_path.stem}.png"
        if gt.exists():
            pairs.append((img_path, gt))
    return pairs


def extract_defect_crop(img_path: Path, gt_path: Path, pad: int = 12, min_pixels: int = 200) -> np.ndarray | None:
    img = cv2.imread(str(img_path))
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        return None
    ys, xs = np.where(gt > 0)
    if len(xs) < min_pixels:
        return None
    x1, x2 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    y1, y2 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    return img[y1:y2, x1:x2].copy()


def list_target_images(target_dir: Path) -> list[Path]:
    return sorted([p for p in target_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])


def save_outputs(result: dict, out_dir: Path, stem: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, item in enumerate(result.get("images") or []):
        url = item.get("url") if isinstance(item, dict) else item
        if not url:
            continue
        try:
            r = requests.get(url, timeout=60); r.raise_for_status()
            (out_dir / f"{stem}_{i}.jpg").write_bytes(r.content)
            n += 1
        except Exception as e:  # noqa: BLE001
            print(f"  download failed: {e}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsdds", default="data/_raw/rsdds")
    ap.add_argument("--target-class", required=True, choices=list(CLASS_PROMPT))
    ap.add_argument("--target-dir", required=True,
                    help="Folder of clean top-view rails to use as target. "
                         "Can be data/surface/<class> or data/normal_synthetic/Normal.")
    ap.add_argument("--out", default="data/rare_synthetic")
    ap.add_argument("--cache-crops", default="data/_raw/rsdds_defect_crops")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    pairs = find_pairs(Path(args.rsdds))
    if not pairs:
        print(f"No RSDDs pairs found under {args.rsdds}")
        return
    print(f"{len(pairs)} RSDDs pairs available")

    crops_dir = Path(args.cache_crops); crops_dir.mkdir(parents=True, exist_ok=True)
    targets = list_target_images(Path(args.target_dir))
    if not targets:
        print(f"No target images in {args.target_dir}")
        return

    out_dir = Path(args.out) / args.target_class.lower()
    prompt = CLASS_PROMPT[args.target_class]

    random.shuffle(pairs)
    saved = 0
    for img_path, gt_path in pairs:
        if saved >= args.n:
            break
        crop = extract_defect_crop(img_path, gt_path)
        if crop is None:
            continue
        crop_path = crops_dir / f"{img_path.stem}.jpg"
        if not crop_path.exists():
            cv2.imwrite(str(crop_path), crop)

        target_path = random.choice(targets)
        try:
            crop_url = upload_file(str(crop_path))
            target_url = upload_file(str(target_path))
            result = submit_and_wait(
                MODEL_EDIT,
                {
                    "prompt": prompt,
                    "image_urls": [crop_url, target_url],
                    "num_images": 1,
                },
                image_paths=[str(crop_path), str(target_path)],
            )
            n = save_outputs(result, out_dir, f"{args.target_class.lower()}_{img_path.stem}_on_{target_path.stem}")
            saved += n
            print(f"[{saved}/{args.n}] defect={img_path.stem} target={target_path.stem} → +{n}")
        except Exception as e:  # noqa: BLE001
            print(f"  fal failed for {img_path.stem}: {e}")
        time.sleep(0.3)

    print(f"Done. Saved {saved} augmented images to {out_dir}")


if __name__ == "__main__":
    main()
