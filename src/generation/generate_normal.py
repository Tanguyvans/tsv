"""Génère des images Normal à partir des images Flakings.

Retire les déchets via SAM 3 + object-removal (LaMa).
Le rail, boulons et ballast restent pixel-identiques.

Usage:
  python src/generation/generate_normal.py --src data/surface/Flakings --n 50
  python src/generation/generate_normal.py --src data/surface/Flakings/image.JPEG
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from src.generation.fal_wrapper import submit_and_wait, upload_file

MODEL_SAM = "fal-ai/sam-3/image"
MODEL_REMOVE = "fal-ai/object-removal/mask"

TRASH_PROMPTS = ["trash", "plastic bag", "litter", "wrapper", "paper", "bottle"]
DILATE_PX = 8


def _download(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def _collect_images(src: Path, n: int) -> list[Path]:
    if src.is_file():
        return [src]
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in exts)[:n]


def get_trash_mask(image_url: str, size: tuple[int, int]) -> np.ndarray | None:
    """Détecte les déchets via SAM 3. Retourne mask binaire (255=déchet) ou None."""
    W, H = size
    union = np.zeros((H, W), dtype=np.uint8)
    found_any = False

    for prompt in TRASH_PROMPTS:
        result = submit_and_wait(
            MODEL_SAM,
            {
                "image_url": image_url,
                "prompt": prompt,
                "apply_mask": False,
                "return_multiple_masks": True,
                "max_masks": 5,
                "output_format": "png",
            },
        )
        masks = result.get("masks") or []
        if masks:
            print(f"    SAM3[{prompt}]: {len(masks)} mask(s)")
        for m in masks:
            murl = m.get("url") if isinstance(m, dict) else None
            if not murl:
                continue
            try:
                mimg = Image.open(io.BytesIO(_download(murl)))
                if mimg.size != (W, H):
                    mimg = mimg.resize((W, H), Image.NEAREST)
                if mimg.mode == "RGBA":
                    arr = np.array(mimg.split()[-1])
                else:
                    arr = np.array(mimg.convert("L"))
                union = np.maximum(union, (arr > 127).astype(np.uint8) * 255)
                found_any = True
            except Exception as e:
                print(f"    mask download failed: {e}")

    if not found_any:
        return None

    if DILATE_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (DILATE_PX * 2 + 1, DILATE_PX * 2 + 1)
        )
        union = cv2.dilate(union, k)

    coverage = union.sum() / 255 / (W * H) * 100
    print(f"    mask coverage: {coverage:.1f}%")
    return union


def process_one(src_img: Path, out_dir: Path) -> Path | None:
    """Retire les déchets d'une image et sauvegarde le résultat."""
    pil = Image.open(src_img).convert("RGB")
    W, H = pil.size
    image_url = upload_file(str(src_img))

    mask = get_trash_mask(image_url, (W, H))
    if mask is None:
        print("  no trash detected → copying original")
        out_path = out_dir / f"normal_{src_img.stem}.jpg"
        pil.save(out_path, quality=95)
        return out_path

    # Save debug mask
    dbg_dir = out_dir / "_debug"
    dbg_dir.mkdir(parents=True, exist_ok=True)
    mask_path = dbg_dir / f"{src_img.stem}_mask.png"
    cv2.imwrite(str(mask_path), mask)

    # Upload mask and remove trash
    mask_url = upload_file(str(mask_path))
    result = submit_and_wait(
        MODEL_REMOVE,
        {
            "image_url": image_url,
            "mask_url": mask_url,
            "model": "best_quality",
        },
        image_paths=[str(src_img), str(mask_path)],
    )

    images = result.get("images") or []
    if not images:
        print("  object-removal returned no image")
        return None

    url = images[0].get("url") if isinstance(images[0], dict) else None
    if not url:
        return None

    out_path = out_dir / f"normal_{src_img.stem}.jpg"
    out_path.write_bytes(_download(url))
    print(f"  → {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Génère des images Normal à partir des Flakings")
    ap.add_argument("--src", required=True, help="Image ou dossier source (Flakings)")
    ap.add_argument("--n", type=int, default=10, help="Nombre max d'images à traiter")
    ap.add_argument("--out", default="data/normal_synthetic/Normal", help="Dossier de sortie")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = _collect_images(Path(args.src), args.n)
    print(f"{len(targets)} image(s) | Output: {out_dir}")

    for i, p in enumerate(targets):
        print(f"[{i + 1}/{len(targets)}] {p.name}")
        process_one(p, out_dir)
        time.sleep(0.2)

    print(f"\nDone. {len(list(out_dir.glob('*.jpg')))} images in {out_dir}")


if __name__ == "__main__":
    main()
