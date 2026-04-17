"""Génère des images Normal à partir des images Flakings.

Retire les déchets via SAM 3 (détection) + Bria Eraser (inpainting).
Le rail, les boulons et le ballast restent pixel-identiques.

Usage:
  python src/generation/generate_normal.py --src data/surface/Flakings --n 50
  python src/generation/generate_normal.py --src data/surface/Flakings/image.JPEG
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

from src.generation.fal_wrapper import upload_file
from src.generation.mask_erase import bria_erase, collect_images, sam3_mask

TRASH_PROMPTS = ["trash", "plastic bag", "litter", "wrapper", "paper", "bottle"]


def process_one(src_img: Path, out_dir: Path) -> Path | None:
    pil = Image.open(src_img).convert("RGB")
    W, H = pil.size
    image_url = upload_file(str(src_img))

    mask = sam3_mask(image_url, (W, H), TRASH_PROMPTS)
    if mask is None:
        print("  no trash detected → copying original")
        out_path = out_dir / f"normal_{src_img.stem}.jpg"
        pil.save(out_path, quality=95)
        return out_path

    mask_path = out_dir / "_debug" / f"{src_img.stem}_mask.png"
    result_bytes = bria_erase(image_url, mask, mask_path, src_path=src_img)
    if result_bytes is None:
        print("  eraser returned no image")
        return None

    out_path = out_dir / f"normal_{src_img.stem}.jpg"
    out_path.write_bytes(result_bytes)
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

    targets = collect_images(Path(args.src), args.n)
    print(f"{len(targets)} image(s) | Output: {out_dir}")

    for i, p in enumerate(targets):
        print(f"[{i + 1}/{len(targets)}] {p.name}")
        process_one(p, out_dir)
        time.sleep(0.2)

    print(f"\nDone. {len(list(out_dir.glob('*.jpg')))} images in {out_dir}")


if __name__ == "__main__":
    main()
