"""Compare plusieurs modèles d'object-removal sur une même image.

Teste en parallèle :
  - LaMa           : fal-ai/object-removal/mask
  - Bria Eraser    : fal-ai/bria/eraser
  - Nano Banana Pro: fal-ai/nano-banana-pro/edit (text-only, pas de mask)

Usage:
  python src/generation/compare_erasers.py \
      --src data/surface/Flakings/image.JPEG \
      --targets "trash,plastic bag,litter,wrapper,paper,bottle" \
      --nb-prompt "remove all trash, debris and litter from this railway ballast"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image

from src.generation.fal_wrapper import submit_and_wait, upload_file
from src.generation.mask_erase import _result_url, download, sam3_mask

MODEL_LAMA = "fal-ai/object-removal/mask"
MODEL_BRIA = "fal-ai/bria/eraser"
MODEL_NB = "fal-ai/nano-banana-pro/edit"


def save_first(result: dict, out_path: Path) -> Path | None:
    url = _result_url(result)
    if not url:
        return None
    out_path.write_bytes(download(url))
    return out_path


def run_lama(image_url: str, mask_url: str, out_path: Path) -> Path | None:
    return save_first(
        submit_and_wait(MODEL_LAMA, {
            "image_url": image_url, "mask_url": mask_url, "model": "best_quality",
        }),
        out_path,
    )


def run_bria(image_url: str, mask_url: str, out_path: Path) -> Path | None:
    return save_first(
        submit_and_wait(MODEL_BRIA, {
            "image_url": image_url, "mask_url": mask_url, "mask_type": "manual",
        }),
        out_path,
    )


def run_nb(image_url: str, prompt: str, out_path: Path) -> Path | None:
    return save_first(
        submit_and_wait(MODEL_NB, {
            "prompt": prompt,
            "image_urls": [image_url],
            "num_images": 1,
            "resolution": "1K",
        }),
        out_path,
    )


def build_comparison(src_path: Path, results: dict[str, Path | None],
                     out_path: Path, target_h: int = 540) -> None:
    """Grille horizontale avec labels : ORIGINAL | LAMA | BRIA | NB_PRO."""
    import numpy as np

    panels = [("ORIGINAL", cv2.imread(str(src_path)))]
    for label, key in [("LAMA", "lama"), ("BRIA", "bria"), ("NB_PRO", "nb_pro")]:
        p = results.get(key)
        if p and p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                panels.append((label, img))

    cols = []
    for label, img in panels:
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        strip = np.full((40, new_w, 3), 240, dtype=np.uint8)
        cv2.putText(strip, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (20, 20, 20), 2, cv2.LINE_AA)
        cols.append(np.vstack([strip, resized]))

    sep = np.full((target_h + 40, 10, 3), 255, dtype=np.uint8)
    out = cols[0]
    for c in cols[1:]:
        out = np.hstack([out, sep, c])
    cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--targets", required=True, help="SAM prompts comma-separated")
    ap.add_argument("--nb-prompt", required=True, help="Text prompt for NB Pro")
    ap.add_argument("--out", default="outputs/eraser_comparison")
    args = ap.parse_args()

    src_path = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pil = Image.open(src_path).convert("RGB")
    W, H = pil.size
    print(f"Source: {src_path.name} ({W}x{H})")

    image_url = upload_file(str(src_path))

    print("\n[1/4] SAM 3 mask")
    prompts = [p.strip() for p in args.targets.split(",") if p.strip()]
    mask = sam3_mask(image_url, (W, H), prompts)

    mask_url = None
    if mask is not None:
        mask_path = out_dir / f"{src_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        mask_url = upload_file(str(mask_path))

    results: dict[str, Path | None] = {}
    if mask_url:
        print("\n[2/4] LaMa")
        results["lama"] = run_lama(image_url, mask_url, out_dir / f"{src_path.stem}_lama.jpg")
        print(f"  → {results['lama']}")

        print("\n[3/4] Bria Eraser")
        results["bria"] = run_bria(image_url, mask_url, out_dir / f"{src_path.stem}_bria.jpg")
        print(f"  → {results['bria']}")
    else:
        print("\n[2-3/4] No mask → skipping LaMa and Bria")

    print("\n[4/4] Nano Banana Pro (text)")
    results["nb_pro"] = run_nb(image_url, args.nb_prompt, out_dir / f"{src_path.stem}_nb_pro.jpg")
    print(f"  → {results['nb_pro']}")

    compare_path = out_dir / f"compare_{src_path.stem}.jpg"
    build_comparison(src_path, results, compare_path)
    print(f"\nComparison → {compare_path}")


if __name__ == "__main__":
    main()
