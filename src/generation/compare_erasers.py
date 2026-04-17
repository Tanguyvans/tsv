"""Compare plusieurs modèles d'object-removal sur une même image.

Teste en parallèle :
  - LaMa           : fal-ai/object-removal/mask (baseline, texture extension)
  - Bria Eraser    : fal-ai/bria/eraser (ControlNet inpaint, licensed training)
  - Nano Banana Pro: fal-ai/nano-banana-pro/edit (text-only, pas de mask)

Les 2 premiers utilisent un mask SAM 3 des objets à retirer.
NB Pro utilise un prompt texte.

Usage:
  python src/generation/compare_erasers.py \
      --src data/surface/Flakings/image.JPEG \
      --targets "trash,plastic bag,litter,wrapper,paper,bottle" \
      --nb-prompt "remove all trash, debris and litter from this railway ballast"
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from src.generation.fal_wrapper import submit_and_wait, upload_file

MODEL_SAM = "fal-ai/sam-3/image"
MODEL_LAMA = "fal-ai/object-removal/mask"
MODEL_BRIA = "fal-ai/bria/eraser"
MODEL_NB = "fal-ai/nano-banana-pro/edit"

DILATE_PX = 8


def _download(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def _mask_png_to_binary(png_bytes: bytes, target_size: tuple[int, int]) -> np.ndarray:
    img = Image.open(io.BytesIO(png_bytes))
    if img.size != target_size:
        img = img.resize(target_size, Image.NEAREST)
    if img.mode == "RGBA":
        arr = np.array(img.split()[-1])
    else:
        arr = np.array(img.convert("L"))
    return (arr > 127).astype(np.uint8) * 255


def sam_union_mask(image_url: str, prompts: list[str], size: tuple[int, int]) -> np.ndarray | None:
    W, H = size
    union = np.zeros((H, W), dtype=np.uint8)
    found_any = False
    for prompt in prompts:
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
                union = np.maximum(union, _mask_png_to_binary(_download(murl), size))
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


def save_first(result: dict, out_path: Path) -> Path | None:
    images = result.get("images") or []
    if images:
        url = images[0].get("url") if isinstance(images[0], dict) else None
        if url:
            out_path.write_bytes(_download(url))
            return out_path
    # Bria returns "image" (singular) sometimes
    img = result.get("image")
    if isinstance(img, dict) and img.get("url"):
        out_path.write_bytes(_download(img["url"]))
        return out_path
    return None


def run_lama(image_url: str, mask_url: str, out_path: Path) -> Path | None:
    result = submit_and_wait(
        MODEL_LAMA,
        {"image_url": image_url, "mask_url": mask_url, "model": "best_quality"},
    )
    return save_first(result, out_path)


def run_bria(image_url: str, mask_url: str, out_path: Path) -> Path | None:
    result = submit_and_wait(
        MODEL_BRIA,
        {"image_url": image_url, "mask_url": mask_url, "mask_type": "manual"},
    )
    return save_first(result, out_path)


def run_nb(image_url: str, prompt: str, out_path: Path) -> Path | None:
    result = submit_and_wait(
        MODEL_NB,
        {"prompt": prompt, "image_urls": [image_url], "num_images": 1, "resolution": "1K"},
    )
    return save_first(result, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Image source")
    ap.add_argument("--targets", required=True,
                    help="Prompts SAM séparés par virgule (ex: 'trash,plastic bag')")
    ap.add_argument("--nb-prompt", required=True,
                    help="Prompt texte pour NB Pro (ex: 'remove all trash')")
    ap.add_argument("--out", default="outputs/eraser_comparison")
    args = ap.parse_args()

    src_path = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pil = Image.open(src_path).convert("RGB")
    W, H = pil.size
    print(f"Source: {src_path.name} ({W}x{H})")

    image_url = upload_file(str(src_path))

    # 1. Build mask with SAM 3
    prompts = [p.strip() for p in args.targets.split(",") if p.strip()]
    print("\n[1/4] SAM 3 mask generation")
    mask = sam_union_mask(image_url, prompts, (W, H))

    mask_path = None
    mask_url = None
    if mask is not None:
        mask_path = out_dir / f"{src_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        mask_url = upload_file(str(mask_path))

    # 2. Run the 3 models
    results = {}
    if mask_url:
        print("\n[2/4] LaMa (object-removal/mask)")
        results["lama"] = run_lama(image_url, mask_url, out_dir / f"{src_path.stem}_lama.jpg")
        print(f"  → {results['lama']}")

        print("\n[3/4] Bria Eraser")
        results["bria"] = run_bria(image_url, mask_url, out_dir / f"{src_path.stem}_bria.jpg")
        print(f"  → {results['bria']}")
    else:
        print("\n[2-3/4] No mask → skipping LaMa and Bria")

    print("\n[4/4] Nano Banana Pro (text-only)")
    results["nb_pro"] = run_nb(image_url, args.nb_prompt, out_dir / f"{src_path.stem}_nb_pro.jpg")
    print(f"  → {results['nb_pro']}")

    # 3. Build side-by-side comparison
    panels = [("ORIGINAL", cv2.imread(str(src_path)))]
    for label, key in [("LAMA", "lama"), ("BRIA", "bria"), ("NB_PRO", "nb_pro")]:
        p = results.get(key)
        if p and p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                panels.append((label, img))

    target_h = 540
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
    out_compare = cols[0]
    for c in cols[1:]:
        out_compare = np.hstack([out_compare, sep, c])
    compare_path = out_dir / f"compare_{src_path.stem}.jpg"
    cv2.imwrite(str(compare_path), out_compare, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\nComparison → {compare_path}")


if __name__ == "__main__":
    main()
