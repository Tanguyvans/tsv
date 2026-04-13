"""Génère des images synthétiques pour les classes rares (Cracks/Joints/Grooves).

Utilise nano-banana-2/edit avec jusqu'à 14 images de référence (toutes les
images réelles disponibles de la classe) pour maximiser la fidélité de style.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.generation.fal_client import submit_and_wait, upload_file
from src.generation.generate_normal import save_outputs

MODEL_EDIT = "fal-ai/nano-banana-2/edit"

CLASS_PROMPTS = {
    "Cracks": (
        "Top-down photograph of a railway rail surface with thin sharp cracks. "
        "Match exactly the style, lighting and texture of the reference images. "
        "Realistic, photographic, industrial outdoor scene."
    ),
    "Joints": (
        "Top-down photograph of a railway rail joint, two rails meeting with "
        "fishplate and bolts visible. Match the style and lighting of the references."
    ),
    "Grooves": (
        "Top-down photograph of a railway rail with grooved wear marks running "
        "along the surface. Match the references' style and lighting."
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", choices=list(CLASS_PROMPTS), required=True)
    ap.add_argument("--src", default="data/surface")
    ap.add_argument("--out", default="data/rare_synthetic")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-refs", type=int, default=8)
    args = ap.parse_args()

    src_dir = Path(args.src) / args.cls
    refs = sorted([p for p in src_dir.rglob("*") if p.suffix.lower() in {".jpg", ".png"}])
    refs = refs[: args.max_refs]
    if not refs:
        print(f"No reference images for {args.cls}")
        return
    ref_urls = [upload_file(str(p)) for p in refs]
    print(f"Using {len(refs)} reference images for {args.cls}")

    out_dir = Path(args.out) / args.cls.lower()
    prompt = CLASS_PROMPTS[args.cls]

    total = 0
    call = 0
    while total < args.n:
        result = submit_and_wait(
            MODEL_EDIT,
            {"prompt": prompt, "image_urls": ref_urls, "num_images": min(4, args.n - total)},
            image_paths=[str(p) for p in refs],
        )
        saved = save_outputs(result, out_dir, f"{args.cls.lower()}_{call:04d}")
        total += len(saved)
        print(f"[{args.cls}] call {call}: +{len(saved)} ({total}/{args.n})")
        call += 1
        if not saved:
            break
        time.sleep(0.3)


if __name__ == "__main__":
    main()
