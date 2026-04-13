"""Génère des images de rails sains via fal.ai Nano Banana 2.

Modes:
  - generate : text-to-image from-scratch
  - edit     : remove defect from an existing defect image (single image input)
  - compose  : combine a rail image (structure source) + an environment image
               (ballast/scene reference) into a clean Normal sample

Usage:
  python src/generation/generate_normal.py --mode generate --n 200
  python src/generation/generate_normal.py --mode edit --src data/surface/Flakings --n 100
  python src/generation/generate_normal.py --mode compose \
      --src data/surface/Flakings --env data/_raw/cabview_refs/E2U-dfQ4nNs/frame_0000.jpg --n 50
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

from src.generation.fal_wrapper import submit_and_wait, upload_file

MODEL_GEN = "fal-ai/nano-banana-2"
MODEL_EDIT = "fal-ai/nano-banana-2/edit"

PROMPT_GEN = (
    "Top-down photograph of a clean healthy railway rail surface, "
    "industrial outdoor lighting, sharp focus, high detail, 4K, "
    "no defects, no cracks, no spalling, no flaking, realistic metal texture"
)
PROMPT_EDIT = (
    "Remove all surface defects (cracks, flakings, spallings, squats, shellings) "
    "from this railway rail. Restore a clean, healthy top-view rail surface. "
    "Keep the exact same lighting, angle, framing, texture and background."
)
PROMPT_COMPOSE = (
    "The FIRST image is a close-up top-down photograph of a railway rail. "
    "The SECOND image is ONLY a reference for the surrounding environment (ballast stones, grass, lighting, color palette). "
    "Your task: produce a new image that is PIXEL-IDENTICAL to the first image for the metal rail itself — "
    "keep the exact same rail metal texture, exact same shine, exact same reflections, exact same micro-scratches, "
    "exact same wear marks, exact same color, exact same bolts, fasteners, joints, position, size, geometry, framing and camera angle. "
    "DO NOT repaint, re-render, smooth, re-texture, recolor, or 'clean up' the rail metal in any way. "
    "The rail surface must be copied verbatim from the first image, untouched. "
    "The ONLY allowed modification is OUTSIDE the rail: replace the ballast, stones, grass and debris around the rail "
    "with textures and lighting consistent with the second image's environment. "
    "Do not add new bolts, do not shift or resize the rail, do not change perspective. "
    "Photorealistic, sharp focus, high detail, same resolution as the first image."
)


def save_outputs(result: dict, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    images = result.get("images") or []
    for i, item in enumerate(images):
        url = item.get("url") if isinstance(item, dict) else item
        if not url:
            continue
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            path = out_dir / f"{stem}_{i}.jpg"
            path.write_bytes(r.content)
            saved.append(path)
        except Exception as e:  # noqa: BLE001
            print(f"  download failed: {e}")
    return saved


def mode_generate(n: int, out_dir: Path, num_per_call: int = 4):
    total = 0
    call = 0
    while total < n:
        result = submit_and_wait(
            MODEL_GEN,
            {"prompt": PROMPT_GEN, "num_images": min(num_per_call, n - total)},
        )
        saved = save_outputs(result, out_dir, f"gen_{call:04d}")
        total += len(saved)
        print(f"[gen] call {call}: +{len(saved)} (total {total}/{n})")
        call += 1
        if not saved:
            break


def mode_edit(src: Path, n: int, out_dir: Path):
    candidates = sorted([p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    candidates = candidates[:n]
    for i, img_path in enumerate(candidates):
        url = upload_file(str(img_path))
        result = submit_and_wait(
            MODEL_EDIT,
            {"prompt": PROMPT_EDIT, "image_urls": [url], "num_images": 1},
            image_paths=[str(img_path)],
        )
        saved = save_outputs(result, out_dir, f"edit_{img_path.stem}")
        print(f"[edit {i+1}/{len(candidates)}] {img_path.name} → {len(saved)} img")
        time.sleep(0.2)


def mode_compose(src: Path, env: Path, n: int, out_dir: Path):
    candidates = sorted([p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    candidates = candidates[:n]
    env_url = upload_file(str(env))
    for i, rail_path in enumerate(candidates):
        rail_url = upload_file(str(rail_path))
        result = submit_and_wait(
            MODEL_EDIT,
            {
                "prompt": PROMPT_COMPOSE,
                "image_urls": [rail_url, env_url],
                "num_images": 1,
            },
            image_paths=[str(rail_path), str(env)],
        )
        saved = save_outputs(result, out_dir, f"compose_{rail_path.stem}")
        print(f"[compose {i+1}/{len(candidates)}] {rail_path.name} → {len(saved)} img")
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["generate", "edit", "compose"], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--src", default="data/surface/Flakings")
    ap.add_argument("--env", default="data/_raw/cabview_refs/E2U-dfQ4nNs/frame_0000.jpg",
                    help="Environment reference image (used in compose mode)")
    ap.add_argument("--out", default="data/normal_synthetic/Normal")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.mode == "generate":
        mode_generate(args.n, out_dir)
    elif args.mode == "edit":
        mode_edit(Path(args.src), args.n, out_dir)
    else:
        mode_compose(Path(args.src), Path(args.env), args.n, out_dir)


if __name__ == "__main__":
    main()
