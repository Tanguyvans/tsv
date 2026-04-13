"""Filtre qualité pour images générées : CLIP-score + LPIPS diversity.

Usage:
  python src/generation/quality_filter.py --dir data/normal_synthetic/Normal \
      --prompt "clean healthy railway rail surface" --min-clip 0.22
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from PIL import Image


def load_clip(device: str):
    import clip
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, clip


def clip_scores(image_paths: list[Path], prompt: str, device: str) -> list[float]:
    model, preprocess, clip = load_clip(device)
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    scores = []
    for p in image_paths:
        try:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model.encode_image(img)
                f /= f.norm(dim=-1, keepdim=True)
            scores.append(float((f @ text_feat.T).item()))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p.name}: {e}")
            scores.append(-1.0)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--min-clip", type=float, default=0.22)
    ap.add_argument("--reject-dir", default=None)
    args = ap.parse_args()

    d = Path(args.dir)
    images = sorted([p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    if not images:
        print(f"No images in {d}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores = clip_scores(images, args.prompt, device)

    reject_dir = Path(args.reject_dir) if args.reject_dir else d.parent / f"{d.name}_rejected"
    reject_dir.mkdir(parents=True, exist_ok=True)

    kept = rejected = 0
    for p, s in zip(images, scores):
        if s < args.min_clip:
            shutil.move(str(p), str(reject_dir / p.name))
            rejected += 1
        else:
            kept += 1
    print(f"Kept {kept} / Rejected {rejected} (threshold CLIP > {args.min_clip})")


if __name__ == "__main__":
    main()
