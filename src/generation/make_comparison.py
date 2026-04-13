"""Crée des images de comparaison côte à côte pour inspection visuelle.

Chaque fichier de sortie est une grille horizontale [original | generated (| composited)]
avec des labels au-dessus de chaque colonne.

Usage :
  python src/generation/make_comparison.py \
      --src-dir data/surface/Flakings \
      --gen-dir data/normal_synthetic/Normal \
      --comp-dir data/normal_synthetic/Normal_composited \
      --out-dir outputs/comparisons/normal
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

LABEL_H = 40
PAD = 10


def label_strip(width: int, text: str) -> np.ndarray:
    strip = np.full((LABEL_H, width, 3), 240, dtype=np.uint8)
    cv2.putText(strip, text, (10, LABEL_H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return strip


def hstack_with_labels(images: list[tuple[str, np.ndarray]], target_h: int = 540) -> np.ndarray:
    cols = []
    for label, img in images:
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        col = np.vstack([label_strip(new_w, label), resized])
        cols.append(col)
    # separators
    sep = np.full((target_h + LABEL_H, PAD, 3), 255, dtype=np.uint8)
    out = cols[0]
    for c in cols[1:]:
        out = np.hstack([out, sep, c])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--comp-dir", default=None)
    ap.add_argument("--gen-prefix", default="clean_")
    ap.add_argument("--comp-prefix", default="composited_")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-h", type=int, default=540)
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    gen_dir = Path(args.gen_dir)
    comp_dir = Path(args.comp_dir) if args.comp_dir else None
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    for gen_path in sorted(gen_dir.glob(f"{args.gen_prefix}*.jpg")):
        stem = gen_path.stem[len(args.gen_prefix):]
        src_matches = list(src_dir.glob(f"{stem}.*"))
        if not src_matches:
            skipped += 1
            continue
        original = cv2.imread(str(src_matches[0]))
        generated = cv2.imread(str(gen_path))
        if original is None or generated is None:
            skipped += 1
            continue

        panels = [("ORIGINAL", original), ("GENERATED", generated)]
        if comp_dir is not None:
            comp_path = comp_dir / f"{args.comp_prefix}{stem}.jpg"
            if comp_path.exists():
                composited = cv2.imread(str(comp_path))
                if composited is not None:
                    panels.append(("COMPOSITED", composited))

        viz = hstack_with_labels(panels, target_h=args.target_h)
        out_path = out_dir / f"compare_{stem}.jpg"
        cv2.imwrite(str(out_path), viz, [cv2.IMWRITE_JPEG_QUALITY, 90])
        done += 1

    print(f"Done. wrote {done} comparisons → {out_dir} (skipped {skipped})")


if __name__ == "__main__":
    main()
