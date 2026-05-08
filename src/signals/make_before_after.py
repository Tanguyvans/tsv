"""Génère des comparaisons avant/après pour les bare poles GERALD.

Chaque sortie : [BEFORE | AFTER] côte à côte, hauteur homogène.

Usage:
  python src/signals/make_before_after.py
  python src/signals/make_before_after.py --n 10 --seed 7
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from src.utils.viz import hstack_with_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", default="data/_raw/gerald_dataset/GERALD/dataset/JPEGImages")
    ap.add_argument("--bare-dir", default="data/gerald_augmented/bare_poles/images")
    ap.add_argument("--out-dir", default="outputs/comparisons/bare_poles")
    ap.add_argument("--n", type=int, default=None, help="max comparisons to write (default: all)")
    ap.add_argument("--seed", type=int, default=None,
                    help="if set, shuffle before selecting --n (deterministic)")
    ap.add_argument("--target-h", type=int, default=540)
    ap.add_argument("--overwrite", action="store_true",
                    help="re-render even if output already exists")
    args = ap.parse_args()

    bare_dir = Path(args.bare_dir)
    orig_dir = Path(args.orig_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bares = sorted(bare_dir.glob("*.jpg"))
    if args.seed is not None:
        random.Random(args.seed).shuffle(bares)
    if args.n is not None:
        bares = bares[: args.n]

    done = skipped_cached = skipped_missing = failed = 0
    for bare_path in bares:
        out_path = out_dir / f"compare_{bare_path.stem}.jpg"
        if out_path.exists() and not args.overwrite:
            skipped_cached += 1
            continue
        before = cv2.imread(str(orig_dir / bare_path.name))
        after = cv2.imread(str(bare_path))
        if before is None:
            skipped_missing += 1
            continue
        if after is None:
            failed += 1
            continue
        viz = hstack_with_labels(
            [("BEFORE", before), ("AFTER", after)],
            target_h=args.target_h,
        )
        cv2.imwrite(str(out_path), viz, [cv2.IMWRITE_JPEG_QUALITY, 90])
        done += 1

    print(f"Done. wrote={done} cached={skipped_cached} "
          f"missing_orig={skipped_missing} unreadable={failed} → {out_dir}")


if __name__ == "__main__":
    main()
