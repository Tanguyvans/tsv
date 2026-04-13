"""Évaluation comparative pipelines A vs B sur un set de test annoté.

Format attendu pour le test set : un dossier d'images + un dossier labels YOLO
(0=panel, 1=mast/pole, 2=bare_pole). Pour chaque image on compare les bare_poles
GT aux bare_poles flaggés par chaque pipeline.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.signals.pipeline_a import run as pipeline_a_run
from src.signals.pipeline_b import infer as pipeline_b_infer
from src.signals.pipeline_b import iou


def load_gt(label_path: Path, img_w: int, img_h: int) -> list[list[float]]:
    """Retourne les bbox bare_pole GT en xyxy pixels."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = parts
        if int(cls) != 2:
            continue
        cx, cy, bw, bh = float(cx) * img_w, float(cy) * img_h, float(bw) * img_w, float(bh) * img_h
        out.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return out


def match(preds: list[dict], gts: list[list[float]], thr: float = 0.3) -> tuple[int, int, int]:
    matched_gt = set()
    tp = 0
    for p in preds:
        if p.get("has_panel", True):
            continue
        for i, g in enumerate(gts):
            if i in matched_gt:
                continue
            if iou(p["box"], g) > thr:
                matched_gt.add(i)
                tp += 1
                break
    fp = sum(1 for p in preds if not p.get("has_panel", True)) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def prf(tp, fp, fn) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--yolo-a", required=True)
    ap.add_argument("--cls-a", required=True)
    ap.add_argument("--yolo-b", required=True)
    ap.add_argument("--out", default="outputs/signals/comparison.md")
    args = ap.parse_args()

    import cv2
    images = sorted(Path(args.images).glob("*.jpg"))
    a_stats = [0, 0, 0]; b_stats = [0, 0, 0]
    a_time = b_time = 0.0
    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gts = load_gt(Path(args.labels) / f"{img_path.stem}.txt", w, h)

        t = time.time(); preds_a = pipeline_a_run(str(img_path), args.yolo_a, args.cls_a); a_time += time.time() - t
        t = time.time(); preds_b = pipeline_b_infer(str(img_path), args.yolo_b); b_time += time.time() - t

        for st, preds in ((a_stats, preds_a), (b_stats, preds_b)):
            tp, fp, fn = match(preds, gts)
            st[0] += tp; st[1] += fp; st[2] += fn

    pa, ra, fa = prf(*a_stats)
    pb, rb, fb = prf(*b_stats)

    md = [
        "# Pipelines A vs B — bare-pole detection", "",
        "| Pipeline | Precision | Recall | F1 | TP | FP | FN | Time/img |",
        "|---|---|---|---|---|---|---|---|",
        f"| A (detect+classify) | {pa:.3f} | {ra:.3f} | {fa:.3f} | {a_stats[0]} | {a_stats[1]} | {a_stats[2]} | {a_time / max(1, len(images)):.3f}s |",
        f"| B (multi-class IoU) | {pb:.3f} | {rb:.3f} | {fb:.3f} | {b_stats[0]} | {b_stats[1]} | {b_stats[2]} | {b_time / max(1, len(images)):.3f}s |",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
