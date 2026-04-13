"""Pipeline B — YOLO multi-classes {pole, panel} + IoU check.

Pour chaque pole détecté, vérifie qu'il existe un panel avec IoU > seuil
(ou que le centre du panel est dans la bbox du pole). Sinon → flag bare_pole.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

POLE_CLS = 0
PANEL_CLS = 1


def iou(a, b) -> float:
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def train(data_yaml: str, model_name: str = "yolov8s.pt", epochs: int = 80, name: str = "pipeline_b"):
    from ultralytics import YOLO
    YOLO(model_name).train(
        data=data_yaml, epochs=epochs, imgsz=640, batch=16,
        project="outputs/signals", name=name,
    )


def infer(image_path: str, ckpt: str, iou_thresh: float = 0.05, out_dir: str = "outputs/signals/pipeline_b"):
    from ultralytics import YOLO
    detector = YOLO(ckpt)
    img = cv2.imread(image_path)
    res = detector(img, verbose=False)[0]
    poles, panels = [], []
    for box in res.boxes:
        cls = int(box.cls[0])
        coords = list(map(float, box.xyxy[0].tolist()))
        (poles if cls == POLE_CLS else panels).append(coords)

    flagged = []
    vis = img.copy()
    for p in poles:
        has = any(iou(p, q) > iou_thresh for q in panels)
        color = (0, 255, 0) if has else (0, 0, 255)
        cv2.rectangle(vis, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), color, 2)
        cv2.putText(vis, "panel" if has else "BARE",
                    (int(p[0]), max(0, int(p[1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        flagged.append({"box": p, "has_panel": has})
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / Path(image_path).name), vis)
    return flagged


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="configs/gerald.yaml")
    t.add_argument("--epochs", type=int, default=80)
    i = sub.add_parser("infer")
    i.add_argument("--image", required=True)
    i.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    if args.cmd == "train":
        train(args.data, epochs=args.epochs)
    else:
        for o in infer(args.image, args.ckpt):
            print(o)


if __name__ == "__main__":
    main()
