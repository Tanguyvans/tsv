"""Stage 1 — YOLOv8 détection de mâts (classe unique pole).

Entraîne sur le dataset GERALD (avec labels collapsés à 1 classe).
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/gerald_pole_only.yaml")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="stage1_pole")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="outputs/signals",
        name=args.name,
    )


if __name__ == "__main__":
    main()
