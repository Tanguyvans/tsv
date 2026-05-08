"""Détection de panneaux via YOLOv26-s pré-entraîné (modèle Otmane42).

Modèle : https://huggingface.co/Otmane42/yolo26s-railway-signs-detector
Détecteur agnostique single-class : retourne juste "il y a un panneau ici"
sans classification fine.

Usage:
    PYTHONPATH=. python src/cabview/detect_yolo26.py \\
        --frames data/cabview/fr/frames_cabview/bordeaux_nantes_2023 --n 20

    # Sur une image
    PYTHONPATH=. python src/cabview/detect_yolo26.py --frames data/.../frame.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

DEFAULT_WEIGHTS = "models/yolo26s-railway-signs-detector/best.pt"


def draw_debug(image_path: Path, detections: list[dict], out_path: Path) -> None:
    img = cv2.imread(str(image_path))
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"sign {d['score']:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960,
                    help="Image size (modèle entraîné en 960)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-debug", action="store_true")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"device: {device}")

    if args.frames.is_file():
        frames = [args.frames]
        out_dir = args.out or args.frames.parent.parent / "detections" / args.frames.stem
    else:
        frames = sorted(args.frames.glob("*.jpg"))
        out_dir = args.out or args.frames.parent.parent / "detections" / args.frames.name

    if args.n:
        frames = frames[:args.n]
    if not frames:
        raise SystemExit(f"Aucune frame dans {args.frames}")

    print(f"{len(frames)} frame(s) à analyser")
    print(f"weights: {args.weights}")
    print(f"conf: {args.conf} | imgsz: {args.imgsz}")

    model = YOLO(args.weights)

    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "_debug"

    summary = []
    n_with_signal = 0
    for fp in tqdm(frames, desc="YOLOv26"):
        try:
            results = model.predict(
                str(fp), conf=args.conf, imgsz=args.imgsz,
                device=device, verbose=False
            )
        except Exception as e:
            print(f"  err {fp.name}: {e}")
            results = []

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().tolist()
                score = float(boxes.conf[i].item())
                detections.append({
                    "score": round(score, 4),
                    "bbox": [int(x) for x in xyxy],
                })

        (out_dir / f"{fp.stem}.json").write_text(json.dumps({
            "frame": fp.name,
            "detections": detections,
        }, indent=2))

        if detections:
            n_with_signal += 1
            if not args.no_debug:
                draw_debug(fp, detections, debug_dir / fp.name)

        summary.append({"frame": fp.name, "n_detections": len(detections)})

    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{n_with_signal} / {len(frames)} frames avec ≥1 détection")
    print(f"JSONs: {out_dir}/<frame>.json")
    if not args.no_debug:
        print(f"Debug: {debug_dir}/")


if __name__ == "__main__":
    main()
