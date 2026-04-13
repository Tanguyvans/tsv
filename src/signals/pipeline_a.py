"""Pipeline A — Two-stage : YOLO detect pole → EfficientNet classify has_panel."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from src.data.preprocessing import eval_transforms
from src.models.classifier import build_classifier


def run(image_path: str, yolo_ckpt: str, cls_ckpt: str, out_dir: str = "outputs/signals/pipeline_a") -> list[dict]:
    from ultralytics import YOLO
    detector = YOLO(yolo_ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = build_classifier("efficientnet_b0", num_classes=2, pretrained=False).to(device)
    classifier.load_state_dict(torch.load(cls_ckpt, map_location=device))
    classifier.eval()
    tf = eval_transforms(160)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    results = detector(img, verbose=False)[0]
    flagged = []
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    vis = img.copy()
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        x = tf(image=rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            p = classifier(x).softmax(-1).cpu().numpy()[0]
        has_panel = bool(p[1] > p[0])
        color = (0, 255, 0) if has_panel else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, "panel" if has_panel else "BARE", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        flagged.append({"box": [x1, y1, x2, y2], "has_panel": has_panel, "p_no_panel": float(p[0])})
    cv2.imwrite(str(out_path / Path(image_path).name), vis)
    return flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--cls", required=True)
    args = ap.parse_args()
    out = run(args.image, args.yolo, args.cls)
    for o in out:
        print(o)


if __name__ == "__main__":
    main()
