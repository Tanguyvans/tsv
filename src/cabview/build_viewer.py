"""Génère une page HTML pour visualiser les détections YOLO sur un dataset.

Pour chaque frame, affiche :
  - le thumbnail
  - les bboxes dessinées si détection
  - le score de la meilleure détection

Vue groupée : "DÉTECTÉES" puis "NON DÉTECTÉES" (échantillon aléatoire).

Usage:
    PYTHONPATH=. python src/cabview/build_viewer.py \\
        --frames data/cabview/fr/frames_cabview/bordeaux_nantes_2023 \\
        --detections data/cabview/fr/detections/bordeaux_nantes_2023
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Détections panneaux — {title}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 16px; background: #1a1a1a; color: #eee; }}
  h1 {{ font-size: 1.3rem; margin-bottom: 4px; }}
  .stats {{ color: #999; margin-bottom: 16px; font-size: 0.9rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 24px; padding: 8px;
        background: #2a2a2a; border-radius: 4px; }}
  h2.detected {{ border-left: 4px solid #00d068; }}
  h2.missing {{ border-left: 4px solid #d04040; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
           gap: 8px; }}
  .card {{ background: #222; border-radius: 4px; overflow: hidden;
           position: relative; cursor: pointer; }}
  .card img {{ width: 100%; height: auto; display: block; }}
  .card .meta {{ padding: 4px 8px; font-size: 0.7rem; color: #999;
                 display: flex; justify-content: space-between; }}
  .card .score {{ color: #00d068; font-weight: bold; }}
  .card.fp .score {{ color: #ffaa00; }}
  .filter-bar {{ position: sticky; top: 0; background: #1a1a1a;
                 padding: 8px 0; z-index: 10; }}
  .filter-bar input {{ background: #2a2a2a; border: 1px solid #444; color: #eee;
                       padding: 4px 8px; border-radius: 4px; width: 200px; }}
</style>
</head>
<body>

<h1>{title}</h1>
<div class="stats">{stats}</div>

<h2 class="detected">✓ Frames avec détection ({n_detected})</h2>
<div class="grid">{detected_html}</div>

<h2 class="missing">✗ Échantillon de frames sans détection ({n_missing_sample} / {n_missing} total)</h2>
<div class="grid">{missing_html}</div>

</body>
</html>
"""


def draw_boxes(image_path: Path, detections: list[dict], out_path: Path,
               max_w: int = 480) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)))
        for d in detections:
            d_scaled = {
                "score": d["score"],
                "bbox": [int(x * scale) for x in d["bbox"]],
            }
            x1, y1, x2, y2 = d_scaled["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d['score']:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 80])


def shrink(image_path: Path, out_path: Path, max_w: int = 480) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 75])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="HTML output (default: viewer.html dans le dossier détections)")
    ap.add_argument("--missing-sample", type=int, default=80,
                    help="Nombre de frames sans détection à montrer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-w", type=int, default=480, help="Largeur des thumbnails")
    args = ap.parse_args()

    out_html = args.out or args.detections / "viewer.html"
    thumbs_dir = args.detections / "_thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    # Charger toutes les détections (un JSON par frame)
    json_files = sorted(args.detections.glob("*.json"))
    json_files = [f for f in json_files if f.name != "_summary.json"]

    detected = []  # [(frame_path, detections, max_score)]
    missing = []   # [frame_path]

    for jf in tqdm(json_files, desc="Loading detections"):
        data = json.loads(jf.read_text())
        frame_name = data["frame"]
        dets = data.get("detections", [])
        frame_path = args.frames / frame_name
        if not frame_path.exists():
            continue
        if dets:
            max_score = max(d["score"] for d in dets)
            detected.append((frame_path, dets, max_score))
        else:
            missing.append(frame_path)

    detected.sort(key=lambda x: -x[2])  # tri par score décroissant
    print(f"Detected: {len(detected)} | Missing: {len(missing)}")

    # Générer thumbnails avec bboxes pour détectées
    print("Generating thumbnails for detected...")
    for fp, dets, _ in tqdm(detected):
        out = thumbs_dir / f"det_{fp.stem}.jpg"
        if not out.exists():
            draw_boxes(fp, dets, out, max_w=args.max_w)

    # Sample missing
    miss_sample = random.sample(missing, min(args.missing_sample, len(missing)))
    print(f"Generating thumbnails for {len(miss_sample)} missing samples...")
    for fp in tqdm(miss_sample):
        out = thumbs_dir / f"miss_{fp.stem}.jpg"
        if not out.exists():
            shrink(fp, out, max_w=args.max_w)

    # Build HTML
    detected_html = []
    for fp, dets, max_score in detected:
        thumb = f"_thumbs/det_{fp.stem}.jpg"
        n = len(dets)
        detected_html.append(
            f'<div class="card"><img src="{thumb}" loading="lazy">'
            f'<div class="meta"><span>{fp.stem}</span>'
            f'<span class="score">{max_score:.2f}{(" ×"+str(n)) if n>1 else ""}</span>'
            f'</div></div>'
        )

    missing_html = []
    for fp in miss_sample:
        thumb = f"_thumbs/miss_{fp.stem}.jpg"
        missing_html.append(
            f'<div class="card"><img src="{thumb}" loading="lazy">'
            f'<div class="meta"><span>{fp.stem}</span>'
            f'<span>—</span></div></div>'
        )

    html = HTML_TEMPLATE.format(
        title=f"{args.frames.parent.name}/{args.frames.name}",
        stats=f"Total : {len(detected) + len(missing)} frames | "
              f"Détectées : {len(detected)} ({100*len(detected)/(len(detected)+len(missing)):.1f}%) | "
              f"Sans détection : {len(missing)}",
        n_detected=len(detected),
        n_missing=len(missing),
        n_missing_sample=len(miss_sample),
        detected_html="\n".join(detected_html),
        missing_html="\n".join(missing_html),
    )

    out_html.write_text(html)
    print(f"\nViewer généré : {out_html}")
    print(f"Ouvrir : open {out_html}")


if __name__ == "__main__":
    main()
