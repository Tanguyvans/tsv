"""Convertit les annotations PASCAL VOC GERALD vers le format YOLO.

Mappe les ~50 classes GERALD vers un schéma simplifié à 3 classes :
  - 0 : panel_signal  (toute classe représentant un signal ferroviaire)
  - 1 : mast_sign     (mât/poteau portant un signal)
  - 2 : bare_pole     (mât sans signal — n'apparaît pas dans GERALD natif)

Ajuster CLASS_MAP au besoin selon la taxonomie réelle de GERALD.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_CLASS_MAP = {
    # mâts/poteaux
    "mast": 1, "Mast": 1, "pole": 1, "Pole": 1,
    "bare_pole": 2,
    # tout le reste = panneau (mappé à 0)
}


def voc_to_yolo_box(b: dict, w: int, h: int) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
    return ((xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h, (xmax - xmin) / w, (ymax - ymin) / h)


def parse_voc(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        b = obj.find("bndbox")
        boxes.append({
            "name": cls_name,
            "xmin": float(b.find("xmin").text),
            "ymin": float(b.find("ymin").text),
            "xmax": float(b.find("xmax").text),
            "ymax": float(b.find("ymax").text),
        })
    return w, h, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/gerald")
    ap.add_argument("--out", default="data/gerald/labels")
    ap.add_argument("--default-class", type=int, default=0,
                    help="Class id for unknown VOC class names (panel_signal)")
    args = ap.parse_args()

    src = Path(args.src)
    annots_dir = src / "Annotations"
    if not annots_dir.exists():
        annots_dir = next((d for d in src.rglob("Annotations") if d.is_dir()), None)
        if annots_dir is None:
            print(f"No Annotations folder under {src}")
            return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_files = n_boxes = 0
    seen_names: set[str] = set()
    for xml in annots_dir.glob("*.xml"):
        try:
            w, h, boxes = parse_voc(xml)
        except ET.ParseError:
            continue
        lines = []
        for b in boxes:
            seen_names.add(b["name"])
            cls_id = DEFAULT_CLASS_MAP.get(b["name"], args.default_class)
            cx, cy, bw, bh = voc_to_yolo_box(b, w, h)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_boxes += 1
        (out_dir / f"{xml.stem}.txt").write_text("\n".join(lines))
        n_files += 1
    print(f"Converted {n_files} files, {n_boxes} boxes")
    print(f"Distinct VOC class names found: {sorted(seen_names)[:30]}...")


if __name__ == "__main__":
    main()
