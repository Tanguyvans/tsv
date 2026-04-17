"""Génère des "bare poles" à partir de GERALD.

Pour chaque image GERALD contenant un signal lumineux :
  1. Parse la VOC annotation → bboxes des signaux (Ks, Hp, Vr, Zs, Signal_*)
  2. SAM 3 avec box prompts raffine les masks des signaux
  3. Bria Eraser retire les panneaux → bare poles
  4. Sauvegarde l'image + YOLO label du mât/pole uniquement

Usage:
  python src/signals/generate_bare_poles.py --n 50
  python src/signals/generate_bare_poles.py --src data/_raw/gerald_dataset/GERALD/dataset
"""
from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from src.generation.fal_wrapper import upload_file
from src.generation.mask_erase import bria_erase, sam3_mask_from_boxes

# Classes GERALD représentant un panneau de signal lumineux à retirer
SIGNAL_CLASSES = {
    "Hp_0_HV", "Hp_0_Ks", "Hp_0_Sh", "Hp_1", "Hp_2",
    "Vr_0", "Vr_1", "Vr_2", "Vr_0_Blink",
    "Ks_1", "Ks_2",
    "Zs_3", "Zs_3v", "Zs_Off", "Zs_1", "Zs_6", "Zs_7",
    "Ne_2", "Ne_3_1", "Ne_3_2", "Ne_3_3", "Ne_3_4", "Ne_3_5", "Ne_4", "Ne_5",
    "Lf_6", "Lf_7",
    "Signal_Off", "Signal_Back", "Signal_Identifier_Sign",
}


def parse_voc(xml_path: Path) -> tuple[int, int, list[dict]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    objs = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        objs.append({
            "name": obj.find("name").text,
            "xmin": int(float(b.find("xmin").text)),
            "ymin": int(float(b.find("ymin").text)),
            "xmax": int(float(b.find("xmax").text)),
            "ymax": int(float(b.find("ymax").text)),
        })
    return w, h, objs


def signal_boxes(objs: list[dict]) -> list[tuple[int, int, int, int]]:
    return [
        (o["xmin"], o["ymin"], o["xmax"], o["ymax"])
        for o in objs if o["name"] in SIGNAL_CLASSES
    ]


POLE_WIDTH_RATIO = 0.3    # poteau = 30% de la largeur du signal
POLE_HEIGHT_FACTOR = 3.0  # hauteur pole = 3x hauteur signal (descend sous le signal)


def pole_boxes_from_signals(objs: list[dict], w: int, h: int) -> list[str]:
    """Infère la bbox du poteau depuis chaque signal retiré.

    Heuristique : le poteau est centré sur le signal, plus étroit, et s'étend
    depuis le signal vers le bas. Classe YOLO 0 = bare_pole.
    """
    lines = []
    for o in objs:
        if o["name"] not in SIGNAL_CLASSES:
            continue
        sx_min, sy_min = o["xmin"], o["ymin"]
        sx_max, sy_max = o["xmax"], o["ymax"]
        s_cx = (sx_min + sx_max) / 2
        s_bw = sx_max - sx_min
        s_bh = sy_max - sy_min

        p_bw = s_bw * POLE_WIDTH_RATIO
        p_xmin = s_cx - p_bw / 2
        p_xmax = s_cx + p_bw / 2
        p_ymin = sy_min  # démarre en haut du signal
        p_ymax = min(h, sy_max + s_bh * (POLE_HEIGHT_FACTOR - 1))  # descend

        cx = (p_xmin + p_xmax) / 2 / w
        cy = (p_ymin + p_ymax) / 2 / h
        bw = (p_xmax - p_xmin) / w
        bh = (p_ymax - p_ymin) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def process_one(img_path: Path, xml_path: Path, out_dir: Path) -> Path | None:
    try:
        w, h, objs = parse_voc(xml_path)
    except ET.ParseError:
        return None

    boxes = signal_boxes(objs)
    if not boxes:
        return None  # pas de signal = rien à retirer

    print(f"  {len(boxes)} signal box(es)")
    image_url = upload_file(str(img_path))

    mask = sam3_mask_from_boxes(image_url, (w, h), boxes, prompt="signal")
    if mask is None:
        print("  no mask")
        return None

    mask_path = out_dir / "_debug" / f"{img_path.stem}_mask.png"
    result_bytes = bria_erase(image_url, mask, mask_path, src_path=img_path)
    if result_bytes is None:
        print("  eraser failed")
        return None

    img_out_dir = out_dir / "images"
    lbl_out_dir = out_dir / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    out_img = img_out_dir / f"{img_path.stem}.jpg"
    out_img.write_bytes(result_bytes)

    yolo_lines = pole_boxes_from_signals(objs, w, h)
    (lbl_out_dir / f"{img_path.stem}.txt").write_text("\n".join(yolo_lines))
    print(f"  → {out_img} ({len(yolo_lines)} pole label(s))")
    return out_img


def main():
    ap = argparse.ArgumentParser(description="Génère des bare poles depuis GERALD")
    ap.add_argument("--src", default="data/_raw/gerald_dataset/GERALD/dataset",
                    help="Dossier contenant JPEGImages/ et Annotations/")
    ap.add_argument("--out", default="data/gerald_augmented/bare_poles")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    src = Path(args.src)
    images_dir = src / "JPEGImages"
    annots_dir = src / "Annotations"
    if not images_dir.exists() or not annots_dir.exists():
        print(f"Missing JPEGImages/ or Annotations/ under {src}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect image/annotation pairs
    xmls = sorted(annots_dir.glob("*.xml"))
    done = 0
    for xml in xmls:
        if done >= args.n:
            break
        img_path = images_dir / f"{xml.stem}.jpg"
        if not img_path.exists():
            continue
        print(f"[{done + 1}/{args.n}] {xml.stem}")
        if process_one(img_path, xml, out_dir) is not None:
            done += 1
        time.sleep(0.2)

    print(f"\nDone. {done} bare-pole image(s) in {out_dir}/images")


if __name__ == "__main__":
    main()
