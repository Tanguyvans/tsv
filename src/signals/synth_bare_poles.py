"""Génère des "bare poles" en supprimant le panneau d'images GERALD via fal.ai.

Pour chaque image GERALD avec un mât+panneau annoté :
  1. Upload l'image
  2. Appelle nano-banana-2/edit avec un prompt "remove the signal panel"
  3. Sauvegarde l'image résultante + l'annotation YOLO du pole (réutilisée tel quel,
     classe re-mappée vers `bare_pole=2`)

Fallback (sans fal): copie simplement avec masque inpainting OpenCV très basique.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import requests

from src.generation.fal_client import submit_and_wait, upload_file

MODEL_EDIT = "fal-ai/nano-banana-2/edit"
PROMPT = (
    "Remove the signal panel and any sign/board mounted on the mast. "
    "Keep the bare pole intact. Preserve sky, rails, ballast and overall scene "
    "exactly. Photorealistic, seamless inpainting."
)


def parse_objects(xml_path: Path):
    tree = ET.parse(xml_path); root = tree.getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        b = obj.find("bndbox")
        objs.append({
            "name": name,
            "xmin": float(b.find("xmin").text), "ymin": float(b.find("ymin").text),
            "xmax": float(b.find("xmax").text), "ymax": float(b.find("ymax").text),
        })
    return w, h, objs


def has_signal(objs) -> bool:
    return any("signal" in o["name"].lower() or "sign" in o["name"].lower() for o in objs)


def fallback_inpaint(img: np.ndarray, objs) -> np.ndarray:
    mask = np.zeros(img.shape[:2], np.uint8)
    for o in objs:
        if "signal" in o["name"].lower() or "sign" in o["name"].lower():
            cv2.rectangle(mask, (int(o["xmin"]), int(o["ymin"])), (int(o["xmax"]), int(o["ymax"])), 255, -1)
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def write_yolo(out_label: Path, w: int, h: int, objs):
    lines = []
    for o in objs:
        if "mast" in o["name"].lower() or "pole" in o["name"].lower():
            cx = (o["xmin"] + o["xmax"]) / 2 / w
            cy = (o["ymin"] + o["ymax"]) / 2 / h
            bw = (o["xmax"] - o["xmin"]) / w
            bh = (o["ymax"] - o["ymin"]) / h
            lines.append(f"2 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    if lines:
        out_label.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/gerald")
    ap.add_argument("--out", default="data/gerald_augmented/bare_poles")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--no-fal", action="store_true", help="Use OpenCV inpainting fallback only")
    args = ap.parse_args()

    src = Path(args.src)
    annots_dir = next((d for d in src.rglob("Annotations") if d.is_dir()), None)
    images_dir = next((d for d in src.rglob("JPEGImages") if d.is_dir()), src)
    if annots_dir is None:
        print("No Annotations dir"); return

    out_img = Path(args.out) / "images"; out_img.mkdir(parents=True, exist_ok=True)
    out_lbl = Path(args.out) / "labels"; out_lbl.mkdir(parents=True, exist_ok=True)

    count = 0
    for xml in annots_dir.glob("*.xml"):
        if count >= args.n:
            break
        try:
            w, h, objs = parse_objects(xml)
        except ET.ParseError:
            continue
        if not has_signal(objs):
            continue
        img_path = images_dir / f"{xml.stem}.jpg"
        if not img_path.exists():
            continue

        if args.no_fal:
            img = cv2.imread(str(img_path))
            edited = fallback_inpaint(img, objs)
            cv2.imwrite(str(out_img / f"{xml.stem}.jpg"), edited)
        else:
            try:
                url = upload_file(str(img_path))
                result = submit_and_wait(
                    MODEL_EDIT,
                    {"prompt": PROMPT, "image_urls": [url], "num_images": 1},
                    image_paths=[str(img_path)],
                )
                images = result.get("images") or []
                if not images:
                    continue
                first = images[0]
                u = first.get("url") if isinstance(first, dict) else first
                r = requests.get(u, timeout=60); r.raise_for_status()
                (out_img / f"{xml.stem}.jpg").write_bytes(r.content)
            except Exception as e:  # noqa: BLE001
                print(f"  fal failed for {xml.stem}: {e}")
                continue

        write_yolo(out_lbl / f"{xml.stem}.txt", w, h, objs)
        count += 1
        print(f"[{count}/{args.n}] {xml.stem}")

    print(f"Done. Saved {count} bare-pole pairs in {args.out}")


if __name__ == "__main__":
    main()
