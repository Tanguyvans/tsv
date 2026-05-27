"""Génère une vidéo split 2×2 comparant les méthodes de distance.

- msense      : frame brute + bbox + distance
- dav2/da3/vda: depth colormap plein écran + distance dans le bandeau

Usage :
    PYTHONPATH=. python src/depth/make_method_comparison_video.py \\
        --benchmark depth_benchmark_focal_full --fps 10 --cell-w 960 --cell-h 540
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


METHODS = ["msense", "dav2", "da3", "vda"]

METHOD_LABELS = {
    "msense": "MultiSense (référence)",
    "dav2":   "Depth Anything V2",
    "da3":    "Depth Anything 3",
    "vda":    "Video Depth Anything",
}

COLORS = {
    "msense": (0,   220, 255),
    "dav2":   (80,  220, 255),
    "da3":    (255, 120,   0),
    "vda":    (180, 255,   0),
}


# ---------------------------------------------------------------------------
# Chargement distances depuis les CSV
# ---------------------------------------------------------------------------

def load_distances(bench_dir: Path, method: str) -> dict[str, dict]:
    """Retourne {frame_name: {dist, bbox, score}} depuis tous les CSV de la méthode."""
    result: dict[str, dict] = {}
    method_dir = bench_dir / method
    if not method_dir.exists():
        return result

    for csv_path in method_dir.glob("*.csv"):
        try:
            with csv_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    frame = row.get("frame", "")
                    if not frame:
                        continue

                    dist = None
                    for col in ("median_m", "mean_m", "dist_m", "distance_m"):
                        if row.get(col):
                            try:
                                dist = float(row[col]); break
                            except ValueError:
                                pass

                    bbox = None
                    try:
                        x1 = int(float(row["x1"]))
                        y1 = int(float(row["y1"]))
                        x2 = int(float(row["x2"]))
                        y2 = int(float(row["y2"]))
                        if x2 > x1 and y2 > y1:
                            bbox = (x1, y1, x2, y2)
                    except (KeyError, ValueError):
                        pass

                    score = float(row.get("score", 0) or 0)
                    if frame not in result or score > result[frame].get("score", 0):
                        result[frame] = {"dist": dist, "bbox": bbox, "score": score}
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Chargement image du panneau
# ---------------------------------------------------------------------------

def load_cell(bench_dir: Path, method: str, frame_name: str, frames_dir: Path) -> tuple[np.ndarray | None, bool]:
    """
    Retourne (image, is_raw).
    msense  → frame brute (is_raw=True  → bbox dessinée)
    autres  → depth colormap moitié droite du debug (is_raw=False → pas de bbox)
    """
    debug_dir = bench_dir / method / "_debug"
    stem = Path(frame_name).stem

    for ext in (".jpg", ".png", ".jpeg"):
        p = debug_dir / (stem + ext)
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                if method == "msense":
                    return img, True
                # Depth colormap = moitié droite du side-by-side
                h, w = img.shape[:2]
                if w > h * 1.5:
                    img = img[:, w // 2:]
                return img, False

    # Fallback : frame brute
    raw = frames_dir / frame_name
    if raw.exists():
        img = cv2.imread(str(raw))
        if img is not None:
            return img, True

    return None, False


# ---------------------------------------------------------------------------
# Rendu d'un panneau (overlay)
# ---------------------------------------------------------------------------

def draw_panel(img: np.ndarray, method: str, dist, bbox, is_raw: bool,
               cell_w: int, cell_h: int, orig_w: int, orig_h: int) -> np.ndarray:

    # Redimensionner
    out = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    color = COLORS.get(method, (255, 255, 255))
    label = METHOD_LABELS.get(method, method.upper())

    # Échelle typographique (référence 960 px de large)
    sc        = cell_w / 960
    bar_h     = int(96 * sc)
    fs_label  = 1.0 * sc
    fs_dist   = 1.55 * sc
    fs_bbox   = 1.0 * sc
    thick     = max(2, int(3 * sc))
    box_thick = max(3, int(4 * sc))
    border    = max(3, int(5 * sc))

    # BBox seulement sur la frame brute (msense)
    if bbox and is_raw and orig_w > 0 and orig_h > 0:
        x1, y1, x2, y2 = bbox
        # Mise à l'échelle des coordonnées vers cell_w × cell_h
        sx = cell_w / orig_w
        sy = cell_h / orig_h
        x1b, x2b = int(x1 * sx), int(x2 * sx)
        y1b, y2b = int(y1 * sy), int(y2 * sy)
        y1b = max(y1b, bar_h)          # éviter que la bbox couvre le bandeau

        cv2.rectangle(out, (x1b, y1b), (x2b, y2b), color, box_thick)

        if dist is not None:
            txt = f"{dist:.1f} m"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs_bbox, thick)
            ty = max(y1b - 6, bar_h + th + 6)
            cv2.rectangle(out, (x1b - 2, ty - th - 4), (x1b + tw + 6, ty + 4), (0, 0, 0), -1)
            cv2.putText(out, txt, (x1b + 3, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_bbox, color, thick, cv2.LINE_AA)

    # Bandeau semi-transparent
    bar = out.copy()
    cv2.rectangle(bar, (0, 0), (cell_w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.62, out, 0.38, 0, out)

    # Bordure colorée
    cv2.rectangle(out, (0, 0), (cell_w - 1, cell_h - 1), color, border)

    # Nom de la méthode
    cv2.putText(out, label, (12, int(bar_h * 0.38)),
                cv2.FONT_HERSHEY_SIMPLEX, fs_label, color, thick, cv2.LINE_AA)

    # Distance — grande et visible
    dist_str = f"dist: {dist:.2f} m" if dist is not None else "dist: N/A"
    cv2.putText(out, dist_str, (12, int(bar_h * 0.82)),
                cv2.FONT_HERSHEY_SIMPLEX, fs_dist, color, thick + 1, cv2.LINE_AA)

    return out


def make_placeholder(method: str, cell_w: int, cell_h: int) -> np.ndarray:
    img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    color = COLORS.get(method, (60, 60, 60))
    label = METHOD_LABELS.get(method, method.upper())
    cv2.putText(img, label, (12, cell_h // 2 - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * cell_w / 960, color, 2, cv2.LINE_AA)
    cv2.putText(img, "N/A", (12, cell_h // 2 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0 * cell_w / 960, (60, 60, 60), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Génération
# ---------------------------------------------------------------------------

def make_video(bench_dir: Path, out_path: Path, fps: float, cell_w: int, cell_h: int) -> None:
    frames_dir = bench_dir / "frames"

    # Liste des frames
    frame_names: list[str] = []
    for m in METHODS:
        d = bench_dir / m / "_debug"
        if d.exists():
            frame_names = sorted(
                p.name for p in d.iterdir()
                if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
            )
            if frame_names:
                break
    if not frame_names and frames_dir.exists():
        frame_names = sorted(
            p.name for p in frames_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
        )

    if not frame_names:
        print(f"[ERR] Aucune frame dans {bench_dir}")
        return

    # Dimensions originales (pour mise à l'échelle des bbox)
    orig_w, orig_h = 0, 0
    sample = frames_dir / frame_names[0] if frames_dir.exists() else None
    if sample and sample.exists():
        s = cv2.imread(str(sample))
        if s is not None:
            orig_h, orig_w = s.shape[:2]

    dist_data = {m: load_distances(bench_dir, m) for m in METHODS}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (cell_w * 2, cell_h * 2))

    print(f"[info] {len(frame_names)} frames — grille 2×2 ({cell_w*2}×{cell_h*2}) → {out_path.name}")

    for idx, frame_name in enumerate(frame_names):
        cells = []
        for method in METHODS:
            if not (bench_dir / method).exists():
                cells.append(make_placeholder(method, cell_w, cell_h))
                continue

            img, is_raw = load_cell(bench_dir, method, frame_name, frames_dir)
            info = dist_data[method].get(frame_name, {})
            dist = info.get("dist")
            bbox = info.get("bbox")

            if img is None:
                cells.append(make_placeholder(method, cell_w, cell_h))
            else:
                cells.append(draw_panel(img, method, dist, bbox, is_raw,
                                        cell_w, cell_h, orig_w, orig_h))

        row1 = np.hstack([cells[0], cells[1]])
        row2 = np.hstack([cells[2], cells[3]])
        writer.write(np.vstack([row1, row2]))

        if idx % 50 == 0:
            print(f"  {idx + 1}/{len(frame_names)}", flush=True)

    writer.release()
    print(f"[done] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--out",       type=Path, default=None)
    ap.add_argument("--fps",       type=float, default=10.0)
    ap.add_argument("--cell-w",    type=int,   default=960)
    ap.add_argument("--cell-h",    type=int,   default=540)
    args = ap.parse_args()

    out = args.out or (
        args.benchmark / "comparison_depth_methods" / f"{args.benchmark.name}_comparison.mp4"
    )
    make_video(args.benchmark, out, args.fps, args.cell_w, args.cell_h)


if __name__ == "__main__":
    main()
