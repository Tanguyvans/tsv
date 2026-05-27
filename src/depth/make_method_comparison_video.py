"""Crée des vidéos pour comparer les méthodes de profondeur.

Layout:
    MSENSE | VDA | DAV2 | DA3

Exemple:
    PYTHONPATH=. python src/depth/make_method_comparison_video.py \\
        --benchmark outputs/depth_method_benchmark_methods_5x12
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


QUADS = [
    ("MSENSE", "msense"),
    ("VDA", "vda"),
    ("DAV2", "dav2"),
    ("DA3", "da3"),
]

METHOD_COLOR_BGR = {
    "msense": (0, 220, 255),
    "vda": (180, 255, 0),
    "dav2": (80, 220, 255),
    "da3": (255, 120, 0),
}


def load_distance_tables(clip_dir: Path, quads: list[tuple[str, str]] | None = None) -> dict[str, dict[tuple[str, int], dict]]:
    quads = quads or QUADS
    tables: dict[str, dict[tuple[str, int], dict]] = {}
    for _, method in quads:
        csv_path = clip_dir / method / f"_{method}_distances.csv"
        if method == "msense":
            csv_path = clip_dir / method / "_msense_distances.csv"
        table: dict[tuple[str, int], dict] = {}
        if csv_path.exists():
            with csv_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    table[(row["frame"], int(row["det_id"]))] = row
        tables[method] = table
    return tables


def load_detection_table(clip_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted((clip_dir / "detections").glob("*.json")):
        if path.name.startswith("_"):
            continue
        data = __import__("json").loads(path.read_text())
        out[data.get("frame", f"{path.stem}.jpg")] = data.get("detections", [])
    return out


def extract_depth_colormap(img: np.ndarray) -> np.ndarray:
    """Extrait la moitié droite (depth colormap) d'une image side-by-side [frame | depth]."""
    h, w = img.shape[:2]
    if w > h * 1.5:          # probablement side-by-side
        return img[:, w // 2:]
    return img


def load_panel(clip_dir: Path, method: str, frame_path: Path, expected_width: int) -> np.ndarray:
    raw = cv2.imread(str(frame_path))
    if raw is None:
        raise RuntimeError(f"image illisible: {frame_path}")

    debug_path = clip_dir / method / "_debug" / frame_path.name

    if debug_path.exists():
        img = cv2.imread(str(debug_path))
        if img is not None:
            if method == "msense":
                return img                      # frame brute, pas de depth colormap
            return extract_depth_colormap(img)  # moitié droite = depth colormap
    return raw


def draw_title(img: np.ndarray, title: str, method: str, dist_str: str = "") -> np.ndarray:
    """Bandeau haut : nom de la méthode + distance estimée."""
    out = img.copy()
    overlay = out.copy()
    color = METHOD_COLOR_BGR.get(method, (255, 255, 255))
    w = out.shape[1]

    # Échelle adaptée à la largeur de la cellule (référence 640 px)
    scale = w / 640
    bar_h = int(90 * scale)
    fs_title = 1.4 * scale
    fs_dist  = 1.1 * scale
    thick    = max(2, int(3 * scale))
    border   = max(4, int(6 * scale))

    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.60, out, 0.40, 0)
    cv2.rectangle(out, (0, 0), (w - 1, out.shape[0] - 1), color, border)

    cv2.putText(out, title, (14, int(bar_h * 0.42)),
                cv2.FONT_HERSHEY_SIMPLEX, fs_title, color, thick, cv2.LINE_AA)
    if dist_str:
        cv2.putText(out, dist_str, (14, int(bar_h * 0.88)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_dist, color, thick, cv2.LINE_AA)
    return out


def draw_readable_distances(
    img: np.ndarray,
    *,
    frame_name: str,
    method: str,
    detections: list[dict],
    distances: dict[str, dict[tuple[str, int], dict]],
) -> tuple[np.ndarray, str]:
    """Retourne l'image avec les bbox + la distance de la meilleure détection."""
    out = img.copy()
    color = METHOD_COLOR_BGR.get(method, (255, 255, 255))
    best_dist_str = ""
    best_score = -1.0

    w = out.shape[1]
    scale = w / 640
    fs = 1.0 * scale
    thick = max(2, int(3 * scale))
    box_thick = max(3, int(4 * scale))

    for det_id, det in enumerate(detections):
        key = (frame_name, det_id)
        row = distances.get(method, {}).get(key)
        if not row:
            continue

        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        med = float(row["median_m"])
        iqr = float(row["iqr_m"]) if row.get("iqr_m") else 0.0
        score = float(det.get("score", det.get("confidence", 0)) or 0)
        label = f"{med:.1f} m  iqr {iqr:.1f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), base = cv2.getTextSize(label, font, fs, thick)
        tx = max(0, min(x1, out.shape[1] - tw - 8))
        ty = max(th + 8, y1 - 10)
        cv2.rectangle(out, (tx - 2, ty - th - 6), (tx + tw + 6, ty + base + 4), (0, 0, 0), -1)
        cv2.putText(out, label, (tx + 2, ty), font, fs, color, thick, cv2.LINE_AA)

        if score > best_score:
            best_score = score
            best_dist_str = f"dist: {med:.2f} m"

    return out, best_dist_str




def add_gutters(panels: list[np.ndarray], gutter: int) -> np.ndarray:
    h, w = panels[0].shape[:2]
    canvas = np.zeros((h, w * len(panels) + gutter * (len(panels) - 1), 3), dtype=np.uint8)
    for idx, panel in enumerate(panels):
        x = idx * (w + gutter)
        canvas[:, x:x + w] = panel
    return canvas


def make_clip_video(
    clip_dir: Path,
    out_dir: Path,
    *,
    fps: float,
    cell_width: int,
    gutter: int,
    quads: list[tuple[str, str]] | None = None,
) -> Path | None:
    quads = quads or QUADS
    frames = sorted((clip_dir / "frames").glob("*.jpg"))
    if not frames:
        return None

    first = cv2.imread(str(frames[0]))
    if first is None:
        return None
    expected_width = first.shape[1]
    cell_height = round(cell_width * first.shape[0] / first.shape[1])

    out_dir.mkdir(parents=True, exist_ok=True)
    detections = load_detection_table(clip_dir)
    distances = load_distance_tables(clip_dir, quads)

    # Calcul des dimensions finales de la grille
    grid_w = cell_width * len(quads) + gutter * (len(quads) - 1)
    out_video = out_dir / f"{clip_dir.name}_methods.mp4"
    writer = cv2.VideoWriter(str(out_video), FOURCC, fps, (grid_w, cell_height))

    for idx, frame_path in enumerate(frames):
        panels = []
        for title, method in quads:
            img = load_panel(clip_dir, method, frame_path, expected_width)
            img, dist_str = draw_readable_distances(
                img,
                frame_name=frame_path.name,
                method=method,
                detections=detections.get(frame_path.name, []),
                distances=distances,
            )
            img = cv2.resize(img, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
            panels.append(draw_title(img, title, method, dist_str))

        writer.write(add_gutters(panels, gutter))
        if idx % 20 == 0:
            print(f"  frame {idx+1}/{len(frames)}", flush=True)

    writer.release()
    return out_video


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--cell-width", type=int, default=640)
    ap.add_argument("--gutter", type=int, default=28)
    args = ap.parse_args()

    out_dir = args.out or args.benchmark / "comparison_depth_methods"
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_dirs = sorted(p for p in args.benchmark.glob("clip_*") if p.is_dir())
    if not clip_dirs and (args.benchmark / "frames").exists():
        clip_dirs = [args.benchmark]

    for clip_dir in clip_dirs:
        # Garde uniquement les méthodes présentes dans ce dossier
        available = [(t, m) for t, m in QUADS if (clip_dir / m).exists()]
        if not available:
            print(f"[skip] {clip_dir.name}: aucune méthode trouvée")
            continue
        missing = [m for _, m in QUADS if not (clip_dir / m).exists()]
        if missing:
            print(f"[info] {clip_dir.name}: méthodes absentes ignorées → {missing}")
        video = make_clip_video(
            clip_dir,
            out_dir,
            fps=args.fps,
            cell_width=args.cell_width,
            gutter=args.gutter,
            quads=available,
        )
        if video:
            print(f"[out] {video}")


if __name__ == "__main__":
    main()
