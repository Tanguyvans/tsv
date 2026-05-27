"""Estime la distance des panneaux detectes dans les videos MultiSense.

Le pipeline utilise COLOR pour la detection YOLO et DEPTH pour la distance.
Les videos DEPTH sont encodees en FFV1 gray16le, avec une profondeur en mm.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


WIDTH = 1920
HEIGHT = 1200
INVALID_DEPTH = 65535


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def read_raw_frame(path: Path, frame_idx: int, pix_fmt: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    cmd = [
        ffmpeg_exe(),
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"select=eq(n\\,{frame_idx})",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-",
    ]
    raw = subprocess.check_output(cmd)
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(raw) != expected:
        raise RuntimeError(f"Lecture incomplete pour {path} frame {frame_idx}: {len(raw)} octets, attendu {expected}")
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def read_color_frame(path: Path, frame_idx: int) -> np.ndarray:
    """Retourne une frame BGR uint8."""
    return read_raw_frame(path, frame_idx, "bgr24", "uint8", (HEIGHT, WIDTH, 3)).copy()


def read_depth_frame(path: Path, frame_idx: int) -> np.ndarray:
    """Retourne une depth map uint16 en millimetres."""
    return read_raw_frame(path, frame_idx, "gray16le", "<u2", (HEIGHT, WIDTH)).copy()


def load_time_rows(time_path: Path) -> list[dict[str, str]]:
    if not time_path.exists():
        return []

    lines = time_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("frameID,")), None)
    if header_idx is None:
        return []

    reader = csv.DictReader(lines[header_idx:])
    return list(reader)


def clip_box(box: list[float], width: int, height: int, shrink: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    if shrink > 0:
        bw = x2 - x1
        bh = y2 - y1
        x1 += bw * shrink
        x2 -= bw * shrink
        y1 += bh * shrink
        y2 -= bh * shrink

    ix1 = max(0, min(width - 1, int(round(x1))))
    iy1 = max(0, min(height - 1, int(round(y1))))
    ix2 = max(ix1 + 1, min(width, int(round(x2))))
    iy2 = max(iy1 + 1, min(height, int(round(y2))))
    return ix1, iy1, ix2, iy2


def depth_stats(depth: np.ndarray, box: tuple[int, int, int, int], max_depth_mm: int) -> dict[str, float | int | None]:
    x1, y1, x2, y2 = box
    crop = depth[y1:y2, x1:x2]
    valid = crop[(crop > 0) & (crop < INVALID_DEPTH) & (crop <= max_depth_mm)]

    if valid.size == 0:
        return {
            "valid_px": 0,
            "distance_m": None,
            "depth_p10_m": None,
            "depth_p90_m": None,
        }

    p10, p50, p90 = np.percentile(valid, [10, 50, 90])
    return {
        "valid_px": int(valid.size),
        "distance_m": float(p50 / 1000.0),
        "depth_p10_m": float(p10 / 1000.0),
        "depth_p90_m": float(p90 / 1000.0),
    }


def draw_detection(img: np.ndarray, row: dict) -> None:
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
    dist = row["distance_m"]
    name = row["class_name"]
    label = f"{name} {row['conf']:.2f}"
    if dist is not None:
        label += f" {dist:.1f}m"

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.putText(img, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)


def parse_frames(args: argparse.Namespace) -> list[int]:
    if args.frames:
        return sorted({int(v.strip()) for v in args.frames.split(",") if v.strip()})
    return list(range(args.start, args.end + 1, args.stride))


def parse_seqs(args: argparse.Namespace) -> list[int]:
    if args.seqs:
        return sorted({int(v.strip()) for v in args.seqs.split(",") if v.strip()})
    return [args.seq]


def max_rss_mb() -> float:
    # Linux/macOS report ru_maxrss in KiB in the environments used here.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_paths(data_root: Path, campaign: str, seq: int) -> tuple[Path, Path, Path]:
    video_root = data_root / campaign / "RawData" / "Video" / "MSense"
    color_path = video_root / "COLOR" / f"color_{seq}.mkv"
    depth_path = video_root / "DEPTH" / f"depth_{seq}.mkv"
    time_path = video_root / "TIME" / f"avi_{seq}.time"
    return color_path, depth_path, time_path


def process_sequence(args: argparse.Namespace, model, names: dict[int, str], seq: int, frames: list[int]) -> tuple[list[dict], dict]:
    color_path, depth_path, time_path = build_paths(Path(args.data_root), args.campaign, seq)
    time_rows = load_time_rows(time_path)
    rows: list[dict] = []
    seq_start = time.perf_counter()

    vis_dir = Path(args.vis_dir)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in frames:
        frame_start = time.perf_counter()
        color = read_color_frame(color_path, frame_idx)
        depth = read_depth_frame(depth_path, frame_idx)
        result = model.predict(color, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        time_info = time_rows[frame_idx] if frame_idx < len(time_rows) else {}

        vis = color.copy() if args.save_vis else None
        for det_id, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = clip_box(box.xyxy[0].tolist(), WIDTH, HEIGHT, args.bbox_shrink)
            stats = depth_stats(depth, (x1, y1, x2, y2), args.max_depth_mm)
            row = {
                "seq": seq,
                "frame": frame_idx,
                "det_id": det_id,
                "class_id": cls_id,
                "class_name": str(names.get(cls_id, cls_id)),
                "conf": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                **stats,
                "frame_runtime_s": time.perf_counter() - frame_start,
                "m_stTime": time_info.get("m_stTime"),
                "timeCam": time_info.get("timeCam"),
                "east": time_info.get("EastPosition"),
                "north": time_info.get("NordPosition"),
                "speed": time_info.get("speed"),
            }
            rows.append(row)
            if vis is not None:
                draw_detection(vis, row)

        if vis is not None:
            cv2.imwrite(str(vis_dir / f"seq{seq:02d}_frame{frame_idx:06d}.jpg"), vis)

    elapsed_s = time.perf_counter() - seq_start
    metrics = {
        "seq": seq,
        "frames": len(frames),
        "detections": len(rows),
        "elapsed_s": elapsed_s,
        "fps": len(frames) / elapsed_s if elapsed_s > 0 else 0.0,
        "max_rss_mb": max_rss_mb(),
    }
    return rows, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA", "/gpfs/projects/acad/brainai/cegelecRecordings"))
    ap.add_argument("--campaign", default="Campaign_2026_03_24_12_59_24")
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--seqs", help="Liste de sequences, ex: 0,1,2. Prioritaire sur --seq.")
    ap.add_argument("--ckpt", required=True, help="Checkpoint YOLO detecteur de panneaux/mats.")
    ap.add_argument("--frames", help="Liste de frames, ex: 0,10,50. Prioritaire sur start/end/stride.")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--bbox-shrink", type=float, default=0.10, help="Reduit la bbox avant d'echantillonner la depth.")
    ap.add_argument("--max-depth-mm", type=int, default=65000)
    ap.add_argument("--out-csv", default="signal_distances.csv")
    ap.add_argument("--out-jsonl", default="signal_distances.jsonl")
    ap.add_argument("--metrics-json", default="metrics.json")
    ap.add_argument("--vis-dir", default="vis")
    ap.add_argument("--save-vis", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.ckpt)
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    seqs = parse_seqs(args)
    frames = parse_frames(args)

    run_start = time.perf_counter()
    rows: list[dict] = []
    metrics: list[dict] = []
    for seq in seqs:
        seq_rows, seq_metrics = process_sequence(args, model, names, seq, frames)
        rows.extend(seq_rows)
        metrics.append(seq_metrics)

    fieldnames = list(rows[0].keys()) if rows else [
        "seq",
        "frame",
        "det_id",
        "class_id",
        "class_name",
        "conf",
        "x1",
        "y1",
        "x2",
        "y2",
        "valid_px",
        "distance_m",
        "depth_p10_m",
        "depth_p90_m",
        "frame_runtime_s",
        "m_stTime",
        "timeCam",
        "east",
        "north",
        "speed",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_elapsed_s = time.perf_counter() - run_start
    summary = {
        "seqs": seqs,
        "frames_per_seq": len(frames),
        "total_frames": len(frames) * len(seqs),
        "total_detections": len(rows),
        "elapsed_s": total_elapsed_s,
        "fps": (len(frames) * len(seqs)) / total_elapsed_s if total_elapsed_s > 0 else 0.0,
        "max_rss_mb": max_rss_mb(),
        "per_seq": metrics,
    }
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"{len(rows)} detections ecrites dans {args.out_csv} et {args.out_jsonl}")
    print(
        f"benchmark: {summary['total_frames']} frames, {summary['elapsed_s']:.2f}s, "
        f"{summary['fps']:.2f} FPS, RAM max {summary['max_rss_mb']:.1f} MB"
    )


if __name__ == "__main__":
    main()
