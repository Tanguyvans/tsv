"""Suivi temporel des panneaux et courbes de distance.

Le script part d'un dossier benchmark produit par ``benchmark_depth_methods.py``:

- ``clip_*/frames`` contient les images;
- ``clip_*/detections`` contient les JSON YOLO;
- ``clip_*/{vda,dav2,da3}/_<method>_distances.csv`` contient les distances.

Il crée des tracks simples par IoU, joint les distances de chaque méthode, puis
exporte des CSV, des courbes et une vidéo annotée.

Exemple:
    PYTHONPATH=. python src/depth/track_distance_timeseries.py \\
        --benchmark outputs/depth_method_benchmark_methods_5x12 \\
        --methods vda,dav2,da3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {
    "vda": (180, 255, 0),
    "dav2": (80, 220, 255),
    "da3": (255, 120, 0),
}


@dataclass
class Detection:
    frame: str
    frame_index: int
    det_id: int
    score: float
    bbox: list[int]


@dataclass
class Track:
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    missed: int = 0

    @property
    def last(self) -> Detection:
        return self.detections[-1]


def iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def frame_index_from_name(name: str) -> int:
    stem = Path(name).stem
    tail = stem.rsplit("_", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return -1


def load_detections(clip_dir: Path) -> list[Detection]:
    det_dir = clip_dir / "detections"
    rows: list[Detection] = []
    for json_path in sorted(p for p in det_dir.glob("*.json") if not p.name.startswith("_")):
        data = json.loads(json_path.read_text())
        frame = data.get("frame", f"{json_path.stem}.jpg")
        frame_index = frame_index_from_name(frame)
        for det_id, det in enumerate(data.get("detections", [])):
            rows.append(
                Detection(
                    frame=frame,
                    frame_index=frame_index,
                    det_id=det_id,
                    score=float(det.get("score", 0.0)),
                    bbox=[int(v) for v in det["bbox"]],
                )
            )
    return sorted(rows, key=lambda d: (d.frame_index, d.det_id))


def build_tracks(
    detections: list[Detection],
    *,
    min_iou: float,
    max_missed: int,
) -> list[Track]:
    active: list[Track] = []
    finished: list[Track] = []
    next_id = 1
    frames = sorted({d.frame_index for d in detections})
    by_frame: dict[int, list[Detection]] = {}
    for det in detections:
        by_frame.setdefault(det.frame_index, []).append(det)

    for frame_index in frames:
        frame_dets = by_frame.get(frame_index, [])
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()

        candidates = []
        for t_idx, track in enumerate(active):
            for d_idx, det in enumerate(frame_dets):
                candidates.append((iou(track.last.bbox, det.bbox), t_idx, d_idx))
        for score, t_idx, d_idx in sorted(candidates, reverse=True):
            if score < min_iou or t_idx in assigned_tracks or d_idx in assigned_dets:
                continue
            active[t_idx].detections.append(frame_dets[d_idx])
            active[t_idx].missed = 0
            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)

        for t_idx, track in enumerate(active):
            if t_idx not in assigned_tracks:
                track.missed += 1

        still_active = []
        for track in active:
            if track.missed > max_missed:
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active

        for d_idx, det in enumerate(frame_dets):
            if d_idx in assigned_dets:
                continue
            active.append(Track(track_id=next_id, detections=[det]))
            next_id += 1

    return finished + active


def load_method_distances(clip_dir: Path, methods: list[str]) -> dict[tuple[str, int, str], dict]:
    out: dict[tuple[str, int, str], dict] = {}
    for method in methods:
        csv_path = clip_dir / method / f"_{method}_distances.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["frame"], int(row["det_id"]), method)
                out[key] = row
    return out


def quality_label(values: list[float]) -> tuple[str, str]:
    if len(values) < 3:
        return "low", "track trop court"
    diffs = np.diff(values)
    decreasing_ratio = float(np.mean(diffs <= 0))
    max_jump = float(np.max(np.abs(diffs))) if len(diffs) else 0.0
    span = float(max(values) - min(values))
    if decreasing_ratio >= 0.75 and max_jump <= max(5.0, span * 0.8):
        return "high", "distance globalement décroissante"
    if decreasing_ratio >= 0.55:
        return "medium", "quelques sauts mais tendance utilisable"
    return "low", "distance non monotone ou instable"


def write_track_rows(
    clip_dir: Path,
    tracks: list[Track],
    distances: dict[tuple[str, int, str], dict],
    methods: list[str],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for track in tracks:
        for det in track.detections:
            row = {
                "clip": clip_dir.name,
                "track_id": track.track_id,
                "frame": det.frame,
                "frame_index": det.frame_index,
                "det_id": det.det_id,
                "score": round(det.score, 4),
                "x1": det.bbox[0],
                "y1": det.bbox[1],
                "x2": det.bbox[2],
                "y2": det.bbox[3],
            }
            for method in methods:
                dist = distances.get((det.frame, det.det_id, method), {})
                row[f"{method}_median_m"] = dist.get("median_m", "")
                row[f"{method}_iqr_m"] = dist.get("iqr_m", "")
            rows.append(row)

    track_csv = out_dir / f"{clip_dir.name}_tracks.csv"
    fieldnames = [
        "clip", "track_id", "frame", "frame_index", "det_id", "score",
        "x1", "y1", "x2", "y2",
    ]
    for method in methods:
        fieldnames.extend([f"{method}_median_m", f"{method}_iqr_m"])
    with track_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for track in tracks:
        item = {
            "clip": clip_dir.name,
            "track_id": track.track_id,
            "n_frames": len(track.detections),
            "first_frame": track.detections[0].frame_index,
            "last_frame": track.detections[-1].frame_index,
        }
        for method in methods:
            values = []
            iqrs = []
            for det in track.detections:
                dist = distances.get((det.frame, det.det_id, method), {})
                if dist.get("median_m"):
                    values.append(float(dist["median_m"]))
                if dist.get("iqr_m"):
                    iqrs.append(float(dist["iqr_m"]))
            if values:
                label, reason = quality_label(values)
                item[f"{method}_first_m"] = round(values[0], 3)
                item[f"{method}_last_m"] = round(values[-1], 3)
                item[f"{method}_delta_m"] = round(values[-1] - values[0], 3)
                item[f"{method}_mean_iqr_m"] = round(float(np.mean(iqrs)), 3) if iqrs else ""
                item[f"{method}_quality"] = label
                item[f"{method}_quality_reason"] = reason
            else:
                item[f"{method}_first_m"] = ""
                item[f"{method}_last_m"] = ""
                item[f"{method}_delta_m"] = ""
                item[f"{method}_mean_iqr_m"] = ""
                item[f"{method}_quality"] = "none"
                item[f"{method}_quality_reason"] = "pas de distance"
        summary_rows.append(item)

    summary_csv = out_dir / f"{clip_dir.name}_track_summary.csv"
    summary_fields = ["clip", "track_id", "n_frames", "first_frame", "last_frame"]
    for method in methods:
        summary_fields.extend([
            f"{method}_first_m",
            f"{method}_last_m",
            f"{method}_delta_m",
            f"{method}_mean_iqr_m",
            f"{method}_quality",
            f"{method}_quality_reason",
        ])
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    return track_csv, summary_csv


def plot_tracks(track_csv: Path, methods: list[str], out_dir: Path) -> list[Path]:
    rows = []
    with track_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    plots = []
    track_ids = sorted({int(r["track_id"]) for r in rows})
    for track_id in track_ids:
        track_rows = [r for r in rows if int(r["track_id"]) == track_id]
        if len(track_rows) < 2:
            continue
        plt.figure(figsize=(8, 4.5))
        for method in methods:
            xs, ys = [], []
            for row in track_rows:
                value = row.get(f"{method}_median_m")
                if value:
                    xs.append(int(row["frame_index"]))
                    ys.append(float(value))
            if xs:
                plt.plot(xs, ys, marker="o", label=method.upper())
        plt.gca().invert_xaxis()
        plt.xlabel("frame index (plus petit = plus tôt)")
        plt.ylabel("distance estimée (m)")
        plt.title(f"{track_rows[0]['clip']} / track {track_id}")
        plt.grid(True, alpha=0.25)
        plt.legend()
        out_path = out_dir / f"{track_rows[0]['clip']}_track_{track_id:02d}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=140)
        plt.close()
        plots.append(out_path)
    return plots


def encode_video(frame_dir: Path, out_video: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%05d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_video),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def make_track_video(
    clip_dir: Path,
    tracks: list[Track],
    distances: dict[tuple[str, int, str], dict],
    methods: list[str],
    out_dir: Path,
    *,
    fps: float,
) -> Path | None:
    frames_dir = clip_dir / "frames"
    tmp_dir = out_dir / f"_{clip_dir.name}_annotated_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    det_by_frame = {}
    for track in tracks:
        for det in track.detections:
            det_by_frame.setdefault(det.frame, []).append((track.track_id, det))

    written = 0
    for image_path in sorted(frames_dir.glob("*.jpg")):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        for track_id, det in det_by_frame.get(image_path.name, []):
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 255, 120), 2)
            pieces = [f"T{track_id}"]
            for method in methods:
                dist = distances.get((det.frame, det.det_id, method), {})
                if dist.get("median_m"):
                    pieces.append(f"{method}:{float(dist['median_m']):.1f}m")
            label = " ".join(pieces)
            cv2.putText(
                img,
                label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (40, 255, 120),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(tmp_dir / f"frame_{written:05d}.jpg"), img)
        written += 1

    if not written:
        return None
    out_video = out_dir / f"{clip_dir.name}_tracks.mp4"
    encode_video(tmp_dir, out_video, fps)
    for p in tmp_dir.glob("*.jpg"):
        p.unlink()
    tmp_dir.rmdir()
    return out_video


def combine_summaries(summary_paths: list[Path], out_dir: Path) -> Path:
    all_rows = []
    fieldnames: list[str] = []
    for path in summary_paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if not fieldnames:
                fieldnames = list(reader.fieldnames or [])
            all_rows.extend(reader)
    out_path = out_dir / "all_track_summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--methods", default="vda,dav2,da3")
    ap.add_argument("--min-iou", type=float, default=0.20)
    ap.add_argument("--max-missed", type=int, default=1)
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    out_dir = args.out or args.benchmark / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = []
    for clip_dir in sorted(p for p in args.benchmark.glob("clip_*") if p.is_dir()):
        detections = load_detections(clip_dir)
        tracks = build_tracks(detections, min_iou=args.min_iou, max_missed=args.max_missed)
        distances = load_method_distances(clip_dir, methods)
        track_csv, summary_csv = write_track_rows(clip_dir, tracks, distances, methods, out_dir)
        plot_tracks(track_csv, methods, out_dir)
        video = make_track_video(clip_dir, tracks, distances, methods, out_dir, fps=args.fps)
        print(f"[clip] {clip_dir.name}: {len(tracks)} track(s)")
        print(f"  CSV: {track_csv}")
        print(f"  summary: {summary_csv}")
        if video:
            print(f"  video: {video}")
        summary_paths.append(summary_csv)

    combined = combine_summaries(summary_paths, out_dir)
    print(f"[out] summary global: {combined}")


if __name__ == "__main__":
    main()
