"""Banc de comparaison des méthodes de distance sur clips cab-view.

Le script orchestre une évaluation reproductible:
1. extrait des frames depuis les clips sélectionnés;
2. lance YOLO pour détecter les panneaux;
3. lance les estimateurs de profondeur disponibles (Depth Anything / VDA);
4. consolide les CSV et génère des vidéos debug.

Exemple:
    PYTHONPATH=. python src/depth/benchmark_depth_methods.py \\
        --manifest outputs/selected_10s_clips/manifest.json \\
        --methods vda,dav2,da3 \\
        --fps 4 \\
        --max-frames 12
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = WORKSPACE / "outputs/selected_10s_clips/manifest.json"
DEFAULT_OUT = WORKSPACE / "outputs/depth_method_benchmark"
DEFAULT_YOLO_WEIGHTS = WORKSPACE / "models/yolo26s-railway-signs-detector/best.pt"
DEFAULT_VDA_CHECKPOINT = (
    WORKSPACE
    / "venv/src/video-depth-anything/checkpoints/metric_video_depth_anything_vits.pth"
)
DEFAULT_DAV2_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


@dataclass(frozen=True)
class ClipSpec:
    id: int
    frame_idx: int
    start_s: float
    duration_s: float
    clip: Path

    @property
    def slug(self) -> str:
        return f"clip_{self.id:02d}_frame_{self.frame_idx:06d}"


def parse_methods(raw: str) -> list[str]:
    methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    methods = ["dav2" if m in {"da2", "depthanythingv2"} else m for m in methods]
    known = {"vda", "dav2", "da3"}
    unknown = sorted(set(methods) - known)
    if unknown:
        raise SystemExit(f"Méthode(s) inconnue(s): {', '.join(unknown)}")
    return methods


def load_manifest(path: Path, selected_ids: set[int] | None) -> list[ClipSpec]:
    data = json.loads(path.read_text())
    clips = []
    for item in data:
        clip_id = int(item["id"])
        if selected_ids is not None and clip_id not in selected_ids:
            continue
        clip_path = Path(item["clip"])
        if not clip_path.is_absolute():
            clip_path = WORKSPACE / clip_path
        clips.append(
            ClipSpec(
                id=clip_id,
                frame_idx=int(item["frame_idx"]),
                start_s=float(item["start_s"]),
                duration_s=float(item["duration_s"]),
                clip=clip_path,
            )
        )
    if not clips:
        raise SystemExit(f"Aucun clip sélectionné dans {path}")
    return clips


def run_command(cmd: list[str], *, cwd: Path, env_prefix: str = "PYTHONPATH=.") -> float:
    printable = " ".join(cmd)
    print(f"\n[run] {env_prefix} {printable}")
    t0 = time.time()
    env = None
    if env_prefix == "PYTHONPATH=.":
        import os

        env = os.environ.copy()
        env["PYTHONPATH"] = "."
    subprocess.run(cmd, cwd=cwd, env=env, check=True)
    return (time.time() - t0) * 1000.0


def extract_frames(
    clip: ClipSpec,
    frames_dir: Path,
    *,
    fps: float,
    max_frames: int | None,
    sample_position: str,
    overwrite: bool,
) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("*.jpg"))
    if existing and not overwrite:
        return existing[:max_frames] if max_frames else existing

    for old in existing:
        old.unlink()

    cap = cv2.VideoCapture(str(clip.clip))
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir le clip: {clip.clip}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(src_fps / fps))
    keep_indices = list(range(0, total_frames, step)) if total_frames else []
    if max_frames is not None and keep_indices:
        if sample_position == "start":
            keep_indices = keep_indices[:max_frames]
        elif sample_position == "middle":
            center = len(keep_indices) // 2
            start = max(0, center - max_frames // 2)
            keep_indices = keep_indices[start:start + max_frames]
        elif sample_position == "end":
            keep_indices = keep_indices[-max_frames:]
        else:
            raise SystemExit(f"sample_position invalide: {sample_position}")
    keep_set = set(keep_indices)

    frames = []
    frame_idx = 0
    kept_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        keep = frame_idx in keep_set if keep_indices else frame_idx % step == 0
        if keep:
            out_path = frames_dir / f"{clip.slug}_{kept_idx:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            frames.append(out_path)
            kept_idx += 1
            if max_frames is not None and len(frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()

    if not frames:
        raise SystemExit(f"Aucune frame extraite depuis {clip.clip}")
    return frames


def make_debug_video(debug_dir: Path, out_video: Path, *, fps: float) -> None:
    images = sorted(debug_dir.glob("*.jpg"))
    if not images:
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        tmp_dir = out_video.parent / f"_{out_video.stem}_video_frames"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for idx, image in enumerate(images):
            frame = cv2.imread(str(image))
            if frame is None:
                continue
            cv2.imwrite(str(tmp_dir / f"frame_{idx:05d}.jpg"), frame)
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%05d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_video),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for frame_path in tmp_dir.glob("*.jpg"):
            frame_path.unlink()
        tmp_dir.rmdir()
        return

    first = cv2.imread(str(images[0]))
    if first is None:
        return
    h, w = first.shape[:2]
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for image in images:
        frame = cv2.imread(str(image))
        if frame is None:
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
    writer.release()


def run_yolo(args: argparse.Namespace, frames_dir: Path, det_dir: Path) -> None:
    if det_dir.exists() and list(det_dir.glob("*.json")) and not args.overwrite:
        print(f"[skip] YOLO déjà présent: {det_dir}")
        return
    cmd = [
        sys.executable,
        "src/cabview/detect_yolo26.py",
        "--frames",
        str(frames_dir),
        "--out",
        str(det_dir),
        "--weights",
        str(args.yolo_weights),
        "--conf",
        str(args.yolo_conf),
        "--imgsz",
        str(args.yolo_imgsz),
        "--device",
        args.yolo_device,
    ]
    run_command(cmd, cwd=WORKSPACE)


def run_vda(args: argparse.Namespace, frames_dir: Path, det_dir: Path, out_dir: Path) -> None:
    if not args.vda_checkpoint.exists():
        print(f"[skip] VDA checkpoint introuvable: {args.vda_checkpoint}")
        return
    if (out_dir / "_vda_distances.csv").exists() and not args.overwrite:
        print(f"[skip] VDA déjà présent: {out_dir}")
        return
    cmd = [
        sys.executable,
        "src/depth/estimate_vda_distance.py",
        "--frames",
        str(frames_dir),
        "--detections",
        str(det_dir),
        "--checkpoint",
        str(args.vda_checkpoint),
        "--out",
        str(out_dir),
        "--encoder",
        args.vda_encoder,
        "--input-size",
        str(args.vda_input_size),
        "--device",
        args.vda_device,
        "--inset-ratio",
        str(args.inset_ratio),
    ]
    if args.vda_fp32:
        cmd.append("--fp32")
    run_command(cmd, cwd=WORKSPACE)
    make_debug_video(out_dir / "_debug", out_dir.with_suffix(".mp4"), fps=args.fps)


def run_da3(args: argparse.Namespace, frames_dir: Path, det_dir: Path, out_dir: Path) -> None:
    if (out_dir / "_da3_distances.csv").exists() and not args.overwrite:
        print(f"[skip] DA3 déjà présent: {out_dir}")
        return
    cmd = [
        sys.executable,
        "src/depth/estimate_da3_distance.py",
        "--frames",
        str(frames_dir),
        "--detections",
        str(det_dir),
        "--out",
        str(out_dir),
        "--model",
        args.da3_model,
        "--device",
        args.da3_device,
        "--hfov-deg",
        str(args.hfov_deg),
        "--inset-ratio",
        str(args.inset_ratio),
    ]
    if args.max_frames is not None:
        cmd.extend(["--n", str(args.max_frames)])
    run_command(cmd, cwd=WORKSPACE)
    make_debug_video(out_dir / "_debug", out_dir.with_suffix(".mp4"), fps=args.fps)


def run_dav2(args: argparse.Namespace, frames_dir: Path, det_dir: Path, out_dir: Path) -> None:
    if (out_dir / "_dav2_distances.csv").exists() and not args.overwrite:
        print(f"[skip] DAV2 déjà présent: {out_dir}")
        return
    cmd = [
        sys.executable,
        "src/depth/estimate_dav2_distance.py",
        "--frames",
        str(frames_dir),
        "--detections",
        str(det_dir),
        "--out",
        str(out_dir),
        "--model",
        args.dav2_model,
        "--device",
        args.dav2_device,
        "--inset-ratio",
        str(args.inset_ratio),
    ]
    if args.max_frames is not None:
        cmd.extend(["--n", str(args.max_frames)])
    run_command(cmd, cwd=WORKSPACE)
    make_debug_video(out_dir / "_debug", out_dir.with_suffix(".mp4"), fps=args.fps)


def read_method_csv(path: Path, method: str, clip: ClipSpec) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["clip_id"] = clip.id
            row["clip_slug"] = clip.slug
            row["method"] = method
            row["source_csv"] = str(path)
            rows.append(row)
    return rows


def consolidate(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_csv = out_dir / "all_distances_long.csv"
    if not results:
        all_csv.write_text("")
        return

    keys = [
        "clip_id",
        "clip_slug",
        "method",
        "frame",
        "frame_index",
        "det_id",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "median_m",
        "mean_trimmed_m",
        "p10_m",
        "p25_m",
        "p75_m",
        "p90_m",
        "iqr_m",
        "valid_pixels",
        "elapsed_ms_frame",
        "elapsed_ms_total",
        "source_csv",
    ]
    with all_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    summary_csv = out_dir / "summary_by_clip_method.csv"
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in results:
        grouped.setdefault((row["clip_slug"], row["method"]), []).append(row)

    with summary_csv.open("w", newline="") as f:
        fieldnames = [
            "clip_slug",
            "method",
            "n_distances",
            "median_of_medians_m",
            "mean_iqr_m",
            "median_elapsed_ms_frame",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (clip_slug, method), rows in sorted(grouped.items()):
            medians = sorted(float(r["median_m"]) for r in rows if r.get("median_m"))
            iqrs = [float(r["iqr_m"]) for r in rows if r.get("iqr_m")]
            elapsed = [
                float(r["elapsed_ms_frame"])
                for r in rows
                if r.get("elapsed_ms_frame") not in (None, "")
            ]
            if not elapsed:
                totals = [
                    float(r["elapsed_ms_total"])
                    for r in rows
                    if r.get("elapsed_ms_total") not in (None, "")
                ]
                frame_count = 0
                source_csv = rows[0].get("source_csv")
                if source_csv:
                    clip_dir = Path(source_csv).parents[1]
                    frame_count = len(list((clip_dir / "frames").glob("*.jpg")))
                if totals and frame_count:
                    elapsed = [totals[0] / frame_count]
            mid = medians[len(medians) // 2] if medians else ""
            writer.writerow(
                {
                    "clip_slug": clip_slug,
                    "method": method,
                    "n_distances": len(rows),
                    "median_of_medians_m": round(mid, 3) if mid != "" else "",
                    "mean_iqr_m": round(sum(iqrs) / len(iqrs), 3) if iqrs else "",
                    "median_elapsed_ms_frame": (
                        round(sorted(elapsed)[len(elapsed) // 2], 1) if elapsed else ""
                    ),
                }
            )
    print(f"[out] {all_csv}")
    print(f"[out] {summary_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--clip-ids", default=None, help="ex: 9,14,17")
    ap.add_argument("--methods", default="vda", help="vda,dav2,da3")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--sample-position", choices=["start", "middle", "end"], default="middle",
                    help="où prendre les frames quand --max-frames est utilisé")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--inset-ratio", type=float, default=0.10)
    ap.add_argument("--hfov-deg", type=float, default=70.0)

    ap.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--yolo-imgsz", type=int, default=960)
    ap.add_argument("--yolo-device", default="auto")

    ap.add_argument("--vda-checkpoint", type=Path, default=DEFAULT_VDA_CHECKPOINT)
    ap.add_argument("--vda-encoder", choices=["vits", "vitb", "vitl"], default="vits")
    ap.add_argument("--vda-input-size", type=int, default=280)
    ap.add_argument("--vda-device", default="mps")
    ap.add_argument("--vda-fp32", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--da3-model", default="DA3METRIC-LARGE")
    ap.add_argument("--da3-device", default="mps")

    ap.add_argument("--dav2-model", default=DEFAULT_DAV2_MODEL)
    ap.add_argument("--dav2-device", default="mps")
    args = ap.parse_args()

    methods = parse_methods(args.methods)
    selected_ids = None
    if args.clip_ids:
        selected_ids = {int(x.strip()) for x in args.clip_ids.split(",") if x.strip()}
    clips = load_manifest(args.manifest, selected_ids)

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for clip in clips:
        print(f"\n=== {clip.slug} ===")
        clip_dir = args.out / clip.slug
        frames_dir = clip_dir / "frames"
        det_dir = clip_dir / "detections"

        frames = extract_frames(
            clip,
            frames_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            sample_position=args.sample_position,
            overwrite=args.overwrite,
        )
        print(f"[frames] {len(frames)} frame(s): {frames_dir}")
        run_yolo(args, frames_dir, det_dir)

        if "vda" in methods:
            vda_dir = clip_dir / "vda"
            run_vda(args, frames_dir, det_dir, vda_dir)
            all_rows.extend(read_method_csv(vda_dir / "_vda_distances.csv", "vda", clip))

        if "dav2" in methods:
            dav2_dir = clip_dir / "dav2"
            run_dav2(args, frames_dir, det_dir, dav2_dir)
            all_rows.extend(read_method_csv(dav2_dir / "_dav2_distances.csv", "dav2", clip))

        if "da3" in methods:
            da3_dir = clip_dir / "da3"
            run_da3(args, frames_dir, det_dir, da3_dir)
            all_rows.extend(read_method_csv(da3_dir / "_da3_distances.csv", "da3", clip))

    consolidate(all_rows, args.out)
    print("\n[done] benchmark terminé")


if __name__ == "__main__":
    main()
