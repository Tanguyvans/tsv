"""Compare MultiSense DEPTH avec Depth Anything sur les mêmes détections.

MultiSense sert de référence capteur: les fichiers DEPTH sont lus en
``gray16le`` et convertis de millimètres vers mètres. Les méthodes DAV2, DA3 et
VDA sont lancées sur les mêmes frames couleur et les mêmes JSON de détection.

Exemple sur Lucia:

    PYTHONPATH=. python src/depth/benchmark_msense_vs_depth_anything.py \\
        --data-root /gpfs/projects/acad/brainai/cegelecRecordings \\
        --ckpt /gpfs/projects/acad/brainai/tvans_distance/yolo_signs_best.pt \\
        --seqs 0,1,2,3 \\
        --start 0 --end 99 --stride 1 \\
        --conf 0.05 \\
        --methods msense,dav2,da3,vda \\
        --out /gpfs/projects/acad/brainai/tvans_distance/depth_benchmark
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.signals.estimate_msense_signal_distance import (
    HEIGHT,
    WIDTH,
    clip_box,
    depth_stats,
    read_color_frame,
    read_depth_frame,
)


WORKSPACE = Path(__file__).resolve().parents[2]


def max_rss_mb(children: bool = False) -> float:
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    return resource.getrusage(who).ru_maxrss / 1024.0


def parse_csv_ints(raw: str | None, default: list[int]) -> list[int]:
    if raw:
        return sorted({int(v.strip()) for v in raw.split(",") if v.strip()})
    return default


def parse_methods(raw: str) -> list[str]:
    methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    known = {"msense", "dav2", "da3", "vda"}
    unknown = sorted(set(methods) - known)
    if unknown:
        raise SystemExit(f"Méthode(s) inconnue(s): {', '.join(unknown)}")
    if "msense" not in methods:
        raise SystemExit("Le benchmark demande toujours 'msense' comme référence capteur.")
    return methods


def run_command(cmd: list[str], *, cwd: Path) -> dict:
    print("\n[run]", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE)
    rss_before = max_rss_mb(children=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, env=env, check=True)
    elapsed_s = time.perf_counter() - t0
    rss_after = max_rss_mb(children=True)
    return {
        "elapsed_s": elapsed_s,
        "max_rss_mb": max(rss_before, rss_after),
    }


def msense_paths(data_root: Path, campaign: str, seq: int) -> tuple[Path, Path]:
    video_root = data_root / campaign / "RawData" / "Video" / "MSense"
    return (
        video_root / "COLOR" / f"color_{seq}.mkv",
        video_root / "DEPTH" / f"depth_{seq}.mkv",
    )


def frame_name(seq: int, frame_idx: int) -> str:
    return f"seq{seq:02d}_frame{frame_idx:06d}.jpg"


def extract_msense_frames(args: argparse.Namespace, seqs: list[int], frames: list[int], frames_dir: Path) -> dict[str, dict]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    t0 = time.perf_counter()

    for seq in seqs:
        color_path, _depth_path = msense_paths(Path(args.data_root), args.campaign, seq)
        for frame_idx in frames:
            name = frame_name(seq, frame_idx)
            out_path = frames_dir / name
            if args.overwrite or not out_path.exists():
                color = read_color_frame(color_path, frame_idx)
                cv2.imwrite(str(out_path), color)
            manifest[name] = {"seq": seq, "frame_idx": frame_idx, "path": str(out_path)}

    return {
        "manifest": manifest,
        "metrics": {
            "method": "extract_frames",
            "frames": len(manifest),
            "elapsed_s": time.perf_counter() - t0,
            "fps": len(manifest) / max(time.perf_counter() - t0, 1e-9),
            "max_rss_mb": max_rss_mb(),
        },
    }


def run_yolo(args: argparse.Namespace, frames_dir: Path, detections_dir: Path) -> dict:
    if detections_dir.exists() and list(detections_dir.glob("*.json")) and not args.overwrite:
        print(f"[skip] YOLO déjà présent: {detections_dir}")
        return {"method": "yolo", "skipped": True}

    cmd = [
        sys.executable,
        "src/cabview/detect_yolo26.py",
        "--frames",
        str(frames_dir),
        "--out",
        str(detections_dir),
        "--weights",
        str(args.ckpt),
        "--conf",
        str(args.conf),
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.yolo_device,
    ]
    if args.no_yolo_debug:
        cmd.append("--no-debug")
    metrics = run_command(cmd, cwd=WORKSPACE)
    metrics["method"] = "yolo"
    return metrics


def load_detections(detections_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(p for p in detections_dir.glob("*.json") if not p.name.startswith("_")):
        data = json.loads(path.read_text())
        out[data.get("frame", f"{path.stem}.jpg")] = data.get("detections", [])
    return out


def write_msense_distances(
    args: argparse.Namespace,
    manifest: dict[str, dict],
    detections_dir: Path,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    detections = load_detections(detections_dir)
    depth_cache: dict[tuple[int, int], np.ndarray] = {}
    rows = []
    t0 = time.perf_counter()

    for name, meta in manifest.items():
        seq = int(meta["seq"])
        frame_idx = int(meta["frame_idx"])
        _color_path, depth_path = msense_paths(Path(args.data_root), args.campaign, seq)
        cache_key = (seq, frame_idx)
        if cache_key not in depth_cache:
            depth_cache[cache_key] = read_depth_frame(depth_path, frame_idx)
        depth = depth_cache[cache_key]

        enriched = []
        for det_id, det in enumerate(detections.get(name, [])):
            x1, y1, x2, y2 = clip_box(det["bbox"], WIDTH, HEIGHT, args.bbox_shrink)
            stats = depth_stats(depth, (x1, y1, x2, y2), args.max_depth_mm)
            item = {
                "frame": name,
                "seq": seq,
                "frame_index": frame_idx,
                "det_id": det_id,
                "score": det.get("score"),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "median_m": stats["distance_m"],
                "p10_m": stats["depth_p10_m"],
                "p90_m": stats["depth_p90_m"],
                "iqr_m": None,
                "valid_pixels": stats["valid_px"],
            }
            rows.append(item)
            enriched.append({**det, "msense": item})

        (out_dir / f"{Path(name).stem}.json").write_text(json.dumps({
            "frame": name,
            "seq": seq,
            "frame_index": frame_idx,
            "detections": enriched,
        }, indent=2))

    csv_path = out_dir / "_msense_distances.csv"
    fieldnames = [
        "frame", "seq", "frame_index", "det_id", "score", "x1", "y1", "x2", "y2",
        "median_m", "p10_m", "p90_m", "iqr_m", "valid_pixels",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed_s = time.perf_counter() - t0
    return {
        "method": "msense",
        "frames": len(manifest),
        "detections": len(rows),
        "elapsed_s": elapsed_s,
        "fps": len(manifest) / elapsed_s if elapsed_s > 0 else 0.0,
        "max_rss_mb": max_rss_mb(),
        "csv": str(csv_path),
    }


def run_depth_method(
    method: str,
    args: argparse.Namespace,
    frames_dir: Path,
    detections_dir: Path,
    method_dir: Path,
) -> dict:
    if method == "dav2":
        cmd = [
            sys.executable,
            "src/depth/estimate_dav2_distance.py",
            "--frames",
            str(frames_dir),
            "--detections",
            str(detections_dir),
            "--out",
            str(method_dir),
            "--model",
            args.dav2_model,
            "--inset-ratio",
            str(args.bbox_shrink),
        ]
        if args.dav2_device != "auto":
            cmd.extend(["--device", args.dav2_device])
    elif method == "da3":
        cmd = [
            sys.executable,
            "src/depth/estimate_da3_distance.py",
            "--frames",
            str(frames_dir),
            "--detections",
            str(detections_dir),
            "--out",
            str(method_dir),
            "--model",
            args.da3_model,
            "--hfov-deg",
            str(args.hfov_deg),
            "--inset-ratio",
            str(args.bbox_shrink),
        ]
        if args.da3_device != "auto":
            cmd.extend(["--device", args.da3_device])
    elif method == "vda":
        if not args.vda_checkpoint:
            return {"method": "vda", "skipped": True, "reason": "--vda-checkpoint absent"}
        cmd = [
            sys.executable,
            "src/depth/estimate_vda_distance.py",
            "--frames",
            str(frames_dir),
            "--detections",
            str(detections_dir),
            "--checkpoint",
            str(args.vda_checkpoint),
            "--out",
            str(method_dir),
            "--encoder",
            args.vda_encoder,
            "--input-size",
            str(args.vda_input_size),
            "--inset-ratio",
            str(args.bbox_shrink),
        ]
        if args.vda_device != "auto":
            cmd.extend(["--device", args.vda_device])
        if args.vda_fp32:
            cmd.append("--fp32")
    else:
        raise ValueError(method)

    if not args.save_depth:
        pass
    else:
        cmd.append("--save-depth")
    if args.no_debug:
        cmd.append("--no-debug")

    metrics = run_command(cmd, cwd=WORKSPACE)
    metrics["method"] = method
    frame_count = len(list(frames_dir.glob("*.jpg")))
    metrics["frames"] = frame_count
    metrics["fps"] = frame_count / metrics["elapsed_s"] if metrics["elapsed_s"] > 0 else 0.0
    return metrics


def read_method_csv(path: Path, method: str) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[(row["frame"], int(row["det_id"]))] = {**row, "method": method}
    return out


def consolidate(out_dir: Path, methods: list[str], metrics: list[dict]) -> None:
    msense = read_method_csv(out_dir / "msense" / "_msense_distances.csv", "msense")
    tables = {
        "dav2": read_method_csv(out_dir / "dav2" / "_dav2_distances.csv", "dav2"),
        "da3": read_method_csv(out_dir / "da3" / "_da3_distances.csv", "da3"),
        "vda": read_method_csv(out_dir / "vda" / "_vda_distances.csv", "vda"),
    }

    rows = []
    for key, ref in sorted(msense.items()):
        ref_m = float(ref["median_m"]) if ref.get("median_m") not in (None, "") else None
        row = {
            "frame": key[0],
            "det_id": key[1],
            "seq": ref.get("seq"),
            "frame_index": ref.get("frame_index"),
            "score": ref.get("score"),
            "x1": ref.get("x1"),
            "y1": ref.get("y1"),
            "x2": ref.get("x2"),
            "y2": ref.get("y2"),
            "msense_m": ref_m,
            "msense_p10_m": ref.get("p10_m"),
            "msense_p90_m": ref.get("p90_m"),
        }
        for method in ("dav2", "da3", "vda"):
            item = tables[method].get(key, {})
            pred = float(item["median_m"]) if item.get("median_m") not in (None, "") else None
            row[f"{method}_m"] = pred
            row[f"{method}_iqr_m"] = item.get("iqr_m", "")
            row[f"{method}_abs_err_m"] = abs(pred - ref_m) if pred is not None and ref_m is not None else ""
            row[f"{method}_rel_err"] = abs(pred - ref_m) / ref_m if pred is not None and ref_m else ""
        rows.append(row)

    wide_csv = out_dir / "distance_comparison_wide.csv"
    fieldnames = [
        "frame", "det_id", "seq", "frame_index", "score", "x1", "y1", "x2", "y2",
        "msense_m", "msense_p10_m", "msense_p90_m",
        "dav2_m", "dav2_iqr_m", "dav2_abs_err_m", "dav2_rel_err",
        "da3_m", "da3_iqr_m", "da3_abs_err_m", "da3_rel_err",
        "vda_m", "vda_iqr_m", "vda_abs_err_m", "vda_rel_err",
    ]
    with wide_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for method in ("dav2", "da3", "vda"):
        errors = [float(r[f"{method}_abs_err_m"]) for r in rows if r.get(f"{method}_abs_err_m") not in ("", None)]
        rels = [float(r[f"{method}_rel_err"]) for r in rows if r.get(f"{method}_rel_err") not in ("", None)]
        metric = next((m for m in metrics if m.get("method") == method), {})
        summary_rows.append({
            "method": method,
            "n_compared": len(errors),
            "median_abs_err_m": float(np.median(errors)) if errors else "",
            "mean_abs_err_m": float(np.mean(errors)) if errors else "",
            "median_rel_err": float(np.median(rels)) if rels else "",
            "elapsed_s": metric.get("elapsed_s", ""),
            "fps": metric.get("fps", ""),
            "max_rss_mb": metric.get("max_rss_mb", ""),
        })

    msense_metric = next((m for m in metrics if m.get("method") == "msense"), {})
    summary_rows.insert(0, {
        "method": "msense",
        "n_compared": len(msense),
        "median_abs_err_m": 0.0,
        "mean_abs_err_m": 0.0,
        "median_rel_err": 0.0,
        "elapsed_s": msense_metric.get("elapsed_s", ""),
        "fps": msense_metric.get("fps", ""),
        "max_rss_mb": msense_metric.get("max_rss_mb", ""),
    })

    summary_csv = out_dir / "method_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[out] {wide_csv}")
    print(f"[out] {summary_csv}")
    print(f"[out] {out_dir / 'metrics.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA", "/gpfs/projects/acad/brainai/cegelecRecordings")))
    ap.add_argument("--campaign", default="Campaign_2026_03_24_12_59_24")
    ap.add_argument("--seqs", default="0")
    ap.add_argument("--frames", default=None, help="Liste de frames, ex: 0,10,50")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--yolo-device", default="auto")
    ap.add_argument("--bbox-shrink", type=float, default=0.10)
    ap.add_argument("--max-depth-mm", type=int, default=65000)
    ap.add_argument("--methods", default="msense,dav2,da3,vda")
    ap.add_argument("--out", type=Path, default=Path("outputs/msense_depth_benchmark"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("--no-yolo-debug", action="store_true")
    ap.add_argument("--save-depth", action="store_true")

    ap.add_argument("--dav2-model", default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    ap.add_argument("--dav2-device", default="auto")
    ap.add_argument("--da3-model", default="DA3METRIC-LARGE")
    ap.add_argument("--da3-device", default="auto")
    ap.add_argument("--hfov-deg", type=float, default=70.0)
    ap.add_argument("--vda-checkpoint", type=Path, default=None)
    ap.add_argument("--vda-encoder", choices=["vits", "vitb", "vitl"], default="vits")
    ap.add_argument("--vda-input-size", type=int, default=280)
    ap.add_argument("--vda-device", default="auto")
    ap.add_argument("--vda-fp32", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    methods = parse_methods(args.methods)
    seqs = parse_csv_ints(args.seqs, [0])
    frames = parse_csv_ints(args.frames, list(range(args.start, args.end + 1, args.stride)))

    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out / "frames"
    detections_dir = args.out / "detections"

    metrics = []
    extracted = extract_msense_frames(args, seqs, frames, frames_dir)
    manifest = extracted["manifest"]
    metrics.append(extracted["metrics"])
    (args.out / "frame_manifest.json").write_text(json.dumps(manifest, indent=2))

    metrics.append(run_yolo(args, frames_dir, detections_dir))

    if "msense" in methods:
        metrics.append(write_msense_distances(args, manifest, detections_dir, args.out / "msense"))

    for method in ("dav2", "da3", "vda"):
        if method in methods:
            metrics.append(run_depth_method(method, args, frames_dir, detections_dir, args.out / method))

    consolidate(args.out, methods, metrics)
    print("[done] benchmark MultiSense vs Depth Anything terminé")


if __name__ == "__main__":
    main()
