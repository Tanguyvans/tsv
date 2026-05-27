"""Méthode C: distance par Depth Anything 3 metric dans les bbox YOLO.

Le script lance DA3 sur les frames, lit les détections YOLO existantes, puis
extrait des statistiques robustes de profondeur dans chaque bbox.

Exemple:
    PYTHONPATH=. python src/depth/estimate_da3_distance.py \\
        --frames outputs/depth_method_benchmark/clip_00/frames \\
        --detections outputs/depth_method_benchmark/clip_00/detections \\
        --model DA3METRIC-LARGE \\
        --out outputs/da3_method_c_midvideo
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def colorize_depth(depth: np.ndarray, near: float | None = None, far: float | None = None) -> np.ndarray:
    """Carte de profondeur vers RGB avec colormap turbo."""
    import matplotlib.cm as cm

    valid = np.isfinite(depth)
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    near = near if near is not None else float(np.nanpercentile(depth[valid], 2))
    far = far if far is not None else float(np.nanpercentile(depth[valid], 98))
    norm = np.clip((depth - near) / max(far - near, 1e-6), 0, 1)
    colored = cm.turbo(norm)[..., :3]
    colored[~valid] = 0
    return (colored * 255).astype(np.uint8)


def collect_images(frames: Path, n: int | None = None) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    if frames.is_file():
        images = [frames]
    else:
        images = sorted(p for p in frames.iterdir() if p.suffix.lower() in suffixes)
    if n is not None:
        images = images[:n]
    if not images:
        raise SystemExit(f"Aucune image trouvée dans {frames}")
    return images


def focal_px_from_hfov(image_width: int, hfov_deg: float) -> float:
    hfov_rad = np.deg2rad(hfov_deg)
    return float(image_width / (2.0 * np.tan(hfov_rad / 2.0)))


def load_detections(path: Path) -> dict[str, list[dict]]:
    json_paths = [path] if path.is_file() else sorted(
        p for p in path.glob("*.json") if not p.name.startswith("_")
    )
    detections: dict[str, list[dict]] = {}
    for json_path in json_paths:
        data = json.loads(json_path.read_text())
        if not isinstance(data, dict):
            continue
        frame_name = data.get("frame", f"{json_path.stem}.jpg")
        detections[frame_name] = data.get("detections", [])
    return detections


def crop_depth_stats(depth: np.ndarray, bbox: list[int], *, inset_ratio: float) -> dict:
    h, w = depth.shape
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    x1 += bw * inset_ratio
    x2 -= bw * inset_ratio
    y1 += bh * inset_ratio
    y2 -= bh * inset_ratio

    ix1 = max(0, min(w - 1, int(round(x1))))
    iy1 = max(0, min(h - 1, int(round(y1))))
    ix2 = max(ix1 + 1, min(w, int(round(x2))))
    iy2 = max(iy1 + 1, min(h, int(round(y2))))

    crop = depth[iy1:iy2, ix1:ix2]
    valid = crop[np.isfinite(crop)]
    valid = valid[valid > 0]
    if valid.size == 0:
        raise ValueError("aucune profondeur valide dans la bbox")

    p10, p25, p50, p75, p90 = np.nanpercentile(valid, [10, 25, 50, 75, 90])
    iqr = float(p75 - p25)
    trimmed = valid[(valid >= p10) & (valid <= p90)]
    return {
        "median_m": float(p50),
        "mean_trimmed_m": float(np.nanmean(trimmed)),
        "p10_m": float(p10),
        "p25_m": float(p25),
        "p75_m": float(p75),
        "p90_m": float(p90),
        "iqr_m": iqr,
        "valid_pixels": int(valid.size),
        "crop": [ix1, iy1, ix2, iy2],
    }


def draw_debug(
    image_path: Path,
    depth: np.ndarray,
    detections: list[dict],
    out_path: Path,
    *,
    side_by_side: bool,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return

    if not detections:
        cv2.putText(img, "no detection", (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 120, 0), 2, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        da3 = det.get("da3", {})
        median_m = da3.get("median_m")
        iqr_m = da3.get("iqr_m")
        label = "DA3 ?"
        if median_m is not None:
            label = f"DA3 {median_m:.1f} m"
            if iqr_m is not None:
                label += f" iqr {iqr_m:.1f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 120, 0), 2)
        cv2.putText(img, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2, cv2.LINE_AA)

    out = img
    if side_by_side:
        depth_vis = colorize_depth(depth)
        if depth_vis.shape[:2] != img.shape[:2]:
            depth_vis = cv2.resize(depth_vis, (img.shape[1], img.shape[0]))
        depth_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
        out = np.concatenate([img, depth_bgr], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/da3_method_c"))
    ap.add_argument("--model", default="DA3METRIC-LARGE",
                    help="suffixe HF depth-anything/<model>")
    ap.add_argument("--focal-px", type=float, default=None,
                    help="focale en pixels pour convertir DA3METRIC-LARGE en mètres")
    ap.add_argument("--hfov-deg", type=float, default=70.0,
                    help="HFOV utilisé pour approximer la focale si --focal-px absent")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto sinon)")
    ap.add_argument("--inset-ratio", type=float, default=0.10,
                    help="réduit la bbox avant stats depth pour limiter le fond")
    ap.add_argument("--save-depth", action="store_true",
                    help="sauvegarde les depth maps .npy dans <out>/_depth")
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("--no-side-by-side", action="store_true")
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[info] device = {device}")

    images = collect_images(args.frames, args.n)
    detections_by_frame = load_detections(args.detections)

    first_img = Image.open(images[0])
    image_width = first_img.size[0]
    focal_px = args.focal_px
    if focal_px is None:
        focal_px = focal_px_from_hfov(image_width, args.hfov_deg)

    try:
        from depth_anything_3.api import DepthAnything3
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "depth_anything_3 n'est pas installé. Installer avec:\n"
            "  venv/bin/pip install xformers\n"
            "  venv/bin/pip install -e git+https://github.com/ByteDance-Seed/Depth-Anything-3.git#egg=depth_anything_3"
        ) from exc

    print(f"[info] chargement de depth-anything/{args.model} ...")
    model = DepthAnything3.from_pretrained(f"depth-anything/{args.model}")
    model = model.to(device).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    debug_dir = args.out / "_debug"
    depth_dir = args.out / "_depth"
    if args.save_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in tqdm(images, desc="DA3 distance"):
        dets = detections_by_frame.get(image_path.name, [])
        t0 = time.time()
        with torch.inference_mode():
            pred = model.inference([str(image_path)])
        elapsed_ms = (time.time() - t0) * 1000.0

        depth = pred.depth
        if hasattr(depth, "detach"):
            depth = depth.detach().cpu().numpy()
        depth = np.squeeze(depth).astype(np.float32)
        if args.model.upper() == "DA3METRIC-LARGE":
            depth = depth * float(focal_px) / 300.0
        rgb_size = Image.open(image_path).size
        if depth.shape[:2] != (rgb_size[1], rgb_size[0]):
            depth = cv2.resize(depth, rgb_size, interpolation=cv2.INTER_LINEAR)
        if args.save_depth:
            np.save(depth_dir / f"{image_path.stem}.npy", depth)

        enriched = []
        for idx, det in enumerate(dets):
            item = dict(det)
            try:
                stats = crop_depth_stats(depth, item["bbox"], inset_ratio=args.inset_ratio)
                item["da3"] = {
                    **{k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()},
                    "model": args.model,
                    "method": "da3_metric_bbox_median",
                }
                rows.append({
                    "frame": image_path.name,
                    "det_id": idx,
                    "score": item.get("score"),
                    "x1": item["bbox"][0],
                    "y1": item["bbox"][1],
                    "x2": item["bbox"][2],
                    "y2": item["bbox"][3],
                    "median_m": round(stats["median_m"], 3),
                    "mean_trimmed_m": round(stats["mean_trimmed_m"], 3),
                    "p10_m": round(stats["p10_m"], 3),
                    "p25_m": round(stats["p25_m"], 3),
                    "p75_m": round(stats["p75_m"], 3),
                    "p90_m": round(stats["p90_m"], 3),
                    "iqr_m": round(stats["iqr_m"], 3),
                    "valid_pixels": stats["valid_pixels"],
                    "elapsed_ms_frame": round(elapsed_ms, 1),
                })
            except ValueError as exc:
                item["da3_error"] = str(exc)
            enriched.append(item)

        result = {
            "frame": image_path.name,
            "model": args.model,
            "focal_px": round(float(focal_px), 3),
            "hfov_deg": None if args.focal_px is not None else args.hfov_deg,
            "elapsed_ms_frame": round(elapsed_ms, 1),
            "depth_shape": list(depth.shape),
            "detections": enriched,
        }
        (args.out / f"{image_path.stem}.json").write_text(json.dumps(result, indent=2))

        if not args.no_debug:
            draw_debug(
                image_path,
                depth,
                enriched,
                debug_dir / image_path.name,
                side_by_side=not args.no_side_by_side,
            )

    with (args.out / "_da3_distances.csv").open("w", newline="") as f:
        fieldnames = [
            "frame", "det_id", "score", "x1", "y1", "x2", "y2",
            "median_m", "mean_trimmed_m", "p10_m", "p25_m", "p75_m", "p90_m",
            "iqr_m", "valid_pixels", "elapsed_ms_frame",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] {len(rows)} distance(s) DA3 extraite(s)")
    print(f"[out] CSV: {args.out / '_da3_distances.csv'}")
    if args.save_depth:
        print(f"[out] depth npy: {depth_dir}/")
    if not args.no_debug:
        print(f"[out] debug: {debug_dir}/")


if __name__ == "__main__":
    main()
