"""Extrait des frames d'une vidéo cabview à un fps configurable.

Usage:
    PYTHONPATH=. python src/cabview/extract_frames.py --region fr --id bordeaux_nantes_2023
    PYTHONPATH=. python src/cabview/extract_frames.py --region fr --id bordeaux_nantes_2023 --fps 0.5
    PYTHONPATH=. python src/cabview/extract_frames.py --region fr --id ... --start 600 --duration 1800
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def find_video(region: str, vid: str) -> Path:
    raw = Path(f"data/cabview/{region}/raw")
    matches = list(raw.glob(f"{vid}.*"))
    matches = [m for m in matches if m.suffix.lower() in {".mp4", ".mkv", ".webm"}]
    if not matches:
        raise SystemExit(f"Vidéo {vid} introuvable dans {raw}")
    return matches[0]


def extract(video: Path, out_dir: Path, fps: float, start: int | None,
            duration: int | None, scale: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{video.stem}_%06d.jpg")

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    vf = f"fps={fps},scale={scale}:-2"
    cmd += ["-vf", vf, "-q:v", "3", pattern]

    print(f"→ Extraction {video.name} → {out_dir}")
    print(f"  fps={fps} scale={scale} start={start} duration={duration}")
    subprocess.run(cmd, check=True)
    n = len(list(out_dir.glob(f"{video.stem}_*.jpg")))
    print(f"  {n} frames extraites")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--fps", type=float, default=1.0,
                    help="Frames par seconde extraites (1 = 1 image/sec)")
    ap.add_argument("--scale", type=int, default=1280,
                    help="Largeur cible (1280 = 720p)")
    ap.add_argument("--start", type=int, default=None, help="Offset start (sec)")
    ap.add_argument("--duration", type=int, default=None, help="Durée à extraire (sec)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    video = find_video(args.region, args.id)
    out = args.out or Path(f"data/cabview/{args.region}/frames/{args.id}")
    extract(video, out, args.fps, args.start, args.duration, args.scale)


if __name__ == "__main__":
    main()
