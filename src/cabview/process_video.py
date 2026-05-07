"""Wrapper end-to-end : download → extract → filter cab-view.

Une seule commande qui chaîne les 3 étapes pour une vidéo donnée. Skip
automatiquement les étapes déjà faites (vidéo déjà téléchargée, frames
déjà extraites).

Usage:
    PYTHONPATH=. python src/cabview/process_video.py --region fr --id bordeaux_nantes_2023

    # Avec custom fps et copie des frames cab-view
    PYTHONPATH=. python src/cabview/process_video.py --region fr --id X --fps 0.5 --copy

    # Skip le download (vidéo déjà là)
    PYTHONPATH=. python src/cabview/process_video.py --region fr --id X --skip-download
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def step(label: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print('═' * 60)


def find_video(region: str, vid: str) -> Path | None:
    raw = Path(f"data/cabview/{region}/raw")
    if not raw.exists():
        return None
    for m in raw.glob(f"{vid}.*"):
        if m.suffix.lower() in {".mp4", ".mkv", ".webm"}:
            return m
    return None


def video_in_sources(region: str, vid: str) -> bool:
    src = Path(f"data/cabview/{region}/sources.yaml")
    if not src.exists():
        return False
    data = yaml.safe_load(src.read_text())
    return any(v.get("id") == vid for v in data.get("videos", []))


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--scale", type=int, default=1280)
    ap.add_argument("--max-height", type=int, default=720)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--duration", type=int, default=None)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="Marge CLIP cab vs not_cab")
    ap.add_argument("--copy", action="store_true",
                    help="Copier les frames cab-view (sinon juste manifest)")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-filter", action="store_true")
    args = ap.parse_args()

    if not video_in_sources(args.region, args.id):
        sys.exit(
            f"Vidéo id={args.id} introuvable dans "
            f"data/cabview/{args.region}/sources.yaml"
        )

    py = sys.executable

    # 1. Download (skip si déjà là)
    step(f"[1/3] Download (region={args.region}, id={args.id})")
    if args.skip_download:
        print("  → skip (--skip-download)")
    elif find_video(args.region, args.id):
        print(f"  → skip (vidéo déjà présente)")
    else:
        run([py, "src/cabview/download.py",
             "--region", args.region, "--id", args.id,
             "--max-height", str(args.max_height)])

    # 2. Extract
    frames_dir = Path(f"data/cabview/{args.region}/frames/{args.id}")
    step(f"[2/3] Extract frames (fps={args.fps}, scale={args.scale})")
    existing = list(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    if args.skip_extract:
        print(f"  → skip (--skip-extract), {len(existing)} frames présentes")
    elif existing:
        print(f"  → skip ({len(existing)} frames déjà extraites)")
    else:
        cmd = [py, "src/cabview/extract_frames.py",
               "--region", args.region, "--id", args.id,
               "--fps", str(args.fps), "--scale", str(args.scale)]
        if args.start is not None:
            cmd += ["--start", str(args.start)]
        if args.duration is not None:
            cmd += ["--duration", str(args.duration)]
        run(cmd)

    # 3. Filter cab-view
    out_dir = Path(f"data/cabview/{args.region}/frames_cabview/{args.id}")
    step(f"[3/3] Filter cab-view (margin={args.margin})")
    if args.skip_filter:
        print("  → skip (--skip-filter)")
    else:
        cmd = [py, "src/cabview/check_cabview.py",
               "--frames", str(frames_dir),
               "--margin", str(args.margin)]
        if args.copy:
            cmd.append("--copy")
        run(cmd)

    # Rapport final
    step("Rapport")
    scores_file = out_dir / "_scores.json"
    if scores_file.exists():
        scores = json.loads(scores_file.read_text())
        n_total = len(scores)
        n_cab = sum(1 for s in scores if s["is_cabview"])
        print(f"  Total frames extraites : {n_total}")
        print(f"  Frames cab-view        : {n_cab} ({100*n_cab/n_total:.1f}%)")
        print(f"  Manifest               : {scores_file}")
        if args.copy:
            n_files = len(list(out_dir.glob("*.jpg")))
            print(f"  Copiées                : {n_files} dans {out_dir}")
    else:
        print(f"  (pas de scores trouvés à {scores_file})")


if __name__ == "__main__":
    main()
