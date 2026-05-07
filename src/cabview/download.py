"""Télécharge les vidéos cabview YouTube listées dans sources.yaml.

Usage:
    PYTHONPATH=. python src/cabview/download.py --region fr
    PYTHONPATH=. python src/cabview/download.py --region fr --id bordeaux_nantes_2023
    PYTHONPATH=. python src/cabview/download.py --region fr --max-height 720
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def load_sources(region: str) -> list[dict]:
    src = Path(f"data/cabview/{region}/sources.yaml")
    with open(src) as f:
        data = yaml.safe_load(f)
    return data.get("videos", [])


def download(video: dict, out_dir: Path, max_height: int) -> Path | None:
    vid = video["id"]
    url = video["url"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / f"{vid}.%(ext)s")

    fmt = (
        f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"best[height<={max_height}]"
    )
    cmd = [
        "yt-dlp",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "-o", out_template,
        "--no-playlist",
        url,
    ]
    print(f"\n→ {vid} ({video.get('title', '')})")
    print(f"  cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    found = list(out_dir.glob(f"{vid}.*"))
    return found[0] if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--id", default=None, help="Télécharger une seule vidéo par id")
    ap.add_argument("--max-height", type=int, default=720,
                    help="Résolution max (720 = HD, économise espace)")
    args = ap.parse_args()

    out = Path(f"data/cabview/{args.region}/raw")
    videos = load_sources(args.region)
    if args.id:
        videos = [v for v in videos if v["id"] == args.id]
        if not videos:
            raise SystemExit(f"id {args.id} introuvable dans sources.yaml")

    for v in videos:
        download(v, out, args.max_height)


if __name__ == "__main__":
    main()
