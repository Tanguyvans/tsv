"""Génère la classe Normal en effaçant les déchets autour des rails via Bria.

Modèle : fal-ai/bria/fibo-edit/erase_by_text
API : image_url + object_name → image avec les objets désignés effacés.
Le reste de l'image (rail inclus) est préservé pixel-perfect.

Usage :
  python src/generation/clean_to_normal.py \
      --src data/surface/Flakings \
      --out data/normal_synthetic/Normal \
      --n 500 --concurrency 5
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

MODEL_PATH = "bria/fibo-edit/erase_by_text"
SUBMIT_URL = f"https://queue.fal.run/{MODEL_PATH}"
# Endpoints status/result vivent sous /bria/fibo-edit/requests (sans /erase_by_text)
QUEUE_BASE = "https://queue.fal.run/bria/fibo-edit"
DEFAULT_OBJECT = "trash, debris, garbage, plastic waste, litter, dirty objects"


def auth_header() -> dict[str, str]:
    key = os.environ.get("FAL_KEY")
    if not key:
        raise RuntimeError("FAL_KEY not set")
    return {"Authorization": f"Key {key}", "Content-Type": "application/json"}


def _content_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp",
    }.get(ext, "image/jpeg")


def upload(path: Path) -> str:
    """Upload fal-cdn-v3 via raw binary body + X-Fal-File-Name."""
    key = os.environ.get("FAL_KEY")
    tok = requests.post(
        "https://rest.alpha.fal.ai/storage/auth/token?storage_type=fal-cdn-v3",
        headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
        json={},
        timeout=30,
    ).json()
    base_url = tok["base_url"]
    token = tok["token"]
    token_type = tok.get("token_type", "Bearer")
    ctype = _content_type(path)
    r = requests.post(
        f"{base_url}/files/upload",
        headers={
            "Authorization": f"{token_type} {token}",
            "Content-Type": ctype,
            "X-Fal-File-Name": path.name,
        },
        data=path.read_bytes(),
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["access_url"]


def submit(image_url: str, object_name: str) -> str:
    body = {"image_url": image_url, "object_name": object_name}
    r = requests.post(SUBMIT_URL, headers=auth_header(), json=body, timeout=60)
    r.raise_for_status()
    return r.json()["request_id"]


def status(req_id: str) -> str:
    r = requests.get(f"{QUEUE_BASE}/requests/{req_id}/status",
                     headers=auth_header(), timeout=30)
    return r.json().get("status", "UNKNOWN")


def fetch_result(req_id: str) -> dict[str, Any]:
    r = requests.get(f"{QUEUE_BASE}/requests/{req_id}",
                     headers=auth_header(), timeout=30)
    r.raise_for_status()
    return r.json()


def download(url: str, path: Path) -> None:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)


def list_sources(src: Path, n: int) -> list[Path]:
    items = sorted(p for p in src.rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return items[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--object-name", default=DEFAULT_OBJECT)
    ap.add_argument("--ext", default="png", choices=["png", "jpg"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--poll-interval", type=float, default=4.0)
    args = ap.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_dir.parent / "clean_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    sources = list_sources(src_dir, args.n)
    pending = [p for p in sources if not (out_dir / f"clean_{p.stem}.{args.ext}").exists()]
    print(f"{len(sources)} sources / {len(pending)} pending "
          f"(already done: {len(sources) - len(pending)})")

    in_flight: dict[str, dict] = {}
    queue = list(pending)
    completed = 0
    failed = 0

    def submit_one(src_path: Path):
        out_path = out_dir / f"clean_{src_path.stem}.{args.ext}"
        try:
            url = upload(src_path)
            req_id = submit(url, args.object_name)
            in_flight[req_id] = {
                "src": str(src_path),
                "out": str(out_path),
                "submitted": time.time(),
            }
            print(f"  → submitted {src_path.name} ({req_id[:8]})")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ submit failed for {src_path.name}: {e}")
            return False
        return True

    while queue or in_flight:
        while queue and len(in_flight) < args.concurrency:
            submit_one(queue.pop(0))
        if not in_flight:
            break

        time.sleep(args.poll_interval)

        done_ids = []
        for req_id, info in list(in_flight.items()):
            try:
                s = status(req_id)
            except Exception as e:  # noqa: BLE001
                print(f"  ? status check failed for {req_id[:8]}: {e}")
                continue
            print(f"    {req_id[:8]} → {s} ({time.time() - info['submitted']:.0f}s)")
            if s == "COMPLETED":
                try:
                    res = fetch_result(req_id)
                    url = res["image"]["url"]
                    download(url, Path(info["out"]))
                    completed += 1
                    print(f"  ✓ {Path(info['out']).name} ({completed}/{len(sources)})")
                    with manifest_path.open("a") as f:
                        f.write(json.dumps({
                            "src": info["src"],
                            "out": info["out"],
                            "request_id": req_id,
                            "duration": time.time() - info["submitted"],
                            "model": MODEL_PATH,
                        }) + "\n")
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    print(f"  ✗ download failed for {req_id[:8]}: {e}")
                done_ids.append(req_id)
            elif s in {"FAILED", "ERROR"}:
                failed += 1
                done_ids.append(req_id)
                print(f"  ✗ {req_id[:8]} status={s}")
        for r in done_ids:
            del in_flight[r]

    print(f"\nDone. completed={completed} failed={failed} total={len(sources)}")


if __name__ == "__main__":
    main()
