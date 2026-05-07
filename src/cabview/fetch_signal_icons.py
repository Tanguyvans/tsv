"""Télécharge les icônes de signaux ferroviaires depuis Wikimedia Commons.

Utilise l'API MediaWiki pour lister tous les fichiers d'une catégorie Commons
(récursivement sur les sous-catégories) et télécharger chaque fichier dans
data/signals_ref/{region}/.

Usage:
    PYTHONPATH=. python src/cabview/fetch_signal_icons.py --region fr
    PYTHONPATH=. python src/cabview/fetch_signal_icons.py --region ch
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

API = "https://commons.wikimedia.org/w/api.php"
UA = "tsv-research-bot/1.0 (railway signal dataset; contact: tanguyvans@gmail.com)"

REGIONS = {
    "fr": [
        "Diagrams of railway signals in France",
    ],
    "fr-photos": [
        "Railway signals in France",
    ],
    "ch": [
        "Diagrams of railway signals in Switzerland",
    ],
    "ch-photos": [
        "Railway signals in Switzerland",
    ],
}


def api_get(params: dict) -> dict:
    params = {**params, "format": "json"}
    qs = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
    req = Request(f"{API}?{qs}", headers={"User-Agent": UA})
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def list_category(cat: str, seen_cats: set[str]) -> tuple[list[str], list[str]]:
    """Retourne (files, subcategories) pour une catégorie donnée."""
    files, subcats = [], []
    cont = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{cat}",
            "cmlimit": 500,
            "cmtype": "file|subcat",
        }
        if cont:
            params["cmcontinue"] = cont
        data = api_get(params)
        for m in data.get("query", {}).get("categorymembers", []):
            title = m["title"]
            if title.startswith("File:"):
                files.append(title[5:])
            elif title.startswith("Category:"):
                sub = title[9:]
                if sub not in seen_cats:
                    subcats.append(sub)
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        time.sleep(0.1)
    return files, subcats


def collect_all_files(roots: list[str], max_depth: int = 2) -> list[str]:
    seen_cats: set[str] = set()
    all_files: set[str] = set()
    frontier = [(c, 0) for c in roots]
    while frontier:
        cat, depth = frontier.pop(0)
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        print(f"[depth={depth}] Category:{cat}")
        files, subcats = list_category(cat, seen_cats)
        all_files.update(files)
        if depth < max_depth:
            for sc in subcats:
                frontier.append((sc, depth + 1))
        time.sleep(0.2)
    return sorted(all_files)


def get_file_urls(filenames: list[str]) -> dict[str, str]:
    """Résout les URLs directes des fichiers via imageinfo (batch de 50)."""
    urls: dict[str, str] = {}
    for i in range(0, len(filenames), 50):
        batch = filenames[i : i + 50]
        params = {
            "action": "query",
            "titles": "|".join(f"File:{f}" for f in batch),
            "prop": "imageinfo",
            "iiprop": "url|mime",
        }
        data = api_get(params)
        pages = data.get("query", {}).get("pages", {})
        for p in pages.values():
            title = p.get("title", "")
            if title.startswith("File:"):
                name = title[5:]
                ii = p.get("imageinfo", [])
                if ii:
                    urls[name] = ii[0]["url"]
        time.sleep(0.1)
    return urls


def download(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": UA})
    last_err = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            return True
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)
            time.sleep(wait)
    raise last_err


def safe_filename(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=REGIONS.keys(), required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--ext", default="svg,png,jpg,jpeg",
                    help="Extensions à garder (csv)")
    args = ap.parse_args()

    out = args.out or Path(f"data/signals_ref/{args.region}")
    out.mkdir(parents=True, exist_ok=True)
    exts = {e.lower().strip() for e in args.ext.split(",")}

    roots = REGIONS[args.region]
    print(f"Collecte des fichiers pour la région {args.region}...")
    files = collect_all_files(roots, max_depth=args.max_depth)
    print(f"  {len(files)} fichiers trouvés (toutes extensions)")

    files = [f for f in files if f.rsplit(".", 1)[-1].lower() in exts]
    print(f"  {len(files)} après filtre extension {exts}")

    print("Résolution des URLs...")
    urls = get_file_urls(files)
    print(f"  {len(urls)} URLs résolues")

    manifest = []
    n_new = 0
    for i, (name, url) in enumerate(urls.items()):
        dest = out / safe_filename(name)
        try:
            new = download(url, dest)
            if new:
                n_new += 1
                print(f"  + [{i+1}/{len(urls)}] {dest.name}")
        except Exception as e:
            print(f"  ! {name}: {e}")
            continue
        manifest.append({"name": name, "url": url, "path": str(dest.relative_to(out))})
        time.sleep(0.5)

    (out / "_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nTerminé. {n_new} nouveaux fichiers. Manifest: {out}/_manifest.json")


if __name__ == "__main__":
    main()
