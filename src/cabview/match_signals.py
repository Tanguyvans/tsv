"""Matche chaque détection YOLO avec les diagrammes SVG du catalogue Wikimedia.

Pipeline :
  1. Render les SVG du catalogue (FR/CH) → PNG sur fond blanc (cache disque)
  2. Embed chaque PNG avec CLIP → cache .npy
  3. Pour chaque détection : crop bbox depuis la frame, embed avec CLIP
  4. Calcul cosine similarity, garde top-K matches
  5. Sauvegarde JSON + génère HTML viewer

Usage:
    PYTHONPATH=. python src/cabview/match_signals.py \\
        --frames data/cabview/fr/frames_cabview/bordeaux_nantes_2023 \\
        --detections data/cabview/fr/detections/bordeaux_nantes_2023 \\
        --refs-dir data/signals_ref/fr_diagrams \\
        --conf-min 0.3
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

# Cairo libs (brew install cairo librsvg)
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

import cairosvg
import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def render_svg(svg_path: Path, out_path: Path, width: int = 224) -> None:
    """Render SVG → PNG sur fond blanc."""
    png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=width)
    img = Image.open(io.BytesIO(png_bytes))
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=92)


def render_all_svgs(refs_dir: Path, cache_dir: Path, width: int = 224) -> list[Path]:
    """Render tous les SVG du dossier vers cache_dir. Retourne la liste des PNG."""
    svgs = sorted(refs_dir.glob("*.svg"))
    pngs = []
    for s in tqdm(svgs, desc="render SVGs"):
        out = cache_dir / f"{s.stem}.jpg"
        if not out.exists():
            try:
                render_svg(s, out, width=width)
            except Exception as e:
                print(f"  err {s.name}: {e}")
                continue
        pngs.append(out)
    return pngs


@torch.no_grad()
def embed_images(model, preprocess, paths: list[Path], device: str,
                 batch: int = 32, label: str = "embed") -> np.ndarray:
    feats = []
    for i in tqdm(range(0, len(paths), batch), desc=label):
        chunk = paths[i:i + batch]
        imgs = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in chunk]
        ).to(device)
        f = model.encode_image(imgs)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def embed_pil_images(model, preprocess, imgs: list[Image.Image],
                     device: str, batch: int = 32) -> np.ndarray:
    feats = []
    for i in range(0, len(imgs), batch):
        chunk = imgs[i:i + batch]
        x = torch.stack([preprocess(im.convert("RGB")) for im in chunk]).to(device)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def crop_with_margin(frame: Image.Image, bbox: list[int], margin: float = 0.2) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx = int(w * margin)
    my = int(h * margin)
    fx1 = max(0, x1 - mx)
    fy1 = max(0, y1 - my)
    fx2 = min(frame.width, x2 + mx)
    fy2 = min(frame.height, y2 + my)
    return frame.crop((fx1, fy1, fx2, fy2))


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Matches signaux — {title}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 16px;
          background: #1a1a1a; color: #eee; }}
  h1 {{ font-size: 1.3rem; }}
  .stats {{ color: #999; font-size: 0.9rem; margin-bottom: 16px; }}
  .row {{ display: grid;
          grid-template-columns: 180px 60px repeat({k}, 1fr);
          gap: 8px; align-items: center; padding: 8px;
          background: #222; border-radius: 4px; margin-bottom: 8px; }}
  .row img {{ width: 100%; height: auto; max-height: 140px; object-fit: contain;
              background: white; border-radius: 4px; }}
  .det {{ background: #333 !important; }}
  .label {{ font-size: 0.7rem; color: #888; word-break: break-word; }}
  .score {{ font-size: 0.7rem; color: #00d068; font-weight: bold; }}
  .conf {{ font-size: 0.7rem; color: #aaa; text-align: center; }}
  .arrow {{ color: #555; font-size: 1.5rem; text-align: center; }}
</style>
</head>
<body>

<h1>{title} — top-{k} matches contre catalogue {region}</h1>
<div class="stats">{stats}</div>

{rows_html}

</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--refs-dir", type=Path, required=True,
                    help="data/signals_ref/fr_diagrams ou ch_diagrams")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--conf-min", type=float, default=0.3,
                    help="Ne matcher que les détections ≥ ce conf")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.2,
                    help="Marge autour de la bbox lors du crop")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"device: {device}")

    out_dir = args.out or args.detections.parent.parent / "matches" / args.detections.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Render SVGs
    refs_cache = args.refs_dir.parent / f"{args.refs_dir.name}_pngs"
    print(f"\n[1/4] Render SVGs → {refs_cache}")
    ref_pngs = render_all_svgs(args.refs_dir, refs_cache, width=224)
    print(f"  {len(ref_pngs)} SVG rendus")

    # 2. CLIP model + ref embeddings (cached)
    print("\n[2/4] CLIP model + embeddings refs")
    model, preprocess = clip.load("ViT-B/32", device=device)
    refs_emb_path = refs_cache / "_embeddings.npy"
    refs_names_path = refs_cache / "_names.json"
    if refs_emb_path.exists() and refs_names_path.exists():
        ref_emb = np.load(refs_emb_path)
        ref_names = json.loads(refs_names_path.read_text())
        if len(ref_names) == len(ref_pngs):
            print(f"  cache hit : {ref_emb.shape}")
        else:
            ref_emb = None
    else:
        ref_emb = None
    if ref_emb is None:
        ref_emb = embed_images(model, preprocess, ref_pngs, device, label="embed refs")
        np.save(refs_emb_path, ref_emb)
        ref_names = [p.stem for p in ref_pngs]
        refs_names_path.write_text(json.dumps(ref_names))
        print(f"  refs emb saved : {ref_emb.shape}")

    # 3. Pour chaque détection : crop + embed + top-K
    print("\n[3/4] Matching détections")
    json_files = sorted(args.detections.glob("*.json"))
    json_files = [f for f in json_files if f.name != "_summary.json"]

    all_matches = []  # [{frame, det_idx, score, top_k:[(svg, sim)]}]
    crops_dir = out_dir / "_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    for jf in tqdm(json_files, desc="matching"):
        data = json.loads(jf.read_text())
        dets = data.get("detections", [])
        if not dets:
            continue

        frame_path = args.frames / data["frame"]
        if not frame_path.exists():
            continue

        # Filter by min conf
        dets_filt = [(i, d) for i, d in enumerate(dets) if d["score"] >= args.conf_min]
        if not dets_filt:
            continue

        frame_img = Image.open(frame_path).convert("RGB")
        crops_pil = []
        crop_paths = []
        for i, d in dets_filt:
            crop = crop_with_margin(frame_img, d["bbox"], margin=args.margin)
            crop_path = crops_dir / f"{frame_path.stem}_d{i}.jpg"
            if not crop_path.exists():
                crop.save(crop_path, "JPEG", quality=85)
            crops_pil.append(crop)
            crop_paths.append(crop_path)

        crops_emb = embed_pil_images(model, preprocess, crops_pil, device)
        sims = crops_emb @ ref_emb.T  # (n_crops, n_refs)

        for k, ((i, d), sim_row, crop_path) in enumerate(zip(dets_filt, sims, crop_paths)):
            top_idx = np.argsort(-sim_row)[:args.top_k]
            top = [{"svg": ref_names[idx], "sim": float(sim_row[idx])}
                   for idx in top_idx]
            all_matches.append({
                "frame": data["frame"],
                "det_idx": i,
                "conf": d["score"],
                "bbox": d["bbox"],
                "crop_path": str(crop_path.relative_to(out_dir)),
                "top_k": top,
            })

    # Save JSON
    matches_json = out_dir / "matches.json"
    matches_json.write_text(json.dumps(all_matches, indent=2))
    print(f"  {len(all_matches)} matches sauvegardés → {matches_json}")

    # 4. HTML viewer
    print("\n[4/4] Génération viewer HTML")
    region = args.refs_dir.name.replace("_diagrams", "").upper()
    rows_html = []
    # tri par conf YOLO décroissant
    all_matches.sort(key=lambda m: -m["conf"])

    # path relatif vers les SVG sources pour les afficher dans HTML
    refs_rel = os.path.relpath(args.refs_dir.resolve(), out_dir.resolve())

    for m in all_matches:
        crop_html = (
            f'<img src="{m["crop_path"]}" alt="crop">'
            f'<div class="conf">YOLO {m["conf"]:.2f}</div>'
        )
        match_cells = []
        for tk in m["top_k"]:
            svg_rel = f"{refs_rel}/{tk['svg']}.svg"
            match_cells.append(
                f'<div><img src="{svg_rel}" alt="{tk["svg"]}">'
                f'<div class="label">{tk["svg"]}</div>'
                f'<div class="score">sim {tk["sim"]:.3f}</div></div>'
            )
        rows_html.append(
            f'<div class="row">'
            f'<div class="det">{crop_html}</div>'
            f'<div class="arrow">→</div>'
            f'{"".join(match_cells)}'
            f'</div>'
        )

    html = HTML_TEMPLATE.format(
        title=args.detections.name,
        region=region,
        k=args.top_k,
        stats=f"{len(all_matches)} détections (conf ≥ {args.conf_min}) "
              f"vs {len(ref_names)} signaux du catalogue",
        rows_html="\n".join(rows_html),
    )
    out_html = out_dir / "viewer.html"
    out_html.write_text(html)
    print(f"  → {out_html}")
    print(f"  open {out_html}")


if __name__ == "__main__":
    main()
